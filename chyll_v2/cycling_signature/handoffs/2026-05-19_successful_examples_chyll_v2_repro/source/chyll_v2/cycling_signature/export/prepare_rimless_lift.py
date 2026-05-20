"""Export CHyLL v2 rimless-wheel latent data for cycling signatures.

This writes the position and unit-tangent CSV files expected by
``time series/cycling_signature/run_cycling_signature.jl``. It is deliberately
standalone from ``src/``: the simulator and network classes are imported only
from ``chyll_v2``.

Usage:
    python chyll_v2/cycling_signature/export/prepare_rimless_lift.py
    julia --project="time series/cycling_signature" \
        "time series/cycling_signature/run_cycling_signature.jl" \
        --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
        --base continuous_lift_chyll_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import erf
from scipy.integrate import solve_ivp

# Make ``chyll_v2`` importable as a package.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.chyll_v2.config import CHyLLv2Config, make_default  # noqa: E402
from chyll_v2.chyll_v2.systems.rimless_wheel import RimlessWheel  # noqa: E402

try:  # noqa: E402
    import torch  # type: ignore
    from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised on lean Python installs.
    torch = None
    CHyLLv2Networks = None


LIMIT_CYCLE_IC = np.array([-0.2, 0.54], dtype=np.float64)


def load_config(config_path: Path | None) -> CHyLLv2Config:
    """Load the saved run config, falling back to the rimless default."""
    if config_path is None:
        return make_default("rimless_wheel")
    payload = json.loads(config_path.read_text())
    allowed = {f.name for f in fields(CHyLLv2Config)}
    payload = {k: v for k, v in payload.items() if k in allowed}
    return CHyLLv2Config(**payload)


def load_state_dict(path: Path):
    """Compatibility wrapper across PyTorch versions."""
    if torch is None:
        return load_torch_state_dict_numpy(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class StorageRef:
    def __init__(self, key: str, dtype: np.dtype, numel: int):
        self.key = key
        self.dtype = np.dtype(dtype)
        self.numel = numel


class TensorRef:
    def __init__(
        self,
        storage: StorageRef,
        storage_offset: int,
        size: tuple[int, ...],
        stride: tuple[int, ...],
    ):
        self.storage = storage
        self.storage_offset = storage_offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def rebuild_tensor_v2(
    storage: StorageRef,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks,
) -> TensorRef:
    return TensorRef(storage, storage_offset, tuple(size), tuple(stride))


def load_torch_state_dict_numpy(path: Path) -> OrderedDict[str, np.ndarray]:
    """Read a ``torch.save(state_dict)`` zip archive without importing torch.

    The CHyLL v2 checkpoints are plain float32 tensor state dicts. PyTorch
    stores them as a pickle with persistent references into ``data/<id>`` raw
    storage blobs. This small reader is intentionally narrow but enough for
    Linear-layer weights and biases.
    """
    import pickle

    class TorchStateUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            kind, storage_type, key, _location, numel = pid
            if kind != "storage":
                raise pickle.UnpicklingError(f"unsupported persistent id: {pid}")
            dtype = storage_type
            if dtype not in ("float32", np.float32):
                raise pickle.UnpicklingError(f"unsupported storage dtype: {dtype}")
            return StorageRef(str(key), np.float32, int(numel))

        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild_tensor_v2
            if module == "torch" and name == "FloatStorage":
                return "float32"
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            return super().find_class(module, name)

    with zipfile.ZipFile(path) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist() if "/" in name}
        if len(roots) != 1:
            raise ValueError(f"expected one root in checkpoint zip, got {roots}")
        root = roots.pop()
        refs = TorchStateUnpickler(
            archive.open(f"{root}/data.pkl")
        ).load()

        arrays: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, ref in refs.items():
            raw = archive.read(f"{root}/data/{ref.storage.key}")
            storage = np.frombuffer(raw, dtype=ref.storage.dtype, count=ref.storage.numel)
            itemsize = storage.dtype.itemsize
            view = np.lib.stride_tricks.as_strided(
                storage[ref.storage_offset:],
                shape=ref.size,
                strides=tuple(s * itemsize for s in ref.stride),
            )
            arrays[name] = np.array(view, copy=True)
    return arrays


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


class NumpyMLP:
    def __init__(self, state: dict[str, np.ndarray], prefix: str, linear_ids: Iterable[int]):
        self.layers = [
            (state[f"{prefix}.{idx}.weight"], state[f"{prefix}.{idx}.bias"])
            for idx in linear_ids
        ]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x.astype(np.float32, copy=False)
        for i, (weight, bias) in enumerate(self.layers):
            y = y @ weight.T + bias
            if i < len(self.layers) - 1:
                y = gelu(y)
        return y


class NumpyCHyLLv2Networks:
    def __init__(self, cfg: CHyLLv2Config, state: dict[str, np.ndarray]):
        self.cfg = cfg
        self.encoder_mlp = NumpyMLP(state, "encoder.mlp", (0, 2, 4, 6))
        self.vfield_mlp = NumpyMLP(state, "vfield.mlp", (0, 2, 4, 6))
        self.decoder_mlp = NumpyMLP(state, "decoder.mlp", (0, 2, 4, 6))

    def encoder(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        pad = np.zeros((*x.shape[:-1], self.cfg.latent_dim), dtype=np.float32)
        pad[..., : self.cfg.state_dim] = x
        return pad + self.encoder_mlp(x)

    def vfield(self, z: np.ndarray) -> np.ndarray:
        return self.vfield_mlp(z)

    def decoder(self, z: np.ndarray) -> np.ndarray:
        z = z.astype(np.float32, copy=False)
        return z[..., : self.cfg.state_dim] + self.decoder_mlp(z)


def simulate_limit_cycle(
    system: RimlessWheel,
    x0: np.ndarray,
    n_impacts: int,
    max_time: float,
    max_step: float,
    rtol: float,
    atol: float,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Simulate base-flow arcs and symbolic jump pairs."""
    state = np.array(x0, dtype=np.float64)
    segments: list[np.ndarray] = []
    jump_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_impacts):
        event = lambda t, y: system.is_guard_hit(t, y)
        event.terminal = True
        event.direction = system.guard_direction
        sol = solve_ivp(
            lambda t, y: system.base_vector_field(y),
            (0.0, max_time),
            state,
            method="RK45",
            events=event,
            dense_output=False,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )
        segments.append(sol.y.T.copy())
        if sol.status != 1:
            break
        x_minus = sol.y[:, -1].copy()
        x_plus = system.reset_map(x_minus)
        jump_pairs.append((x_minus, x_plus))
        state = x_plus

    return segments, jump_pairs


def finite_difference_tangents(z: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Forward-difference unit tangents with a small zero-step repair."""
    dz = np.diff(z, axis=0)
    dz = np.vstack([dz, dz[-1:]])
    return normalize_tangent_rows(dz, eps, "latent finite differences")


def tag_aware_finite_difference_tangents(
    z: np.ndarray,
    tags: list[tuple[str, int]],
    eps: float,
) -> tuple[np.ndarray, int]:
    """Forward-difference unit tangents computed inside each tagged piece.

    The exported lift is a concatenation of base-flow arcs and bridge
    interiors. Naive finite differences across tag boundaries create chords
    between distinct pieces, including the latent gluing leap. This routine
    differentiates only within maximal runs with the same tag; the last sample
    of each run copies the previous within-run difference.
    """
    if len(tags) != len(z):
        raise ValueError(f"expected {len(z)} tags, got {len(tags)}")

    dz = np.zeros_like(z)
    start = 0
    while start < len(z):
        end = start + 1
        while end < len(z) and tags[end] == tags[start]:
            end += 1

        if end - start >= 2:
            local = np.diff(z[start:end], axis=0)
            dz[start:end - 1] = local
            dz[end - 1] = local[-1]
        else:
            # Degenerate one-point pieces are not expected for the current
            # exporter, but leave them repairable by normalize_tangent_rows.
            dz[start] = 0.0
        start = end

    return normalize_tangent_rows(dz, eps, "tag-aware latent finite differences")


def normalize_tangent_rows(
    dz: np.ndarray,
    eps: float,
    label: str,
) -> tuple[np.ndarray, int]:
    """Normalize tangent rows, repairing rare zero rows from nearby samples."""
    norms = np.linalg.norm(dz, axis=1)
    tiny = norms < eps
    if tiny.any():
        good = np.where(~tiny)[0]
        if len(good) == 0:
            raise ValueError(f"all {label} are numerically zero")
        for i in np.where(tiny)[0]:
            prev = good[good < i]
            dz[i] = dz[prev[-1] if len(prev) else good[0]]
        norms = np.linalg.norm(dz, axis=1)
    return dz / norms[:, None], int(tiny.sum())


def model_call(fn, x: np.ndarray) -> np.ndarray:
    """Evaluate a torch or NumPy model function on a NumPy batch."""
    if torch is None:
        return fn(x)
    with torch.no_grad():
        return fn(torch.as_tensor(x, dtype=torch.float32)).cpu().numpy()


def vfield_tangents(vfield, z: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    """Unit tangents from the learned latent vector field V_theta(z)."""
    vel = model_call(vfield, z)
    norms = np.linalg.norm(vel, axis=1)
    tiny = norms < eps
    if tiny.any():
        good = np.where(~tiny)[0]
        if len(good) == 0:
            raise ValueError("all learned latent velocities are numerically zero")
        for i in np.where(tiny)[0]:
            prev = good[good < i]
            vel[i] = vel[prev[-1] if len(prev) else good[0]]
        norms = np.linalg.norm(vel, axis=1)
    return vel / norms[:, None], int(tiny.sum())


def summarize(values: np.ndarray) -> str:
    return (
        f"mean={values.mean():.3e} median={np.median(values):.3e} "
        f"p95={np.percentile(values, 95):.3e} max={values.max():.3e}"
    )


def write_report(
    path: Path,
    *,
    model_path: Path,
    cfg: CHyLLv2Config,
    z: np.ndarray,
    x_aug: np.ndarray,
    x_rec: np.ndarray,
    tags: list[tuple[str, int]],
    tangent_source: str,
    tiny_tangent_count: int,
    gluing_errors: np.ndarray,
) -> None:
    arc_mask = np.array([tag[0] == "arc" for tag in tags], dtype=bool)
    bridge_mask = ~arc_mask
    full_err = np.linalg.norm(x_rec - x_aug, axis=1)

    lines = [
        "CHyLL v2 rimless cycling-signature export",
        "=" * 48,
        f"model: {model_path}",
        f"latent_dim: {cfg.latent_dim}",
        f"lift shape: {z.shape}",
        f"tangent_source: {tangent_source}",
        f"tiny tangent repairs: {tiny_tangent_count}",
        "",
        "Reconstruction D(E(x,s)) vs augmented input",
        f"  all:     {summarize(full_err)}",
        f"  arcs:    {summarize(full_err[arc_mask])}",
        f"  bridges: {summarize(full_err[bridge_mask])}",
        "",
        "Symbolic gluing errors ||E(g,1)-E(r(g),0)||",
        f"  {summarize(gluing_errors)}",
        "",
        "Latent coordinate ranges",
    ]
    coord_min = z.min(axis=0)
    coord_max = z.max(axis=0)
    for j, (lo, hi) in enumerate(zip(coord_min, coord_max)):
        lines.append(f"  z{j}: [{lo:.6g}, {hi:.6g}] span={hi - lo:.6g}")

    path.write_text("\n".join(lines) + "\n")


def encode_lift(
    encoder,
    segments: Iterable[np.ndarray],
    jump_pairs: list[tuple[np.ndarray, np.ndarray]],
    n_s: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """Encode base arcs plus mapping-cylinder bridge interiors."""
    s_vals = np.linspace(0.0, 1.0, n_s + 2, dtype=np.float64)[1:-1]
    pieces_z: list[np.ndarray] = []
    pieces_x: list[np.ndarray] = []
    tags: list[tuple[str, int]] = []

    for j, seg in enumerate(segments):
        x_arc = np.hstack([seg, np.zeros((len(seg), 1), dtype=np.float64)])
        z_arc = model_call(encoder, x_arc)
        pieces_x.append(x_arc)
        pieces_z.append(z_arc)
        tags.extend([("arc", j)] * len(x_arc))

        if j < len(jump_pairs):
            x_minus, _ = jump_pairs[j]
            x_bridge = np.hstack(
                [
                    np.tile(x_minus, (n_s, 1)),
                    s_vals[:, None],
                ]
            )
            z_bridge = model_call(encoder, x_bridge)
            pieces_x.append(x_bridge)
            pieces_z.append(z_bridge)
            tags.extend([("bridge", j)] * len(x_bridge))

    return np.vstack(pieces_z), np.vstack(pieces_x), tags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "runs" / "rimless_wheel" / "model.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "runs" / "rimless_wheel" / "config.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "chyll_v2"
            / "cycling_signature"
            / "data"
            / "rimless_wheel"
        ),
    )
    parser.add_argument("--base", default="continuous_lift_chyll_v2")
    parser.add_argument("--n-impacts", type=int, default=5)
    parser.add_argument("--n-s", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=50.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--tangent-eps", type=float, default=1e-12)
    parser.add_argument(
        "--tangent-source",
        choices=("diff", "vfield"),
        default="diff",
        help="Use finite differences of E(X') or the learned latent vector field.",
    )
    parser.add_argument(
        "--tangent-mode",
        choices=("naive", "tagaware"),
        default="naive",
        help=(
            "Finite-difference convention for --tangent-source diff. "
            "'tagaware' differentiates only within arc/bridge pieces."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.device = "cpu"
    system = RimlessWheel()

    state_dict = load_state_dict(args.model)
    if torch is None:
        if cfg.use_sin_last_layer:
            raise NotImplementedError(
                "NumPy fallback does not implement use_sin_last_layer=True; "
                "run with a Python environment that has torch installed."
            )
        nets = NumpyCHyLLv2Networks(cfg, state_dict)
        print("torch not available; using NumPy checkpoint reader")
    else:
        nets = CHyLLv2Networks(cfg)
        nets.load_state_dict(state_dict)
        nets.eval()

    segments, jump_pairs = simulate_limit_cycle(
        system=system,
        x0=LIMIT_CYCLE_IC,
        n_impacts=args.n_impacts,
        max_time=args.max_time,
        max_step=args.max_step,
        rtol=cfg.sim_rtol,
        atol=cfg.sim_atol,
    )
    if not jump_pairs:
        raise RuntimeError("rimless limit-cycle simulation produced no impacts")

    z, x_aug, tags = encode_lift(nets.encoder, segments, jump_pairs, args.n_s)
    if args.tangent_source == "vfield":
        tangents, tiny_count = vfield_tangents(nets.vfield, z, args.tangent_eps)
    elif args.tangent_mode == "tagaware":
        tangents, tiny_count = tag_aware_finite_difference_tangents(
            z,
            tags,
            args.tangent_eps,
        )
    else:
        tangents, tiny_count = finite_difference_tangents(z, args.tangent_eps)

    x_rec = model_call(nets.decoder, z)
    g_aug = np.vstack([np.concatenate([g, [1.0]]) for g, _ in jump_pairs])
    rg_aug = np.vstack([np.concatenate([rg, [0.0]]) for _, rg in jump_pairs])
    zg = model_call(nets.encoder, g_aug)
    zrg = model_call(nets.encoder, rg_aug)
    gluing_errors = np.linalg.norm(zg - zrg, axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base
    np.save(args.out_dir / f"{base}.npy", z)
    np.save(args.out_dir / f"{base}_tangents.npy", tangents)
    np.savetxt(args.out_dir / f"{base}_positions.csv", z, delimiter=" ")
    np.savetxt(args.out_dir / f"{base}_tangents.csv", tangents, delimiter=" ")
    write_report(
        args.out_dir / f"report_{base}.txt",
        model_path=args.model,
        cfg=cfg,
        z=z,
        x_aug=x_aug,
        x_rec=x_rec,
        tags=tags,
        tangent_source=(
            args.tangent_source
            if args.tangent_source == "vfield"
            else f"{args.tangent_source}:{args.tangent_mode}"
        ),
        tiny_tangent_count=tiny_count,
        gluing_errors=gluing_errors,
    )

    print(f"simulated {len(segments)} arcs, {len(jump_pairs)} jumps")
    print(f"encoded lift: {z.shape}")
    tangent_label = (
        args.tangent_source
        if args.tangent_source == "vfield"
        else f"{args.tangent_source}:{args.tangent_mode}"
    )
    print(f"tangent source: {tangent_label}; tiny repairs: {tiny_count}")
    print(f"wrote {args.out_dir / (base + '_positions.csv')}")
    print(f"wrote {args.out_dir / (base + '_tangents.csv')}")
    print(f"wrote {args.out_dir / ('report_' + base + '.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
