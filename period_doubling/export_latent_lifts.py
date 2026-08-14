"""Export latent suspension lifts of the compass-gait cascade trajectories.

Matched-design counterpart of the raw-state negative control: the SAME
fixed-time-span trajectories (data/compass_gait/compass_{regime}.npz) are
lifted into the continuous latent suspension space of the per-slope trained
CHyLL v2 models. Flow samples enter as augmented states (x, s=0); at each
recorded impact a mapping-cylinder bridge (jump_minus, s) for s in (0, 1) is
inserted, so the lift is a single continuous polyline in R^latent_dim when
the learned gluing E(g, 1) = E(r(g), 0) holds.

Time convention: every exported sample (arc or bridge) costs one dt of
suspension time, so a subsegment of length L spans L * dt * stride
suspension-seconds. Bridge sample count n_s is chosen so bridge point
spacing matches the median arc step in latent space (override with --n-s).

Tangents: primary source is the encoder pushforward (JVP) of the physical
unit tangents -- J_E(x,0) @ (v, 0) on arcs, J_E(g,s) @ e_s on bridges --
normalized in latent space. A tag-aware finite-difference tangent set is
also written as a cross-check.

Usage:
    python export_latent_lifts.py                  # all five regimes
    python export_latent_lifts.py --regimes period2,chaos
    python export_latent_lifts.py --n-s 40
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
CODE_ROOT = PKG_DIR.parent
sys.path.insert(0, str(CODE_ROOT))

import torch  # noqa: E402

from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # noqa: E402
from chyll_v2.cycling_signature.export.prepare_rimless_lift import (  # noqa: E402
    load_config,
    load_state_dict,
    normalize_tangent_rows,
    tag_aware_finite_difference_tangents,
)
from export_lifts import load_npz  # noqa: E402


REGIMES = ["period1", "period2", "period4", "period8", "chaos"]

# Per-regime trained models. period1 uses the phi=0.07 rad (4.0107 deg) model
# on phi=4.00 deg data -- the only period-1 model available; the slope
# mismatch is 0.0002 rad and is recorded in the report.
MODEL_MAP = {
    "period1": "compass_gait_phi007",
    "period2": "compass_gait_phi_1_4.75deg",
    "period4": "compass_gait_phi_2_5deg",
    "period8": "compass_gait_phi_3_5.02deg",
    "chaos": "compass_gait_phi_4_cloud_5.2deg",
}


def load_networks(run_dir: Path, untrained: bool = False) -> CHyLLv2Networks:
    cfg = load_config(run_dir / "config.json")
    cfg.device = "cpu"
    if untrained:
        torch.manual_seed(0)
        nets = CHyLLv2Networks(cfg)
    else:
        nets = CHyLLv2Networks(cfg)
        nets.load_state_dict(load_state_dict(run_dir / "model.pt"))
    nets.eval()
    return nets


def encode(nets: CHyLLv2Networks, x_aug: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = nets.encoder(torch.as_tensor(x_aug, dtype=torch.float32))
    return z.numpy().astype(np.float64)


def decode(nets: CHyLLv2Networks, z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = nets.decoder(torch.as_tensor(z, dtype=torch.float32))
    return x.numpy().astype(np.float64)


def encoder_jvp(nets: CHyLLv2Networks, x_aug: np.ndarray, v_aug: np.ndarray) -> np.ndarray:
    """Pushforward J_E(x) @ v, row-wise over a batch."""
    xt = torch.as_tensor(x_aug, dtype=torch.float32)
    vt = torch.as_tensor(v_aug, dtype=torch.float32)
    _, jv = torch.func.jvp(lambda a: nets.encoder(a), (xt,), (vt,))
    return jv.detach().numpy().astype(np.float64)


def build_augmented_stream(ts, n_s: int):
    """Interleave arc samples (x, 0) with bridge samples (g_j, s_i).

    Returns (x_aug, v_aug, tags, piece_kind) where v_aug is the physical
    tangent direction in augmented coordinates (arcs: (v, 0); bridges: e_s),
    tags is the (kind, piece_index) list for tag-aware finite differences,
    and piece_kind is a uint8 array (0 = arc sample, 1 = bridge sample).
    """
    x, v, t = ts.x, ts.v, ts.t
    impact_times = ts.impact_times
    jump_minus = ts.jump_minus
    n, d = x.shape
    k = len(impact_times)

    s_vals = np.linspace(0.0, 1.0, n_s + 2)[1:-1]
    e_s = np.zeros(d + 1)
    e_s[d] = 1.0

    # Sample at t == impact_time belongs to the ending arc (pre-impact state).
    arc_of_sample = np.searchsorted(impact_times, t, side="left")

    pieces_x, pieces_v, tags, kinds = [], [], [], []
    for j in range(k + 1):
        sel = arc_of_sample == j
        if sel.any():
            xa = np.hstack([x[sel], np.zeros((sel.sum(), 1))])
            va = np.hstack([v[sel], np.zeros((sel.sum(), 1))])
            pieces_x.append(xa)
            pieces_v.append(va)
            tags.extend([("arc", j)] * sel.sum())
            kinds.extend([0] * sel.sum())
        if j < k:
            g = jump_minus[j]
            xb = np.hstack([np.tile(g, (n_s, 1)), s_vals[:, None]])
            vb = np.tile(e_s, (n_s, 1))
            pieces_x.append(xb)
            pieces_v.append(vb)
            tags.extend([("bridge", j)] * n_s)
            kinds.extend([1] * n_s)

    return (
        np.vstack(pieces_x),
        np.vstack(pieces_v),
        tags,
        np.array(kinds, dtype=np.uint8),
    )


def auto_n_s(nets, ts, probe_pts: int = 64) -> tuple[int, dict]:
    """Choose n_s so bridge spacing matches the median arc step in latent space."""
    z_arc = encode(nets, np.hstack([ts.x, np.zeros((len(ts.x), 1))]))
    arc_steps = np.linalg.norm(np.diff(z_arc, axis=0), axis=1)
    # Steps across an impact are not arc steps; drop the largest K of them.
    k = len(ts.impact_times)
    if k > 0:
        arc_steps = np.sort(arc_steps)[: len(arc_steps) - k]
    med_step = float(np.median(arc_steps))

    s_probe = np.linspace(0.0, 1.0, probe_pts)
    lengths = np.empty(k)
    for j in range(k):
        g = ts.jump_minus[j]
        xb = np.hstack([np.tile(g, (probe_pts, 1)), s_probe[:, None]])
        zb = encode(nets, xb)
        lengths[j] = np.linalg.norm(np.diff(zb, axis=0), axis=1).sum()

    med_len = float(np.median(lengths)) if k else 0.0
    n_s = int(np.clip(np.ceil(med_len / med_step), 8, 200)) if k else 0
    stats = {
        "median_arc_step": med_step,
        "median_bridge_length": med_len,
        "max_bridge_length": float(lengths.max()) if k else 0.0,
        "auto_n_s": n_s,
    }
    return n_s, stats


def summarize(values: np.ndarray) -> str:
    return (
        f"mean={values.mean():.3e} median={np.median(values):.3e} "
        f"p95={np.percentile(values, 95):.3e} max={values.max():.3e}"
    )


def export_regime(regime: str, data_dir: Path, out_dir: Path, runs_dir: Path,
                  n_s_arg: int | None, untrained: bool = False) -> dict:
    npz_path = data_dir / f"compass_{regime}.npz"
    ts = load_npz(npz_path)
    run_dir = runs_dir / MODEL_MAP[regime]
    nets = load_networks(run_dir, untrained=untrained)

    n_s_auto, step_stats = auto_n_s(nets, ts)
    n_s = n_s_arg if n_s_arg is not None else n_s_auto

    x_aug, v_aug, tags, kinds = build_augmented_stream(ts, n_s)
    z = encode(nets, x_aug)

    # Primary tangents: encoder pushforward of physical directions.
    jv = encoder_jvp(nets, x_aug, v_aug)
    tangents, tiny_jvp = normalize_tangent_rows(jv, 1e-12, "JVP pushforward tangents")

    # Cross-check tangents: tag-aware finite differences.
    tangents_fd, tiny_fd = tag_aware_finite_difference_tangents(z, tags, 1e-12)
    cos = np.abs(np.sum(tangents * tangents_fd, axis=1))
    arc_mask = kinds == 0

    # Diagnostics.
    dz = np.linalg.norm(np.diff(z, axis=0), axis=1)
    x_rec = decode(nets, z)
    rec_err = np.linalg.norm(x_rec - x_aug, axis=1)

    g_aug = np.hstack([ts.jump_minus, np.ones((len(ts.jump_minus), 1))])
    rg_aug = np.hstack([ts.jump_plus, np.zeros((len(ts.jump_plus), 1))])
    gluing_errors = np.linalg.norm(encode(nets, g_aug) - encode(nets, rg_aug), axis=1)

    dt = float(ts.meta["dt"])
    n_total = len(z)
    t_susp = np.arange(n_total) * dt

    meta = dict(ts.meta)
    meta.update(
        {
            "lift": "chyll_v2_latent_suspension"
            + ("_untrained" if untrained else ""),
            "model_run": MODEL_MAP[regime],
            "model_path": str(run_dir / "model.pt"),
            "latent_dim": z.shape[1],
            "n_s": n_s,
            "n_s_auto": n_s_auto,
            "n_arc_samples": int(arc_mask.sum()),
            "n_bridge_samples": int((~arc_mask).sum()),
            "n_impacts_bridged": len(ts.impact_times),
            "tangent_source": "encoder_jvp",
            "suspension_time_convention": "every sample costs dt",
            "dt": dt,
        }
    )
    if regime == "period1":
        meta["slope_note"] = (
            "model trained at phi=0.07 rad (4.0107 deg), data at 4.00 deg; "
            "mismatch 2.0e-4 rad"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"compass_{regime}"
    np.savetxt(out_dir / f"{base}_positions.csv", z, delimiter=" ")
    np.savetxt(out_dir / f"{base}_tangents.csv", tangents, delimiter=" ")
    np.savetxt(out_dir / f"{base}_tangents_fdta.csv", tangents_fd, delimiter=" ")
    np.savez_compressed(
        out_dir / f"{base}.npz",
        t=t_susp,
        x=z,
        v=tangents,
        piece_kind=kinds,
        meta_json=json.dumps(meta, default=str),
    )

    report = [
        f"Latent suspension lift: compass_{regime}",
        "=" * 56,
        f"source npz: {npz_path}",
        f"model: {run_dir}",
        f"latent_dim: {z.shape[1]}  lift shape: {z.shape}",
        f"arc samples: {int(arc_mask.sum())}  bridge samples: {int((~arc_mask).sum())} "
        f"({len(ts.impact_times)} impacts x n_s={n_s}, auto suggestion {n_s_auto})",
        f"median arc step: {step_stats['median_arc_step']:.4e}  "
        f"median bridge length: {step_stats['median_bridge_length']:.4e}",
        "",
        "Consecutive latent gaps (pre-stride)",
        f"  all:   {summarize(dz)}",
        "",
        "Symbolic gluing errors ||E(g,1) - E(r(g),0)|| over all impacts",
        f"  {summarize(gluing_errors)}",
        "",
        "Reconstruction D(E(x,s)) vs augmented input",
        f"  arcs:    {summarize(rec_err[arc_mask])}",
        f"  bridges: {summarize(rec_err[~arc_mask])}",
        "",
        "JVP vs tag-aware FD tangent |cos| agreement",
        f"  arcs:    {summarize(cos[arc_mask])}",
        f"  bridges: {summarize(cos[~arc_mask])}",
        f"  (tiny repairs: jvp={tiny_jvp} fd={tiny_fd})",
        "",
        "Latent coordinate ranges",
    ]
    for jdim in range(z.shape[1]):
        lo, hi = z[:, jdim].min(), z[:, jdim].max()
        report.append(f"  z{jdim}: [{lo:.6g}, {hi:.6g}] span={hi - lo:.6g}")
    if "slope_note" in meta:
        report.append("")
        report.append(f"NOTE: {meta['slope_note']}")
    (out_dir / f"report_{base}.txt").write_text("\n".join(report) + "\n")

    return {
        "regime": regime,
        "n_total": n_total,
        "n_s": n_s,
        "gluing_max": float(gluing_errors.max()),
        "gap_max": float(dz.max()),
        "cos_min_arc": float(cos[arc_mask].min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PKG_DIR / "data" / "compass_gait")
    parser.add_argument("--out-dir", type=Path, default=PKG_DIR / "data" / "compass_gait_latent")
    parser.add_argument("--runs-dir", type=Path, default=CODE_ROOT / "chyll_v2" / "runs")
    parser.add_argument("--regimes", default="all")
    parser.add_argument("--n-s", type=int, default=None,
                        help="bridge samples per impact (default: auto per regime)")
    parser.add_argument("--untrained", action="store_true",
                        help="use a randomly initialized (untrained) network as control")
    args = parser.parse_args()

    regimes = REGIMES if args.regimes == "all" else args.regimes.split(",")
    for r in regimes:
        if r not in REGIMES:
            raise SystemExit(f"unknown regime: {r}")

    for regime in regimes:
        info = export_regime(regime, args.data_dir, args.out_dir, args.runs_dir,
                             args.n_s, untrained=args.untrained)
        print(
            f"{regime}: {info['n_total']} samples (n_s={info['n_s']}), "
            f"max gluing err {info['gluing_max']:.3e}, max gap {info['gap_max']:.3e}, "
            f"min arc |cos(jvp,fd)| {info['cos_min_arc']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
