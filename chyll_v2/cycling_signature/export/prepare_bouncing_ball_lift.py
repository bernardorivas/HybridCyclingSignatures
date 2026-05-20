"""Export CHyLL v2 bouncing-ball latent data for cycling signatures.

Parallel of ``prepare_rimless_lift.py``: same shared helpers, same output
naming convention, same tangent options (``diff[:naive|tagaware]`` or
``vfield``), same Julia interop format. Only the system class and the
default initial condition change.

Usage:
    python chyll_v2/cycling_signature/export/prepare_bouncing_ball_lift.py
    python chyll_v2/cycling_signature/export/prepare_bouncing_ball_lift.py \
        --tangent-source vfield --base continuous_lift_chyll_v2_bb_wv1_twoseam_vfield
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chyll_v2.chyll_v2.config import make_default  # noqa: E402
from chyll_v2.chyll_v2.systems.bouncing_ball import BouncingBall  # noqa: E402
from chyll_v2.cycling_signature.export.prepare_rimless_lift import (  # noqa: E402
    encode_lift,
    finite_difference_tangents,
    load_config,
    load_state_dict,
    model_call,
    normalize_tangent_rows,
    simulate_limit_cycle,
    tag_aware_finite_difference_tangents,
    vfield_tangents,
    write_report,
)

try:  # noqa: E402
    import torch  # type: ignore
    from chyll_v2.chyll_v2.networks import CHyLLv2Networks  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    CHyLLv2Networks = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "runs" / "bouncing_ball_phaseB_finetune" / "model.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "runs" / "bouncing_ball_phaseB_finetune" / "config.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "bouncing_ball",
    )
    parser.add_argument("--base", default="continuous_lift_chyll_v2_bb_phaseB")
    parser.add_argument("--n-impacts", type=int, default=5)
    parser.add_argument("--n-s", type=int, default=50)
    parser.add_argument("--max-time", type=float, default=10.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--tangent-eps", type=float, default=1e-12)
    parser.add_argument(
        "--tangent-source", choices=("diff", "vfield"), default="diff"
    )
    parser.add_argument(
        "--tangent-mode", choices=("naive", "tagaware"), default="tagaware"
    )
    # Canonical bouncing-ball drop: h0 = 1, v0 = 0.
    parser.add_argument("--h0", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="coefficient of restitution; must match the trained model")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config is not None else make_default("bouncing_ball")
    cfg.device = "cpu"

    system = BouncingBall(alpha=args.alpha)
    state_dict = load_state_dict(args.model)
    if torch is None:
        raise RuntimeError(
            "torch not available; NumPy fallback not wired through this exporter"
        )
    nets = CHyLLv2Networks(cfg)
    nets.load_state_dict(state_dict)
    nets.eval()

    segments, jump_pairs = simulate_limit_cycle(
        system=system,
        x0=np.array([args.h0, args.v0], dtype=np.float64),
        n_impacts=args.n_impacts,
        max_time=args.max_time,
        max_step=args.max_step,
        rtol=cfg.sim_rtol,
        atol=cfg.sim_atol,
    )
    if not jump_pairs:
        raise RuntimeError("bouncing-ball simulation produced no impacts")

    z, x_aug, tags = encode_lift(nets.encoder, segments, jump_pairs, args.n_s)

    if args.tangent_source == "vfield":
        tangents, tiny_count = vfield_tangents(nets.vfield, z, args.tangent_eps)
    elif args.tangent_mode == "tagaware":
        tangents, tiny_count = tag_aware_finite_difference_tangents(
            z, tags, args.tangent_eps
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
    print(f"tangent source: "
          f"{args.tangent_source if args.tangent_source == 'vfield' else f'{args.tangent_source}:{args.tangent_mode}'}; "
          f"tiny repairs: {tiny_count}")
    print(f"wrote {args.out_dir / base}_{{positions,tangents}}.csv")
    print(f"wrote {args.out_dir / f'report_{base}.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
