"""
Off-limit-cycle rimless evaluation — does D(E(x)) track trajectories that
start far from the stable gait limit cycle, or does the encoder only get
lucky on the LC because it's the dominant training distribution?

Picks several ICs spread across the basin of attraction (low / high
omega, post-reset / mid-arc theta), simulates each, and overlays true and
reconstructed phase trajectories on a single (theta, omega) plot, plus
per-IC error stats.

Outputs:
    figures/rimless_wheel/fig_rimless_decoded_rollout_basin.png
    data/rimless_wheel/report_decoded_rollout_basin_metrics.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
TS_ROOT = Path(__file__).parent.parent / "time series"
sys.path.insert(0, str(TS_ROOT))
sys.path.insert(0, str(TS_ROOT / "rimless wheel"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config, SystemType
config.system_type = SystemType.RIMLESS_WHEEL
config.device = "cpu"
from networks import SuspensionNetworks  # noqa: E402
from simulate import simulate_rimless_wheel  # noqa: E402


ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "runs" / "rimless_wheel" / "model.pt"
FIG_DIR = ROOT / "figures" / "rimless_wheel"
DATA_DIR = ROOT / "data" / "rimless_wheel"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# A spread of ICs across the basin. Limit-cycle post-reset state is roughly
# (theta, omega) ~ (-0.2, 0.54). These ICs include points well above and
# below the LC fixed point, and not all start at the reset boundary.
BASIN_ICS = [
    ("LC_post_reset",   np.array([-0.2,  0.54])),  # on the limit cycle
    ("low_omega",       np.array([-0.2,  0.30])),  # slow, just enough to roll
    ("fast_post_reset", np.array([-0.2,  1.20])),  # fast at the reset
    ("mid_arc_slow",    np.array([ 0.10, 0.45])),  # interior of an arc, slow
    ("mid_arc_fast",    np.array([ 0.10, 1.50])),  # interior, fast
    ("near_guard",      np.array([ 0.50, 0.80])),  # close to the guard
]
N_IMPACTS = 6


def _extend_with_bridges(segments, jump_pairs, n_bridge=50):
    """Build a (T, 3) raw trajectory with explicit bridge samples at s in (0,1)."""
    s_vals = np.linspace(0.0, 1.0, n_bridge + 2)[1:-1]
    pieces = []
    tags = []
    for j, seg in enumerate(segments):
        inp_arc = np.hstack([seg, np.zeros((len(seg), 1))])
        pieces.append(inp_arc)
        tags.extend([("arc", j)] * len(seg))
        if j < len(jump_pairs):
            xm, _ = jump_pairs[j]
            inp_br = np.hstack([np.tile(xm, (n_bridge, 1)), s_vals[:, None]])
            pieces.append(inp_br)
            tags.extend([("bridge", j)] * n_bridge)
    X = np.concatenate(pieces, axis=0)
    return X, tags


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: missing {MODEL_PATH}. Train first.")
        sys.exit(1)

    net = SuspensionNetworks()
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    net.eval()
    E, D = net.E, net.D

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(BASIN_ICS) - 1, 1)) for i in range(len(BASIN_ICS))]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    ax.axvline(config.theta_guard, color="#888", linestyle="--", lw=0.8,
               label=r"$\theta_{\mathrm{guard}}$")
    ax.axvline(config.theta_reset, color="#888", linestyle=":", lw=0.8,
               label=r"$\theta_{\mathrm{reset}}$")

    report = []
    report.append("Rimless-wheel off-limit-cycle decoded-rollout evaluation")
    report.append("=" * 56)
    report.append(f"model: {MODEL_PATH}")
    report.append(f"n_impacts per IC: {N_IMPACTS}")
    report.append("")
    report.append(f"{'name':<20} {'IC':<26} {'arcs':>5} {'arc_err mean':>13} "
                  f"{'arc_err max':>13} {'bridge_err max':>15}")

    for (name, ic), color in zip(BASIN_ICS, colors):
        segments, jump_pairs = simulate_rimless_wheel(ic[0], ic[1], n_impacts=N_IMPACTS)
        if not segments:
            report.append(f"{name:<20} {str(ic):<26} (no rollout — guard never reached)")
            continue
        X_raw, tags = _extend_with_bridges(segments, jump_pairs, n_bridge=50)
        with torch.no_grad():
            x_in = torch.as_tensor(X_raw, dtype=torch.float32)
            x_rt = D(E(x_in)).numpy()

        arc_mask = np.array([t[0] == "arc" for t in tags])
        err_state = np.linalg.norm(x_rt[:, :2] - X_raw[:, :2], axis=1)
        arc_err = err_state[arc_mask]
        bridge_err = err_state[~arc_mask] if (~arc_mask).any() else np.array([0.0])

        # Plot true (solid line, low alpha to suggest path continuity within
        # arcs, broken across jumps) and learned (dashed) on the arc samples.
        for j, seg in enumerate(segments):
            ax.plot(seg[:, 0], seg[:, 1], '-', color=color, lw=1.4, alpha=0.75)
        ax.plot(x_rt[arc_mask, 0], x_rt[arc_mask, 1],
                '.', color=color, ms=1.6, alpha=0.55)
        # Marker at the IC for clarity.
        ax.plot(ic[0], ic[1], marker='o', color=color, ms=8,
                markeredgecolor='k', markeredgewidth=0.6, label=name, zorder=10)

        report.append(
            f"{name:<20} {str(ic):<26} {len(segments):>5}  "
            f"{arc_err.mean():>13.3e} {arc_err.max():>13.3e} {bridge_err.max():>15.3e}"
        )
        print(f"{name}: {len(segments)} arcs, arc_err mean={arc_err.mean():.3e}, "
              f"arc max={arc_err.max():.3e}, bridge max={bridge_err.max():.3e}")

    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_title("Off-limit-cycle rollouts: true (solid) vs $D(E(x))$ (dots)")
    ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_phase = FIG_DIR / "fig_rimless_decoded_rollout_basin.png"
    fig.savefig(out_phase, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_phase}")

    txt = "\n".join(report) + "\n"
    out_report = DATA_DIR / "report_decoded_rollout_basin_metrics.txt"
    out_report.write_text(txt, encoding="utf-8")
    print(f"wrote {out_report}")


if __name__ == "__main__":
    main()
