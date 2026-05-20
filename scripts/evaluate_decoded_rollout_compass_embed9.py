"""
Evaluation for experiment `embed9`: loads runs/compass_gait/model_embed9.pt,
runs the same held-out rollout protocol as the baseline evaluator, writes to
new experimental paths.

Does NOT touch baseline files:
    figures/compass_gait/fig_compass_decoded_rollout_phase_embed9.png
    figures/compass_gait/fig_compass_decoded_rollout_traces_embed9.png
    data/compass_gait/report_decoded_rollout_metrics_embed9.txt
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
TS_ROOT = Path(__file__).parent.parent / "time series"
sys.path.insert(0, str(TS_ROOT))
sys.path.insert(0, str(TS_ROOT / "compass gait"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config, SystemType
config.system_type = SystemType.COMPASS_GAIT
config.device = "cpu"
config.embed_extra = 4        # <-- must match the trained model's embed_dim
from networks import SuspensionNetworks  # noqa: E402
from system import CompassGaitHybridSystem  # noqa: E402
from simulate import simulate_compass_gait, LIMIT_CYCLE_IC, PHI  # noqa: E402


ROOT = Path(__file__).parent.parent
EXP_TAG = "embed9"
MODEL_PATH = ROOT / "runs" / "compass_gait" / f"model_{EXP_TAG}.pt"
FIG_DIR = ROOT / "figures" / "compass_gait"
DATA_DIR = ROOT / "data" / "compass_gait"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extend_with_bridges(segments, jump_pairs, n_bridge=50):
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

    print(f"Loading {MODEL_PATH}")
    print(f"  embed_dim = {config.embed_dim}  (embed_extra={config.embed_extra})")
    net = SuspensionNetworks()
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    net.eval()
    E, D, F = net.E, net.D, net.F

    rng = np.random.default_rng(2026)
    perturbation = rng.normal(0, 0.02, size=4)
    ic = LIMIT_CYCLE_IC + perturbation
    print(f"Simulating rollout from IC={ic}")
    segments, jump_pairs = simulate_compass_gait(ic, n_impacts=8)
    print(f"  {len(segments)} arcs, {len(jump_pairs)} jumps")

    X_raw, tags = _extend_with_bridges(segments, jump_pairs, n_bridge=50)
    print(f"  extended trajectory: {X_raw.shape}")

    # Round-trip D(E(x))
    with torch.no_grad():
        x_in = torch.as_tensor(X_raw, dtype=torch.float32)
        z = E(x_in)
        x_rt = D(z).numpy()

    err_state = np.linalg.norm(x_rt[:, :4] - X_raw[:, :4], axis=1)
    err_s = np.abs(x_rt[:, 4] - X_raw[:, 4])

    # Semiflow long-horizon rollout (reported but not plotted)
    x0 = X_raw[0:1]
    z_path = []
    with torch.no_grad():
        z_curr = E(torch.as_tensor(x0, dtype=torch.float32))
        z_path.append(z_curr.numpy().copy())
        dt = config.integration_tau
        for _ in range(len(X_raw) - 1):
            z_curr = F(z_curr, dt)
            z_path.append(z_curr.numpy().copy())
    Z_flow = np.concatenate(z_path, axis=0)
    with torch.no_grad():
        x_flow = D(torch.as_tensor(Z_flow, dtype=torch.float32)).numpy()
    err_flow_state = np.linalg.norm(x_flow[:, :4] - X_raw[:, :4], axis=1)

    # One-step semiflow D(F(E(x_i))) vs x_{i+1}
    with torch.no_grad():
        z_all = E(torch.as_tensor(X_raw, dtype=torch.float32))
        z_next_pred = F(z_all, config.integration_tau)
        x_onestep = D(z_next_pred).numpy()
    err_onestep_state = np.linalg.norm(
        x_onestep[:-1, :4] - X_raw[1:, :4], axis=1)

    arc_mask = np.array([t[0] == "arc" for t in tags])

    # (a) phase portrait — round-trip only
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(X_raw[arc_mask, 0], X_raw[arc_mask, 2],
                 '.', color="#0072B2", ms=2.5, label="true")
    axes[0].plot(x_rt[arc_mask, 0], x_rt[arc_mask, 2],
                 '.', color="#D55E00", ms=1.5, alpha=0.7, label="D(E(x))")
    axes[0].set_xlabel(r"$\theta_{ns}$"); axes[0].set_ylabel(r"$\dot{\theta}_{ns}$")
    axes[0].set_title("Swing leg")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].plot(X_raw[arc_mask, 1], X_raw[arc_mask, 3],
                 '.', color="#0072B2", ms=2.5, label="true")
    axes[1].plot(x_rt[arc_mask, 1], x_rt[arc_mask, 3],
                 '.', color="#D55E00", ms=1.5, alpha=0.7, label="D(E(x))")
    axes[1].set_xlabel(r"$\theta_s$"); axes[1].set_ylabel(r"$\dot{\theta}_s$")
    axes[1].set_title("Stance leg")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Encoder-decoder round-trip D(E(x)) — experiment {EXP_TAG}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_phase = FIG_DIR / f"fig_compass_decoded_rollout_phase_{EXP_TAG}.png"
    fig.savefig(out_phase, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_phase}")

    # (b) time traces — full rollout, round-trip only
    t_idx = np.arange(len(X_raw))
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    labels = [r"$\theta_{ns}$", r"$\theta_s$",
              r"$\dot{\theta}_{ns}$", r"$\dot{\theta}_s$"]
    for k, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(t_idx, X_raw[:, k], color="#0072B2", lw=1.2, label="true")
        ax.plot(t_idx, x_rt[:, k], color="#D55E00", lw=0.9, alpha=0.8,
                label="D(E(x))")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sample index")
    axes[0].legend(fontsize=9, loc="upper right", ncol=2)
    axes[0].set_title(f"D(E(x)) round-trip — {EXP_TAG}")
    fig.tight_layout()
    out_traces = FIG_DIR / f"fig_compass_decoded_rollout_traces_{EXP_TAG}.png"
    fig.savefig(out_traces, dpi=200)
    plt.close(fig)
    print(f"wrote {out_traces}")

    report = []
    report.append(f"Compass-gait decoded-rollout evaluation [{EXP_TAG}]")
    report.append("=" * 60)
    report.append(f"model:  {MODEL_PATH}")
    report.append(f"embed_dim: {config.embed_dim} (embed_extra={config.embed_extra})")
    report.append(f"IC:     {ic}")
    report.append(f"arcs:   {len(segments)}  jumps: {len(jump_pairs)}")
    report.append(f"samples: {X_raw.shape[0]}  (arc: {int(arc_mask.sum())}, "
                  f"bridge: {int((~arc_mask).sum())})")
    report.append("")
    report.append("[D(E(x)) round-trip]")
    report.append(f"  state err  mean={err_state.mean():.3e}  "
                  f"median={np.median(err_state):.3e}  "
                  f"p95={np.percentile(err_state, 95):.3e}  "
                  f"max={err_state.max():.3e}")
    report.append(f"  s err      mean={err_s.mean():.3e}  max={err_s.max():.3e}")
    report.append(f"  arc-only state err   mean={err_state[arc_mask].mean():.3e}  "
                  f"max={err_state[arc_mask].max():.3e}")
    report.append(f"  bridge state err     mean={err_state[~arc_mask].mean():.3e}  "
                  f"max={err_state[~arc_mask].max():.3e}")
    report.append("")
    report.append("[Semiflow one-step D(F(E(x_i))) vs x_{i+1}]")
    report.append(f"  state err  mean={err_onestep_state.mean():.3e}  "
                  f"median={np.median(err_onestep_state):.3e}  "
                  f"p95={np.percentile(err_onestep_state, 95):.3e}  "
                  f"max={err_onestep_state.max():.3e}")
    report.append("")
    report.append("[Semiflow long-horizon rollout D(F^k(E(x0)))]")
    report.append(f"  state err  mean={np.nan_to_num(err_flow_state).mean():.3e}  "
                  f"p95={np.nanpercentile(err_flow_state, 95):.3e}")
    for k_h in (10, 25, 50, 100, 200):
        if k_h < len(err_flow_state):
            report.append(f"  state err at step {k_h}: {err_flow_state[k_h]:.3e}")
    txt = "\n".join(report) + "\n"
    print("\n" + txt)
    out_report = DATA_DIR / f"report_decoded_rollout_metrics_{EXP_TAG}.txt"
    out_report.write_text(txt, encoding="utf-8")
    print(f"wrote {out_report}")


if __name__ == "__main__":
    main()
