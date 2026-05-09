"""
Rimless-wheel decoded-rollout evaluation, parallel of
scripts/evaluate_decoded_rollout_compass.py.

Loads runs/rimless_wheel/model.pt, simulates a held-out rimless rollout from
a perturbed limit-cycle IC, runs every sample through Encoder -> Decoder and
Encoder -> SemiflowF^k -> Decoder, and compares against ground truth.

Outputs:
    figures/rimless_wheel/fig_rimless_decoded_rollout_phase.png
    figures/rimless_wheel/fig_rimless_decoded_rollout_traces.png
    data/rimless_wheel/report_decoded_rollout_metrics.txt
"""
import argparse
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
FIG_DIR = ROOT / "figures" / "rimless_wheel"
DATA_DIR = ROOT / "data" / "rimless_wheel"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_MODEL = ROOT / "runs" / "rimless_wheel" / "model.pt"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=_DEFAULT_MODEL,
                   help="Path to trained model.pt.")
    p.add_argument("--out-suffix", default="",
                   help="Suffix appended to output figure / report names.")
    return p.parse_args()

# Limit-cycle post-reset state for alpha=0.4, gamma=0.2:
#   omega_minus^2 = 2*(cos(theta_reset) - cos(theta_guard)) / sin^2(2*alpha)
#   omega_plus    = omega_minus * cos(2*alpha)
# yields theta_plus = -0.2, omega_plus ~ 0.54.
LIMIT_CYCLE_IC = np.array([-0.2, 0.54])


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
    args = _parse_args()
    model_path = args.model
    suffix = args.out_suffix

    if not model_path.exists():
        print(f"ERROR: missing {model_path}. Train first.")
        sys.exit(1)

    print(f"Loading {model_path}")
    net = SuspensionNetworks()
    net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    net.eval()
    E, D, F = net.E, net.D, net.F

    # Held-out rollout: perturbed limit cycle IC.
    rng = np.random.default_rng(2026)
    perturbation = rng.normal(0, 0.02, size=2)
    ic = LIMIT_CYCLE_IC + perturbation
    print(f"Simulating rollout from IC={ic}")
    segments, jump_pairs = simulate_rimless_wheel(ic[0], ic[1], n_impacts=8)
    print(f"  {len(segments)} arcs, {len(jump_pairs)} jumps")

    X_raw, tags = _extend_with_bridges(segments, jump_pairs, n_bridge=50)
    print(f"  extended trajectory: {X_raw.shape}")

    # -- 1. Round-trip: D(E(x))
    with torch.no_grad():
        x_in = torch.as_tensor(X_raw, dtype=torch.float32)
        z = E(x_in)
        x_rt = D(z).numpy()

    err_state = np.linalg.norm(x_rt[:, :2] - X_raw[:, :2], axis=1)
    err_s = np.abs(x_rt[:, 2] - X_raw[:, 2])

    # -- 2. Semiflow rollout: E(x_0) then push k steps by F, decode each
    x0 = X_raw[0:1]
    z_path = []
    with torch.no_grad():
        z_curr = E(torch.as_tensor(x0, dtype=torch.float32))
        z_path.append(z_curr.numpy().copy())
        dt = config.integration_tau
        for _ in range(len(X_raw) - 1):
            z_curr = F(z_curr, dt)
            z_path.append(z_curr.numpy().copy())
    Z_flow = np.concatenate(z_path, axis=0)           # (T, d)

    with torch.no_grad():
        x_flow = D(torch.as_tensor(Z_flow, dtype=torch.float32)).numpy()

    err_flow_state = np.linalg.norm(x_flow[:, :2] - X_raw[:, :2], axis=1)

    # One-step semiflow check: E -> F(., dt) -> D at every sample.
    with torch.no_grad():
        z_all = E(torch.as_tensor(X_raw, dtype=torch.float32))
        z_next_pred = F(z_all, config.integration_tau)
        x_onestep = D(z_next_pred).numpy()
    err_onestep_state = np.linalg.norm(
        x_onestep[:-1, :2] - X_raw[1:, :2], axis=1)

    # -- 3. Plots
    arc_mask = np.array([t[0] == "arc" for t in tags])

    # (a) phase portrait of round-trip
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
    ax.plot(X_raw[arc_mask, 0], X_raw[arc_mask, 1],
            '.', color="#0072B2", ms=2.5, label="true")
    ax.plot(x_rt[arc_mask, 0], x_rt[arc_mask, 1],
            '.', color="#D55E00", ms=1.5, alpha=0.7, label="D(E(x))")
    ax.axvline(config.theta_guard, color="#888", linestyle="--", lw=0.8,
               label=r"$\theta_{\mathrm{guard}}$")
    ax.axvline(config.theta_reset, color="#888", linestyle=":", lw=0.8,
               label=r"$\theta_{\mathrm{reset}}$")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_title("Encoder-decoder round-trip $D(E(x))$ on held-out rollout")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_phase = FIG_DIR / f"fig_rimless_decoded_rollout_phase{suffix}.png"
    fig.savefig(out_phase, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_phase}")

    # (b) time traces of theta, omega, full rollout, round-trip only
    t_idx = np.arange(len(X_raw))
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    labels = [r"$\theta$", r"$\dot{\theta}$"]
    for k, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(t_idx, X_raw[:, k], color="#0072B2", lw=1.2, label="true")
        ax.plot(t_idx, x_rt[:, k], color="#D55E00", lw=0.9, alpha=0.8,
                label="D(E(x))")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sample index")
    axes[0].legend(fontsize=9, loc="upper right", ncol=2)
    axes[0].set_title("Encoder-decoder round-trip $D(E(x))$ — full rollout")
    fig.tight_layout()
    out_traces = FIG_DIR / f"fig_rimless_decoded_rollout_traces{suffix}.png"
    fig.savefig(out_traces, dpi=200)
    plt.close(fig)
    print(f"wrote {out_traces}")

    # -- 4. Metrics report
    report = []
    report.append("Rimless-wheel decoded-rollout evaluation")
    report.append("=" * 48)
    report.append(f"model:  {model_path}")
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
    report.append(f"  s err      mean={err_s.mean():.3e}  "
                  f"max={err_s.max():.3e}")
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
                  f"median={np.nan_to_num(err_flow_state).max():.3e}  "
                  f"p95={np.nanpercentile(err_flow_state, 95):.3e}")
    for k_horizon in (10, 25, 50, 100, 200):
        if k_horizon < len(err_flow_state):
            report.append(f"  state err at step {k_horizon}: "
                          f"{err_flow_state[k_horizon]:.3e}")

    txt = "\n".join(report) + "\n"
    print("\n" + txt)
    out_report = DATA_DIR / f"report_decoded_rollout_metrics{suffix}.txt"
    out_report.write_text(txt, encoding="utf-8")
    print(f"wrote {out_report}")


if __name__ == "__main__":
    main()
