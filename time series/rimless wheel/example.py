"""
Minimal example: compare ground-truth vs finite-difference jump detection.

    python "time series/rimless wheel/example.py"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from simulate import simulate_rimless_wheel, concatenate_timeseries
from data_construction import detect_jumps, build_augmented_dataset
from plot_suspension import plot_rollout_comparison
from bridge_net import BridgeNet, regularization_loss
import matplotlib.pyplot as plt


# --- Simulate one representative trajectory (5 impacts) ---
segments_gt, jumps_gt = simulate_rimless_wheel(
    theta0=0.0, omega0=1.5, n_impacts=5
)
print(f"Ground truth: {len(segments_gt)} arcs, {len(jumps_gt)} impacts")

# --- Finite-difference jump detection on the raw concatenated series ---
raw_series = concatenate_timeseries(segments_gt)
print(f"Raw time series: {raw_series.shape}")

threshold = 0.3  # rimless wheel jump is ~0.8 in theta, well above this
base_detected, jumps_detected = detect_jumps(raw_series, threshold)
print(f"Detected: {len(jumps_detected)} jumps (threshold={threshold})")

# Compare detected vs ground-truth jump pairs
print("\nJump pair comparison:")
print(f"  {'':4s} {'GT x^-':>20s}  {'GT x^+':>20s}  |  {'Det x^-':>20s}  {'Det x^+':>20s}  |  err(x^-)  err(x^+)")
for i in range(min(len(jumps_gt), len(jumps_detected))):
    xm_gt, xp_gt = jumps_gt[i]
    xm_d, xp_d = jumps_detected[i]
    err_m = np.linalg.norm(xm_gt - xm_d)
    err_p = np.linalg.norm(xp_gt - xp_d)
    print(f"  {i:3d}  {xm_gt.round(4)}  {xp_gt.round(4)}  |  {xm_d.round(4)}  {xp_d.round(4)}  |  {err_m:.2e}   {err_p:.2e}")

# --- Train BridgeNet on detected jump pairs ---
dataset_gt = build_augmented_dataset(
    np.concatenate(segments_gt), jumps_gt, n_bridge_samples=20)
dataset_det = build_augmented_dataset(
    base_detected, jumps_detected, n_bridge_samples=20)

net_gt = BridgeNet(state_dim=2)
net_det = BridgeNet(state_dim=2)

# Use same init for fair comparison
net_det.load_state_dict(net_gt.state_dict())

for net, jp, label in [(net_gt, jumps_gt, "GT"), (net_det, jumps_detected, "Detected")]:
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(300):
        idx = torch.randperm(len(jp))[:min(50, len(jp))].tolist()
        optimizer.zero_grad()
        loss = regularization_loss(net, [jp[i] for i in idx], n_s=10)
        loss.backward()
        optimizer.step()
    print(f"\n{label} final loss: {loss.item():.6f}")

# --- Rollout comparison plots ---
fig_gt, ax_gt = plot_rollout_comparison(
    segments_gt, jumps_gt, net_gt,
    title="Ground-truth event detection")

fig_det, ax_det = plot_rollout_comparison(
    segments_gt, jumps_detected, net_det,
    title=f"Finite-difference detection (threshold={threshold})")

plt.show()
