"""
Bouncing ball example: total base-space overlap, eta resolves it.

    python "time series/bouncing ball/example.py"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from simulate import (simulate_bouncing_ball, generate_diverse_dataset,
                      extract_jump_pairs)
from data_construction import build_augmented_dataset
from bridge_net import BridgeNet, regularization_loss, separation_loss
from plot_suspension import suspension_diagnostics
import matplotlib.pyplot as plt


# 1. Generate diverse training data
print("Generating bouncing ball trajectories...")
segments_all, jump_pairs_all = generate_diverse_dataset(
    n_trajectories=100, n_impacts_per=5,
    x_range=(0.5, 5.0), v_range=(0.0, 8.0),
)
base_trajectory, jump_pairs_all = extract_jump_pairs(segments_all, jump_pairs_all)
dataset = build_augmented_dataset(base_trajectory, jump_pairs_all, n_bridge_samples=20)
print(f"Training: {len(jump_pairs_all)} jumps, "
      f"{dataset['base_points'].shape[0]} base + {dataset['bridge_points'].shape[0]} bridge points")

# Show that all jumps occur at x=0
x_at_guard = [xm[0] for xm, _ in jump_pairs_all]
print(f"x at guard: min={min(x_at_guard):.6f}, max={max(x_at_guard):.6f} (all ~0)")

# 2. Demonstrate total overlap in base space
print("\nBase-space overlap check (line segments from x^- to x^+):")
print("All segments are vertical lines at x=0:")
for i, (xm, xp) in enumerate(jump_pairs_all[:5]):
    print(f"  jump {i}: ({xm[0]:.4f}, {xm[1]:.4f}) -> ({xp[0]:.4f}, {xp[1]:.4f})")

# Count how many segment pairs overlap
n = min(len(jump_pairs_all), 200)
n_overlap = 0
for i in range(n):
    xm1, xp1 = jump_pairs_all[i]
    v_min1, v_max1 = sorted([xm1[1], xp1[1]])
    for j in range(i+1, n):
        xm2, xp2 = jump_pairs_all[j]
        v_min2, v_max2 = sorted([xm2[1], xp2[1]])
        # Both at x=0, check if v ranges overlap
        if v_min1 < v_max2 and v_min2 < v_max1:
            n_overlap += 1
total_pairs = n * (n-1) // 2
print(f"Overlapping segment pairs: {n_overlap}/{total_pairs} "
      f"({100*n_overlap/total_pairs:.1f}%)")

# 3. Train BridgeNet with regularization + separation
net = BridgeNet(state_dim=2)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
lambda_reg = 0.01
lambda_sep = 1.0
margin = 0.15

print(f"\nTraining reg + separation (500 steps, margin={margin})")
for step in range(500):
    idx = torch.randperm(len(jump_pairs_all))[:50].tolist()
    batch_jp = [jump_pairs_all[i] for i in idx]

    optimizer.zero_grad()
    l_reg = regularization_loss(net, batch_jp, n_s=10)
    l_sep = separation_loss(net, batch_jp, n_s=10, margin=margin)
    loss = lambda_sep * l_sep + lambda_reg * l_reg
    loss.backward()
    optimizer.step()

    if step % 100 == 0 or step == 499:
        print(f"  step {step:3d}  loss={loss.item():.6f}  (sep={l_sep.item():.6f}  reg={l_reg.item():.6f})")

# 4. Test trajectory
segments_test, jumps_test = simulate_bouncing_ball(x0=0.0, v0=8.0, n_impacts=8)
print(f"\nTest: {len(segments_test)} arcs, {len(jumps_test)} impacts")

# 5. Diagnostics
diag = suspension_diagnostics(segments_test, jumps_test, net, n_s=50)
print(f"Max junction gap: {max(diag['junction_gaps']):.2e}")
print(f"Min bridge separation (R^3): {diag['min_pairwise']:.6f}")

# 6. Three-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Phase portrait (x vs v) — shows total overlap of jump segments
ax = axes[0]
for i, seg in enumerate(segments_test):
    ax.plot(seg[:, 0], seg[:, 1], color='#0072B2', lw=1.5,
            label='trajectory' if i == 0 else None)
for i, (xm, xp) in enumerate(jumps_test):
    ax.plot([xm[0], xp[0]], [xm[1], xp[1]], color='#D55E00', lw=2,
            label='jump (all at x=0)' if i == 0 else None)
    ax.plot(xm[0], xm[1], 'v', color='k', ms=6, zorder=5)
    ax.plot(xp[0], xp[1], '^', color='k', ms=6, zorder=5)
ax.set_xlabel(r'$x$ (height)')
ax.set_ylabel(r'$v$ (velocity)')
ax.set_title('Phase portrait — jumps overlap on v-axis')
ax.legend(fontsize=8)

# Panel 2: Bridge curves in (v, eta) — shows separation via eta
ax = axes[1]
s_vals = torch.linspace(0, 1, 50).unsqueeze(-1)
net.eval()
cmap = plt.cm.viridis
with torch.no_grad():
    for j, (xm, xp) in enumerate(jumps_test):
        xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0).expand(50, -1)
        xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).expand(50, -1)
        gamma = net(xm_t, xp_t, s_vals).numpy()
        # Color by impact velocity magnitude
        color = cmap(j / max(len(jumps_test) - 1, 1))
        ax.plot(gamma[:, 1], gamma[:, 2], color=color, lw=2,
                label=f'v$^-$={xm[1]:.1f}' if j < 6 else None)
        ax.plot(gamma[0, 1], gamma[0, 2], 'v', color='k', ms=6, zorder=5)
        ax.plot(gamma[-1, 1], gamma[-1, 2], '^', color='k', ms=6, zorder=5)
ax.set_xlabel(r'$v$ (velocity)')
ax.set_ylabel(r'$\eta$')
ax.set_title(r'Bridge arches in $(v, \eta)$ — separated by height')
ax.legend(fontsize=7)

# Panel 3: Full 3D view (x, v, eta)
ax = fig.add_subplot(133, projection='3d')
# Base trajectory at eta=0
for i, seg in enumerate(segments_test):
    ax.plot(seg[:, 0], seg[:, 1], np.zeros(len(seg)),
            color='#0072B2', lw=1.2, alpha=0.7,
            label='base trajectory' if i == 0 else None)
# Bridge arches
with torch.no_grad():
    for j, (xm, xp) in enumerate(jumps_test):
        xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0).expand(50, -1)
        xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0).expand(50, -1)
        gamma = net(xm_t, xp_t, s_vals).numpy()
        color = cmap(j / max(len(jumps_test) - 1, 1))
        ax.plot(gamma[:, 0], gamma[:, 1], gamma[:, 2],
                color=color, lw=2, label=f'bridge {j}' if j < 4 else None)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$v$')
ax.set_zlabel(r'$\eta$')
ax.set_title('Suspension in 3D')
ax.legend(fontsize=7, loc='upper left')

plt.suptitle('Bouncing Ball: total overlap in base space, resolved by $\\eta$', fontsize=12)
plt.tight_layout()
plt.show()
