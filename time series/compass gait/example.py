"""
Compass gait example: simulate -> BridgeNet -> regularization -> plots.

    python "time series/compass gait/example.py"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from simulate import (simulate_compass_gait, generate_diverse_dataset,
                      extract_jump_pairs, concatenate_timeseries, LIMIT_CYCLE_IC)
from data_construction import detect_jumps, build_augmented_dataset
from bridge_net import BridgeNet, regularization_loss
from plot_suspension import suspension_diagnostics, plot_diagnostics
import matplotlib.pyplot as plt


# 1. Generate diverse training data
print("Generating diverse compass gait trajectories...")
segments_all, jump_pairs_all = generate_diverse_dataset(
    n_trajectories=100, n_impacts_per=3
)
base_trajectory, jump_pairs_all = extract_jump_pairs(segments_all, jump_pairs_all)
dataset = build_augmented_dataset(base_trajectory, jump_pairs_all, n_bridge_samples=20)
print(f"Training data: {dataset['base_points'].shape[0]} base + "
      f"{dataset['bridge_points'].shape[0]} bridge points, "
      f"{len(jump_pairs_all)} jumps")

# 2. Train BridgeNet (state_dim=4 for compass gait)
net = BridgeNet(state_dim=4)
print(f"BridgeNet parameters: {sum(p.numel() for p in net.parameters()):,}")

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
n_steps = 300

print(f"Training regularization loss ({n_steps} steps)")
for step in range(n_steps):
    idx = torch.randperm(len(jump_pairs_all))[:50].tolist()
    optimizer.zero_grad()
    loss = regularization_loss(net, [jump_pairs_all[i] for i in idx], n_s=10)
    loss.backward()
    optimizer.step()
    if step % 100 == 0 or step == n_steps - 1:
        print(f"  step {step:3d}  loss={loss.item():.6f}")

# 3. Test trajectory: 5 impacts from limit cycle
segments_test, jumps_test = simulate_compass_gait(
    LIMIT_CYCLE_IC, n_impacts=5
)
print(f"\nTest trajectory: {len(segments_test)} arcs, {len(jumps_test)} impacts")

# Print jump pairs
for i, (xm, xp) in enumerate(jumps_test):
    print(f"  jump {i}: x-={xm.round(3)}  ->  x+={xp.round(3)}")

# 4. Verify endpoint attachment
print("\nEndpoint attachment check:")
with torch.no_grad():
    for i, (xm, xp) in enumerate(jumps_test[:3]):
        xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0)
        xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0)
        g0 = net(xm_t, xp_t, torch.zeros(1, 1)).numpy().flatten()
        g1 = net(xm_t, xp_t, torch.ones(1, 1)).numpy().flatten()
        err0 = np.linalg.norm(g0[:4] - xm)
        err1 = np.linalg.norm(g1[:4] - xp)
        eta_mid = net(xm_t, xp_t, torch.tensor([[0.5]])).numpy().flatten()[-1]
        print(f"  jump {i}: err(s=0)={err0:.2e}  err(s=1)={err1:.2e}  eta(s=0.5)={eta_mid:.4f}")

# 5. Phase portraits by physical leg identity
#
# The hybrid state is [theta_ns, theta_s, dtheta_ns, dtheta_s] where
# "ns" and "s" swap roles at each impact. To plot consistent left/right
# leg trajectories, track which physical leg is which.
#
# Convention: at the start (segment 0), "ns" = leg A, "s" = leg B.
# After each impact the roles swap.

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = {'A': '#0072B2', 'B': '#D55E00'}

# Build per-leg trajectories + suspension readout jumps
leg_A_theta = []  # (theta, dtheta) for leg A
leg_B_theta = []
leg_A_jumps_gt = []   # ground truth jump lines
leg_B_jumps_gt = []
leg_A_jumps_susp = []  # suspension readout jump lines
leg_B_jumps_susp = []

ns_is_A = True  # at start, ns = leg A

net.eval()
with torch.no_grad():
    for i, seg in enumerate(segments_test):
        if ns_is_A:
            leg_A_theta.append(seg[:, [0, 2]])
            leg_B_theta.append(seg[:, [1, 3]])
        else:
            leg_B_theta.append(seg[:, [0, 2]])
            leg_A_theta.append(seg[:, [1, 3]])

        if i < len(jumps_test):
            xm, xp = jumps_test[i]

            # Bridge landing
            xm_t = torch.tensor(xm, dtype=torch.float32).unsqueeze(0)
            xp_t = torch.tensor(xp, dtype=torch.float32).unsqueeze(0)
            land = net(xm_t, xp_t, torch.ones(1, 1)).numpy().flatten()[:4]

            if ns_is_A:
                # leg A: ns pre-impact -> s post-impact (idx swap)
                leg_A_jumps_gt.append(([xm[0], xp[1]], [xm[2], xp[3]]))
                leg_B_jumps_gt.append(([xm[1], xp[0]], [xm[3], xp[2]]))
                leg_A_jumps_susp.append(([xm[0], land[1]], [xm[2], land[3]]))
                leg_B_jumps_susp.append(([xm[1], land[0]], [xm[3], land[2]]))
            else:
                leg_B_jumps_gt.append(([xm[0], xp[1]], [xm[2], xp[3]]))
                leg_A_jumps_gt.append(([xm[1], xp[0]], [xm[3], xp[2]]))
                leg_B_jumps_susp.append(([xm[0], land[1]], [xm[2], land[3]]))
                leg_A_jumps_susp.append(([xm[1], land[0]], [xm[3], land[2]]))
            ns_is_A = not ns_is_A

# Left panel: Leg A
ax = axes[0]
for i, seg in enumerate(leg_A_theta):
    ax.plot(seg[:, 0], seg[:, 1], color=colors['A'], lw=1.5,
            label='ground truth' if i == 0 else None)
for (th, dth) in leg_A_jumps_gt:
    ax.plot(th, dth, color=colors['A'], ls='--', lw=0.8, alpha=0.5)
for (th, dth) in leg_A_jumps_susp:
    ax.plot(th, dth, color='#009E73', ls='-', lw=1.2, alpha=0.8,
            label='suspension readout' if th is leg_A_jumps_susp[0][0] else None)
for (th, dth) in leg_A_jumps_gt:
    ax.plot(th[0], dth[0], 'v', color='k', ms=5, zorder=5)
    ax.plot(th[1], dth[1], '^', color='k', ms=5, zorder=5)
ax.set_xlabel(r'$\theta_A$')
ax.set_ylabel(r'$\dot{\theta}_A$')
ax.set_title('Leg A phase portrait')
ax.legend(fontsize=8)

# Right panel: Leg B
ax = axes[1]
for i, seg in enumerate(leg_B_theta):
    ax.plot(seg[:, 0], seg[:, 1], color=colors['B'], lw=1.5,
            label='ground truth' if i == 0 else None)
for (th, dth) in leg_B_jumps_gt:
    ax.plot(th, dth, color=colors['B'], ls='--', lw=0.8, alpha=0.5)
for (th, dth) in leg_B_jumps_susp:
    ax.plot(th, dth, color='#009E73', ls='-', lw=1.2, alpha=0.8,
            label='suspension readout' if th is leg_B_jumps_susp[0][0] else None)
for (th, dth) in leg_B_jumps_gt:
    ax.plot(th[0], dth[0], 'v', color='k', ms=5, zorder=5)
    ax.plot(th[1], dth[1], '^', color='k', ms=5, zorder=5)
ax.set_xlabel(r'$\theta_B$')
ax.set_ylabel(r'$\dot{\theta}_B$')
ax.set_title('Leg B phase portrait')
ax.legend(fontsize=8)

plt.suptitle('Compass Gait: physical leg phase portraits', fontsize=12)
plt.tight_layout()

# 6. Suspension quality diagnostics
print("\n--- Suspension diagnostics ---")
diag = suspension_diagnostics(segments_test, jumps_test, net, n_s=50)

print(f"Junction gaps (L2):")
for i, g in enumerate(diag['junction_gaps']):
    label = "base->bridge" if i % 2 == 0 else "bridge->base"
    print(f"  {label} {i//2}: {g:.2e}")
print(f"Max junction gap: {max(diag['junction_gaps']):.2e}")

print(f"\nBridge separation:")
print(f"  Global min pairwise distance: {diag['min_pairwise']:.6f}")
# Show closest pairs
pw = diag['pairwise_matrix']
n = pw.shape[0]
closest = []
for i in range(n):
    for j in range(i+1, n):
        if pw[i,j] < np.inf:
            closest.append((pw[i,j], i, j))
closest.sort()
for dist, i, j in closest[:5]:
    print(f"  bridges ({i},{j}): min dist = {dist:.6f}")

fig_diag, _ = plot_diagnostics(
    diag, jumps_test,
    state_labels=[r'$\theta_{ns}$', r'$\theta_s$',
                  r'$\dot\theta_{ns}$', r'$\dot\theta_s$'])

plt.show()
