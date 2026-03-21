import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ==========================================
# 1. Hybrid Dynamical System Generation
# ==========================================
class IzhikevichHybridSystem:
    def __init__(self, C=100.0, k=0.7, v_r=-60.0, v_t=-40.0,
                 v_peak=35.0, a=0.03, b=-2.0, c=-50.0, d=100.0, I=70.0):
        self.C, self.k, self.v_r, self.v_t, self.v_peak = C, k, v_r, v_t, v_peak
        self.a, self.b, self.c, self.d, self.I = a, b, c, d, I

    def vector_field(self, t, state):
        v, u = state
        dv = (self.k * (v - self.v_r) * (v - self.v_t) - u + self.I) / self.C
        du = self.a * (self.b * (v - self.v_r) - u)
        return [dv, du]

    def reset_map(self, state):
        v, u = state
        return np.array([self.c, u + self.d])

    def generate_tau_timeseries(self, x0, tau, n_steps):
        def guard_event(t, state):
            return state[0] - self.v_peak
        guard_event.terminal = True
        guard_event.direction = 1

        timeseries = [np.array(x0)]
        curr_state = np.array(x0)

        for _ in range(n_steps):
            t_curr = 0.0
            while t_curr < tau:
                sol = solve_ivp(
                    self.vector_field,
                    [t_curr, tau],
                    curr_state,
                    events=guard_event,
                    max_step=tau/10.0,
                    rtol=1e-6, atol=1e-9
                )

                if sol.status == 1:
                    curr_state = self.reset_map(sol.y[:, -1])
                    t_curr = sol.t[-1]
                else:
                    curr_state = sol.y[:, -1]
                    t_curr = tau
            timeseries.append(curr_state)
        return np.array(timeseries)

def generate_random_tau_transitions(hds, tau, num_samples):
    v_bounds = (-80.0, 40.0)
    u_bounds = (-20.0, 150.0)

    x_i_list, x_next_list = [], []
    for _ in range(num_samples):
        x0 = [np.random.uniform(*v_bounds), np.random.uniform(*u_bounds)]
        traj = hds.generate_tau_timeseries(x0, tau, n_steps=1)
        if len(traj) == 2:
            x_i_list.append(traj[0])
            x_next_list.append(traj[1])

    return torch.tensor(np.array(x_i_list), dtype=torch.float32), \
           torch.tensor(np.array(x_next_list), dtype=torch.float32)

# ==========================================
# 2. Neural Networks
# ==========================================
class HybridfoldNetworksExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))
        self.F = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))
        self.D = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 2))

# ==========================================
# 3. Regularized Loss Functions
# ==========================================
def compute_vicreg_loss(y, gamma=2.0, epsilon=1e-4):
    y_mean = y - y.mean(dim=0)
    std = torch.sqrt(y_mean.var(dim=0) + epsilon)
    loss_var = torch.mean(F.relu(gamma - std))

    batch_size = y.size(0)
    cov = (y_mean.T @ y_mean) / (batch_size - 1)
    mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=y.device)
    loss_cov = (cov[mask] ** 2).sum() / cov.size(0)

    return loss_var, loss_cov

def compute_gluing_loss(model, v_peak=35.0, c=-50.0, d=100.0, num_samples=256):
    u_samples = torch.empty(num_samples, 1, device=next(model.parameters()).device).uniform_(-20.0, 150.0)
    x_guard = torch.cat([torch.full((num_samples, 1), v_peak, device=u_samples.device), u_samples], dim=1)
    x_reset = torch.cat([torch.full((num_samples, 1), c, device=u_samples.device), u_samples + d], dim=1)
    return F.mse_loss(model.E(x_guard), model.E(x_reset))

def calculate_smooth_mask(x, v_peak=35.0, c=-50.0, temp=5.0):
    v = x[:, 0]
    dist_G = torch.abs(v - v_peak)
    dist_rG = torch.abs(v - c)
    w_G = torch.sigmoid(temp * (dist_G - 2.0))
    w_rG = torch.sigmoid(temp * (dist_rG - 2.0))
    return w_G * w_rG

def compute_contrastive_loss(y, x, delta=1.0, v_peak=35.0, c=-50.0):
    y_dist = torch.cdist(y, y)
    v = x[:, 0]
    is_near_G = (torch.abs(v - v_peak) < 5.0).float()
    is_near_rG = (torch.abs(v - c) < 5.0).float()

    near_G_matrix = is_near_G.unsqueeze(1) @ is_near_rG.unsqueeze(0)
    near_rG_matrix = is_near_rG.unsqueeze(1) @ is_near_G.unsqueeze(0)

    eye = torch.eye(y.size(0), device=y.device)
    valid_pairs_mask = (1.0 - torch.clamp(near_G_matrix + near_rG_matrix, 0.0, 1.0)) * (1.0 - eye)

    loss = F.relu(delta - y_dist) * valid_pairs_mask
    return loss.sum() / (valid_pairs_mask.sum() + 1e-8)

# ==========================================
# 4. Deep Crossing Validation
# ==========================================
def validate_crossing_dynamics(model, hds, tau=1.0):
    """
    Validates that a trajectory originating just before the guard set 
    (spiking threshold) accurately traverses through the embedded cyclic 
    manifold without topological singularities, and maps back smoothly
    to the reset state.
    """
    model.eval()
    
    # 1. Generate an orbit that is guaranteed to spike (cross v_peak)
    # Start high voltage, high recovery
    x0 = [15.0, 50.0]  
    
    # Force dense sampling right across the spike
    # Simulate with very small tau to get fine-grained steps before/after spike
    fine_tau = 0.05
    n_fine_steps = int(2.5 / fine_tau)  # Enough time to spike and reset once
    
    true_orbit_x = hds.generate_tau_timeseries(x0, fine_tau, n_fine_steps)
    true_orbit_t = torch.tensor(true_orbit_x, dtype=torch.float32)
    
    with torch.no_grad():
        # Lift exactly computed discontinuous sequence into embedded space
        true_orbit_y = model.E(true_orbit_t)
        
        # Pull back from the continuous space using only the decoder
        # This checks the Masked autoencoder and evaluates if Decoder
        # correctly handles the glued seam
        recon_orbit_x = model.D(true_orbit_y)
        
        # Test the purely continuous Flow Predictor (F) around the boundary
        # We start at x0, map to y0, and continuously integrate F 
        pred_orbit_y = [true_orbit_y[0]]
        y_curr = true_orbit_y[0].unsqueeze(0)
        for _ in range(len(true_orbit_t) - 1):
            y_curr = model.F(y_curr)
            pred_orbit_y.append(y_curr.squeeze(0))
        pred_orbit_y = torch.stack(pred_orbit_y)
        
        # Pull back the continuously predicted orbit back to reality
        recon_pred_orbit_x = model.D(pred_orbit_y)

    # Calculate errors around the crossing
    y_mse = F.mse_loss(true_orbit_y, pred_orbit_y).item()
    x_mse = F.mse_loss(true_orbit_t, recon_pred_orbit_x).item()
    
    print("\n" + "="*50)
    print("DEEP CROSSING VALIDATION RESULTS")
    print("="*50)
    print(f"Continuous Embedded Path (Y-space) MSE: {y_mse:.6f}")
    print(f"Discontinuous Decoded Path (X-space) MSE: {x_mse:.6f}")
    
    # Identify the index where the jump occurs in reality
    jump_idx = None
    for i in range(1, len(true_orbit_x)):
        # If voltage suddenly drops significantly, a reset happened
        if true_orbit_x[i, 0] - true_orbit_x[i-1, 0] < -50:
            jump_idx = i
            break
            
    if jump_idx is not None:
        print("\nSpike/Reset Event Isolated:")
        print(f"  Step {jump_idx-1} -> State just before Guard:  {true_orbit_x[jump_idx-1]}")
        print(f"  Step {jump_idx}   -> State just after Reset: {true_orbit_x[jump_idx]}")
        
        dist_y_crossing = torch.norm(true_orbit_y[jump_idx] - true_orbit_y[jump_idx-1]).item()
        dist_y_pred_crossing = torch.norm(pred_orbit_y[jump_idx] - pred_orbit_y[jump_idx-1]).item()
        
        print(f"\nEmbedded Space (Y) Euclidean Distance across boundary seam:")
        print(f"  True discrete jump translated to Y:  {dist_y_crossing:.4f}  <-- Should be small (Gluing success)")
        print(f"  Predicted continuous step in Y (F):  {dist_y_pred_crossing:.4f}  <-- Should match true Y distance")
        
        recon_pre = recon_pred_orbit_x[jump_idx-1].numpy()
        recon_post = recon_pred_orbit_x[jump_idx].numpy()
        
        print("\nDecoder (D) Pull-back across the glued boundary:")
        print(f"  Reconstructed State pre-guard:   [{recon_pre[0]:.2f}, {recon_pre[1]:.2f}]")
        print(f"  Reconstructed State post-reset:  [{recon_post[0]:.2f}, {recon_post[1]:.2f}]")
        print("  (Warning: Decoder values exactly at the glued boundary may interpolate roughly due to multi-valued nature)")
    else:
        print("\nWARNING: No spike detected in validation orbit. Increase simulation time or initial condition.")
    print("="*50 + "\n")
    
    return true_orbit_x, recon_pred_orbit_x.numpy(), true_orbit_y.numpy(), pred_orbit_y.numpy(), jump_idx

# ==========================================
# 4. Training Loop
# ==========================================
def train_hybrid_system(model, dataloader, epochs=200, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        epoch_loss = 0.0
        alpha = min(1.0, epoch / (epochs * 0.4)) # Ramp up topology constraints

        for x_i, x_next in dataloader:
            x_i, x_next = x_i.to(device), x_next.to(device)
            optimizer.zero_grad()

            y_i = model.E(x_i)
            y_next_true = model.E(x_next)
            y_next_pred = model.F(y_i)

            loss_comm = F.mse_loss(y_next_pred, y_next_true)
            loss_var, loss_cov = compute_vicreg_loss(y_i)
            loss_glue = compute_gluing_loss(model)

            x_recon = model.D(y_i)
            mask = calculate_smooth_mask(x_i).unsqueeze(1)
            loss_recon = torch.mean(mask * (x_recon - x_i)**2)

            loss_push = compute_contrastive_loss(y_i, x_i)

            total_loss = (loss_comm +
                          1.0 * loss_glue +
                          1.0 * loss_recon +
                          alpha * (0.1 * loss_var + 0.1 * loss_cov + 0.05 * loss_push))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += total_loss.item()

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")

    model.to("cpu")
    return model

# ==========================================
# 5. Global Visualization
# ==========================================
def visualize_manifold_and_orbits(hds, model, tau):
    model.eval()

    # Deep crossing validation
    true_x, recon_x, true_y, pred_y, jump_idx = validate_crossing_dynamics(model, hds)

    # Generate background grid for manifold visualization
    v_bg = np.linspace(-80, 40, 60)
    u_bg = np.linspace(-20, 150, 60)
    V, U = np.meshgrid(v_bg, u_bg)
    X_bg = np.column_stack([V.ravel(), U.ravel()])
    X_bg_t = torch.tensor(X_bg, dtype=torch.float32)

    with torch.no_grad():
        Y_bg = model.E(X_bg_t).numpy()

    # Generate specific orbits
    initial_conditions = [[-70.0, 0.0], [-20.0, 50.0], [10.0, 100.0]]
    colors = ['cyan', 'magenta', 'lime']
    n_steps = 150

    orbits_X_true = []
    orbits_Y_true = []
    orbits_Y_pred = []

    for x0 in initial_conditions:
        true_X = hds.generate_tau_timeseries(x0, tau, n_steps)
        orbits_X_true.append(true_X)

        true_X_t = torch.tensor(true_X, dtype=torch.float32)
        with torch.no_grad():
            orbits_Y_true.append(model.E(true_X_t).numpy())

            # Predict orbit in Y entirely using F
            y_curr = model.E(true_X_t[0].unsqueeze(0))
            pred_Y = [y_curr.squeeze().numpy()]
            for _ in range(n_steps):
                y_curr = model.F(y_curr)
                pred_Y.append(y_curr.squeeze().numpy())
            orbits_Y_pred.append(np.array(pred_Y))

    # Plotting
    fig = plt.figure(figsize=(16, 7))

    # Plot 1: Original 2D Space
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(X_bg[:, 0], X_bg[:, 1], c=X_bg[:, 0], cmap='viridis', s=2, alpha=0.3)
    for i, orbit_x in enumerate(orbits_X_true):
        ax1.plot(orbit_x[:, 0], orbit_x[:, 1], '.-', color=colors[i], markersize=4, label=f'Orbit {i+1}')
    
    # Highlight the deep crossing validation orbit
    ax1.plot(true_x[:, 0], true_x[:, 1], 'r.-', linewidth=1.5, markersize=3, label='Validation Orbit (Cross)')
    ax1.plot(recon_x[:, 0], recon_x[:, 1], 'x--', color='orange', linewidth=1, markersize=3, label='Reconstructed Orbit')
    
    if jump_idx is not None:
        ax1.scatter(true_x[jump_idx-1, 0], true_x[jump_idx-1, 1], color='red', s=50, marker='*', zorder=5, label='Guard State')
        ax1.scatter(true_x[jump_idx, 0], true_x[jump_idx, 1], color='green', s=50, marker='*', zorder=5, label='Reset State')

    ax1.set_title(r"Original Discontinuous State Space $\mathcal{X}$")
    ax1.set_xlabel("Voltage $v$")
    ax1.set_ylabel("Recovery $u$")
    ax1.legend()
    fig.colorbar(scatter1, ax=ax1, label='Voltage $v$')

    # Plot 2: Embedded 3D Manifold
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(Y_bg[:, 0], Y_bg[:, 1], Y_bg[:, 2], c=X_bg[:, 0], cmap='viridis', s=2, alpha=0.2)
    for i in range(len(initial_conditions)):
        # Plot true mapped orbit
        ax2.plot(orbits_Y_true[i][:, 0], orbits_Y_true[i][:, 1], orbits_Y_true[i][:, 2],
                 'o-', color=colors[i], markersize=3, alpha=0.5, label=f'Mapped True {i+1}')
        # Plot predicted continuous orbit
        ax2.plot(orbits_Y_pred[i][:, 0], orbits_Y_pred[i][:, 1], orbits_Y_pred[i][:, 2],
                 '--', color=colors[i], linewidth=2, label=fr'Predicted $F^n$ {i+1}')

    # Highlight the deep crossing validation orbit in Y space
    ax2.plot(true_y[:, 0], true_y[:, 1], true_y[:, 2], 'r.-', linewidth=2, label='True Mapped Crossing')
    ax2.plot(pred_y[:, 0], pred_y[:, 1], pred_y[:, 2], 'x--', color='orange', linewidth=2, label=r'Predicted Crossing ($F$)')

    if jump_idx is not None:
        ax2.scatter(true_y[jump_idx-1, 0], true_y[jump_idx-1, 1], true_y[jump_idx-1, 2], color='red', s=100, marker='*', zorder=5)
        ax2.scatter(true_y[jump_idx, 0], true_y[jump_idx, 1], true_y[jump_idx, 2], color='green', s=100, marker='*', zorder=5)

    ax2.set_title(r"Continuous Embedded Hybridfold $\mathcal{Y}$")
    ax2.set_xlabel("$y_1$")
    ax2.set_ylabel("$y_2$")
    ax2.set_zlabel("$y_3$")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Initializing Izhikevich System...")
    hds = IzhikevichHybridSystem()
    
    print("Generating exact Time-tau maps...")
    tau = 0.5
    X_train, Y_train = generate_random_tau_transitions(hds, tau=tau, num_samples=3000)
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Instantiating Networks...")
    model = HybridfoldNetworksExt()
    
    print("Training Model...")
    model = train_hybrid_system(model, dataloader, epochs=50)
    
    print("Visualizing Manifold and Deep Dynamics...")
    visualize_manifold_and_orbits(hds, model, tau=tau)
