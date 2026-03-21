import dataclasses
from enum import Enum, auto

class SystemType(Enum):
    RIMLESS_WHEEL = auto()
    COMPASS_GAIT = auto()

@dataclasses.dataclass
class HybridSuspensionConfig:
    # ---------------------------------------------------------
    # Global Settings
    # ---------------------------------------------------------
    system_type: SystemType = SystemType.RIMLESS_WHEEL
    device: str = "mps"
    
    # ---------------------------------------------------------
    # Rimless Wheel Physical Parameters
    # ---------------------------------------------------------
    gamma: float = 0.2          # Inclined plane slope
    alpha: float = 0.4          # Spoke angle (half-angle between legs)
    
    # ---------------------------------------------------------
    # Compass Gait Physical Parameters
    # ---------------------------------------------------------
    phi: float = 0.07           # Ground slope (in radians, ~4 degrees)
    m: float = 5.0              # Leg mass (m_total = 2*m + m_H = 20kg if mu=2)
    m_H: float = 10.0           # Hip mass
    l: float = 1.0              # Leg length
    a: float = 0.5              # Distance from foot to leg mass
    b: float = 0.5              # Distance from hip to leg mass (l = a + b)
    g: float = 9.81             # Gravity
    
    @property
    def mu(self) -> float: return self.m_H / self.m
    @property
    def beta(self) -> float: return self.b / self.a

    # ---------------------------------------------------------
    # Derived Geometries (Rimless Wheel)
    # ---------------------------------------------------------
    @property
    def theta_guard(self) -> float:
        return self.alpha + self.gamma
        
    @property
    def theta_reset(self) -> float:
        return 2 * self.gamma - self.theta_guard
        
    @property
    def omega_restitution(self) -> float:
        import math
        return math.cos(2 * self.alpha)
        
    # Phase Space Bounds for Sampling (Rimless)
    theta_bounds: tuple = (-0.2, 0.6)
    omega_bounds: tuple = (-1.0, 2.0)
    
    # Phase Space Bounds (Compass Gait - approximate)
    cg_theta_bounds: tuple = (-0.5, 0.5)
    cg_omega_bounds: tuple = (-2.0, 2.0)

    @property
    def state_dim(self) -> int:
        if self.system_type == SystemType.RIMLESS_WHEEL:
            return 2
        elif self.system_type == SystemType.COMPASS_GAIT:
            return 4
        return 2

    # ---------------------------------------------------------
    # Semiflow Integration Settings
    # ---------------------------------------------------------
    integration_tau: float = 0.05   # Time step for the discrete-time semiflow
    num_train_samples: int = 3000  # Number of initial conditions
    points_per_orbit: int = 20     # Trajectory length (steps) per initial condition

    # ---------------------------------------------------------
    # Visualization Settings
    # ---------------------------------------------------------
    viz_tau: float = 0.01          # Finer time step for smoother orbit curves
    viz_orbit_duration: float = 5.0  # Total simulated time per orbit in plots
    
    # ---------------------------------------------------------
    # Network Architectures (E, F, D)
    # ---------------------------------------------------------
    embed_extra: int = 0         # Extra embedding dims beyond d+1 (0 = minimal)
    hidden_dim: int = 256
    num_layers: int = 4          # Depth of each MLP (E, F, D)

    @property
    def embed_dim(self) -> int:
        """Dimension of the embedding space: state_dim + 1 + embed_extra."""
        return self.state_dim + 1 + self.embed_extra
    
    # ---------------------------------------------------------
    # Loss Optimization Weights
    # ---------------------------------------------------------
    weight_comm: float = 1.0     # F(E(x)) = E(Phi(x))
    weight_glue: float = 3.0     # Enforce E(guard, 1) == E(reset, 0)
    weight_recon: float = 0.1    # Prevent manifold collapse
    
    # ---------------------------------------------------------
    # Hardware & Training Hyperparameters
    # ---------------------------------------------------------
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 8e-4
    weight_decay: float = 1e-5

config = HybridSuspensionConfig()
