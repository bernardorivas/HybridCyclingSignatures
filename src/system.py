import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseHybridSystem:
    """Abstract base class for hybrid dynamical systems with mapping cylinders."""
    def __init__(self, config):
        self.cfg = config
        self.state_dim = config.state_dim

    def base_vector_field(self, t, state):
        raise NotImplementedError

    def reset_map(self, state):
        raise NotImplementedError

    def is_guard_hit(self, t, state):
        raise NotImplementedError

    def simulate_continuous_trajectory(self, t_span, y0):
        def guard_event(t, y):
            return self.is_guard_hit(t, y)
        guard_event.terminal = True
        guard_event.direction = self.guard_direction

        sol = solve_ivp(
            self.base_vector_field, 
            t_span, 
            y0, 
            events=guard_event,
            dense_output=True,
            max_step=0.01
        )
        return sol

    def generate_tau_timeseries(self, initial_state, tau, n_steps):
        """Generic relaxed flow φ' on X' = X ∪ (G x [0,1])."""
        timeseries = [np.array(initial_state)]
        curr_state = np.array(initial_state)
        d = self.state_dim
        
        for _ in range(n_steps):
            state_base = curr_state[:d]
            s = curr_state[d]
            
            # --- PHASE 1: Cylinder Traverse ---
            in_cylinder = s > 0 or (abs(self.is_guard_hit(0, state_base)) < 1e-6 and s == 0)

            if in_cylinder:
                time_to_glue = 1.0 - s
                if tau >= time_to_glue:
                    rem_tau = tau - time_to_glue
                    base_reset = self.reset_map(state_base)
                    if rem_tau > 0:
                        sol = self.simulate_continuous_trajectory([0, rem_tau], base_reset)
                        next_state = np.zeros(d + 1)
                        next_state[:d] = sol.y[:, -1]
                        next_state[d] = 0.0
                    else:
                        next_state = np.zeros(d + 1)
                        next_state[:d] = base_reset
                        next_state[d] = 0.0
                else:
                    next_state = curr_state.copy()
                    next_state[d] = s + tau

                timeseries.append(next_state)
                curr_state = next_state
                continue

            # --- PHASE 2: Base Space Dynamics ---
            sol = self.simulate_continuous_trajectory([0, tau], state_base)
            if sol.status == 1: # Hit the guard!
                time_to_guard = sol.t[-1]
                rem_tau = tau - time_to_guard
                guard_state_base = sol.y[:, -1]
                
                if rem_tau > 0:
                    if rem_tau >= 1.0:
                        base_reset = self.reset_map(guard_state_base)
                        rem_tau_after_cyl = rem_tau - 1.0
                        if rem_tau_after_cyl > 0:
                            sol2 = self.simulate_continuous_trajectory([0, rem_tau_after_cyl], base_reset)
                            next_state = np.zeros(d + 1)
                            next_state[:d] = sol2.y[:, -1]
                            next_state[d] = 0.0
                        else:
                            next_state = np.zeros(d + 1)
                            next_state[:d] = base_reset
                            next_state[d] = 0.0
                    else:
                        next_state = np.zeros(d + 1)
                        next_state[:d] = guard_state_base
                        next_state[d] = rem_tau
                else:
                    next_state = np.zeros(d + 1)
                    next_state[:d] = guard_state_base
                    next_state[d] = 0.0
            else:
                next_state = np.zeros(d + 1)
                next_state[:d] = sol.y[:, -1]
                next_state[d] = 0.0
                    
            timeseries.append(next_state)
            curr_state = next_state
            
        return np.vstack(timeseries)

    @property
    def guard_direction(self):
        return 1

    def is_moving_forward(self, state):
        raise NotImplementedError


class RimlessWheelHybridSystem(BaseHybridSystem):
    def __init__(self, config):
        super().__init__(config)
        self.guard_theta = config.theta_guard
        self.reset_theta = config.theta_reset
        self.loss_factor = config.omega_restitution

    def base_vector_field(self, t, state):
        theta, omega = state
        return [omega, np.sin(theta)]

    def reset_map(self, state):
        theta, omega = state
        return np.array([self.reset_theta, omega * self.loss_factor])

    def is_guard_hit(self, t, state):
        # We want this to cross zero when theta hits guard_theta
        if state[1] > 0:
            return state[0] - self.guard_theta
        return 1.0

    def is_moving_forward(self, state):
        return state[1] > 0


class CompassGaitHybridSystem(BaseHybridSystem):
    """
    Implements the Compass-Gait Biped hybrid dynamics.
    State: [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
    """
    def __init__(self, config):
        super().__init__(config)
        self.m = config.m
        self.m_H = config.m_H
        self.a = config.a
        self.b = config.b
        self.l = config.l
        self.phi = config.phi
        self.g = config.g

    def get_matrices(self, q, dq):
        th_ns, th_s = q
        dth_ns, dth_s = dq
        
        m, m_H, a, b, l, g = self.m, self.m_H, self.a, self.b, self.l, self.g
        
        # Mass matrix M(q)
        M = np.zeros((2, 2))
        M[0, 0] = m * b**2
        M[0, 1] = -m * l * b * np.cos(th_s - th_ns)
        M[1, 0] = -m * l * b * np.cos(th_s - th_ns)
        M[1, 1] = (m_H + m) * l**2 + m * a**2
        
        # Coriolis/Centrifugal matrix N(q, dq)
        N = np.zeros((2, 2))
        N[0, 1] = m * l * b * np.sin(th_s - th_ns) * dth_s
        N[1, 0] = -m * l * b * np.sin(th_s - th_ns) * dth_ns
        
        # Gravity vector G(q)
        G = np.zeros(2)
        G[0] = m * b * g * np.sin(th_ns)
        G[1] = -(m_H * l + m * a + m * l) * g * np.sin(th_s)
        
        return M, N, G

    def base_vector_field(self, t, state):
        q = state[:2]
        dq = state[2:]
        M, N, G = self.get_matrices(q, dq)
        
        # M * ddq + N * dq + G = 0  => ddq = -M_inv * (N*dq + G)
        ddq = np.linalg.solve(M, -(N @ dq + G))
        return [dq[0], dq[1], ddq[0], ddq[1]]

    def is_guard_hit(self, t, state):
        # Guard: theta_ns + theta_s = -2*phi (Goswami Eq. 2.9)
        # At end of step: theta_ns > theta_s (swing ahead of stance)
        if state[0] - state[1] > 0.01:
            return (state[0] + state[1]) + 2 * self.phi
        return 1.0

    @property
    def guard_direction(self):
        return -1

    def is_moving_forward(self, state):
        # Support leg moving backward in world frame (stamped on ground)
        # Swing leg moving forward
        return state[2] > 0

    def reset_map(self, state):
        """Impact map: roles swap, angular momentum conserved."""
        th_ns_minus, th_s_minus, dth_ns_minus, dth_s_minus = state
        
        m, m_H, a, b, l = self.m, self.m_H, self.a, self.b, self.l
        # alpha = (th_ns - th_s)/2 per Goswami Eq. 2.10, but sign
        # doesn't matter: only cos(2*alpha) appears in Q matrices.
        alpha = (th_s_minus - th_ns_minus) / 2.0

        # Angle swap (Eq. 2.4): q+ = J * q-
        th_s_plus = th_ns_minus
        th_ns_plus = th_s_minus

        # Velocity jump from angular momentum conservation (Eq. A.34-A.37)
        cos2a = np.cos(2*alpha)
        Q_minus = np.zeros((2, 2))
        Q_minus[0, 0] = -m*a*b
        Q_minus[0, 1] = (m_H*l**2 + 2*m*a*l)*cos2a - m*a*b
        Q_minus[1, 0] = 0
        Q_minus[1, 1] = -m*a*b
        
        Q_plus = np.zeros((2, 2))
        Q_plus[0, 0] = m*b*(b - l*cos2a)
        Q_plus[0, 1] = m*l*(l - b*cos2a) + m_H*l**2 + m*a**2
        Q_plus[1, 0] = m*b**2
        Q_plus[1, 1] = -m*b*l*cos2a
        
        dq_minus = np.array([dth_ns_minus, dth_s_minus])
        dq_plus = np.linalg.solve(Q_plus, Q_minus @ dq_minus)
        
        return np.array([th_ns_plus, th_s_plus, dq_plus[0], dq_plus[1]])


def generate_suspension_dataset(sys, tau, num_orbits, points_per_orbit=50, cfg=None):
    if cfg is None:
        from config import config as cfg

    X_list = []
    Y_list = []
    d = cfg.state_dim
    initial_conditions = []

    from config import SystemType
    if cfg.system_type == SystemType.RIMLESS_WHEEL:
        for _ in range(num_orbits // 2):
            th0 = np.random.uniform(*cfg.theta_bounds)
            w0 = np.random.uniform(0, cfg.omega_bounds[1])
            initial_conditions.append([th0, w0, 0.0])
        for _ in range(num_orbits // 2):
            w0 = np.random.uniform(*cfg.omega_bounds)
            s0 = np.random.uniform(0.01, 1.0)
            initial_conditions.append([cfg.theta_guard, w0, s0])
    else:
        # Compass Gait ICs
        for _ in range(num_orbits // 2):
            q = np.random.uniform(*cfg.cg_theta_bounds, size=2)
            dq = np.random.uniform(*cfg.cg_omega_bounds, size=2)
            initial_conditions.append(np.concatenate([q, dq, [0.0]]))
        for _ in range(num_orbits // 2):
            # Start on the cylinder (guard surface: th_ns + th_s = -2*phi, th_ns > th_s)
            th_s = np.random.uniform(cfg.cg_theta_bounds[0], -cfg.phi - 0.01)
            th_ns = -2*cfg.phi - th_s  # ensures th_ns > -phi > th_s
            dq = np.random.uniform(*cfg.cg_omega_bounds, size=2)
            s0 = np.random.uniform(0.01, 1.0)
            initial_conditions.append(np.concatenate([[th_ns, th_s], dq, [s0]]))

    for x0 in initial_conditions:
        orbit = sys.generate_tau_timeseries(x0, tau, points_per_orbit)
        X_list.append(orbit[:-1])
        Y_list.append(orbit[1:])
        
    X_tensor = torch.tensor(np.vstack(X_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.vstack(Y_list), dtype=torch.float32)
    
    return X_tensor, Y_tensor
