"""Reproduce Goswami et al. compass-gait period doubling.

Run:
    python compass_goswami.py             # CSV + diagnostic PNG files
    python compass_goswami.py --video     # also MP4 (or GIF without ffmpeg)

Dependencies: numpy, scipy, matplotlib. GIF fallback additionally uses Pillow.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp


OUT = Path("compass_goswami_output")
MAKE_VIDEO = "--video" in sys.argv
DT = 0.01
VIDEO_DURATION = 12.0
VIDEO_FPS = 50

# Choose any subset, e.g. SELECT = ["period2", "period4"].
SELECT = ["period1", "period2", "period4", "period8", "chaos"]

# Every x0 is a post-impact state [theta_ns, theta_s, dtheta_ns, dtheta_s].
# Period-2/4/8 states were obtained from the attracting return sets for the
# slopes plotted in Goswami et al. (1998), Fig. 10. The chaotic state is one
# point on the empirical 5.20 degree post-impact return cloud.
CASES = {
    "period1": {
        "phi_deg": 4.00,
        "period": 1,
        "x0": [-0.368711604132, 0.229085263972,
               -0.216411423691, -1.139899112182],
        "steps": 40,
    },
    "period2": {
        "phi_deg": 4.75,
        "period": 2,
        "x0": [-0.381789947562, 0.215983668622,
               -0.153332476821, -1.153674078371],
        "steps": 40,
    },
    "period4": {
        "phi_deg": 5.00,
        "period": 4,
        "x0": [-0.392348120036, 0.217815194837,
               -0.108006761939, -1.161444888863],
        "steps": 48,
    },
    "period8": {
        "phi_deg": 5.02,
        "period": 8,
        "x0": [-0.394646896303, 0.219415839402,
               -0.097321131256, -1.162198996812],
        "steps": 64,
    },
    "chaos": {
        "phi_deg": 5.20,
        "period": None,
        "x0": [-0.401092708453, 0.219578466246,
               -0.087963122288, -1.170237773352],
        "steps": 200,
    },
}

# Nominal robot: total mass 20 kg, mu = 2, beta = 1, l = 1 m.
P = {"m": 5.0, "mH": 10.0, "l": 1.0,
     "a": 0.5, "b": 0.5, "g": 9.81}


def vector_field(_t: float, x: np.ndarray) -> np.ndarray:
    """Swing-phase equation M(q) qdd + N(q,qd) qd + G(q) = 0."""
    th_ns, th_s, dth_ns, dth_s = x
    m, mH, l, a, b, g = (P[k] for k in ("m", "mH", "l", "a", "b", "g"))
    delta = th_s - th_ns

    M = np.array([
        [m * b**2, -m * l * b * np.cos(delta)],
        [-m * l * b * np.cos(delta), (mH + m) * l**2 + m * a**2],
    ])
    N = np.array([
        [0.0, m * l * b * np.sin(delta) * dth_s],
        [-m * l * b * np.sin(delta) * dth_ns, 0.0],
    ])
    G = np.array([
        m * b * g * np.sin(th_ns),
        -(mH * l + m * a + m * l) * g * np.sin(th_s),
    ])
    dq = np.array([dth_ns, dth_s])
    ddq = np.linalg.solve(M, -(N @ dq + G))
    return np.r_[dq, ddq]


def make_guard(phi: float):
    def guard(_t: float, x: np.ndarray) -> float:
        # The ordering condition suppresses the same guard immediately after
        # impact, when the former stance leg has become the swing leg.
        if x[0] - x[1] > 0.01:
            return x[0] + x[1] + 2.0 * phi
        return 1.0

    guard.terminal = True
    guard.direction = -1
    return guard


def reset_map(x_minus: np.ndarray) -> np.ndarray:
    """Plastic foot strike, angular-momentum impact map, and leg-role swap."""
    th_ns, th_s, dth_ns, dth_s = x_minus
    m, mH, l, a, b = (P[k] for k in ("m", "mH", "l", "a", "b"))
    c = np.cos(th_s - th_ns)  # cos(2 alpha)

    Qm = np.array([
        [-m * a * b, (mH * l**2 + 2.0 * m * a * l) * c - m * a * b],
        [0.0, -m * a * b],
    ])
    Qp = np.array([
        [m * b * (b - l * c),
         m * l * (l - b * c) + mH * l**2 + m * a**2],
        [m * b**2, -m * b * l * c],
    ])
    dq_plus = np.linalg.solve(Qp, Qm @ np.array([dth_ns, dth_s]))
    return np.array([th_s, th_ns, dq_plus[0], dq_plus[1]])


def geometry(x: np.ndarray, support_xy: np.ndarray):
    """World coordinates of hip, old/swing foot, and stance foot."""
    th_ns, th_s = x[:2]
    l = P["l"]
    hip = support_xy + np.array([-l * np.sin(th_s), l * np.cos(th_s)])
    swing = hip + np.array([l * np.sin(th_ns), -l * np.cos(th_ns)])
    return hip, swing, support_xy


def simulate(phi_deg: float, x0, n_steps: int):
    phi = np.deg2rad(phi_deg)
    guard = make_guard(phi)
    x = np.asarray(x0, dtype=float)
    global_t = 0.0
    support = np.zeros(2)
    segments, returns = [], []

    for step in range(n_steps):
        sol = solve_ivp(
            vector_field, (0.0, 5.0), x, events=guard, dense_output=True,
            rtol=1e-10, atol=1e-12, max_step=0.005,
        )
        if sol.status != 1:
            raise RuntimeError(f"No foot strike found at step {step}")

        T = float(sol.t_events[0][0])
        local_t = np.arange(0.0, T, DT)
        local_t = np.r_[local_t, T]
        states = sol.sol(local_t).T
        segments.append({
            "step": step,
            "t": global_t + local_t,
            "x": states,
            "support": support.copy(),
        })

        x_minus = states[-1]
        x_plus = reset_map(x_minus)
        global_t += T
        returns.append({"step": step + 1, "t": global_t,
                        "T": T, "x": x_plus.copy()})

        # The impact point becomes the next stance-foot origin.
        _, new_support, _ = geometry(x_minus, support)
        support = new_support
        x = x_plus

    return segments, returns


def save_csv(label: str, segments, returns):
    OUT.mkdir(exist_ok=True)
    with (OUT / f"{label}_timeseries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "step", "event", "theta_ns", "theta_s",
                    "dtheta_ns", "dtheta_s", "hip_x", "hip_y",
                    "swing_x", "swing_y", "support_x", "support_y"])
        for seg in segments:
            for j, (t, x) in enumerate(zip(seg["t"], seg["x"])):
                event = "post_impact" if j == 0 else (
                    "pre_impact" if j == len(seg["t"]) - 1 else "flow")
                hip, swing, support = geometry(x, seg["support"])
                w.writerow([t, seg["step"], event, *x,
                            *hip, *swing, *support])

    with (OUT / f"{label}_returns.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "impact_time", "step_period", "theta_ns",
                    "theta_s", "dtheta_ns", "dtheta_s"])
        for r in returns:
            w.writerow([r["step"], r["t"], r["T"], *r["x"]])


def plot_diagnostics(label: str, phi_deg: float, segments, returns):
    t = np.concatenate([s["t"] for s in segments])
    x = np.concatenate([s["x"] for s in segments])
    R = np.array([r["x"] for r in returns])
    T = np.array([r["T"] for r in returns])

    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    ax[0, 0].plot(t, x[:, 0], label=r"$\theta_{ns}$")
    ax[0, 0].plot(t, x[:, 1], label=r"$\theta_s$")
    ax[0, 0].set(xlabel="time (s)", ylabel="angle (rad)")
    ax[0, 0].legend()

    ax[0, 1].plot(t, x[:, 2], label=r"$\dot\theta_{ns}$")
    ax[0, 1].plot(t, x[:, 3], label=r"$\dot\theta_s$")
    ax[0, 1].set(xlabel="time (s)", ylabel="angular velocity (rad/s)")
    ax[0, 1].legend()

    ax[1, 0].plot(np.arange(1, len(T) + 1), T, ".-", ms=3)
    ax[1, 0].set(xlabel="step number", ylabel="step period (s)")

    ax[1, 1].plot(R[:-1, 0], R[1:, 0], ".", ms=4)
    lo, hi = R[:, 0].min(), R[:, 0].max()
    pad = max(0.002, 0.05 * (hi - lo))
    ax[1, 1].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
    ax[1, 1].set(xlabel=r"$\theta_{ns,k}$", ylabel=r"$\theta_{ns,k+1}$",
                 xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad))

    fig.suptitle(label + rf": compass gait at $\phi={phi_deg:.2f}^\circ$")
    fig.savefig(OUT / f"{label}_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_physical_leg_phase(ax, segments, returns, n_steps):
    """Plot phase curves while preserving physical leg identity at impacts."""
    colors = {"A": "#0072B2", "B": "#D55E00"}
    n_steps = min(n_steps, len(segments), len(returns))

    for k in range(n_steps):
        seg = segments[k]
        x = seg["x"]
        x_minus = x[-1]
        x_plus = returns[k]["x"]
        ns_is_a = (k % 2 == 0)

        if ns_is_a:
            a_curve, b_curve = x[:, [0, 2]], x[:, [1, 3]]
            a_pre, a_post = x_minus[[0, 2]], x_plus[[1, 3]]
            b_pre, b_post = x_minus[[1, 3]], x_plus[[0, 2]]
        else:
            a_curve, b_curve = x[:, [1, 3]], x[:, [0, 2]]
            a_pre, a_post = x_minus[[1, 3]], x_plus[[0, 2]]
            b_pre, b_post = x_minus[[0, 2]], x_plus[[1, 3]]

        ax.plot(a_curve[:, 0], a_curve[:, 1], color=colors["A"], lw=1.25,
                label="physical leg A" if k == 0 else None)
        ax.plot(b_curve[:, 0], b_curve[:, 1], color=colors["B"], lw=1.25,
                label="physical leg B" if k == 0 else None)
        ax.plot([a_pre[0], a_post[0]], [a_pre[1], a_post[1]],
                color=colors["A"], lw=0.8, ls="--", alpha=0.65)
        ax.plot([b_pre[0], b_post[0]], [b_pre[1], b_post[1]],
                color=colors["B"], lw=0.8, ls="--", alpha=0.65)

    ax.set_xlim(-0.48, 0.58)
    ax.set_ylim(-3.10, 2.90)
    ax.grid(True, color="#D0D0D0", lw=0.6, alpha=0.65)
    ax.set_xlabel(r"angular position $\theta$ (rad)")
    ax.set_ylabel(r"angular velocity $\dot\theta$ (rad/s)")


def plot_figure10_atlas(results):
    """Modern counterpart of Goswami et al. Figure 10, plus period 1."""
    order = ["period1", "period2", "period4", "period8", "chaos"]
    titles = {
        "period1": "Period 1",
        "period2": "Period 2",
        "period4": "Period 4",
        "period8": "Period 8",
        "chaos": "Chaotic gait",
    }
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    flat = axes.ravel()

    for i, label in enumerate(order):
        cfg = CASES[label]
        result = results[label]
        n_show = 100 if label == "chaos" else max(2, cfg["period"])
        plot_physical_leg_phase(flat[i], result["segments"],
                                result["returns"], n_show)
        flat[i].set_title(
            rf"{letters[i]} {titles[label]}  |  $\phi={cfg['phi_deg']:.2f}^\circ$",
            fontsize=12,
        )

    flat[0].legend(loc="upper right", frameon=True, fontsize=9)
    flat[5].axis("off")
    flat[5].text(
        0.03, 0.88,
        "Compass-gait period-doubling cascade\n\n"
        r"Nominal model: $\mu=2$, $\beta=1$, $l=1$ m" "\n\n"
        "Solid curves: continuous swing dynamics\n"
        "Dashed curves: instantaneous impact reset\n\n"
        "Colors follow physical legs across the\n"
        "stance/nonstance label swap.",
        va="top", ha="left", fontsize=12, linespacing=1.35,
    )

    fig.suptitle("Passive Compass Gait: Phase-Plane Period Doubling",
                 fontsize=17, fontweight="bold")
    fig.savefig(OUT / "figure10_modern_phase_portraits.png", dpi=220)
    fig.savefig(OUT / "figure10_modern_phase_portraits.pdf")
    plt.close(fig)


def save_video(label: str, phi_deg: float, segments):
    """Save every gait on the same uniformly sampled physical-time window."""
    frames = []
    target_times = np.linspace(
        0.0, VIDEO_DURATION, round(VIDEO_DURATION * VIDEO_FPS) + 1
    )
    seg_index = 0
    for t in target_times:
        while (seg_index + 1 < len(segments)
               and t > segments[seg_index]["t"][-1] + 1e-12):
            seg_index += 1
        seg = segments[seg_index]
        if t > seg["t"][-1] + 1e-9:
            raise RuntimeError(
                f"Trajectory for {label} is shorter than {VIDEO_DURATION} s"
            )
        x = np.array([
            np.interp(t, seg["t"], seg["x"][:, j]) for j in range(4)
        ])
        hip, swing, support = geometry(x, seg["support"])
        frames.append((t, hip, swing, support))

    fig, ax = plt.subplots(figsize=(7, 4))
    leg_s, = ax.plot([], [], "o-", lw=3, color="#0072B2", label="stance")
    leg_ns, = ax.plot([], [], "o-", lw=3, color="#D55E00", label="swing")
    ground, = ax.plot([], [], "k-", lw=1)
    clock = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("horizontal position (m)")
    ax.set_ylabel("height (m)")
    ax.legend(loc="upper right")

    phi = np.deg2rad(phi_deg)

    def update(i):
        t, hip, swing, support = frames[i]
        center = hip[0]
        gx = np.array([center - 2.0, center + 2.0])
        gy = -np.tan(phi) * gx
        ground.set_data(gx, gy)
        leg_s.set_data([support[0], hip[0]], [support[1], hip[1]])
        leg_ns.set_data([hip[0], swing[0]], [hip[1], swing[1]])
        ax.set_xlim(center - 1.4, center + 1.4)
        # Follow the descending ground vertically as well as horizontally.
        # Without this, the fixed world-coordinate y limits eventually leave
        # the slope below the frame and make the walker look underground.
        ground_y_at_center = -np.tan(phi) * center
        ax.set_ylim(ground_y_at_center - 0.25,
                    ground_y_at_center + 1.25)
        clock.set_text(f"{label}, slope={phi_deg:.2f} deg, t={t:.2f} s")
        return leg_s, leg_ns, ground, clock

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / VIDEO_FPS, blit=False)
    if shutil.which("ffmpeg"):
        anim.save(OUT / f"{label}.mp4",
                  writer=FFMpegWriter(fps=VIDEO_FPS))
    else:
        anim.save(OUT / f"{label}.gif",
                  writer=PillowWriter(fps=VIDEO_FPS))
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    for label in SELECT:
        cfg = CASES[label]
        print(f"Simulating {label} at {cfg['phi_deg']:.2f} deg ...")
        segments, returns = simulate(cfg["phi_deg"], cfg["x0"], cfg["steps"])
        results[label] = {"segments": segments, "returns": returns}
        save_csv(label, segments, returns)
        plot_diagnostics(label, cfg["phi_deg"], segments, returns)
        if MAKE_VIDEO:
            save_video(label, cfg["phi_deg"], segments)
        tail = np.array([r["T"] for r in returns[-8:]])
        print("  last step periods:", np.array2string(tail, precision=6))

    if all(label in results for label in
           ["period1", "period2", "period4", "period8", "chaos"]):
        plot_figure10_atlas(results)


if __name__ == "__main__":
    main()
