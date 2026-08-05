"""Summarize cycling signatures across the Goswami period-doubling cascade."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = (
    REPO_ROOT / "chyll_v2" / "cycling_signature" / "data"
    / "compass_gait_goswami_csv"
)
FIGURE_ROOT = REPO_ROOT / "chyll_v2" / "cycling_signature" / "figures"

CASES = ["period1", "period2", "period4", "period8", "chaos"]
LABELS = {
    "period1": "period 1",
    "period2": "period 2",
    "period4": "period 4",
    "period8": "period 8",
    "chaos": "chaos",
}
COLORS = {
    "period1": "#0072B2",
    "period2": "#E69F00",
    "period4": "#009E73",
    "period8": "#D55E00",
    "chaos": "#CC79A7",
}
MARKERS = {
    "period1": "o",
    "period2": "s",
    "period4": "^",
    "period8": "D",
    "chaos": "X",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def count_latent_returns(case: str, tolerance: float = 1e-5) -> int | None:
    if case == "chaos":
        return None
    base = f"continuous_lift_goswami_{case}_vfield"
    points = np.load(DATA_ROOT / case / f"{base}_postimpact_latent.npy")
    representatives: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - rep) <= tolerance for rep in representatives):
            representatives.append(point)
    return len(representatives)


def load_scale_sweep(case: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / case / f"cycling_goswami_{case}_beta1_scale_sweep.csv"
    rows = read_rows(path)
    return (
        np.array([float(row["boxsize"]) for row in rows]),
        np.array([int(row["beta1_Y"]) for row in rows]),
    )


def load_rank_probability(case: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / case / f"cycling_goswami_{case}_rank_at_radius.csv"
    rows = read_rows(path)
    lengths = np.array([int(row["segment_length"]) for row in rows])
    totals = np.array(
        [sum(int(value) for key, value in row.items() if key.startswith("rank"))
         for row in rows]
    )
    rank1 = np.array([int(row["rank1"]) for row in rows])
    return lengths, rank1 / totals


def write_summary(path: Path) -> None:
    fields = [
        "case", "expected_period", "latent_return_count", "beta1_at_0.30",
        "max_beta1_common_scales", "first_length_rank1_probability_ge_0.8",
        "interpretation",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in CASES:
            boxsize, beta1 = load_scale_sweep(case)
            lengths, probability = load_rank_probability(case)
            common = boxsize >= 0.10
            onset = lengths[np.flatnonzero(probability >= 0.8)[0]]
            return_count = count_latent_returns(case)
            writer.writerow(
                {
                    "case": case,
                    "expected_period": "" if case == "chaos" else case[6:],
                    "latent_return_count": (
                        "non-finite cloud" if return_count is None else return_count
                    ),
                    "beta1_at_0.30": beta1[np.isclose(boxsize, 0.30)][0],
                    "max_beta1_common_scales": int(beta1[common].max()),
                    "first_length_rank1_probability_ge_0.8": int(onset),
                    "interpretation": (
                        "aperiodic return cloud"
                        if case == "chaos"
                        else "period recovered by latent Poincare returns"
                    ),
                }
            )


def plot(path_png: Path, path_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), constrained_layout=True)

    ax = axes[0]
    for case in CASES:
        boxsize, beta1 = load_scale_sweep(case)
        ax.plot(
            boxsize, beta1, color=COLORS[case], marker=MARKERS[case],
            lw=1.7, ms=5, label=LABELS[case],
        )
    ax.axhline(1, color="#4D4D4D", lw=1, ls=":")
    ax.set(
        title=r"Comparison-space complexity $\beta_1(Y)$",
        xlabel="box size",
        ylabel=r"$\beta_1(Y)$",
    )
    ax.set_xticks([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    ax.grid(alpha=0.22)

    ax = axes[1]
    for case in CASES:
        lengths, probability = load_rank_probability(case)
        ax.plot(
            lengths, probability, color=COLORS[case], marker=MARKERS[case],
            lw=1.7, ms=4, label=LABELS[case],
        )
    ax.set(
        title="Rank-1 subsegment signatures",
        xlabel="subsegment length (samples)",
        ylabel="fraction with rank 1",
        ylim=(-0.04, 1.04),
    )
    ax.grid(alpha=0.22)

    ax = axes[2]
    periodic_cases = CASES[:-1]
    counts = [count_latent_returns(case) for case in periodic_cases]
    bars = ax.bar(
        [LABELS[case] for case in periodic_cases], counts,
        color=[COLORS[case] for case in periodic_cases],
        edgecolor="#333333", linewidth=0.7,
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, count + 0.18, str(count),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set(
        title="Distinct latent Poincaré returns",
        xlabel="steady gait",
        ylabel="return-state count",
        ylim=(0, 9),
    )
    ax.grid(axis="y", alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.055),
        ncol=5, frameon=False,
    )
    fig.text(
        0.5, -0.035,
        "Cycling parameters: box size 0.30, sphere-bundle radius 1, "
        "evaluation radius 0.15; 30 random subsegments per length. "
        "The Poincaré count is the direct period diagnostic.",
        ha="center", fontsize=9, color="#444444",
    )
    fig.savefig(path_png, dpi=210, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    summary = DATA_ROOT / "cycling_signature_period_doubling_summary.csv"
    png = FIGURE_ROOT / "goswami_period_doubling_cycling_signatures.png"
    pdf = FIGURE_ROOT / "goswami_period_doubling_cycling_signatures.pdf"
    write_summary(summary)
    plot(png, pdf)
    print(f"wrote {summary}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
