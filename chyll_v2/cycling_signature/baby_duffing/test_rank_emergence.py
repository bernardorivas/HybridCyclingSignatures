"""Test Bernardo's hypothesis: rank distribution at fixed segment length encodes period.

A subsegment of length L on a period-n orbit traces L/(nT) full closures.
With max_rank >= 4 reported, the rank-k emergence curve in segment-length
should shift right by a factor of n for period-n orbits:

    period-1: rank-1 emerges at L ~ T, rank-2 at L ~ 2T, ...
    period-2: rank-1 emerges at L ~ 2T, rank-2 at L ~ 4T, ...
    period-4: rank-1 emerges at L ~ 4T, rank-2 at L ~ 8T, ...

This is independent of beta_1(Y) (which is 1 for all of them) and is the
real cascade-discrimination story.

For this test we use boxsize in the beta_1 = 1 plateau of each gamma
(verified via prior sweep). max_rank = 4 so we see up to rank-4 emergence.

Usage:
    python chyll_v2/cycling_signature/baby_duffing/test_rank_emergence.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


DATA_DIR = REPO_ROOT / "chyll_v2" / "cycling_signature" / "data" / "baby_duffing"

# Each entry: (gamma, boxsize). Boxsizes picked from prior beta_1 sweep
# to be inside the beta_1(Y) = 1 plateau for the corresponding gamma.
EXPERIMENTS = [
    (1.0, 0.30),
    (1.6, 0.30),
    (2.6, 0.30),
]

MAX_RANK = 4
SB_RADIUS = 1
R_MAX = 1.0
EVAL_RADIUS = 0.3
SEGMENT_LENGTHS = "20:10:1500"
N_RUNS = 200


def base_4d(gamma: float) -> str:
    return f"duffing_g{gamma:.2f}".replace(".", "p")


def main() -> int:
    for gamma, boxsize in EXPERIMENTS:
        base = base_4d(gamma)
        out_prefix = f"subsegments_{base}_maxrank{MAX_RANK}"
        cmd = [
            "julia",
            f"--project=time series/cycling_signature",
            "chyll_v2/cycling_signature/julia/run_subsegments.jl",
            "--data-dir", str(DATA_DIR.relative_to(REPO_ROOT)),
            "--base", base,
            "--boxsize", str(boxsize),
            "--sb-radius", str(SB_RADIUS),
            "--r-max", str(R_MAX),
            "--eval-radius", str(EVAL_RADIUS),
            "--segment-lengths", SEGMENT_LENGTHS,
            "--n-runs", str(N_RUNS),
            "--max-rank", str(MAX_RANK),
            "--out-prefix", out_prefix,
        ]
        print(f"\n=== gamma = {gamma}, boxsize = {boxsize}, max_rank = {MAX_RANK} ===")
        print("running:", " ".join(cmd[:4]), base)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        meta = (DATA_DIR / f"{out_prefix}_metadata.txt").read_text()
        for line in meta.splitlines():
            if line.startswith("beta1_Y="):
                print(f"  ==> gamma = {gamma}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
