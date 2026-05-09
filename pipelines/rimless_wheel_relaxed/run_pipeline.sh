#!/usr/bin/env bash
#
# End-to-end reproduction of the rimless-wheel relaxed-space learned
# cycling-signature result. Parallel to
# pipelines/compass_gait_relaxed/run_pipeline.sh.
#
# Stages 2-5 are not yet ported; this orchestrator runs only stage 1
# (training) today and prints a TODO list for the remaining stages.
#
# Usage (from repo root):
#     bash pipelines/rimless_wheel_relaxed/run_pipeline.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] Train relaxed-space encoder on rimless-wheel semiflow ==="
python scripts/train_rimless.py

echo
echo "=== [2/5] Evaluate decoded rollout vs ground truth ==="
python scripts/evaluate_decoded_rollout_rimless.py

echo
echo "=== [3/5] Export encoded lifts -- TODO ==="
echo "    Write time series/cycling_signature/prepare_rimless_cs_inputs_relaxed.py."

echo
echo "=== [4/5] Cycling signature in Julia -- ready when stage 3 lands ==="
echo "    julia --project=\"time series/cycling_signature\" \\"
echo "          \"time series/cycling_signature/run_cycling_signature.jl\" \\"
echo "          --base continuous_lift_relaxed_rimless"

echo
echo "=== [5/5] Render heatmap figure -- TODO ==="
echo "    Adapt time series/cycling_signature/plot_compass_rank_heatmap.py."

echo
echo "Stage 1 complete. Outputs:"
echo "    runs/rimless_wheel/model.pt"
echo "    figures/rimless_wheel/fig_rimless_optimization_losses.png"
