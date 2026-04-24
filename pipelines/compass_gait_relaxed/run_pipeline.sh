#!/usr/bin/env bash
#
# End-to-end reproduction of the compass-gait relaxed-space learned
# cycling-signature result.
#
# The "relaxed-space" encoder E_theta is the CHyLL-adjacent extrusion
# architecture E(x, s) = (x, 0) + s * MLP(x, s) described in SPEC.md; it
# embeds the relaxed space X' = X union (G x [0,1]), not the hybrifold.
#
# Runs four stages, each idempotent:
#   1. Train the SuspensionNetworks on the compass-gait relaxed semiflow.
#   2. Export two encoded lifts (5-impact and 20-impact rollouts) + diagnostics.
#   3. Compute David Hien's cycling signature in Julia for each lift.
#   4. Render the manuscript heatmap figure + appendix variant.
#
# Usage (from repo root):
#     bash pipelines/compass_gait_relaxed/run_pipeline.sh
#
# The Python environment must have torch / scipy / numpy / matplotlib.
# Julia must be on PATH; the first run installs StepFunctions.jl and develops
# CyclingSignatures.jl into time series/cycling_signature/.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] Train relaxed-space encoder on compass-gait semiflow ==="
python scripts/train_compass.py

echo
echo "=== [2/5] Evaluate decoded rollout vs ground truth ==="
python scripts/evaluate_decoded_rollout_compass.py

echo
echo "=== [3/5] Export encoded lifts (5- and 20-impact rollouts) ==="
python "time series/cycling_signature/prepare_compass_cs_inputs_relaxed.py"
python "time series/cycling_signature/prepare_compass_cs_inputs_relaxed.py" \
    --n_impacts 20 --suffix _relaxed_n20

echo
echo "=== [4/5] Cycling signature in Julia (may take a while on first run) ==="
julia --project="time series/cycling_signature" \
      "time series/cycling_signature/run_cycling_signature.jl" \
      --base continuous_lift_relaxed
julia --project="time series/cycling_signature" \
      "time series/cycling_signature/run_cycling_signature.jl" \
      --base continuous_lift_relaxed_n20

echo
echo "=== [5/5] Render heatmap figure ==="
python "time series/cycling_signature/plot_compass_rank_heatmap.py"

echo
echo "Done. Main figure:"
echo "    figures/compass_gait/fig_compass_cycling_rank_heatmap.pdf"
echo "Appendix (with analytic reference):"
echo "    figures/compass_gait/fig_compass_cycling_rank_heatmap_appendix.pdf"
