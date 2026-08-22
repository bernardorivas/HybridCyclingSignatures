#!/usr/bin/env bash
#
# NON-RUNNABLE PARTIAL ORCHESTRATOR for the compass-gait relaxed-space
# learned cycling-signature result.
#
# The "relaxed-space" encoder E_theta is the generic residual architecture
# E(x) = pad(x, d) + MLP(x) described in SPEC.md. It embeds the relaxed space
# X' = X union (G x [0,1]), not the hybrifold.
#
# The individual implementations for stages 1-4 are present. Stage 5 also
# requires data/compass_gait/barcode_H1_analytic_tgt025.csv, but no stage in
# this pipeline creates that analytic baseline and the file is not stored in
# the repository. This script therefore exits before training or writing
# artifacts. The commands below are retained as a stage inventory.
#
# Intended stages:
#   1. Train the SuspensionNetworks on the compass-gait relaxed semiflow.
#   2. Evaluate a decoded held-out rollout.
#   3. Export two encoded lifts (5-impact and 20-impact rollouts) + diagnostics.
#   4. Compute David Hien's cycling signature in Julia for each lift.
#   5. Render the manuscript heatmap figure + appendix variant.
#
# Usage (from repo root):
#     bash pipelines/compass_gait_relaxed/run_pipeline.sh
#
# The Python environment must have torch / scipy / numpy / matplotlib.
# Julia must be on PATH. The locked project is period_doubling/julia/;
# do not use the Windows-pathed time series/cycling_signature/Manifest.toml.
#
set -euo pipefail

echo "NON-RUNNABLE: compass pipeline is partial; run validated stages manually." >&2
echo "Missing analytic input: data/compass_gait/barcode_H1_analytic_tgt025.csv (required by stage 5)." >&2
exit 2

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
julia --project="period_doubling/julia" \
      "time series/cycling_signature/run_cycling_signature.jl" \
      --base continuous_lift_relaxed
julia --project="period_doubling/julia" \
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
