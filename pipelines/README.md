# Pipelines

Production pipelines for the manuscript. Each subfolder contains a walkthrough
(`README.md`), a full specification (`SPEC.md`), and an orchestrator script
that reproduces the published artifacts end-to-end from a fresh clone.

These folders do **not** duplicate code. They reference the canonical source
files at their existing locations and give the minimum context needed to run
them in the right order and understand what each step contributes.

## Available

- **`compass_gait_relaxed/`**
  Learned relaxed-space embedding (CHyLL-adjacent, see SPEC.md) of the
  compass-gait biped + David Hien's cycling-signature computation on the
  unit tangent bundle. Produces the manuscript figure
  `figures/compass_gait/fig_compass_cycling_rank_heatmap.{pdf,png}` and its
  3-panel appendix variant.

## In progress

- **`rimless_wheel_relaxed/`**
  Parallel of the compass pipeline applied to the rimless wheel. Stage 1
  (training) is live via `scripts/train_rimless.py`; stages 2–5 (decoded
  rollout, lift export, Julia, heatmap) are not yet ported. See its
  `README.md` for current status.

## Not yet in production

- Purely analytic bridge pipeline (`_tgt025` etc.) — kept as a diagnostic
  baseline only; the comparison is included in the compass appendix figure.
