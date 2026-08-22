# Pipelines

Pipeline runbooks for the manuscript. Each subfolder contains a walkthrough
(`README.md`), a full specification (`SPEC.md`), and an orchestrator script.
Both orchestrators are explicitly disabled before they can train or write
artifacts. The compass renderer requires an analytic baseline its preceding
stages do not create; the rimless final rendering stage is unimplemented.

These folders do **not** duplicate code. They reference the canonical source
files at their existing locations and give the minimum context needed to run
them in the right order and understand what each step contributes.

## Available as individual stages

- **`compass_gait_relaxed/`**
  Learned relaxed-space embedding (CHyLL-adjacent, see SPEC.md) of the
  compass-gait biped + David Hien's cycling-signature computation on the
  unit tangent bundle. Stages 1–4 have individual commands. The figure stage
  additionally requires the absent analytic `_tgt025` baseline, so the
  orchestrator exits without running anything.

## In progress

- **`rimless_wheel_relaxed/`**
  Parallel of the compass pipeline applied to the rimless wheel. Stages 1–4
  are available as individual commands; stage 5 (the heatmap renderer) and
  end-to-end validation are missing, so `run_pipeline.sh` exits without
  running anything. See its `README.md` for current status.

## Not yet in production

- Purely analytic bridge pipeline (`_tgt025` etc.) — intended as a diagnostic
  baseline, but its stored barcode or a reproduction stage is required before
  the compass figure wrapper can run.
