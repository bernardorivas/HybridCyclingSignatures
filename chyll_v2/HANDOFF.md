# chyll_v2 handoff brief

Context for any agent picking up this work mid-stream. Last updated 2026-05-13.

## Context

The user (Kaito Iwasaki) is the lead author of a manuscript on learning
continuous embeddings of hybrid dynamical systems via the suspension /
mapping-cylinder construction. The reference pipeline lives in `src/` and uses
a 6-term loss derived from a commutative diagram:
`L_dyn + L_glue + L_rec + L_seam + L_conf + L_coll`.

The work is being compared against Teng, Liu, Sreenath (ICML 2026)
*Embedding Hybrid Systems into Continuous Latent Vector Fields*. PDF at
`local_docs/Learning_Cycling_Signatures_of_Hybrid_Systems/hybrid_cyclingsignatures/references/Sangli-ICML-Fed.18 (1).pdf`.

The older CHyLL 2025 version (Teng et al. 2025) of that paper is at the same
path under `CHyLL.pdf`. The 2025 paper had a conformal loss similar to our
`L_conf`; the 2026 paper replaced it with a velocity-compatibility loss `L_v`
and then showed via ablation that `L_v` is itself redundant under network
Lipschitz continuity.

## Goal

Reimplement Sangli's approach from scratch in a fresh top-level directory,
run it on the systems we use, produce a fair head-to-head comparison.

## Design constraints (from the user)

1. **Do not modify** `src/`. The new code is fully separate so deletion is a
   clean fallback. It lives in `chyll_v2/` and imports nothing from `src/`.
2. **Use torchdiffeq** (not hand-rolled RK4) so the latent flow is integrated
   exactly as in Sangli's algorithm. `torchdiffeq.odeint_adjoint` is the
   default.
3. **Keep the mapping cylinder in the data.** Even though Sangli's paper uses
   the hybrifold `M_H`, the user wants to test Sangli's *loss recipe +
   continuous latent ODE* on the same data manifold
   `X' = X ∪ (G × [0, 1])` that the `src/` pipeline uses. The encoder sees
   the augmented state `(x, s) ∈ R^{n+1}`. Latent dim `> 2 (n + 1)`.

## Architecture

- `chyll_v2/chyll_v2/config.py` — `CHyLLv2Config` dataclass with
  `base_dim`, `state_dim = base_dim + 1`, `latent_dim`, curriculum,
  torchdiffeq settings. `make_default(name)` returns presets for
  rimless wheel, bouncing ball, compass gait.
- `chyll_v2/chyll_v2/systems/base.py` — `tau`-semiflow simulator on `X'`
  (mirrors the logic of `src/system.py:38-109` without copying), plus
  `sample_guard_point` and `sample_guard_pairs` for symbolic gluing-pair
  generation.
- `chyll_v2/chyll_v2/systems/rimless_wheel.py` — `alpha = 0.4`,
  `gamma = 0.2`, matching `src/config.py`.
- `chyll_v2/chyll_v2/data.py` — `TrajectorySliceDataset` for windowed
  batches of cylinder trajectories; `GuardSampler` for fresh symbolic
  guard pairs each step. The gluing index set is exact (we know `G`),
  not Lipschitz-thresholded the way Sangli's hybrifold-setting code is.
- `chyll_v2/chyll_v2/networks.py` — Encoder / VectorField / Decoder MLPs
  with optional SIREN-style sin activation at the final hidden layer
  (`use_sin_last_layer`). Encoder uses pad-identity residual:
  `f(x, s) ≈ pad_m(x, s)` at init.
- `chyll_v2/chyll_v2/ode.py` — torchdiffeq wrapper.
  `rk4` / `dopri5`, `odeint` / `odeint_adjoint`.
- `chyll_v2/chyll_v2/losses.py` — five-term composite. `L_g` = MSE between
  `E(g, 1)` and `E(r(g), 0)` on sampled guards. `L_v` uses
  `torch.func.jvp` to compute `∇_x E(r(g), 0) · V(r(g))` and compares it
  to a finite-difference `∂_s E(g, 1)`. Conformal regularizer
  intentionally absent (Sangli dropped it from CHyLL 2025 → ICML 2026).
- `chyll_v2/chyll_v2/train.py` — Algorithm 1 of the paper. Curriculum
  `(5, 10, 25, 50, 100)` × 1000 steps each, AdamW + cosine LR, grad clip 2.0.

## Tweaks applied during Phase 1

- `collapse_threshold` (`Lambda`) dropped from 1.0 to 0.3. The latent dim is
  `<= 11` and natively-bounded states make `Lambda = 1.0` over-aggressive
  against the pad-identity init — `L_c` dominated the loss budget.
- `w_v = 0`. Matches Sangli's own ablation (`L_v` redundant).
- Smoke runs use `--ode-method rk4 --no-adjoint` for speed.
- Full run uses `rk4 + adjoint`. `dopri5` works but is too slow on CPU.

## Phase 1 result (rimless wheel, full run)

5000 updates, ~1h 43m on CPU. Final losses:

| Term | Value |
|---|---|
| `L_x` | ≈ 10⁻³ |
| `L_z` | ≈ 3·10⁻⁴ |
| `L_g` | ≈ 10⁻⁵ |
| `L_v` | 0 (disabled) |
| `L_c` | 0 (saturated to floor) |
| total | ≈ 1.4·10⁻³ |

The rollout reproduces 5+ cycles of the mapping-cylinder trajectory exactly,
including the cylinder plateau in `theta`, the reset jump from
`(0.6, 1) → (-0.2, 0)`, and the sawtooth in `s`. Extrapolation beyond the
H=100 training horizon (5 s) out to 7.5 s holds. Artifacts:

- `chyll_v2/runs/rimless_wheel/{model.pt, train_log.jsonl, config.json}`
- `chyll_v2/figures/rimless_wheel/{loss_history, rollout_vs_truth, phase_portrait_truth_settled}.png`

## Surprise

Expected Sangli's approach (no `L_seam`) to fail at the seam discontinuity on
the cylinder. It doesn't, on reconstruction MSE. This forces the manuscript's
`L_seam` argument to shift from "reconstruction breaks without it" to "the
unit tangent bundle is corrupted in ways that wreck cycling-signature ranks
downstream." That hypothesis is not yet tested.

## Phase 1b result (rimless wheel cycling-signature rank)

Added `chyll_v2/cycling_signature/export/prepare_rimless_lift.py`, a
standalone exporter for David Hien's cycling-signature pipeline. It imports
nothing from `src/`; if PyTorch is not installed in the active Python, it
reads the `torch.save(state_dict)` zip archive directly and runs the CHyLL v2
MLPs in NumPy. It writes the Julia interop files under
`chyll_v2/cycling_signature/data/rimless_wheel/`.

Two tangent conventions were tested on the same encoded 5-impact limit-cycle
lift (`960 x 7`):

| base | tangent source | result |
|---|---|---|
| `continuous_lift_chyll_v2` | finite differences of `E(X')` | rank 0 everywhere; `beta_1(Y)=0` everywhere on the standard grid |
| `continuous_lift_chyll_v2_vfield` | learned latent vector field `V_theta(z)` | rank 1 only at coarse cells `(boxsize, sb_radius)=(0.3,1),(0.2,1)`; rank 0 elsewhere |

Artifacts:

- `chyll_v2/cycling_signature/data/rimless_wheel/continuous_lift_chyll_v2_{positions,tangents}.csv`
- `chyll_v2/cycling_signature/data/rimless_wheel/barcode_H1_chyll_v2.csv`
- `chyll_v2/cycling_signature/data/rimless_wheel/continuous_lift_chyll_v2_vfield_{positions,tangents}.csv`
- `chyll_v2/cycling_signature/data/rimless_wheel/barcode_H1_chyll_v2_vfield.csv`
- diagnostic reports `report_continuous_lift_chyll_v2*.txt`

Interpretation: with the same finite-difference UT convention used by the
existing Section-4 pipeline, CHyLL v2 loses the rimless-wheel cycling signature
despite good rollout/reconstruction. If one instead supplies CHyLL's own
learned vector field as the tangent, the expected rank appears at coarse scale,
so the manuscript claim should be phrased carefully: the learned embedding
trajectory's finite-difference unit tangent bundle is the failing object, not
necessarily the Neural-ODE vector field itself.

## Open issues / next moves

1. Decide whether the head-to-head table should report finite-difference UT
   rank only (matching the existing Section-4 pipeline) or both
   finite-difference and `V_theta`-tangent ranks for CHyLL v2.
2. Keep all CHyLL v2 topology postprocessing under `chyll_v2/cycling_signature/`
   to avoid mixing training code, one-orbit barcode checks, and future
   David-style subsegment experiments.
3. `chyll_v2/cycling_signature/julia/run_subsegments.jl` is implemented and
   smoke-tested on both `continuous_lift_chyll_v2` and
   `continuous_lift_chyll_v2_vfield`. It writes rank heatmaps, rank-at-radius
   summaries, rank-1 cycling-space counts, birth summaries, segment starts,
   and metadata as CSV/TXT under `chyll_v2/cycling_signature/data/rimless_wheel/`.
   Large grids are expensive; run pilot grids before increasing `n_runs`.
4. Pilot grids have been run:
   `subsegments_chyll_v2_diff_pilot` and
   `subsegments_chyll_v2_vfield_pilot`, each with segment lengths
   `20:20:300`, `n_runs=25`, `r_subdivisions=101`, `(boxsize,sb_radius)=(0.3,1)`.
   Finite-difference UT is rank 0 everywhere (`beta_1(Y)=0`). `V_theta`
   tangents have `beta_1(Y)=1`; rank 1 appears cleanly from segment length
   `200` onward, with `25/25` rank-1 segments at the evaluation radius `0.3`.
5. First publication-style pilot plotter is implemented at
   `chyll_v2/cycling_signature/plot/plot_rimless_subsegments.py`. It writes
   `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_chyll_v2_subsegments_pilot.{pdf,png}`.
   Treat this as a pilot figure; the manuscript pass should rerun the same
   plotter after a denser subsegment grid.
6. Dense zoom grids have been run:
   `subsegments_chyll_v2_diff_zoom100` and
   `subsegments_chyll_v2_vfield_zoom100`, each with segment lengths
   `20:10:300`, `n_runs=100`, `r_max=0.002`, `eval_radius=0.001`,
   `r_subdivisions=161`, `(boxsize,sb_radius)=(0.3,1)`. The zoom scale is
   important: the flow-tangent birth radii are around `10^-5` to `10^-3`,
   so the old `r_max=0.3` pilot collapsed the transition into a single row.
7. Manuscript-style rimless cycling-signature figures are implemented at
   `chyll_v2/cycling_signature/plot/plot_rimless_signature_figures.py`.
   It writes
   `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_signature_rank_distribution.{pdf,png}`
   and
   `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_signature_rank_heatmaps.{pdf,png}`.
   These follow David's Fig. 10 / Fig. 15 grammar more closely than the pilot.
8. `prepare_rimless_lift.py` now supports `--tangent-mode tagaware` for
   finite-difference tangents. The old naive FD exporter differentiated across
   the entire concatenated lift, including arc/bridge boundary chords. The
   tag-aware mode differentiates only within each exported arc or bridge piece.
   Tag-aware FD exports were created for both `w_v=0` and `w_v=1`:
   `continuous_lift_chyll_v2_tagaware` and
   `continuous_lift_chyll_v2_wv1_tagaware`. David-style subsegment runs
   `subsegments_chyll_v2_tagaware` and
   `subsegments_chyll_v2_wv1_tagaware` still have `beta_1(Y)=0` and rank 0
   for every segment length at `r=0.001`. Conclusion: cross-boundary chords
   were a bug, but not the root cause of the FD-tangent failure.
9. `chyll_v2/cycling_signature/diagnostics/tangent_coherence.py` measures
   same-piece tangent rotation. Output files:
   `chyll_v2/cycling_signature/data/rimless_wheel/tangent_coherence_all.{txt,csv}`.
   The tag-aware FD tangents are locally coherent: median same-piece rotation
   is `0.371` degrees for `w_v=0` and `0.340` degrees for `w_v=1`; p95 is
   `3.521` and `4.172` degrees. The problem is at guard entry into the
   cylinder: tag-aware FD arc-to-bridge boundary angles are `159.78` degrees
   (`w_v=0`) and `166.92` degrees (`w_v=1`), while bridge-to-next-arc angles
   are only `8.06` and `9.90` degrees. This suggests the FD rank-0 failure is
   not local Jacobian noise along pieces; it is a global / guard-entry tangent
   discontinuity in the exported mapping-cylinder trajectory.
10. **Two-seam velocity result**: the entry-seam diagnosis is confirmed.
   `chyll_v2/chyll_v2/data.py` now detects both gluing exits and cylinder
   entries; `chyll_v2/chyll_v2/losses.py` applies `L_v` on their union while
   preserving `L_g` on the gluing exit mask. Full run:
   `chyll_v2/runs/rimless_wheel_wv1_twoseam`. At the end of training,
   losses were in the same regime as the one-seam run (`L_x` about
   `2.6e-3`, `L_z` about `1.1e-3`, `L_g` near zero, `L_v` about `6e-4`).
   Exported lifts:
   `continuous_lift_chyll_v2_wv1_twoseam_tagaware` and
   `continuous_lift_chyll_v2_wv1_twoseam_vfield`. Tangent coherence:
   tag-aware FD arc-to-bridge boundary angle dropped from `166.92` degrees to
   `0.756` degrees, and bridge-to-arc dropped from `9.90` degrees to
   `0.757` degrees. `V_theta` has `0.187` degrees at entry and `0.530`
   degrees at exit.
11. David-style subsegment ranks for the two-seam run:
   `subsegments_chyll_v2_wv1_twoseam_tagaware` and
   `subsegments_chyll_v2_wv1_twoseam_vfield`, each with segment lengths
   `20:10:300`, `n_runs=100`, `r_max=0.002`, `eval_radius=0.001`,
   `r_subdivisions=161`, `(boxsize,sb_radius)=(0.3,1)`. Both rows now have
   `beta_1(Y)=1`. Rank 1 first appears at segment length `200`, peaks at
   `90/100` sampled segments at length `270`, and does not reach 100% at this
   evaluation radius. The tag-aware FD row and the `V_theta` row have identical
   rank-at-radius counts under the fixed random seed. Generated figures:
   `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_signature_rank_distribution_twoseam.{pdf,png}`
   and
   `chyll_v2/cycling_signature/figures/rimless_wheel/fig_rimless_signature_rank_heatmaps_twoseam.{pdf,png}`.
12. **Phase 2**: implement `chyll_v2/chyll_v2/systems/bouncing_ball.py`
   (dynamics `dot{x} = (x_2, -g)`, guard `{x_1 = 0, x_2 <= 0}`, reset
   `r(x) = (x_1, -alpha x_2)`). Add `chyll_v2/scripts/train_bouncing_ball.py`.
   The bouncing ball is *not* present in `src/`, so this adds a system to
   the cross-system comparison set.
13. **Phase 3**: implement `chyll_v2/chyll_v2/systems/compass_gait.py`
   (4D Lagrangian biped, mass-matrix dynamics, angular-momentum impact
   reset). Mirror the physics in `src/system.py:144-227`.
14. Ablation status: one-seam `w_v=1.0` did not fix finite-difference UT
   rank, because it trained the exit seam only. Two-seam `w_v=1.0` repairs
   the entry tangent and recovers rank 1 at the same onset length as the
   `V_theta` row.
15. Optional: side-by-side training of `src/` 6-term vs `chyll_v2/` 5-term
   on identical trajectory data, with reconstruction MSE + cycling-signature
   rank reported.

## Sharp gotchas

- The phase-portrait plotter in
  `chyll_v2/chyll_v2/visualize.py:plot_phase_portrait_rimless` was wrong on
  first cut — it drew a single line through all base-flow samples per
  trajectory, so the line jumped across each reset and produced a criss-cross
  mess. Fixed by splitting each stride into its own segment.
- The Lipschitz-threshold gluing detector that Sangli uses for the hybrifold
  setting is unnecessary on the cylinder — we have `G` and `r` symbolically.
  `chyll_v2/chyll_v2/data.py:GuardSampler` draws fresh `g ∈ G` each step.
- The user is a math-journal author, terse-direct, decisive-scope. Prefers
  structured phased plans with checkpoints, dislikes uncontrolled scope
  creep. The Section-4 baseline in `src/` is frozen; `chyll_v2/` is the
  comparison arm.
- Auto-memory index at
  `C:\Users\roger\.claude\projects\c--Users-roger-OneDrive---Umich-Kaito-Iwasaki-The-Vault-Research-2026-HybridCyclingSignatures\memory\MEMORY.md`
  points to deeper project context including referee report, scope notes,
  collaborator channels, and Windows/OneDrive gotchas.
