# Prepared experiments and completed diagnostic

> **Artifact availability.** `experiments_planned/outputs/` is intentionally
> ignored by Git because it contains large generated bundles, plans,
> signatures, summaries, and renders. The drivers are versioned here;
> completed-run validation requires the corresponding local outputs or a
> fresh rebuild from the documented materialization steps.

Most scripts here prepare items 7--10 from the top-level `next-steps.md` and
remain unexecuted.  The fine-Compass duration/C planner, runner, manager, and
renderer have now generated the completed tied metric/tangent-cover
diagnostic and the fixed-`C=0.4` tangent-source/cover attribution documented
below.  Their output location is
`code/experiments_planned/outputs/`, outside every pre-existing artifact
directory.  The existing data and checkpoint trees are inputs only.

Use the repository environment for Python and the clean Julia project at
`code/period_doubling/julia`.  Run commands from `code/`.
`code/requirements.txt` now pins `torchdiffeq==0.2.5`, but the audited local
`code/venv` did not yet contain that package; synchronize the environment
before executing the Neural-ODE rollout or training plans.

## Duration-first gate

The stored compass analysis shows that `segment_length * dt * stride` is a
nominal suspension clock, not physical hybrid time: bridge samples are charged
`dt` although impacts are instantaneous. Before using Items 8--10 to assess
period resolution, add a duration-indexed subsegment selector with these fixed
design requirements:

- select paired starts on arc samples and end each segment at the first arc
  sample at or beyond a target physical duration;
- retain every intervening bridge point while giving all points on one bridge
  the corresponding event time;
- use one physical-duration grid for all regimes and extend it beyond the
  6.004312-second period-8 return;
- retain the full stored radius grid and preregister the radius normalization
  before reading the period-4/period-8 contrast;
- run the reference-orbit recurrence audit and exact-suspension control before
  any model retraining.

The current sample-length commands remain reproducible baselines, but they are
not the proposed duration-indexed resolution experiment. Retraining is the
last gate: consider it only if the duration-matched exact suspension separates
period 4 from period 8 while the learned lift does not.

## Fine Compass scale/cover and radius diagnostics

`prepare_fine_compass_c_radius_sweep.py` and
`run_duration_c_radius.jl` prepare the fine-only follow-up requested after the
stored radius--duration plots were audited.  The default denser tied-cover
plan below has not been executed; the separate 20-start fixed-position, tied
metric/tangent-cover diagnostic described below has.  The planner selects 100
paired starts uniformly from arc-time samples, reuses
them at target physical durations `0.25:0.25:7.5` seconds, and splits them
50/50 into tuning and validation subsets.  It covers periods 1, 2, 4, and 8
plus chaos and prepares 45 primary jobs (five regimes times nine values of
`C`) for the tied library sweep

```text
C = boxsize * sb_radius = 0.10:0.025:0.30,  sb_radius = 1
rho = r / C = 0:0.025:1.75
```

Every duration window retains all intervening mapping-cylinder samples, but
starts inside a zero-time bridge are excluded.  The same manifest is reused
for every value of `C`.  The Julia runner records the realized physical
duration, maximum consecutive dynamic distance `h`, comparison-space
`beta1(Y)`, and complete cycling-signature birth vector for every window.
Each result file contains both labelled halves.  Stage isolation is logical:
the analyzer reads only `split=tune` while selecting a plateau and only
`split=validate` after the cell is frozen.  This is not computation-time
blinding.
Only cells with `r > h` are curve-resolved candidates under the documented
curve-resolution hypothesis.  This check does not by itself establish the
separate admissible-neighborhood condition `r < r0(Y; Gamma)`.

To inspect the plan without writing or running anything:

```bash
venv/bin/python experiments_planned/prepare_fine_compass_c_radius_sweep.py
```

To materialize manifests and a command list beneath the safe planned-output
tree (still without running a signature):

```bash
venv/bin/python experiments_planned/prepare_fine_compass_c_radius_sweep.py \
  --materialize
```

Materialization also writes a structured `jobs.json`.  Its output directories
and prefixes carry the metric-scale suffix (`C_0p2`, `C_1`, and so on), so
results from different values cannot overwrite one another.  The job manager
validates this document and reports pending/completed jobs without launching
Julia unless `--execute` is stated explicitly:

```bash
venv/bin/python experiments_planned/manage_fine_compass_jobs.py \
  --jobs-file experiments_planned/outputs/fine_compass_c_radius/jobs.json \
  --arm primary --max-workers 5
```

The manager is the preferred execution path once compute is authorized.  It
runs at most `--max-workers` independent Julia processes, fixes each process
at one Julia thread by default, stages every job privately, validates the full
CSV and metadata pair against the fixed manifest, and then atomically promotes
the job directory.  A valid existing result is skipped; an interrupted staging
directory is preserved under a failure name before a retry.  An invalid final
directory is never replaced automatically.  This makes bounded parallel
execution resumable without trusting file existence alone.  The following is
an execution command for the unrun default root; the completed fixed-position
diagnostic used the same manager with its own two `jobs.json` files.  On
`Ctrl-C` or `SIGTERM`, the manager cancels queued work,
terminates each active Julia process group, and retains any incomplete staging
directories for quarantine on the next run:

```bash
venv/bin/python experiments_planned/manage_fine_compass_jobs.py \
  --jobs-file experiments_planned/outputs/fine_compass_c_radius/jobs.json \
  --arm primary --max-workers 5 --julia-threads 1 --execute
```

### Performance notes

The expensive step is the per-window distance-complex calculation, not file
loading, comparison-space construction, or evaluation on multiple radii.  At
100 starts, 30 target durations, and five regimes, the present runner visits
about 8.16 billion upper-triangular distance cells for one `C`.  Julia starts
with one thread here, so the manager's five independent regime processes are
the safest immediate acceleration.  On the audited 14-core, 48-GB machine,
archived timings suggest roughly 45--70 minutes of wall time per `C` for the
100-start design, or 10--15 minutes for the 20-start diagnostic, rather than
3--4 CPU-hours and 35--45 CPU-minutes sequentially.  These are planning
estimates; benchmark one `C` before increasing `--max-workers`.

Changing the plotted `rho` grid is essentially free after a job finishes:
one stored birth vector supplies every radius up to `r_max`.  Reducing the
number of plotted radii therefore does not avoid the dominant work.  The next
exact kernel optimization would exploit the fact that all 30 durations for a
given start are nested prefixes.  Growing one triangular distance grid to the
longest endpoint would reduce first-pass pair visits from 8.16 to about 0.78
billion per `C`, a factor of 10.52, without dropping any duration.  A later
multi-`C` kernel can also cache the positional and tangent distances in
`d_C=max(d_x,C*d_v)`, and same-`C` factorization controls can share one
trajectory barcode.  Those optimizations are not implemented here: they must
preserve the current traversal and tie-breaking and reproduce existing birth
vectors and cycling matrices before replacing the reference runner.

To vary `C` without changing the positional cover, hold `boxsize=0.2` and let
the planner derive the positive integer `sb_radius=C/boxsize`.  The completed
diagnostic used `C=0.2,0.4,0.6,0.8,1.0`, hence
`sb_radius=1,2,3,4,5`, on exactly the same windows.  The lower four values are
reproduced by

```bash
venv/bin/python experiments_planned/prepare_fine_compass_c_radius_sweep.py \
  --output-root experiments_planned/outputs/compass_c0p2_to_c0p8_diagnostic \
  --c-grid 0.2:0.2:0.8 --fixed-boxsize 0.2 \
  --duration-grid 0.25:0.25:7.5 --rho-max 1.75 \
  --n-starts 20 --seed 20260820
```

The `C=1` reproduction command is given in the next subsection.  The two
materialized plans contain 25 primary jobs total, and all 25 completed and
passed the manager's full manifest/result validation: the manager reports
`complete=20` for the lower-four plan and `complete=5` for the `C=1` plan.
The immutable `plan.json` and `jobs.json` `status=prepared_not_executed`
strings describe their materialization-time state; the promoted result pairs
and manager validation establish execution completion.  Noninteger
`C/boxsize` values are rejected rather than rounded.  This is a fixed-position,
tied metric/tangent-cover diagnostic, not a clean metric-`C` sweep: changing
`sb_radius` changes both `C=boxsize*sb_radius` and the tangent cover of `Y`.
The tied `(boxsize,sb_radius)=(C,1)` and same-`C` alternative factorizations
were not part of that five-construction run.  The completed single-`C`
attribution below now supplies the `(0.2,1)` and `(0.2,2)` cells at `C=0.4`;
the other factorizations and a broad clean metric-`C` sweep remain unrun.

Each `C` contains 20 starts at 30 target durations in all five regimes, or
3,000 signatures.  In period-1/2/4/8/chaos order, the comparison-space Betti
vectors for increasing `C` are `(1,4,2,4,1)`, `(0,2,1,2,0)`,
`(0,2,2,2,0)`, `(0,2,1,2,0)`, and `(0,3,2,3,0)`.  The rank-zero totals out
of 600 per regime are `(40,47,51,54,54)`, `(600,64,62,71,600)`,
`(600,44,72,77,600)`, `(600,43,72,76,600)`, and
`(600,41,63,72,600)`.

The first of six common curve-resolved grid rows is at
`r=0.325/0.650/0.975/1.300/1.625` for increasing `C`.  At those rows, the
first durations with `P(rank>0)>=0.5` are
`(.75,.75,1,1,1)`, `(none,1,1,1,none)`,
`(none,.75,1,1.25,none)`, `(none,.75,1,1.5,none)`, and
`(none,.75,.75,.75,none)` seconds.  Only the period-1 value at `C=0.2`
matches the independent 0.748/1.502/3.002/6.004-second orbit periods.  The
period-4 and period-8 valid-band matrices remain nearly identical.  Thus the
diagnostic does not isolate a metric-`C` effect, identify an ideal `C`, or
distinguish periods 4 and 8.

### Completed fixed-`C=0.4` tangent/cover attribution

The follow-up holds the frozen encoded positions, physical-duration manifests,
and the numerical metric coefficient `C=0.4` fixed while varying two factors:
the encoder-JVP versus learned-flow direction and `sb_radius=1` versus `2` at
`boxsize=0.2`.  The JVP/`sb_radius=2` cell is the validated `C=0.4` output from
the tied diagnostic above.  The other three cells are explicit-metric jobs
under
`outputs/compass_c0p4_fixed_metric_tangent_2x2/`.  Each cell contains 20 fixed
starts at 30 exact target durations for five regimes, or 3,000 signatures.
The three new cells therefore contain 9,000 signatures in 15 jobs; the complete
four-cell comparison contains 12,000.

Only the schema-v2 learned-flow tangent bundle is an input to this diagnostic:

```bash
venv/bin/python experiments_planned/export_fine_compass_learned_flow_tangents.py \
  --output-dir \
  experiments_planned/outputs/fine_compass_learned_flow_tangents_v2 \
  --materialize
```

Its `provenance.json` SHA-256 is
`dd246ce54116225ca962d7f5f27c04e19ed04ff986bc2c9308ae48b64350dce3`.
For each stored position `z`, the exporter evaluates and normalizes
`V_theta(z)`.  It preserves row order, positions, timestamps, piece tags, and
manifests.  These directions are not path derivatives and are not trajectories
of the learned flow.  Unit normalization discards learned speed.  The saved
models have `w_v=0`, so seam velocity was not directly trained, and each regime
uses a separately trained encoder/vector field.  The period-1 source is at
4.00 degrees while its checkpoint is at 0.07 rad (4.0107 degrees), a
`2.0e-4`-rad mismatch.

The materialized plan records the exact arms, input hashes, reused comparator,
and required rowwise curve-bound invariants.  Its reproduction command is:

```bash
venv/bin/python experiments_planned/prepare_fine_compass_c_radius_sweep.py \
  --output-root \
  experiments_planned/outputs/compass_c0p4_fixed_metric_tangent_2x2 \
  --c-grid 0.4:0.4:0.4 --explicit-metric \
  --diagnostic-arm jvp_sb1:0.2:1 \
  --diagnostic-arm \
  flow_sb1:0.2:1:experiments_planned/outputs/fine_compass_learned_flow_tangents_v2 \
  --diagnostic-arm \
  flow_sb2:0.2:2:experiments_planned/outputs/fine_compass_learned_flow_tangents_v2 \
  --reuse-manifests-from \
  experiments_planned/outputs/compass_c0p2_to_c0p8_diagnostic \
  --external-comparator-plan \
  experiments_planned/outputs/compass_c0p2_to_c0p8_diagnostic \
  --external-comparator-label jvp_sb2_existing \
  --external-comparator-boxsize 0.2 \
  --external-comparator-sb-radius 2 \
  --curve-bound-pair jvp_sb1:jvp_sb2_existing \
  --curve-bound-pair flow_sb1:flow_sb2 \
  --duration-grid 0.25:0.25:7.5 --rho-max 1.75 \
  --n-starts 20 --seed 20260820 --materialize
```

Execution used the dry-run-by-default manager with five independent Julia
processes and one Julia thread per process:

```bash
venv/bin/python experiments_planned/manage_fine_compass_jobs.py \
  --jobs-file \
  experiments_planned/outputs/compass_c0p4_fixed_metric_tangent_2x2/jobs.json \
  --max-workers 5 --julia-threads 1 --execute
```

A subsequent read-only manager call reports `complete=15`,
`ready-to-promote=0`, `incomplete-stage=0`, `invalid-final=0`, and `pending=0`.
It also passes exact rowwise curve-bound invariants for
`jvp_sb1:jvp_sb2_existing` and `flow_sb1:flow_sb2`.  As in the earlier plans,
the immutable `plan.json` and `jobs.json` status strings preserve the
materialization-time state `prepared_not_executed`; validated output pairs and
the manager audit establish completion.

The four comparison-space Betti vectors are:

| tangent / cover arm | `beta1(Y)` in period-1/2/4/8/chaos order |
|---|---|
| JVP, `sb_radius=1` | `1/4/2/4/1` |
| JVP, `sb_radius=2` (reused) | `0/2/1/2/0` |
| learned flow, `sb_radius=1` | `1/4/2/3/1` |
| learned flow, `sb_radius=2` | `1/4/2/3/1` |

The pooled maximum curve bounds, in the same regime order, are
`.539586/.542252/.647264/.586774/.600465` for JVP and
`.132481/.187092/.171279/.138520/.137968` for learned flow.  They are exactly
equal row-by-row between covers for a fixed tangent source.  The common
curve-resolved radius range is therefore `0.65--0.70` for either JVP arm and
`0.19--0.70` for either learned-flow arm.  These are only curve-resolution
gates: no certified `r0(Y;Gamma)` is stored for any arm.

The focused clean-flow audit is the four-row band `r=0.19--0.22`, or
`rho=r/C=0.475--0.55`.  The pooled `P(rank>0)>=0.5` onset vectors are
`.75/1/1/1.25/1` seconds at `r=0.19` and `0.20`, then
`.75/1/1/1/1` at `r=0.21` and `0.22`.  Every regime is continuously saturated
from 1.5 seconds onward, and none of the 2,900 adjacent comparisons in the 100
fixed-start nested rank sequences decreases at any focused row.  Period 4 and
period 8 nevertheless agree in 90--93.3% of the 30 probability cells at each
row; their mean absolute difference is only `0.0117--0.0133`, and their maximum
absolute difference is `0.20--0.25`.  Neither the half-probability onsets nor
the saturation onsets recover the independently measured
`0.748/1.502/3.002/6.004`-second period staircase.

This four-row band is a post-hoc diagnostic, not the output of the tune-only
plateau selector below.  It is the maximal contiguous grid band starting at
the first common curve-resolved row and ending before nested-rank decreases
appear.  At `r=0.23`, period 2 has 160 adjacent full-rank decreases across all
20 fixed starts (80 in each labelled half).  The reported tune/validation
columns are therefore descriptive sensitivity summaries, not a blind
holdout.  For comparison, JVP/`sb_radius=1` has 320 pooled adjacent decreases
at its first resolved row `r=0.65`: 160 each in periods 2 and 8, affecting all
20 starts in each regime.

The learned-flow `sb_radius=1` and `2` probability grids are exactly identical
at all stored radii, durations, regimes, and splits.  The JVP/`sb_radius=1`
cell also restores nonzero comparison-space loops for period 1 and chaos,
where the reused JVP/`sb_radius=2` cell is identically rank zero.  Consequently,
the original all-zero `C=0.4` panels cannot be attributed to encoded positions
alone; tangent source and tangent-cover resolution matter.  This does not make
the learned-flow substitution a preferred UTB definition, select an ideal
`C`, or distinguish period classes.

The validated summarizer writes 11 files under the plan's `summary/` directory:
Betti vectors, curve-resolution bounds, focused arm and regime evaluations,
probability matrices, half-probability onsets, period-4/period-8 contrasts,
pairwise arm differences at row and aggregate levels, input hashes, and the
machine-readable `summary.json`.  Reproduce the summaries with:

```bash
venv/bin/python experiments_planned/summarize_compass_c0p4_tangent_2x2.py \
  --plan-root \
  experiments_planned/outputs/compass_c0p4_fixed_metric_tangent_2x2
```

Do not choose the best single `(C,r)` cell.  On the tuning half, estimate the
duration at which nontrivial-signature probability crosses one half for each
periodic regime and seek a connected plateau that both aligns those onsets
with the recurrence periods and separates adjacent regimes at the shorter
orbit time.  Freeze the plateau center, then report only the validation half.
Repeat that selected metric scale with
`(boxsize,sb_radius)=(C/2,2)` to distinguish `C` from cover resolution.

`analyze_fine_compass_c_radius_sweep.py` makes that selection reproducible.
It expands every stored birth vector on the planned `rho` grid, checks the
five-regime manifests and hashes, enforces curve coverage, scores only the
four periodic regimes, finds a connected near-optimal plateau, and freezes a
discrete medoid.  Every scientific threshold is required explicitly.  The
following values are examples awaiting author approval, not defaults:

```bash
venv/bin/python experiments_planned/analyze_fine_compass_c_radius_sweep.py tune \
  --plan-root experiments_planned/outputs/fine_compass_c_radius \
  --results-root experiments_planned/outputs/fine_compass_c_radius/signatures \
  --output-dir experiments_planned/outputs/fine_compass_c_radius/analysis/tune_preregistered \
  --onset-probability 0.5 \
  --onset-tolerance-seconds 0.25 \
  --contrast-time-factor 1.1 \
  --minimum-contrast 0.20 \
  --near-optimal-contrast-slack 0.04 \
  --minimum-plateau-c-values 2 \
  --minimum-plateau-rho-values 2
```

If a plateau qualifies, the analyzer writes an inert `commands.sh` for later
review.  It contains one frozen primary-validation command and, for every
nonduplicate control construction, five factorization jobs including chaos
plus one factorization-validation command.  The default control is
`(boxsize,sb_radius)=(C/2,2)`.  For a fixed-boxsize plan it also
records the tied `(C,1)` control.  Any construction identical to the selected
primary is omitted.  The adjacent `control_jobs.json` contains only the Julia
jobs, grouped by control-arm label, and can be inspected or later executed by
`manage_fine_compass_jobs.py --arm LABEL`.  The analyzer never launches them.

A rigorous admissible-neighborhood check additionally requires a CSV with
this exact header:

```text
regime,C,boxsize,sb_radius,r0_lower,certified,provenance
```

Run the analyzer with `--r0-csv PATH --require-certified-r0` only when
`r0_lower` is a certified construction-specific lower bound.  The
factorization controls need separate bounds.  Without them, a passing analysis
is labelled `curve-resolved provisional`, never mathematically admissible.

The five regimes use separately trained encoders, so a common raw numeric
`C` is meaningful only after the authors either justify comparable latent
position units, normalize those units by a preregistered rule, or select `C`
separately within each encoder.  The tied metric/tangent-cover diagnostic has
been computed,
but that decision still precedes a cross-regime scientific interpretation.

The archived fine production run alone cannot select an ideal `C`: it contains
segment signatures only at `C=0.2`, while the stored pilot at other box sizes
records only `beta1(Y)`.  The completed fixed-position diagnostic adds segment
signatures at five tied `(C,sb_radius)` constructions but does not isolate the
metric coefficient, yield period-aligned onsets, or give a period-4/period-8
discriminator.  More seriously, none of the archived
stride-two windows meet
the curve-resolution hypothesis anywhere on the stored `r <= 0.2` grid.
At stride one, one arc--bridge tangent corner per impact still gives
`h/C` up to about `1.62`.  Before interpreting the completed diagnostic or
running the denser sweep, the authors
must decide whether to smooth/refine that UTB tangent corner, justify both the
larger curve-resolving radius and the admissible-neighborhood bound, or change
the comparison construction.

### Fixed `C=1` Compass diagnostic

The requested `C=1` follow-up is a new signature computation, not a
replot of the stored `C=0.2` births.  The stored fine positions, tangents, raw
timestamps, checkpoints, and configurations are sufficient; no retraining is
needed.  To increase tangent influence without simultaneously coarsening the
existing positional cover by a factor of five, use

```text
primary:  (boxsize, sb_radius) = (0.2, 5),  C = 1
controls: (boxsize, sb_radius) = (0.5, 2) and (1.0, 1),  C = 1
```

The controls are necessary because `C=boxsize*sb_radius`: `(1,1)` alone
changes both the dynamic metric and the positional comparison cover.  The
primary keeps the earlier fine positional scale `boxsize=0.2` and refines the
tangent cover.  The two controls test whether an apparent result survives
other factorizations of the same metric coefficient.

This is the reproduction command for the bounded 20-start `C=1` plan.  Without
`--materialize` it only audits the plan and prints five primary commands:

```bash
venv/bin/python experiments_planned/prepare_fine_compass_c_radius_sweep.py \
  --output-root experiments_planned/outputs/compass_c1_diagnostic \
  --c-grid 1:1:1 --fixed-boxsize 0.2 --fixed-sb-radius 5 \
  --duration-grid 0.25:0.25:7.5 --rho-max 1.75 \
  --n-starts 20 --seed 20260820
```

The materialized plan contains 30 target durations, 20 fixed starts per
duration, and labelled 10/10 tuning and validation halves.  Its five primary
jobs completed and passed the manager's validation, producing 3,000
signatures.  The analyzer below has not been used to make a post-hoc selection;
its numerical thresholds still require author approval.  When approved, a
one-`C` plateau is accepted only for a one-value `C` plan; the broad sweep
continues to require at least two `C` values.  A frozen selection writes, but
never launches, ten control commands (five regimes for each control
factorization), their two validation commands, and a structured
`control_jobs.json` for bounded/resumable execution.  Those controls were not
run.

```bash
venv/bin/python experiments_planned/analyze_fine_compass_c_radius_sweep.py tune \
  --plan-root experiments_planned/outputs/compass_c1_diagnostic \
  --results-root experiments_planned/outputs/compass_c1_diagnostic/signatures \
  --output-dir experiments_planned/outputs/compass_c1_diagnostic/analysis/tune \
  --onset-probability 0.5 --onset-tolerance-seconds 0.25 \
  --contrast-time-factor 1.1 --minimum-contrast 0.20 \
  --near-optimal-contrast-slack 0.04 \
  --minimum-plateau-c-values 1 --minimum-plateau-rho-values 2
```

These numerical thresholds remain examples awaiting author approval; the
completed primary computation does not turn them into defaults.

For these deterministic windows at `C=1`, the first `0.025`-spaced radii with
full curve-bound coverage are `1.350`, `1.375`, `1.625`, `1.475`, and `1.525`
for period 1, 2, 4, 8, and chaos, respectively.  Keeping
`rho=r/C=0:0.025:1.75` therefore leaves six common curve-resolved rows starting
at `r=1.625`.  This is only the curve gate.  A rigorous common cell still
needs construction-specific certified bounds `r0>r`; no such `C=1` bounds are
stored.  If no interval `h<r<r0` exists, increasing `C` does not produce an
admissible cycling-signature claim.

The completed 20-start primary diagnostic has
`beta1(Y)=0/3/2/3/0` and rank-zero totals
`600/41/63/72/600` in period-1/2/4/8/chaos order.  At the first common
curve-resolved row, `r=1.625`, the first durations with
`P(rank>0)>=0.5` are `none/.75/.75/.75/none` seconds.  Thus period 2, period
4, and period 8 all turn on at the same displayed duration, and their onsets
do not match the measured 1.502/3.002/6.004-second returns.  The period-4 and
period-8 valid-band probability matrices are nearly identical.  Each unrun
same-`C` control has the same planned 3,000-signature cost.

Paper probability filenames record the metric scale.  The completed result is
`compassgait_C1p0.pdf`; the archived panel remains `compassgait_C0p2.pdf`.
`render_fine_compass_probability.py` adapted the validated planned manifest
plus `*_births.csv` schema to the same minimalist two-by-five paper layout:

```bash
venv/bin/python experiments_planned/render_fine_compass_probability.py \
  --plan-root experiments_planned/outputs/compass_c1_diagnostic \
  --c 1.0 --boxsize 0.2 --sb-radius 5
```

The renderer requires an explicit `(C, boxsize, sb_radius)` and exactly one
matching complete primary job for every regime.  It rejects control-arm or
mixed-factorization job sets, partial `tune`- or `validate`-only files, a
truncated planned radius grid, result/manifest/lift-input discrepancies, and
an existing destination PDF.  Control computations stop at a frozen radius
and are validation artifacts rather than full-grid paper heatmaps.  The
filename suffix comes from the five mutually consistent result metadata
files, not the command-line spelling of `C`.

The heatmap columns use the preregistered `duration_grid_seconds` values from
`plan.json`.  They are therefore the intended segment durations, while the
nearby realized endpoint times remain row-level provenance checks; the
renderer does not median-bin or rebin them.  Every planned window in both
labelled halves contributes to its duration cell, and an empty birth vector
is retained as a rank-zero observation.  The top row continues to read the
stored raw orbits without modifying them.  Rendering only adapts completed
outputs; it never launches Julia or computes a signature.

The matched sweep renderings are stored separately as
`paper/hybrid_cyclingsignatures/figures/cycling_signatures/`
`compass_c_sweep_n20/compassgait_C{0p2,0p4,0p6,0p8,1p0}.pdf` so neither the
archived large-sample panel nor another metric scale is overwritten.

## 7. Learned-latent-flow rollouts

`evaluate_latent_rollouts.py` draws initial conditions from a seed distinct
from the saved training seed, simulates matching ground-truth suspension
trajectories, and integrates the learned latent vector field from the encoded
initial states.  It records decoded impact counts and timing error, a bounded
rollout indicator, and return-period recurrence scores.  Each predicted
latent trajectory is exported as `*_positions.csv` and `*_tangents.csv` for
the existing cycling-signature driver.

Example preparations:

```bash
venv/bin/python experiments_planned/evaluate_latent_rollouts.py \
  --run-dir chyll_v2/runs/rimless_wheel_phaseB_finetune

venv/bin/python experiments_planned/evaluate_latent_rollouts.py \
  --run-dir chyll_v2/runs/compass_gait_phi_3_5.02deg \
  --slope-config phi_3 --expected-period 8
```

After reviewing the rollout metrics, run a separate signature job for each
exported base.  For example:

```bash
julia --project=period_doubling/julia \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir experiments_planned/outputs/latent_rollouts/compass_gait_phi_3_5.02deg \
  --base latent_rollout_000 --boxsize 0.45 --sb-radius 1 \
  --segment-lengths 20:20:800 --n-runs 150 --eval-radius 0.1
```

The predicted event detector uses the same sampled-cylinder thresholds as the
data-driven gluing detector (`s >= 0.9` followed by `s <= 0.1`).  The return
period is the smallest candidate lag whose median mismatch lies within
`best * (1 + rtol) + atol` of the best recurrence score, which prevents a
fundamental period from being replaced automatically by one of its multiples.
The default selection tolerances are explicit command-line arguments and
should be preregistered before execution.  This remains a return-map
diagnostic, not topological evidence.

## 8. Exact-suspension control

`export_exact_suspension_control.py` adds an analytic bridge to each recorded
raw compass-gait reset.  The base slice is the identity embedding
`x -> (x, 0)` and the bridge is
`((1-s)g + s r(g), h sin(pi s))`, so both seam endpoints agree exactly.  This
is a matched diagnostic control for exact gluing, not a global embedding
theorem.  The default eight interior bridge samples match the stored coarse
trained lifts.  The construction guarantees positional seam endpoints; it
does not silently assume that its arch tangent matches the base-flow tangent.

```bash
venv/bin/python experiments_planned/export_exact_suspension_control.py

# End-to-end matched coarse grid for all five regimes
venv/bin/python experiments_planned/export_exact_suspension_control.py \
  --run-signatures --boxsize 0.45 --r-max 0.45 \
  --eval-radius 0.1125 --segment-lengths 20:20:800 --n-runs 150
```

The script reads the existing `period_doubling/data/compass_gait/*.npz`
archives and writes new outputs only below `experiments_planned/outputs/`.
With `--run-signatures`, it invokes the existing Julia driver through the
clean `period_doubling/julia` project for every exported regime.  Neither form
was executed while preparing this directory.

## 9. Multi-seed ensemble

`run_multiseed_ensemble.py` covers rimless Phase A/B, canonical compass
`phi=0.07`, and one cascade slope.  The default cascade stress case is
`phi_3` (period 8); use `--cascade-slope` to change it.  By default the script
only prints and saves the full command plan.  Nothing is trained or analyzed
unless a future researcher explicitly supplies `--execute`.

```bash
venv/bin/python experiments_planned/run_multiseed_ensemble.py \
  --seeds 1,2,3,4,5
```

The plan gives every seed an isolated run, figure, lift, and signature
directory.  It also fixes a distinct Julia subsegment seed and can aggregate
the 95%-rank-1 onset and final rank-1 fraction after all runs finish.  Review
the generated `experiments_planned/multiseed-plan.json` before any execution.
The cascade arm encodes the same fixed-`dt=0.02` source trajectory for every
seed through `period_doubling/export_latent_lifts.py`; it does not substitute
an adaptive-step rollout.  The planned signature grids reproduce the stored
headline settings: rimless through length 300 at `r=0.001`, canonical compass
through 600 at `r=0.05`, and the cascade through 800 on the full `r<=0.45`
grid with the common diagnostic radius `0.1125`.

## 10. Period-8 resolution criterion

`assess_period8_resolution.py` implements the conservative criterion to use
before calling period 8 resolved. Its default branch input is the learned
encoding of the reference-simulator post-impact states in
`chyll_v2/compass_analysis/compass_gait_cascade/`
`phi_3_latent_postimpact_z.npy`; sampled post-bridge arc starts are not used:

> The minimum separation of the eight post-impact latent branch centroids,
> after subtracting twice the largest within-branch 95% radius, must be
> strictly greater than the smallest signature radius at which at least 90%
> of subsegments close at some tested length.

If the inequality fails, the required label is **partially resolved**.  Both
sides of the comparison are reported in the same latent-distance units.

```bash
venv/bin/python experiments_planned/assess_period8_resolution.py

# Fine-sampling variant
venv/bin/python experiments_planned/assess_period8_resolution.py \
  --latent-postimpact chyll_v2/compass_analysis/compass_gait_cascade/phi_3_latent_postimpact_z.npy \
  --rank1-heatmap period_doubling/data_fine/compass_gait_latent/signatures/subsegments_compass_period8_rank_heatmap_rank1.csv \
  --output experiments_planned/outputs/period8_resolution/fine-criterion.json
```

The success fraction, burn-in count, number of branches, and `--n-runs` are
explicit arguments so a sensitivity audit can be prepared without changing
the code.  `--n-runs` must agree with the heatmap metadata; it is never
inferred from the largest observed rank-1 count.
