# Shared Coauthor-Protocol Probability Driver

> **Artifact availability.** Generated inputs, plans, signatures, and figures
> under `experiments_planned/outputs/` are intentionally not stored in Git.
> Completed-run commands require those local artifacts; on a fresh clone,
> regenerate and materialize them using the builders and commands below.

Historical completed plans freeze Julia Manifest SHA-256
`7e0472391bd0a41651e778eea7f7fb460f307ed61cfd7a678c3a9d550afb472a`.
The current Manifest changes only the machine-local `CyclingSignatures.jl`
path to `../../CyclingSignatures.jl`, so its hash is intentionally different.
Validate historical local artifacts with their frozen Manifest revision; use
the portable Manifest for new plans. The one-line migration is recorded in
`../julia/manifest_path_migration.json`.

This directory defines one analysis path for both the continuous Rössler
system and the learned Compass-gait suspension.  The trajectory source and
top-row projection differ by system; the bottom-row computation does not.
Both use the same window sampling, UTB construction, filtration grid,
coefficient field, statistic, axis convention, and renderer.

The shared statistic is exactly the one documented in the coauthor's
Rössler note:

```text
P(nontrivial cycling signature at r)
    = 1 - number of rank-zero windows at r / 20.
```

There is no event alignment, duration binning, median span, period-aware
threshold, or regime-specific choice of radius.  At each sample length, the
20 starts are drawn independently and uniformly with replacement from one
long trajectory.  A length-`L` window is labelled by the coauthor convention
`L * effective_dt`, not by its `(L - 1)`-interval endpoint span.

## Frozen documented protocol

The primary profile in `coauthor_protocol.json` records:

- segment lengths `100:20:1200` and 20 starts per length;
- effective sample interval `0.01` and duration labels `1:0.2:12`;
- occupied cover `5 Q_1 x Q_2`, represented by position box size 5 and
  tangent-cover resolution 1;
- dynamic-distance coefficient `C=5`, radii on the inferred 201-point grid
  from 0 through 5, and coefficients in `F_43`;
- the literal tangent lift in the note, `f / ||f||_infinity`;
- probability `P(rank > 0)` and a horizontal sample-radius guide.

The standalone note and the cited implementation disagree about tangent
normalization.  The note literally uses infinity-norm directions; the cited
algorithm uses Euclidean-unit directions in the dynamic metric and applies
infinity normalization only when constructing the cubical cover.  The
primary profile follows the literal note.  The explicit
`coauthor_protocol_l2_compatibility.json` profile supports the second reading
without changing any other analysis setting.

The repository does not contain the coauthor's generator, numerical arrays,
initial state, random seed, sampled starts, exact radius grid, or definition
of the dashed guide.  The profiles therefore record, rather than hide, the
following reproducibility completions: initial state `(1,1,0)`, seed
`20260820`, 201 radii, the local centered-grid tie rule, and the interpretation
of the dashed line as the global consecutive-sample dynamic-distance bound.
These runs can reproduce the documented protocol, but cannot be claimed as a
pixel-identical reconstruction of the original PNG.

## One driver, two trajectory sources

Run commands from the workspace root.  Read-only checks are the default safe
starting point:

```bash
code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py check \
  --cases code/period_doubling/shared_probability/roessler_cases.json

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py check \
  --cases \
  code/period_doubling/shared_probability/roessler_coauthor_reference_cases.json

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py check \
  --cases code/period_doubling/shared_probability/compass_cases.json
```

`roessler_cases.json` contains the requested five local regimes
`c=4/6/8.5/8.7/9`; `roessler_coauthor_reference_cases.json` separately
contains the coauthor's documented `a=2.82/2.86` positive control.  Both use
fixed-step RK4 at `dt=0.01`, final time 3000, and removal of the first 2,000
updates.  The source records name the three equation roles explicitly as the
linear-`y` coefficient, `z` offset, and `z` control coefficient, so the local
`c` notation cannot be confused with the coauthor's `a` notation.  The Compass
cases read the fine learned-lift artifacts without
modifying them and use stride two on their `dt=0.005` stream, giving the same
effective interval `0.01`.  They include period 1, 2, 4, 8, and chaos.  The
same numeric
`(boxsize, tangent resolution, C, r)` profile is deliberate: the first run
tests the coauthor protocol itself rather than silently tuning a separate
Compass analysis.  A later scale study must apply a versioned paired protocol
to both systems.

Because the five Compass encoders were trained separately, equal numeric
latent distances are not guaranteed to represent equal physical scales.  The
shared `C=5` construction is therefore a strict numeric protocol control, not
by itself a scale-invariant comparison across slope regimes.

For Rössler, the duration coordinate is ordinary integration time.  For
Compass, it is nominal suspension/index time: mapping-cylinder bridge samples
also consume one stored sample interval even though the physical hybrid impact
is instantaneous.  This is an unavoidable consequence of applying the same
sample-window protocol literally, and it must not be relabelled as physical
hybrid elapsed time.

The flow-direction Compass arm pairs the frozen encoded suspension positions with
the already-provenanced directions `V_theta(z)`.  This is the closest stored
analogue of the coauthor vector-field direction and keeps the sample-radius
guide inside the shared `r <= 5` panel.  It is nevertheless a tangent-source
control, not a matched learned-flow trajectory: the positions come from the
encoded reference-simulator path, not a rollout generated by `V_theta`, and
the training configurations used `w_v=0`.  A fully matched learned-continuous
experiment still requires integrating each learned vector field to produce
its own positions and feeding those positions and directions to this same
driver.

`compass_cases_jvp_control.json` retains the encoder-JVP path tangents as a
second named arm; neither arm is silently treated as ground truth.  Under the
literal shared profile its seam corners put
the sample radius above `r_max=5`; rendering refuses that mismatch rather than
silently hiding the horizontal guide.  A wider-radius control therefore
requires a separate versioned protocol file.

Even for the smoother learned-flow directions, only radii above each panel's
horizontal sample-radius guide satisfy the consecutive-sample curve check.
The guide lies high in the shared Compass range, so low-radius color structure
is displayed for protocol fidelity but is not curve-resolved evidence.

The top row plots the first three coordinates of the same analyzed trajectory
used by the bottom row.  For Compass these are latent coordinates, not a raw
physical-state overlay.  This preserves the coauthor figure's source-data
relationship; a separate physical-orbit figure can supply gait interpretation.

Completed Rössler plans freeze `driver.py` by hash, including its original
whole-trajectory display adapter.  That adapter uniformly selected 5,000
points from almost 3,000 time units and then joined them, producing artificial
piecewise-linear chords.  The display-only `render_settled.py` keeps the
validated heatmaps unchanged and plots the final 5,000 post-stride samples
consecutively.  It does not interpolate or recompute signatures.  Overwriting
a stable PDF is explicit and atomic, and the adjacent `*.render.json` records
the plan, renderer, selection rule, and output hashes:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability/render_settled.py \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_five_regime_linf/plan.json \
  --replace-existing

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability/render_settled.py \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_coauthor_reference_linf/plan.json \
  --replace-existing
```

To check the implementation-compatible Euclidean normalization while keeping
everything else fixed, add:

```bash
--protocol \
  code/period_doubling/shared_probability/coauthor_protocol_l2_compatibility.json
```

## Executed primary sequence

Materialization writes only beneath
`code/experiments_planned/outputs/shared_coauthor_protocol/`.  It generates
Rössler input CSVs or binds the existing Compass CSVs, hashes every input,
driver, and learned-flow tangent provenance bundle, and writes inert commands.
It does not compute signatures.

```bash
code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py materialize \
  --cases code/period_doubling/shared_probability/roessler_cases.json \
  --output-root \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_five_regime_linf

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py materialize \
  --cases \
  code/period_doubling/shared_probability/roessler_coauthor_reference_cases.json \
  --output-root \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_coauthor_reference_linf

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py materialize \
  --cases code/period_doubling/shared_probability/compass_cases.json \
  --output-root \
  code/experiments_planned/outputs/shared_coauthor_protocol/compass_five_regime_flow_linf
```

Execution is a separate explicit action and uses the clean
`code/period_doubling/julia` project:

```bash
code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py execute \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_five_regime_linf/plan.json

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py execute \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_coauthor_reference_linf/plan.json

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py execute \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/compass_five_regime_flow_linf/plan.json
```

After every case is complete, the same rendering action creates the coauthor
two-row layout.  The main Rössler and Compass files both have five columns;
the separate coauthor reference file has two.

```bash
code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py render \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_five_regime_linf/plan.json

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py render \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/roessler_coauthor_reference_linf/plan.json

code/venv/bin/python \
  code/period_doubling/shared_probability/driver.py render \
  --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/compass_five_regime_flow_linf/plan.json
```

The Julia kernel writes one trial-birth table, sampled-start table, rank-zero
heatmap, and metadata file per case.  Empty birth vectors remain in the
20-window denominator.  Metadata records raw and effective sample intervals,
normalization, `F_43`, the cover and metric parameters, `beta1(Y)`, the sample
radius, hashes, and timing.  Existing outputs are never overwritten.

For the Rössler reference cases the renderer also requires the documented
`beta1(Y)=1`; failure of that positive control blocks the figure.  Result
metadata, input and runner hashes, sampled-start rows, and the emitted
rank-zero heatmap are cross-checked against the frozen plan before rendering.
The plan also freezes the Julia version, clean `CyclingSignatures.jl` commit,
and the period-doubling Julia Project and Manifest hashes.

Output names include the numeric metric coefficient:
`roessler_C5p0.pdf`, `roessler_coauthor_reference_C5p0.pdf`,
`compassgait_C5p0.pdf`, and the deliberately invalid on this radius profile
`compassgait_C5p0_jvp_control.pdf`.  The main Compass case metadata still
identifies the frozen-path/learned-flow-direction construction as a control;
the concise filename follows the manuscript artwork convention.  Titles
contain only the case parameter; prose interpretation belongs in the paper.

The literal-infinity primary profile was executed on 2026-08-21 for the
two-case Rössler reference, five-regime Rössler study, and five-regime Compass
flow-direction control.  All twelve comparison spaces have
`beta1(Y)=1`.  Each case has 1,120 stored windows and one 201-row rank-zero
table.  The JVP Compass arm was not run because its read-only sample-radius
check exceeds the frozen `r_max=5`.

The production-validated summaries are under each plan root in `summary/`.
They can be reproduced without recomputing signatures by running, for each
completed plan:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability/summarize.py \
  --plan PATH/TO/plan.json
```

The two-case reference passes its positive control: throughout the common
curve-resolved band `r=0.275--0.375`, the first durations reaching probability
`0.25`, `0.5`, and `0.75` are `5.8` and `11.6`, exactly a factor of two.  The
five-regime Rössler panel is only a partial diagnostic: at the first common
curve-resolved row `r=0.8`, periods 1 and 2 cross at `6` and `12`, while the
period-4, period-8, and chaos cases are right-censored by the 12-unit duration
grid.  No common radius gives a threshold-robust four-period staircase.

The Compass result is a stronger negative diagnostic.  Its sample-radius
bounds are `3.551/3.668/3.887/3.688/4.426`, so the common curve-resolved
display band is `r=4.45--4.975`.  Every probability cell in that band is one
for every regime; period 1, 2, 4, 8, and chaos are therefore exactly
indistinguishable under the frozen shared numerical profile.  Color variation
below the horizontal guides is retained to show the protocol output but is not
curve-resolved evidence.  No run supplies the separate certificate
`r<r0(Y;Gamma)`.

These commands generate diagnostic artifacts, not a claim that one raw
numeric `C=5` is scale-calibrated across separately trained Compass encoders.
The Rössler PDFs use the consecutive settled-tail renderer described above;
their heatmaps still have the diagnostic limitations stated here.
