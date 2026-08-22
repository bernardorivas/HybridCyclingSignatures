# Versioned David-family Rössler probability analysis

> **Artifact availability.** Generated orbit bundles, plans, signatures,
> summaries, and figures under `experiments_planned/outputs/` are intentionally
> not stored in Git. Completed-run validation requires the local artifacts or
> a fresh rebuild using the versioned builders and commands below.

Historical completed plans freeze Julia Manifest SHA-256
`7e0472391bd0a41651e778eea7f7fb460f307ed61cfd7a678c3a9d550afb472a`.
The current Manifest changes only the machine-local `CyclingSignatures.jl`
path to `../../CyclingSignatures.jl`, so its hash is intentionally different.
Validate historical local artifacts with their frozen Manifest revision; use
the portable Manifest for new plans. The one-line migration is recorded in
`../julia/manifest_path_migration.json`.

This directory is an isolated version-2 analysis path. It does not import or
modify the frozen version-1 Python driver, protocols, plans, results, or PDFs.
It consumes the independently certified orbit bundle at
`code/experiments_planned/outputs/roessler_david_fourier_continuation_v1/`
and calls the unchanged Julia kernel
`code/period_doubling/julia/run_shared_probability.jl`.

The bottom row uses the same statistic for every case:

```text
P(nontrivial cycling signature at r)
    = 1 - rank-zero windows at r / 20.
```

The versioned extension uses segment lengths `100:20:6000`, giving nominal
durations `1:0.2:60` at `dt=0.01`. Starts are sampled independently for every
length, with replacement, using seed `20260820`. The lift is normalized in
the infinity norm, the cover is `5 Q_1 x Q_2`, `C=5`, radii are
`0:0.025:5`, and coefficients are in `F_43`.

The top row is not selected from a decimated long trajectory. Periodic panels
use the bundle's dense, certified single primitive orbit. The chaotic panel
uses the bundle's exactly 50-time-unit bounded segment. White vertical dashed
lines mark the manifest-certified full periods in the four periodic panels;
there is no chaos line and no horizontal radius guide.

All actions are explicit and the default is read-only `check`:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/roessler_probability_v2.py

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/roessler_probability_v2.py \
  materialize

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/roessler_probability_v2.py \
  execute --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/\
roessler_david_fourier_probability_linf_v2_david_grid/plan.json

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/roessler_probability_v2.py \
  render --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/\
roessler_david_fourier_probability_linf_v2_david_grid/plan.json
```

`check` verifies the complete bundle inventory and certificates, hashes the
kernel and Julia environment, validates all numerical inputs, and runs the
Julia kernel with `--check-only`. `materialize` writes only a plan and inert
commands beneath a new named output root. `execute` is the only action that
computes signatures. `render` validates every result before creating a new
PDF and provenance sidecar and refuses to overwrite either.

The completed plan is the `_v2_david_grid` path shown above.  The neighboring
`roessler_david_fourier_probability_linf_v2/` root is an immutable rejected
draft with a one-unit duration step; it contains no signatures and must not be
executed or used for the figure.

## Paper-scale rendering

The frozen orchestrator's original 16.8-inch diagnostic layout is retained in
its source hash, but is too wide for manuscript placement.  The versioned
display-only renderer revalidates the frozen plan and every result, then
creates the 9.00-by-4.65-inch vector PDF used by the paper.  It shares the
duration/radius labels across panels, keeps only parameter titles above the
orbit row, omits every horizontal guide, and preserves the certified vertical
period guides:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/\
render_roessler_probability_paper_v1.py \
  --plan code/experiments_planned/outputs/shared_coauthor_protocol/\
roessler_david_fourier_probability_linf_v2_david_grid/plan.json \
  --replace
```

The explicit replacement flag updates only `roessler_C5p0.pdf` and its
render-provenance sidecar after staging and validation.  It does not change or
recompute the plan, bundle, signatures, or probability summaries.

## Read-only probability summaries

`summarize_probability_v2.py` independently reuses the strict plan, bundle,
birth-table, rank-zero-table, and result-binding validators above. Its default
`check` action writes nothing. Once all five cases exist, the explicit `write`
action creates the new `probability_summary_v1/` directory without changing
the plan, signatures, logs, PDF, or manuscript:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/summarize_probability_v2.py \
  check --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/\
roessler_david_fourier_probability_linf_v2_david_grid/plan.json

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/summarize_probability_v2.py \
  write --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/\
roessler_david_fourier_probability_linf_v2_david_grid/plan.json
```

The versioned tables report P25/P50/P75 first crossings and observed-suffix
sustained onsets at every sampled radius, per-case curve resolution and
`beta1_Y`, certified-period errors for periodic cases, and pairwise raw
probability/onset differences. Curve resolution is the strict test `r > h`,
where `h` is the sampled global curve bound. A separate strict flag records
the profile interval `r < C=5`; their conjunction is labeled only as a
numerical/theory candidate. There is no certified `r0`, so the tables do not
call any sampled radius certified or admissible, and in particular `r=5` is
not a candidate. They summarize exactly
`P(rank > 0) = 1 - rank0/20`; no smoothing or interpolation is performed.

## Fourier-closed embedded Compass control

The Compass counterpart is isolated from the Rössler plan and from the older
frozen-path controls.  `build_compass_fourier_bundle_v1.py` constructs a
hash-inventoried derived bundle at
`code/experiments_planned/outputs/compass_embedded_fourier_orbits_v1/`.
For a period-q case, it aligns 32 deterministic late q-impact cycles from one
bridge-to-arc phase class, averages them on the literal stored-row suspension
clock, and retains `H=6q` Fourier harmonics.  The periodic analysis tangent is
the analytic Fourier derivative.  Chaos is explicitly nonperiodic: both its
frozen encoded path and its `V_theta(z)` directions are interpolated from
`dt=.005` to `dt=.0025`, so its tangent semantics differ from the periodic
panels and are recorded as such.

The refined sample cadence is `dt=.0025`. Segment lengths
`400:80:4800` preserve David's displayed duration grid `1:.2:12`; all other
probability settings remain the same: 20 independently resampled starts per
length, seed `20260820`, infinity normalization, cover `5 Q_1 x Q_2`, `C=5`,
radii `0:.025:5`, and coefficients in `F_43`. The vertical guides use nominal
stored-row suspension periods `.880`, `1.760`, `3.520`, and `7.045`; physical
hybrid returns are provenance context only because they are not in the
heatmap's time coordinate.

The default check is read-only and includes both kernel `--check-only` and an
independent beta/curve gate. Materialization repeats that gate before writing
an inert immutable plan:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/compass_probability_v2.py check

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/compass_probability_v2.py \
  materialize

PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/compass_probability_v2.py \
  execute --plan \
  code/experiments_planned/outputs/shared_coauthor_protocol/\
compass_fourier_embedded_probability_linf_v2_david_grid/plan.json
```

The frozen plan requires `beta1(Y)=1` for every periodic comparison space and
strictly validates the bundle inventory, Fourier certificates, curve bounds,
input hashes, Julia environment, all trial tables, sampled starts, and the
rank-zero matrix before publishing a per-case v2 result binding.

### Completed Compass result, summary, and rendering

The canonical plan is
`code/experiments_planned/outputs/shared_coauthor_protocol/compass_fourier_embedded_probability_linf_v2_david_grid/plan.json`
(SHA-256
`0a1d18864234ca8482b7587e74fb68db161f9fc0f373340b9058148c4368677e`).
It binds bundle manifest SHA-256
`19b05c2db02f67abf712f50d0d965cdc399db9be477572ca9d1828c343a87f85`.
The canonical plan was materialized with the repository plotting environment;
the strict result loader and renderer reject a plan from a different Python
environment.

All five jobs completed and reload through the production validator.  In
period-1/2/4/8/chaos order, the result-binding SHA-256 values are:

- `c7c81eaa1d6868a612fd1e4f04ec98502493fdcdf8f48604f6dd3a966fb6eda9`;
- `78cf3c1077f11bc06b5d0123a3ed7918f4e20ecb37df5f67ad65c23efe06782f`;
- `64fea2cc34f854bebdce2f33a6b1e0c8aacd900ea7039d645205897e5df72eb7`;
- `3e6a102ad33cab02bfc147e6422701343f1b064a327db7762cab265dae7b5464`;
- `5be86746120520b7cb12b5fecbd7d266acc09e968792eb70fd7388154087626c`.

Each comparison space has `beta1(Y)=1`.  The sampled curve bounds are
`.871704/1.009068/1.450654/1.089065/1.696259`; the corresponding first strict
`r>h` grid rows are `.875/1.025/1.475/1.100/1.700`.  At `r=0`, the P25, P50,
and P75 first crossings and observed-suffix sustained onsets all coincide at
`1.0/1.8/3.6/7.2`, with no chaos onset through duration 12.  At each case's
first strict curve-resolved row, P50 first and sustained onset are already
`1.0`.  This is frozen as
`low_r_fourier_closure_empirical_not_curve_resolved`: the staircase is an
empirical discrete Fourier-closure control and not a curve-resolved topology
claim.

The compact summary is written without probability smoothing, interpolation,
or scale selection:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/\
summarize_compass_probability_v2.py
```

Its `compass_probability_summary_v1/summary.json` SHA-256 is
`a1ed4793524b8ec96802c0d3866c873f3502ebc969eec1fa5048cc2d7cb8facd`,
and `case_summary.csv` SHA-256 is
`6272782ede61a3198242b76968f0d9b82e2174dca3f1643fdb53094deb3006b6`.
All five paired-start CSVs are byte-identical at SHA-256
`c4fe2f53961121ca439f22f62c52ce8d21ff042bb515f0b85b7cf669dd4b8ffb`.

The paper-scale renderer revalidates the plan and all result bindings before
writing its staged vector output:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v2/\
render_compass_probability_paper_v1.py \
  --plan code/experiments_planned/outputs/shared_coauthor_protocol/\
compass_fourier_embedded_probability_linf_v2_david_grid/plan.json \
  --replace
```

The code PDF and wired paper copy of `compassgait_C5p0.pdf` are byte-identical
at SHA-256
`c7f410bca46a1ae62ced79672298fe4331e7fcd7e8a03c1ee619988cc707f716`.
The render-sidecar SHA-256 is
`8b910544d4b6ebcd23908cbfa99d6db4dfbfc9081101f83252b8d7a0885954f1`.
The PDF is one fully vector page with no horizontal guide; the four periodic
vertical guides use the nominal stored-row suspension clock, and the chaos
panel has no period guide.
