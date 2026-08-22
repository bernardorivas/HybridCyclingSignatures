# Period-Doubling Cascade Package

## Purpose

This package generates fixed-time-span trajectories for two systems exhibiting period-doubling routes to chaos: the Roessler attractor and the passive compass-gait biped. The compass data support manuscript Example 3, the slope-dependent bifurcation analysis. The Rössler data provide an auxiliary continuous-system comparison rather than a numbered manuscript example. The stable canonical compass orbit in Example 2 uses the separate `chyll_v2/runs/compass_gait_phi007` artifacts.

## Systems

### Roessler Attractor

Continuous dissipative 3D system governed by:

```
dx/dt = -y - z
dy/dt = x + a*y
dz/dt = b + z*(x - c)
```

| Period | c   | a   | b   | Expected Period |
|--------|-----|-----|-----|-----------------|
| 1      | 4.0 | 0.1 | 0.1 | 1               |
| 2      | 6.0 | 0.1 | 0.1 | 2               |
| 4      | 8.5 | 0.1 | 0.1 | 4               |
| 8      | 8.7 | 0.1 | 0.1 | 8               |
| Chaotic| 9.0 | 0.1 | 0.1 | None            |

Initial condition: x0 = (1.0, 1.0, 0.0) for all regimes.

### Passive Compass Gait

Hybrid 4D biped with continuous swing phase (Lagrangian dynamics) and instantaneous impact map. State: [theta_ns, theta_s, dtheta_ns, dtheta_s] where ns/s denote non-stance/stance legs.

Nominal model parameters: mu = 2, beta = 1, l = 1 m (corresponds to M=5 kg, M_H=10 kg, L=1 m, A=0.5 m, B=0.5 m, G=9.81 m/s²).

| Period | phi (deg) | Expected Period |
|--------|-----------|-----------------|
| 1      | 4.00      | 1               |
| 2      | 4.75      | 2               |
| 4      | 5.00      | 4               |
| 8      | 5.02      | 8               |
| Chaotic| 5.20      | None            |

Initial conditions are post-impact states recorded from Goswami's return-map fixed points (see chyll_v2/systems/compass_gait_slope_configs.py).

## Design: Fixed Time Span

All five regimes of each system integrate for identical T and dt, ensuring uniform segment length across the cascade. This design enables direct comparison of trajectory complexity: periodic orbits produce shorter symbolic sequences, while chaotic regimes fill the available space.

- Roessler: T = 500 s, dt = 0.02 s → 25,001 samples
- Compass gait: T = 400 s, dt = 0.02 s → 20,001 samples (between impacts)

## Usage

### Generate Data

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/period_doubling

# Full run (slow)
python generate_all.py --system both

# Quick test (reduced time span)
python generate_all.py --system both --quick

# Roessler only
python generate_all.py --system roessler

# Compass gait only
python generate_all.py --system compass
```

### Plot Atlases

```bash
python plot_atlas.py

# Or specify custom paths:
python plot_atlas.py --data-root /path/to/data --fig-dir /path/to/figures
```

Produces:
- `figures/roessler_atlas.png` — 2×3 phase-plane grid (x-y projections)
- `figures/roessler_data_colored.png` — 1×5 time-colored 3D scatters
- `figures/compass_atlas_check.png` — 2×3 dual-leg phase planes

### Downstream Julia Analysis

Extract cycling signatures using the period_doubling lifts as input to the CHyLL pipeline:

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures

# Roessler period-2 orbit
julia --project="code/period_doubling/julia" \
  code/chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir code/period_doubling/data/roessler \
  --base roessler_period2 \
  --boxsize 0.3 --sb-radius 1 \
  --segment-lengths 20:10:300 --n-runs 200

# Compass period-4 orbit (hybrid case with learned latent embedding)
julia --project="code/period_doubling/julia" \
  code/chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir code/period_doubling/data/compass_gait \
  --base compass_period4 \
  --boxsize 0.3 --sb-radius 1 \
  --segment-lengths 20:10:300 --n-runs 200
```

For the compass gait, the CHyLL latent embedding route (learned from the hybrid suspension network) is available under `code/chyll_v2/cycling_signature/`.

## Cycling-Signature Computation

Once time-series data (positions and tangents CSVs) are generated, compute cycling signatures via the Julia driver:

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures

# Roessler all regimes
for regime in period1 period2 period4 period8 chaos; do
  julia --project="code/period_doubling/julia" \
    code/chyll_v2/cycling_signature/julia/run_subsegments.jl \
    --data-dir code/period_doubling/data/roessler \
    --base roessler_${regime} \
    --boxsize 0.3 --sb-radius 1 \
    --segment-lengths 20:10:300 --n-runs 200
done

# Output: code/period_doubling/data/roessler/signatures/
#   subsegments_roessler_{regime}_rank_at_radius.csv
#   subsegments_roessler_{regime}_rank{k}_spaces_at_radius.csv (k=1,2,...)
#   subsegments_roessler_{regime}_s21_inclusion.csv
#   subsegments_roessler_{regime}_rank_heatmap_rank{k}.csv
#   subsegments_roessler_{regime}_metadata.txt (includes stride)
```

**Data conventions:**
- Segment lengths are in post-stride samples (multiply by `dt * stride` to get time span)
- `dt` is read from `roessler_{regime}.npz` meta_json field
- Stride is read from metadata.txt line `stride=N` (default 1 if absent)
- Radii and segment lengths are axes of rank heatmaps

**Multi-radius evaluation:**
When calling the Julia script with `--eval-radius r_value`, variants of CSV files are written with suffix `_r{value_with_dots_as_p}`:
- `subsegments_roessler_{regime}_rank_at_radius_r0p8.csv`
- `subsegments_roessler_{regime}_rank1_spaces_at_radius_r0p8.csv`
- `subsegments_roessler_{regime}_s21_inclusion_r0p8.csv`

(Heatmap files are always generated with full radius range and do not use suffixes.)

### Plotting Cycling Signatures

Generate matplotlib reproductions of paper Figures 10, 11, 13, 14, 15:

```bash
cd /Users/bdoprad/Work/Projects/hybrid-cycling-signatures/code/period_doubling

# Default: plot all regimes at default evaluation radius
python plot_signatures.py

# Specify custom data and figure directories
python plot_signatures.py --data-root /path/to/data --fig-dir /path/to/figures

# Plot multi-radius variant (e.g., evaluation at radius 0.8)
python plot_signatures.py --eval-radius-suffix r0p8
```

**Output figures:**
- `roessler_fig10_rank_stacked.png` — Rank distribution (stacked bar chart)
- `roessler_fig11_sig1_stacked.png` — Rank-1 cycling spaces (per-system stacked bars, V1/V2/... labels)
- `roessler_fig13_sig2_stacked.png` — Rank-2 cycling spaces (per-system stacked bars, W1/W2/... labels)
- `roessler_fig14_inclusion.png` — Inclusion graphs (bipartite: V's ↔ W's)
- `roessler_fig15_rank_heatmaps.png` — Rank heatmaps grid (5 regimes × 3 ranks, shared colorbar)

For multi-radius evaluation, output filenames include the suffix: `roessler_fig10_rank_stacked_r0p8.png`, etc.

## Data Format

All regimes export three file types per system to `data/roessler/` and `data/compass_gait/`:

### CSV Lifts (space-delimited, no header)

- `{base}_positions.csv` — (N, d) matrix of state samples, loadable by Julia `readdlm(path, ' ', Float64)` 
- `{base}_tangents.csv` — (N, d) matrix of unit tangent vectors

### NPZ Archives

- `{base}.npz` — compressed NumPy archive containing:
  - `t` — (N,) time array
  - `x` — (N, d) positions
  - `v` — (N, d) unit tangents
  - `meta_json` — JSON string with system parameters and metadata
  - `impact_times`, `jump_minus`, `jump_plus` (compass gait only)

### Reports

- `{base}_report.txt` — human-readable summary (sample count, dimensions, dt, parameters)

### Summary

- `data/summary.csv` — one row per regime with columns: system, regime, parameter (c or phi_deg), expected_period, detected_clusters, ok (bool), n_samples, dt, t_span, n_impacts

## Periodicity Detection

The package includes post-hoc numerical checks:

- **Roessler**: `detect_roessler_period()` clusters local maxima of x-coordinate to count oscillations.
- **Compass gait**: `detect_compass_period()` clusters post-impact states.

`generate_all.py` runs these checks and exits with code 1 if any periodic regime does not match its expected period (chaos is exempt). This validates that the fixed-time-span data actually captures the intended dynamical regime.

## Raw compass-gait signatures (negative control)

Cycling signatures were computed directly on the raw (discontinuous) compass-gait state data, per regime:

```bash
julia --project=julia julia/run_signatures.jl \
  --data-dir data/compass_gait \
  --base compass_{period1,period2,period4,period8,chaos} \
  --boxsize 0.3 --sb-radius 1 --stride 2 \
  --segment-lengths 10,20,...,300 \
  --n-runs 150 \
  --eval-radius 0.1,0.2,0.3 \
  --out-dir data/compass_gait/signatures
```

Observed `beta1_Y` (first Betti number of the ambient cubical complex at boxsize 0.3): period1 = 0, period2 = 1, period4 = 0, period8 = 0, chaos = 0.

Result: in period1, period4, period8, and chaos the ambient complex carries no 1-cycles at all, so every subsegment has cycling rank 0 at all three evaluation radii — vanilla cycling signatures on the raw hybrid state see no cycling, because the reset discontinuity prevents the arcs from closing. This is the negative-control baseline for the forthcoming suspension-embedding comparison, where the same data lifted to the suspension is expected to exhibit the period-doubling cascade. One caveat: in the period2 regime the boxsize-0.3 cover incidentally closes a single loop (`beta1_Y = 1`), and rank 1 then dominates for segment time spans above roughly 1 at all three radii (rank-1 fractions 3907/4500, 4001/4500, 4018/4500 at r = 0.1, 0.2, 0.3); this is a resolution artifact of the cubical cover closing a loop that the trajectory itself does not close, not a recovered hybrid cycle, and it is absent in the other four regimes. A box-size probe confirms the artifact reading: `beta1_Y` for period2 is 5, 3, 0, 1 at boxsize 0.15, 0.2, 0.25, 0.3 respectively (`signatures_pilot/probe_period2_*`) — non-monotone in the cover scale, unlike the Roessler comparison spaces, which are stable across box sizes. On raw hybrid data the vanilla method thus produces both a false negative (the genuine period-1 limit cycle is invisible because its arc never closes) and a scale-unstable false positive (period2).

## Latent suspension lifts (positive hybrid computation)

`export_latent_lifts.py` lifts the SAME compass-gait trajectories used for the
raw negative control into the continuous latent suspension space of the
regime-matched trained CHyLL v2 models (`chyll_v2/runs/compass_gait_phi*`, 11-dim
latent). Flow samples enter as augmented states (x, s=0); at each recorded
impact a mapping-cylinder bridge (jump_minus, s), s in (0,1), n_s = 8 samples,
is inserted, so the lift is a single continuous polyline once the learned
gluing E(g,1) = E(r(g),0) holds. Every sample (arc or bridge) costs one dt of
suspension time. Tangents are the encoder pushforward (JVP) of the physical
unit tangents (arcs) and of e_s (bridges); a tag-aware finite-difference
tangent set is written alongside as a cross-check (mean |cos| agreement
>= 0.99; learned latent vector field agrees with the JVP tangents at
mean |cos| 0.97-0.99). period1 uses the phi=0.07 rad (4.0107 deg) model on
4.00 deg data (mismatch 2e-4 rad, noted in the report).

Model map: period1 -> phi007, period2 -> phi_1, period4 -> phi_2,
period8 -> phi_3, chaos -> phi_4_cloud.

Diagnostics per regime (`report_compass_{regime}.txt`): symbolic gluing seam
errors 0.072-0.128 (about a quarter to a third of the median latent arc step
0.29); reconstruction error concentrates on just-post-impact samples, the
expected quotient effect (decoder must choose a seam preimage); only the
encoder enters the signature computation.

Pilot (`julia/pilot_beta1_latent.jl`): beta_1(Y) = 1 for ALL five regimes,
stable across boxsizes 0.3-1.0 and strides 1-2 — the suspension closes the
loop the raw hybrid state cannot (raw: beta_1 = 0), with none of the raw
period2 scale instability.

Production runs (2026-08-13): boxsize 0.45, sb-radius 1, stride 1, segment
lengths 20:20:800, n-runs 150, eval radii 0.1/0.15/0.3/0.45; outputs in
`data/compass_gait_latent/signatures/`, figures
`figures/compass_latent_fig{10,11,13,14,15}_r*.png`.

Results:
- beta_1(Y) = 1 in every regime; a single 1-dim cycling space V1 everywhere
  (rank never exceeds 1, consistent with one closed-orbit topology in each
  periodic regime; the chaotic recurrence behavior is interpreted separately).
- Reference-simulator event-time recurrence gives full physical return times
  0.748241 / 1.502140 / 3.001914 / 6.004312 s for period1/2/4/8, with
  successive ratios 2.0076 / 1.9984 / 2.0002. The fundamental return lags are
  consistent with sampled period-doubling regimes independently of the return-
  map cluster diagnostic; a continuation study would be needed to establish
  the intervening bifurcations.
- The minimal closing length is an ordered coverage diagnostic, not a period
  estimator. At eval radii in the resolving band r in [0.09, 0.14], rank-1
  onset (>=135/150 runs) is 60 / 100 / 160 / 220 samples for
  period1/2/4/8. Their nominal suspension endpoint spans are
  1.18 / 1.98 / 3.18 / 4.38 s;
  reconstructing physical hybrid time from the stored starts gives median
  spans 1.009 / 1.660 / 2.587 / 3.580 s. In particular, the period8 cell
  closes before the full eight-impact, 6.004-s orbit is traversed.
- Period8 is empirically unresolved by the stored signature grid. Under the
  learned encoding of the reference-simulator post-impact states,
  corresponding daughter branches four impacts apart are separated by only
  0.00203-0.00815, well below the reachable radius floor. This scale mismatch
  is consistent with non-resolution but is not sufficient by itself because
  tangent directions also enter the signature metric. Values near 0.02 require
  a separately defined along-arc average and are not the return-section
  separation.
- Radius floor: below r ~ 0.09 periodic regimes stop closing entirely
  (sampling-phase drift: returns miss sampled points by up to half a latent
  sample step ~0.15). Chaos is qualitatively distinct — its rank-1 boundary
  is ragged and descends toward r -> 0 at long segment spans (recurrence),
  where periodic regimes are hard rank-0.
- Untrained-encoder control (`--untrained`, same architecture, seed 0, same
  bridge construction): seam gap 2.3-3.2 (the physical state jump, unglued),
  beta_1(Y) = 0, rank 0 at all lengths and radii
  (`data/compass_gait_latent_untrained/`). Together with the raw control this
  isolates the TRAINED gluing as the operative ingredient.

The reference-orbit checks and one-return-per-regime plots can be reproduced from
the stored NPZ files without simulation, integration, training, or Julia:

```bash
code/venv/bin/python code/period_doubling/plot_steady_compass_orbits.py \
  --check-only

code/venv/bin/python code/period_doubling/plot_steady_compass_orbits.py \
  --output-dir code/period_doubling/figures/steady_compass_orbits
```

The plotting run writes four individual return figures, an aligned comparison,
a focused period-4/period-8 return-section and lag-recurrence diagnostic, small
derived orbit extracts, a summary CSV, and machine-readable checks. It does not
modify the source archives. Hybrid closure is reported modulo leg relabeling at
impacts. The return-section panels are labelled as return-map diagnostics
rather than topological evidence; cluster counts and elapsed-time ratios are
descriptive and do not pass-gate the recurrence validation.

### Fine variant (dt = 0.005) — Roessler-comparable figures

The dt = 0.02 lift caps the analysis in the fully-merged regime: latent arc
steps ~0.29 force boxsize >= ~0.3, every doubled branch merges, beta_1(Y) = 1
everywhere, rank never exceeds 1, and figs 13/14 are empty — visibly poorer
than the Roessler set. The dt = 0.005 regeneration (`data_fine/`, same ICs
and t_span; arc steps ~0.072, max gap 0.14; uniform n_s = 26) lowers the
usable boxsize to 0.2 and lands in the partially-resolved regime the Roessler
study occupies.

Production (2026-08-13): boxsize 0.2, sb-radius 1, stride 2, lengths
40:40:1600, n-runs 150, eval radii 0.05/0.1/0.15/0.2; outputs in
`data_fine/compass_gait_latent/signatures/`, figures in `figures_fine/`.

Results (cf. Roessler at boxsize 1.5: beta_1 = 1/2/5/3/2):
- beta_1(Y) = 1 / 4 / 2 / 4 / 1 for period1/2/4/8/chaos, stable at strides
  1-2; 4/4/9 at boxsize 0.15.
- Rank takeovers mirror parts of the Roessler figures: period1 saturates at
  rank 1; period2 passes rank 1 -> rank 2 -> rank 3; and period4 and period8
  saturate at rank 2. Here `(length - 1) * dt * stride` is a nominal suspension
  duration because bridge samples also cost `dt`, not physical hybrid time.
  Period4 and period8 have identical terminal ranks at every stored radius
  and differ in only a few intermediate duration cells, so the current fine
  computation does not resolve those two regimes.
- Multiple 1-dim cycling spaces (V1-V3) appear in the pre-saturation band
  and the V -> W inclusion graphs (fig14) are nontrivial for period2/4/8;
  period1 and chaos have a single V1 and no rank-2 segments.
- Chaos differs from Roessler chaos: its latent band does not fragment at
  boxsize >= 0.1 (beta_1 stays 1), so it keeps the rank <= 1
  recurrence-limited diffuse boundary instead of a rank-1/2 mixture.

Post-hoc duration/metric audit (2026-08-20): the birth summaries and stored
start indices can be joined trial by trial.  Assigning every bridge sample its
instantaneous impact time gives an exact physical endpoint span for each
archived window; no median sample-length relabeling is required.  For a
continuous-time-comparable probability panel, starts inside zero-time bridges
are excluded and all remaining windows, including rank-zero windows, enter
`P(rank > 0)`.

The audit also found that the stored fine grid is exploratory rather than
curve-hypothesis-valid. At `C=0.2`, stride two, every archived periodic window
has a maximum consecutive dynamic distance above `r=0.2`; only three retained
chaos windows clear that radius. Even at stride one, an arc--bridge tangent
corner reaches roughly `1.35C`--`1.62C`. The existing probability surface
must therefore not be used as a rigorous claim figure.
The prepared denser tied-cover follow-up under `experiments_planned/` varies
`C=0.10:0.025:0.30`, uses paired physical-duration windows through 7.5 s,
evaluates `r/C` through 1.75, and records the per-window curve bound for all
five regimes. Its prepared analyzer selects a connected plateau from the
labelled tuning half, freezes the center, and then reads the validation half;
no job from that plan has been run. Varying boxsize at fixed `sb_radius=1` ties
comparison-cover resolution to `C`, so the selected scale must be repeated
with another factorization before it is called robust. A common numerical `C`
also requires a preregistered normalization or a justification that the five
separately trained encoders use comparable latent-position units.

A matched fixed-position, tied metric/tangent-cover diagnostic is now complete
at `boxsize=0.2` for
`C=0.2/0.4/0.6/0.8/1.0`, hence `sb_radius=1/2/3/4/5`.  Each `C` uses the same
20 starts at 30 target durations in all five regimes: 3,000 signatures per
scale.  In period-1/2/4/8/chaos order, the comparison-space Betti vectors by
increasing `C` are `(1,4,2,4,1)`, `(0,2,1,2,0)`, `(0,2,2,2,0)`,
`(0,2,1,2,0)`, and `(0,3,2,3,0)`.  The corresponding rank-zero totals out of
600 are `(40,47,51,54,54)`, `(600,64,62,71,600)`,
`(600,44,72,77,600)`, `(600,43,72,76,600)`, and
`(600,41,63,72,600)`.

All five scales have six common curve-resolved grid rows, starting at
`r=0.325/0.650/0.975/1.300/1.625`.  At those rows, the first durations with
`P(rank>0)>=0.5` are `(.75,.75,1,1,1)`, `(none,1,1,1,none)`,
`(none,.75,1,1.25,none)`, `(none,.75,1,1.5,none)`, and
`(none,.75,.75,.75,none)` seconds.  Only the period-1 value at `C=0.2` aligns
with the measured 0.748/1.502/3.002/6.004-second orbit periods.  The period-4
and period-8 valid-band matrices remain nearly identical, so the sweep does
not distinguish the doubled periods.  Curve resolution alone is insufficient:
no tested construction has a certified `r0(Y;Gamma)` lower bound.  The
same-`C` alternative factorizations, including `(0.5,2)` and `(1.0,1)` at
`C=1`, were not run.

This is not a clean metric-`C` sweep: changing `sb_radius` changes both
`C=boxsize*sb_radius` and the tangent cover of `Y`.  Isolating the metric
coefficient requires an explicit `DynamicDistance` override while holding the
full comparison cover at `(boxsize,sb_radius)=(0.2,1)`.

The planned-results renderer validates `C=boxsize*sb_radius` from all five
metadata files, ingests the manifest plus `*_births.csv` layout, and derives
the suffix automatically.  The matched panels are
`compass_c_sweep_n20/compassgait_C{0p2,0p4,0p6,0p8,1p0}.pdf`; the main
fixed-`C=1` diagnostic is also `compassgait_C1p0.pdf`.  The archived
large-sample panel remains `compassgait_C0p2.pdf`, and separate paths prevent
the diagnostics from overwriting it or one another.

## Shared coauthor-protocol analysis

`shared_probability/` contains one versioned analysis driver that applies the
coauthor's documented Rössler probability protocol to both Rössler and the
fine learned Compass suspension.  It fixes the same independent random-window
sampling, `L * dt` duration labels, UTB cover and metric, radius grid,
`F_43` coefficients, `P(rank > 0)` statistic, and two-row renderer for both
systems.  This is separate from the historical five-regime artifacts and the
physical-duration diagnostics above.

The prepared cases include a requested five-column Rössler
period-1/2/4/8/chaos panel, a five-column Compass panel, and a separate
two-case `a=2.82/2.86` Rössler positive control.  The Compass flow-direction
and encoder-JVP tangent inputs are explicitly named controls; a true learned-
flow rollout remains a distinct missing input rather than a silently changed
statistic.

The primary profile follows the note's literal infinity-normalized tangent
lift; an explicit Euclidean-normalized compatibility profile captures the
cited implementation convention.  Missing coauthor seed, initial condition,
radius grid, and guide-line provenance are frozen and labelled as assumptions.
See `shared_probability/README.md` for the frozen protocol, exact commands,
and completed-output audit.  The literal-infinity profile was executed for the
two-case Rössler reference and both requested five-regime panels on
2026-08-21.  The reference reproduces a factor-two probability boundary in a
narrow curve-resolved band.  The five-regime Rössler run is horizon-limited,
and the Compass common curve-resolved band is identically probability one in
all five regimes; neither five-regime run establishes the desired four-period
discriminator.

`shared_probability_v2/` contains the completed David-family and Compass
Fourier-orbit follow-ups.  The R\"ossler path keeps the same binary statistic,
random-window rule, infinity-normalized lift, `F_43`, `5 Q_1 x Q_2`, `C=5`,
and radius grid, but extends the common duration grid to `1:0.2:60`.  Its
periodic inputs are independently certified primitive Fourier orbits at
`a=2.82/2.86/4.10/4.18`; `a=4.30` is a positive-Lyapunov chaotic control.  At
`r=0.025`, the half-probability onsets are `5.8/11.6/23.4/46.6`, matching the
certified periods at the plotted resolution.  The period-4/8 full-period bands
remain below their sampled curve bounds, so the last two steps are empirical
discrete signatures rather than curve-resolved claims.

The Compass path supersedes the earlier frozen-path `C=5` paper figure with
a strict derived bundle: 32 phase-aligned late `q`-impact cycles, `H=6q`
Fourier harmonics, and analytic derivatives for the four periodic cases.
Chaos is explicitly a mixed-semantics control using the interpolated frozen
encoded path and interpolated learned-flow directions.  At `dt=.0025`, the
`400:80:4800` length grid preserves displayed duration `1:.2:12`; every other
probability/control setting above is unchanged.  Its `r=0` P25/P50/P75
first-and-sustained onsets are all `1.0/1.8/3.6/7.2`, with no chaos onset by
12.  Because the first strict curve-resolved radii are
`.875/1.025/1.475/1.100/1.700` and their P50 onsets are already `1.0`, the
frozen classification is
`low_r_fourier_closure_empirical_not_curve_resolved`; no probability
smoothing or post-hoc radius selection is used.

The Compass bundle/plan SHA-256 values are
`19b05c2db02f67abf712f50d0d965cdc399db9be477572ca9d1828c343a87f85`
and `0a1d18864234ca8482b7587e74fb68db161f9fc0f373340b9058148c4368677e`.
The period-1/2/4/8/chaos result-binding SHA-256 values are
`c7c81eaa1d6868a612fd1e4f04ec98502493fdcdf8f48604f6dd3a966fb6eda9`,
`78cf3c1077f11bc06b5d0123a3ed7918f4e20ecb37df5f67ad65c23efe06782f`,
`64fea2cc34f854bebdce2f33a6b1e0c8aacd900ea7039d645205897e5df72eb7`,
`3e6a102ad33cab02bfc147e6422701343f1b064a327db7762cab265dae7b5464`,
and `5be86746120520b7cb12b5fecbd7d266acc09e968792eb70fd7388154087626c`.
The summary JSON/CSV SHA-256 values are
`a1ed4793524b8ec96802c0d3866c873f3502ebc969eec1fa5048cc2d7cb8facd`
and `6272782ede61a3198242b76968f0d9b82e2174dca3f1643fdb53094deb3006b6`;
the final `compassgait_C5p0.pdf` SHA-256 is
`c7f410bca46a1ae62ced79672298fe4331e7fcd7e8a03c1ee619988cc707f716`.
See `shared_probability_v2/README.md` and the two versioned summary roots for
the frozen plans, exact methods, and onset tables.

`shared_probability_v3/` contains the completed refined Compass
`C=0.75` pilot. It preserves the binary statistic, paired 20-start random-
window rule, seed `20260820`, infinity-normalized lift, `F_43`, and five-case
layout. The refined bundle uses `dt=.00125`, durations `.2:.1:12`, continuous
OLS period guides `.878247/1.762131/3.521923/7.044295`, and
`r=0:.002:.5`. All five comparison spaces have `beta1(Y)=1`; across 11,900
validated trials, no first birth is zero or near zero, and every `r=0`
probability cell is zero. This fixes the earlier exact recurrence/grid-lock
artifact.

The refined pilot does not pass its scientific acceptance test. The five
curve bounds are
`0.0685227462000182/0.0757526962952281/0.109046270801063/`
`0.0959224712507916/0.155813652930367`, giving first common
valid radius `.156`. No common curve-resolved radius strictly orders the four
periodic cases for pooled P25, P50, or P75 first or sustained onsets; at
`.156`, the pooled sustained-P50 onsets are `.9/1.1/1.2/1.2`. The exact-birth
low-radius rendering is restricted to `r<=.02`, below every bound, and uses no
smoothing or interpolation. Both new PDFs remain under the validated code
output root and are not wired into the manuscript; the paper retains
`compassgait_C5p0.pdf` pending author review.

The v3 bundle manifest and plan SHA-256 values are
`371cfb4bf751ab6f4b04226dece7e1049fceb516db638fee44f111ada352b442`
and `3245bbd941debf5eebda6ba0ddf2ae618154095757563d644166d914f8eab45a`.
The execution and validation used the isolated NumPy `2.4.2` environment
recorded in the plan. The base interpreter now exposes NumPy `2.5.2`, while
the isolated environment and all raw Julia output hashes remain unchanged.
See `shared_probability_v3/README.md` and
`experiments_planned/outputs/shared_coauthor_protocol/`
`compass_refined_v3_probability_linf_C0p75/compact_summary_v2/` for the exact
bindings, split diagnostics, and acceptance tables.

## References

- Roessler, O. E. (1976). "An equation for continuous chaos." *Phys. Lett. A*, 57(5), 397–398.
- Goswami, A., Thuilot, B., & Espiau, B. (1998). "A study of the passive gait of a compass-like biped robot: Symmetry and chaos." *Int. J. Robot. Res.*, 7(2), 19–42.
- Conley, C. (1978). *Isolated invariant sets and the Morse index*. CBMS Regional Conference Series in Mathematics.

## License

Research code for hybrid-cycling-signatures project.
