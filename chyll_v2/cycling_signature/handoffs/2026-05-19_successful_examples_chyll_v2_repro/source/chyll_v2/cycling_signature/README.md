# CHyLL v2 Cycling-Signature Lab

Postprocessing and plotting for David Hien's cycling-signature pipeline on the
CHyLL v2 baseline. This folder is intentionally separate from the model/training
code in `chyll_v2/chyll_v2/` and from training entry points in `chyll_v2/scripts/`.

## Layout

```text
chyll_v2/cycling_signature/
  export/   # Python exporters from trained CHyLL v2 models to lift CSVs
  julia/    # Julia cycling-signature experiment runners
  plot/     # Python/Matplotlib publication figure builders
  data/     # exported lifts, barcode CSVs, subsegment summaries
  figures/  # generated manuscript-facing figures
```

## Rimless Wheel: Current Single-Orbit Check

Export finite-difference unit-tangent data:

```bash
python chyll_v2/cycling_signature/export/prepare_rimless_lift.py
```

Run David's existing one-orbit rank sweep:

```bash
julia --project="time series/cycling_signature" \
  "time series/cycling_signature/run_cycling_signature.jl" \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2
```

Export and run the learned-vector-field tangent variant:

```bash
python chyll_v2/cycling_signature/export/prepare_rimless_lift.py \
  --tangent-source vfield \
  --base continuous_lift_chyll_v2_vfield

julia --project="time series/cycling_signature" \
  "time series/cycling_signature/run_cycling_signature.jl" \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2_vfield
```

Current result:

- `continuous_lift_chyll_v2`: finite-difference UT rank is zero on the
  standard cover grid.
- `continuous_lift_chyll_v2_vfield`: learned-vector-field tangents recover
  rank one only at coarse cover cells `(boxsize, sb_radius)=(0.3, 1)` and
  `(0.2, 1)`.

## Rimless Wheel: David-Style Subsegment Summaries

`julia/run_subsegments.jl` runs random subsegment experiments using
`RandomSubsegmentExperiment` from `CyclingSignatures.jl` and writes plain CSV
summaries. Use a pilot grid first; runtime scales roughly with
`length(segment_lengths) * n_runs`.

Finite-difference unit-tangent pilot:

```bash
julia --project="time series/cycling_signature" \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2 \
  --boxsize 0.3 \
  --sb-radius 1 \
  --segment-lengths 20:20:300 \
  --n-runs 25 \
  --r-subdivisions 101 \
  --out-prefix subsegments_chyll_v2_diff_pilot
```

Learned-vector-field tangent pilot:

```bash
julia --project="time series/cycling_signature" \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2_vfield \
  --boxsize 0.3 \
  --sb-radius 1 \
  --segment-lengths 20:20:300 \
  --n-runs 25 \
  --r-subdivisions 101 \
  --out-prefix subsegments_chyll_v2_vfield_pilot
```

The pilot showed that the relevant birth radii for the flow-tangent lift are
around `10^-5` to `10^-3`, so a manuscript-style heatmap should zoom the
filtration scale rather than plot the full cover radius `0.3`.

Pilot result from `--segment-lengths 20:20:300 --n-runs 25`:

- `subsegments_chyll_v2_diff_pilot`: finite-difference UT has rank 0 for
  every sampled segment length and radius; the comparison carrier has
  `beta_1(Y)=0`.
- `subsegments_chyll_v2_vfield_pilot`: `V_theta` tangents have
  `beta_1(Y)=1`; rank 1 appears coherently for segment lengths `200` and
  above, with all `25/25` sampled segments rank 1 at the evaluation radius
  `0.3`.

Subsegment outputs:

- `*_rank_heatmap_rank0.csv`, `*_rank_heatmap_rank1.csv`: radius by segment
  length matrices.
- `*_rank_at_radius.csv`: rank counts at the evaluation radius.
- `*_rank1_spaces_at_radius.csv`: frequent rank-1 cycling spaces at the
  evaluation radius.
- `*_birth_summary.csv`: one row per sampled segment with birth vector data.
- `*_segment_starts.csv`: sampled starts for reproducibility.
- `*_metadata.txt`: parameters and source files.

Build the current pilot figure:

```bash
python chyll_v2/cycling_signature/plot/plot_rimless_subsegments.py
```

Generated pilot figure:

- `figures/rimless_wheel/fig_rimless_chyll_v2_subsegments_pilot.pdf`
- `figures/rimless_wheel/fig_rimless_chyll_v2_subsegments_pilot.png`

Interpretation: at this pilot density, finite-difference UT gives zero
rank-1 subsegments throughout the radius/length grid, while the learned
`V_theta` tangents recover rank 1 coherently from segment length `200`
onward. This is the first David-style barcode-summary figure; before using it
in the manuscript, rerun the same pipeline with denser segment lengths and
larger `n_runs`.

## Rimless Wheel: Manuscript-Style Zoom Figures

Dense zoom runs:

```bash
julia --project="time series/cycling_signature" \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2 \
  --boxsize 0.3 \
  --sb-radius 1 \
  --segment-lengths 20:10:300 \
  --n-runs 100 \
  --r-max 0.002 \
  --eval-radius 0.001 \
  --r-subdivisions 161 \
  --out-prefix subsegments_chyll_v2_diff_zoom100

julia --project="time series/cycling_signature" \
  chyll_v2/cycling_signature/julia/run_subsegments.jl \
  --data-dir chyll_v2/cycling_signature/data/rimless_wheel \
  --base continuous_lift_chyll_v2_vfield \
  --boxsize 0.3 \
  --sb-radius 1 \
  --segment-lengths 20:10:300 \
  --n-runs 100 \
  --r-max 0.002 \
  --eval-radius 0.001 \
  --r-subdivisions 161 \
  --out-prefix subsegments_chyll_v2_vfield_zoom100
```

Build the two manuscript-style figures:

```bash
python chyll_v2/cycling_signature/plot/plot_rimless_signature_figures.py
```

Generated figures:

- `figures/rimless_wheel/fig_rimless_signature_rank_distribution.pdf`
- `figures/rimless_wheel/fig_rimless_signature_rank_distribution.png`
- `figures/rimless_wheel/fig_rimless_signature_rank_heatmaps.pdf`
- `figures/rimless_wheel/fig_rimless_signature_rank_heatmaps.png`

Interpretation: the orbit-tangent lift remains rank 0 because its comparison
space has `beta_1(Y)=0`. The flow-tangent lift has `beta_1(Y)=1` and shows a
radius-dependent rank-1 transition near segment length `200`, with nontrivial
texture on the `10^-3` filtration scale.

## Rimless Wheel: Tag-Aware Finite Differences

The first finite-difference exporter used consecutive differences over the
entire concatenated lift. That includes arc/bridge boundary chords, including
the latent gluing leap. `prepare_rimless_lift.py` now has
`--tangent-mode tagaware`, which differentiates only inside each exported
arc or bridge piece and copies the previous within-piece tangent at the final
sample of each piece.

Exports:

```bash
python chyll_v2/cycling_signature/export/prepare_rimless_lift.py \
  --model chyll_v2/runs/rimless_wheel/model.pt \
  --config chyll_v2/runs/rimless_wheel/config.json \
  --base continuous_lift_chyll_v2_tagaware \
  --tangent-source diff \
  --tangent-mode tagaware

python chyll_v2/cycling_signature/export/prepare_rimless_lift.py \
  --model chyll_v2/runs/rimless_wheel_wv1/model.pt \
  --config chyll_v2/runs/rimless_wheel_wv1/config.json \
  --base continuous_lift_chyll_v2_wv1_tagaware \
  --tangent-source diff \
  --tangent-mode tagaware
```

Subsegment outputs:

- `subsegments_chyll_v2_tagaware_*`
- `subsegments_chyll_v2_wv1_tagaware_*`

Result at `r=0.001`, `segment_lengths=20:10:300`, `n_runs=100`:

| run | tangent export | `beta_1(Y)` | peak rank-1 fraction |
|---|---|---:|---:|
| `w_v=0` | tag-aware FD | 0 | 0% |
| `w_v=1` | tag-aware FD | 0 | 0% |

Conclusion: cross-boundary finite-difference chords were a real export bug,
but removing them does not recover the cycling signature. The finite-difference
lift failure is deeper than the bridge-to-arc chord convention.

## Rimless Wheel: Tangent-Coherence Diagnostics

Run:

```bash
python chyll_v2/cycling_signature/diagnostics/tangent_coherence.py \
  --bases continuous_lift_chyll_v2 \
          continuous_lift_chyll_v2_tagaware \
          continuous_lift_chyll_v2_vfield \
          continuous_lift_chyll_v2_wv1 \
          continuous_lift_chyll_v2_wv1_tagaware \
          continuous_lift_chyll_v2_wv1_vfield \
  --out-prefix tangent_coherence_all
```

Outputs:

- `data/rimless_wheel/tangent_coherence_all.txt`
- `data/rimless_wheel/tangent_coherence_all.csv`

Result: the tag-aware finite-difference tangent field is locally coherent
inside each exported piece. For `w_v=0`, the within-piece median rotation is
`0.371` degrees and p95 is `3.521` degrees. For `w_v=1`, the within-piece
median rotation is `0.340` degrees and p95 is `4.172` degrees. Thus the
rank-0 result is not caused by random local Jacobian rotation along arcs.

The large residual mismatch is at the arc-to-bridge transition: tag-aware FD
has median boundary angles of `159.78` degrees (`w_v=0`) and `166.92` degrees
(`w_v=1`). The bridge-to-next-arc transition is modest (`8.06` and `9.90`
degrees). This points to a discontinuity at guard entry into the cylinder,
not within-piece tangent incoherence.

## Rimless Wheel: Two-Seam Velocity Experiment

The original one-seam CHyLL v2 velocity loss trained only the bridge-to-arc
exit seam. Diagnostics above showed that the failed finite-difference UT rank
was instead caused by the untrained arc-to-bridge entry seam. The training code
now exposes both masks and applies `L_v` on their union, while keeping `L_g` on
the gluing exit seam only.

Full run:

```bash
python chyll_v2/scripts/train_rimless.py \
  --w-v 1.0 \
  --run-dir chyll_v2/runs/rimless_wheel_wv1_twoseam \
  --figure-dir chyll_v2/figures/rimless_wheel_wv1_twoseam \
  --ode-method rk4 \
  --device cpu
```

Exports:

- `continuous_lift_chyll_v2_wv1_twoseam_tagaware`: tag-aware finite
  differences of `E(X')`.
- `continuous_lift_chyll_v2_wv1_twoseam_vfield`: learned vector-field
  tangents `V_theta(z)`.

Tangent-coherence output:

- `data/rimless_wheel/tangent_coherence_twoseam.txt`
- `data/rimless_wheel/tangent_coherence_twoseam.csv`

Result: the entry seam is repaired. Tag-aware FD arc-to-bridge boundary angles
drop from `166.92` degrees to `0.756` degrees; bridge-to-arc angles drop from
`9.90` degrees to `0.757` degrees. The `V_theta` export is also coherent
across both seams (`0.187` degrees entry, `0.530` degrees exit).

Subsegment rank result at `r=0.001`, `segment_lengths=20:10:300`,
`n_runs=100`:

| tangent export | `beta_1(Y)` | first rank-1 length | peak rank-1 fraction |
|---|---:|---:|---:|
| tag-aware FD | 1 | 200 | 90% at length 270 |
| `V_theta` | 1 | 200 | 90% at length 270 |

Neither row reaches 100% rank-1 at the evaluation radius, but both recover the
expected comparison-space cycle and both have identical rank-at-radius counts
under the fixed random seed. This confirms the main diagnosis: the rank-0 FD
failure was not due to reconstruction error or local tangent noise, but to the
missing entry-seam tangent condition.

Generated two-seam manuscript-style figures:

- `figures/rimless_wheel/fig_rimless_signature_rank_distribution_twoseam.pdf`
- `figures/rimless_wheel/fig_rimless_signature_rank_distribution_twoseam.png`
- `figures/rimless_wheel/fig_rimless_signature_rank_heatmaps_twoseam.pdf`
- `figures/rimless_wheel/fig_rimless_signature_rank_heatmaps_twoseam.png`
