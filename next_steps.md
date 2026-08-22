# Compass-Gait Shared-Representation Experiments

## Status and scope

This document is a plan. None of the experiments below have been run.

All implementation, training outputs, diagnostics, and figures belong under
the `code/` repository. The `paper/` repository is out of scope: do not edit,
stage, or commit anything there unless an author explicitly asks for that
later.

## Why do this?

The five current Compass-gait models were trained separately. Each model has
its own encoder and decoder, so the same numerical radius is being applied to
five latent coordinate systems that are similar but not literally identical.
That makes cross-angle comparisons harder to interpret.

The main question is:

> Does the period-doubling hierarchy become clearer when every angle is
> represented in one common latent coordinate system?

This is an important control, not a promised fix. Existing diagnostics suggest
that the period-4 and period-8 daughter branches are genuinely close, so a
shared representation may still produce compressed cycling-space blocks.

We will test two ways to learn a common representation:

1. Joint training at the five target angles with one shared encoder and
   decoder and five separate latent vector fields.
2. Continuation in the slope angle, using one shared encoder and decoder and
   one smooth angle-conditioned latent vector field.

The first experiment is the cleaner ablation. The second is the more natural
model of a smoothly parameterized family.

## Plan at a glance

1. Put every slope in the same guard-aligned physical coordinates.
2. Fix the seam loss so that gluing and decoding no longer ask for
   contradictory targets.
3. Train one shared \(E,D\) with five separate vector-field heads.
4. Validate the representation and dyadic return geometry before computing
   expensive cycling signatures.
5. Train one conditional field \(V(z,\phi)\) by continuation, with replay and
   a fixed chart anchor.
6. Compare both learned models with the old separate models and with a
   non-learned guard-aligned physical-suspension control.

## Common geometric setup

### Align the guards before learning

For slope angle \(\phi\), the Compass-gait guard is

\[
G_\phi
=
\{\theta_{\mathrm{ns}}+\theta_{\mathrm{s}}=-2\phi,
\ \theta_{\mathrm{ns}}-\theta_{\mathrm{s}}>0.01\}.
\]

Use slope-relative angles

\[
u_{\mathrm{ns}}=\theta_{\mathrm{ns}}+\phi,
\qquad
u_{\mathrm{s}}=\theta_{\mathrm{s}}+\phi.
\]

In these coordinates every slope has the same guard,

\[
u_{\mathrm{ns}}+u_{\mathrm{s}}=0.
\]

Angles are stored in radians internally, even when the schedule is written in
degrees.

The impact reset remains the same because it swaps the two leg angles and
depends on their difference. The slope dependence moves into the continuous
dynamics, where it belongs.

Every model below therefore uses the common augmented input

\[
y=(u_{\mathrm{ns}},u_{\mathrm{s}},
\dot u_{\mathrm{ns}},\dot u_{\mathrm{s}},s).
\]

The conversion between \(u\) and the physical angles \(\theta\) is fixed and
analytic. It is not learned.

### Use one global scale

- Fit one normalization using the pooled training data.
- Freeze it before comparing experiments.
- Never normalize or rescale an angle separately.
- Use the same scale for the two leg-angle coordinates and the same scale for
  the two angular-velocity coordinates.
- Save the normalization constants with every checkpoint.

This rule is essential because the cycling analysis uses metric distances in
the latent space.

### Treat the quotient seam correctly

At the end of the mapping cylinder, the quotient identifies

\[
(g,1)\sim (R(g),0).
\]

The encoder should therefore satisfy

\[
E(g,1)=E(R(g),0).
\]

A deterministic decoder cannot reconstruct both pre-quotient representatives
from that one latent point. Do not ask it to do so.

The primary rule is to mask reconstruction in a small neighborhood of the
glued endpoint. A secondary ablation may instead decode one canonical
representative, preferably the post-impact state. Never train one run against
both endpoint labels.

The primary experiment should use exact endpoint glue pairs. Do not treat a
coarse \(\tau\)-step that has already advanced into post-impact flow as the
exact identification.

## Shared rules for both experiments

The target slopes are

\[
4.00^\circ,
\quad 4.75^\circ,
\quad 5.00^\circ,
\quad 5.02^\circ,
\quad 5.20^\circ.
\]

Use an exact \(4.00^\circ\) data record. The old period-1 checkpoint at
\(0.07\) radians, approximately \(4.0107^\circ\), is not an exact substitute.

For both experiments:

- Split data by complete trajectories, not by individual windows.
- Keep held-out trajectories for every angle.
- Balance the five angles in every joint optimization stage.
- Stratify batches so that base flow, cylinder flow, entry seams, and glued
  endpoints are all represented.
- Sample all branches of a period-2, period-4, or period-8 orbit evenly.
- Save the physical data manifest, split, seed, environment, configuration,
  checkpoint hashes, optimizer state, and normalization.
- Use versioned output directories and refuse accidental overwrite.
- Run at least one smoke test before any full training.
- Run one development seed first. Repeat an accepted design with at least three
  seeds before making a scientific conclusion.
- Compute the same dyadic-distance diagnostics on a non-learned,
  guard-aligned physical suspension. This separates representation failure
  from genuinely compressed hybrid geometry.

## Experiment 1: shared \(E,D\) with five vector-field heads

### Question

Does a literally shared latent chart improve comparability and preserve the
dyadic return geometry better than five separately trained charts?

### Model

Use

\[
z=E(y),
\qquad
\dot z=V_{\phi_j}(z),
\qquad
\widehat y=D(z),
\]

where:

- \(E\) is shared by all five angles;
- \(D\) is shared by all five angles;
- each target angle has one full-capacity vector-field head
  \(V_{\phi_j}\);
- neither \(E\) nor \(D\) receives \(\phi\) as an input.

The five heads isolate the chart-sharing question without also requiring a
single network to interpolate in \(\phi\).

### Balanced joint update

At every optimizer step:

1. Draw one homogeneous batch from each angle.
2. Encode all five batches with the same \(E\).
3. Roll out each batch with its own \(V_{\phi_j}\).
4. Decode all five batches with the same \(D\).
5. Average the five per-angle losses.
6. Take one optimizer step.

Do not train the five angles sequentially in this experiment. Sequential
fine-tuning would allow catastrophic forgetting and would no longer be the
clean joint-training ablation.

### Loss

Use the existing reconstruction, latent-dynamics, gluing, seam-velocity, and
anti-collapse ideas, with three changes:

- Mask or canonicalize reconstruction at the quotient seam.
- Compute anti-collapse control on the pooled latent batch, rather than forcing
  every narrow periodic orbit to fill every latent coordinate independently.
- Report every loss separately for every angle, not only as a pooled mean.

### Training phases

#### Phase 0: smoke test

- Tiny datasets from all five angles.
- Short horizons and a few updates.
- Verify head routing, balanced batches, exact seam pairs, checkpoint resume,
  and deterministic replay.

#### Phase A: shared-chart baseline

- Train from scratch.
- Use \(w_v=0\), matching the existing baseline.
- Save the best held-out checkpoint, not just the final checkpoint.

#### Phase B: seam-velocity fine-tuning

- Start from the accepted Phase-A checkpoint.
- Enable seam-velocity compatibility with a modest weight first.
- Compare at least \(w_v=0.1\) and \(w_v=1\); do not assume the larger value is
  better.
- Use a smaller learning rate than Phase A.

### Required outputs

- One joint checkpoint containing shared `encoder`, shared `decoder`, and five
  named vector-field heads.
- Five compatibility checkpoints for existing exporters. They must contain
  byte-identical \(E,D\) weights and the selected vector-field head.
- Per-angle and pooled training/validation logs.
- Reconstruction, gluing, seam-velocity, rollout, and tangent-alignment
  reports.
- A fixed-probe latent-distance report proving that the five analyses use the
  same chart and scale.

## Experiment 2: continuation with a conditional field

### Question

Can a smoothly parameterized latent flow be learned by transporting one model
through the slope family, while keeping one common latent chart?

### Model

Use

\[
z=E(y),
\qquad
\dot z=V(z,\widetilde\phi),
\qquad
\widehat y=D(z),
\]

where \(\widetilde\phi\) is a fixed normalized version of the slope angle.
Only the vector field receives \(\phi\). The encoder and decoder remain
angle-independent.

This is equivalent in spirit to adjoining \(\dot\phi=0\) and learning one
continuous parameterized family.

### Important interpretation

Warm-starting an entire new model at each angle does not prove that the latent
charts are the same. Small changes in \(E,D\) can accumulate into scale,
rotation, shear, or nonlinear coordinate drift.

Continuation should therefore be used as a curriculum for one growing model,
not as a chain of unrelated checkpoints.

### Angle schedule

All increments below are in degrees.

- Begin at exactly \(4.00^\circ\).
- Use \(0.01^\circ\) increments over most of the interval to
  \(5.20^\circ\).
- Use approximately \(0.002^\circ\) to \(0.005^\circ\) increments near the
  tightly packed period-4, period-8, and aperiodic transition, especially over
  roughly \(4.98^\circ\) to \(5.05^\circ\).
- Determine the final refined schedule from a return-map continuation scan,
  not from the appearance of the cycling heatmap.

Do not confuse \(0.01^\circ\) with \(0.01\) radians. The latter is about
\(0.573^\circ\) and is too coarse here.

### Continuation curriculum

At each new angle \(\phi_k\):

1. Continue or regenerate the correct physical return orbit at \(\phi_k\).
2. Add the new data to the training set; do not discard earlier angles.
3. Initially freeze \(E,D\) and update only \(V(z,\phi)\).
4. Use replay: approximately half of each update should represent the current
   angle and half should be balanced over earlier angles.
5. If the frozen chart fails held-out reconstruction, gluing, or tangent tests,
   unlock \(E,D\) with a learning rate 20 to 100 times smaller than the
   vector-field rate.
6. If \(E,D\) are unlocked, retain replay and apply the chart anchors below.
7. After reaching \(5.20^\circ\), perform one balanced joint polish over all
   target and retained intermediate angles.
8. Re-encode every final analysis trajectory with the final shared \(E,D\).

Intermediate angles should use a small transport dataset and short warm
fine-tuning. Reserve the full dataset and full curriculum for the five final
analysis angles.

### Prevent chart drift

Create one fixed physical anchor bank before training. It should include:

- base-flow states from the full region of interest;
- exact guard and reset pairs;
- cylinder states;
- points from every daughter branch near the period doublings;
- nearby off-attractor states, not only the stable periodic orbit.

If \(E,D\) are allowed to move, penalize changes in:

- the anchor latent coordinates;
- pairwise latent distances between anchors;
- local encoder Jacobian scale and conditioning.

Anchor to the original common reference, not only to the immediately previous
checkpoint. Neighbor-to-neighbor anchoring permits cumulative drift.

### Crossing a period doubling

The governing dynamics remain smooth in \(\phi\), but the attracting orbit
changes stability and gains daughter branches. Near a flip:

- use return-map continuation rather than relying only on long burn-in;
- sample all daughter branches equally;
- include short horizons so that loss averaging does not merge new branches;
- retain long horizons that test the complete period;
- monitor the minimum latent separation between daughter branches.

Once the encoder collapses two daughter branches, no vector-field model can
recover their distinction.

### Continuation controls

Run these three versions if resources permit:

1. **Frozen-chart control:** train \(E,D\) at \(4.00^\circ\), freeze them for
   the entire continuation, and train only \(V(z,\phi)\).
2. **Anchored continuation:** allow slow \(E,D\) adaptation only when the
   frozen chart fails. This is the primary continuation model.
3. **Balanced joint conditional model:** train the same conditional
   architecture on all retained angles from scratch. This tests whether the
   continuation curriculum improves optimization or merely changes the local
   minimum.

## Validation before cycling-signature computation

Do not launch a full cycling-signature run merely because training loss is
small or the latent orbit looks smooth.

### Representation checks

For every angle, report on held-out trajectories:

- reconstruction error away from the ambiguous seam;
- exact endpoint gluing error;
- seam tangent mismatch;
- learned-flow versus encoder-Jacobian tangent angle;
- latent scale and encoder-Jacobian condition numbers;
- fixed-anchor coordinate and distance drift.

### Dynamical checks

- Decode held-out latent rollouts.
- Recover the expected period-1, period-2, period-4, and period-8 return
  structure.
- Confirm that period estimates agree with the physical return map.
- Check that the conditional field changes smoothly between neighboring
  angles.
- Verify that no earlier angle's held-out error deteriorates substantially
  during continuation. Use 10 percent as an initial review threshold, not an
  automatic theorem.

### Geometry and topology checks

For each periodic case, measure the lifted distances associated with

\[
T,
\quad T/2,
\quad T/4,
\quad T/8.
\]

Report minima and low quantiles as well as medians. A few local near-touches
can activate \(P(\operatorname{rank}>0)\) early even when the median separation
looks satisfactory.

Before the expensive probability computation, require:

- a nonzero and stable comparison-space \(\beta_1\) over a common cover-scale
  plateau;
- no angle-specific rescaling;
- a curve-resolution bound \(h\) below the recurrence scale being interpreted;
- preferably at least a 25 percent margin, \(h\leq0.75d_{\mathrm{target}}\);
- a radius interval wide enough to be numerically stable, rather than a
  one-threshold coincidence.

If these conditions fail, record the failure and stop. Do not tune a separate
radius or cover for each angle to manufacture a staircase.

## Final cycling analysis

Only after the validation gates pass:

- use the same cycling-signature statistic for every angle;
- use paired random starts and a saved common seed;
- use one common duration grid and one common radius rule;
- save exact birth values so plots can be re-thresholded without rerunning;
- require \(P(\operatorname{rank}>0)=0\) at \(r=0\) and reject exact or
  near-zero sampling-lock artifacts;
- show the curve-resolution boundary in diagnostics;
- label any structure below that boundary as exploratory;
- compare against the existing separately trained Compass baseline.

The success criterion is not simply a visually attractive heatmap. A successful
shared representation should preserve the dyadic return ordering on held-out
data at radii above the relevant resolution bound.

## Interpreting the outcomes

### Shared chart restores the blocks

The separate latent charts were an important source of the earlier
compression. Use the shared model for subsequent cross-angle analysis.

### Shared chart is accurate, but the blocks remain compressed

The likely limitation is the physical daughter-branch geometry or the binary
statistic \(P(\operatorname{rank}>0)\), not cross-angle coordinate drift.

### Shared chart fails reconstruction or gluing

The bottleneck is representation learning. Improve seam treatment, data
coverage, and local metric conditioning before computing more signatures.

### Five heads succeed but the conditional field fails

The common chart is viable, but the chosen conditional vector-field network or
continuation curriculum is underpowered.

### Continuation outperforms balanced joint training

The smooth parameter curriculum is helping optimization and should become the
default training procedure.

## Suggested implementation order

1. Implement and test the slope-relative coordinate transform.
2. Generate exact endpoint glue pairs and mask or canonicalize seam decoding.
3. Add the shared-\(E,D\), five-head network container.
4. Add balanced five-angle training and held-out validation.
5. Run the smoke test, Phase A, and Phase B.
6. Complete representation, dynamics, and topology diagnostics.
7. Implement the conditional field and continuation curriculum.
8. Run the frozen-chart, anchored-continuation, and joint-conditional controls.
9. Compute full cycling signatures only for models that pass the preregistered
   geometry and resolution gates.

## Definition of done

The project is ready for author review when the `code/` repository contains:

- reproducible training commands and immutable configurations;
- exact data, split, normalization, and environment manifests;
- joint and continuation checkpoints with resumable optimizer state;
- per-angle held-out validation reports;
- seam, tangent, chart-drift, return-map, dyadic-distance, \(\beta_1\), and
  curve-resolution diagnostics;
- a concise comparison of the old separate models, the five-head shared model,
  and the continuation model;
- code-side figures and exact numerical summaries;
- no changes in the `paper/` repository.
