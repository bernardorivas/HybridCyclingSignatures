# Trained CHyLL-v2 models — compass-gait period-doubling cascade

Weights only. `model.pt` plus the small JSONs beside it are everything needed to
load a model. `train_log.jsonl` is not required and is not tracked.

Slopes, in the slope-fixed frame `u = theta + phi` (guard and reset are
phi-independent there, which is what makes a shared chart well posed):

| key | phi | attractor |
|-----|-----|-----------|
| `phi_0` | 4.01° | period 1 |
| `phi_1` | 4.75° | period 2 |
| `phi_2` | 5.00° | period 4 |
| `phi_3` | 5.02° | period 8 |
| `phi_4` | 5.20° | chaos |

## Five independent models — one E, D, V per slope

| runs | w_v |
|------|-----|
| `compass_gait_phi007`, `compass_gait_phi_1_4.75deg`, `compass_gait_phi_2_5deg`, `compass_gait_phi_3_5.02deg`, `compass_gait_phi_4_cloud_5.2deg` | 0 |
| `wv1_baseline_{period1,period2,period4,period8,chaos}` | 1 |

## Shared encoder/decoder — the chart is learned once for all slopes

| run | how phi enters | slopes | w_v |
|-----|----------------|--------|-----|
| `compass_gait_joint_cascade` | conditioned field `V(z, phi_n)` | phi_1..4 | 0 |
| `compass_gait_joint_cascade_wv1` | conditioned field `V(z, phi_n)` | phi_1..4 | 1 |
| `compass_gait_joint_cascade_multihead` | one field per slope | phi_1..4 | 0 |
| `compass_gait_joint_cascade_multihead5_wv1` | one field per slope | phi_0..4 | 1 |

`w_v` weights the seam-velocity term (latent velocities matched across the
cylinder entry and gluing exit seams). It is the only difference between the
w_v = 0 and w_v = 1 configs — same curriculum, data, steps, latent_dim.

Conditioned fields read phi as a continuous input, so they are defined at slopes
never trained on. Multihead heads never see phi at all; the head index is a
lookup, so nothing is defined between training slopes.

## Loading

- independent: `chyll_v2/chyll_v2/networks.py` → `CHyLLv2Networks`
- joint: `chyll_v2/chyll_v2/joint.py` → `JointNetworks`, `MultiHeadNetworks`
- reference loader: `load_run()` in `chyll_v2/scripts/export_joint_lifts.py`

Files in a joint run:

| file | what |
|------|------|
| `model.pt` | state dict — `torch.load(..., map_location="cpu", weights_only=True)` |
| `config.json` | `CHyLLv2Config` fields |
| `joint_meta.json` | arch, `train_slopes`, phi per slope, `latent_dim` |
| `phi_scaler.json` | `(center, half)` normalisation phi is fed through |
| `head_map.json` | slope key → head index (multihead only) |

## What they read

Cycling signatures on the learned lifts, one cover (boxsize 0.70, sb_radius 1),
n = 200 subsegment starts, at the best radius common to all three slopes.
Values are `L*` in base periods — the shortest segment at which every start
phase gives a nontrivial cycling class.

| model | 4.75° | 5.00° | 5.02° |
|-------|-------|-------|-------|
| five independent, w_v = 0 | 2.06 | 3.96 | **8.07** |
| five independent, w_v = 1 | 2.06 | 3.96 | 4.11 |
| shared chart (all four variants) | 2.06 | 4.11 | 7.03 |
| truth | 2 | 4 | 8 |

All four shared-chart models agree exactly. `w_v = 1` costs the independent
models the period-8 reading and leaves the shared chart untouched.
