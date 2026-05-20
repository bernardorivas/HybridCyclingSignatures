"""
Prepare cycling-signature pipeline inputs for the rimless-wheel system,
parallel of prepare_compass_cs_inputs_relaxed.py.

Encodes a canonical limit-cycle rollout through the trained encoder from
runs/rimless_wheel/model.pt, writes canonical artifacts with the `_relaxed`
suffix, and reports encoder-quality diagnostics (reconstruction, latent
collapse screen, tangent stability, junction angles, bridge separation
under DynamicDistance).

Usage:
    python "time series/cycling_signature/prepare_rimless_cs_inputs_relaxed.py"

Canonical outputs under data/rimless_wheel/:
    continuous_lift_relaxed.npy                 (T, d_latent) encoded positions
    continuous_lift_relaxed_tangents.npy        (T, d_latent) unit tangents
    continuous_lift_relaxed_positions.csv       Julia interop
    continuous_lift_relaxed_tangents.csv        Julia interop
    report_relaxed_encoder_diagnostics.txt      human-readable Phase-A report
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TS_ROOT = os.path.join(REPO_ROOT, "time series")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, TS_ROOT)
sys.path.insert(0, os.path.join(TS_ROOT, "rimless wheel"))

from config import config, SystemType  # noqa: E402
config.system_type = SystemType.RIMLESS_WHEEL
config.device = "cpu"
from networks import SuspensionNetworks  # noqa: E402
from simulate import simulate_rimless_wheel  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data", "rimless_wheel")
os.makedirs(DATA_DIR, exist_ok=True)

# Canonical limit-cycle post-reset state for alpha=0.4, gamma=0.2.
LIMIT_CYCLE_IC = np.array([-0.2, 0.54])

parser = argparse.ArgumentParser()
parser.add_argument("--n_s", type=int, default=50)
parser.add_argument("--n_impacts", type=int, default=5)
parser.add_argument(
    "--model",
    default=os.path.join(REPO_ROOT, "runs", "rimless_wheel", "model.pt"),
)
parser.add_argument("--suffix", default="_relaxed")
parser.add_argument(
    "--ic", nargs=2, type=float, default=None,
    help="Override IC (theta0 omega0). Defaults to the limit-cycle IC.",
)
parser.add_argument(
    "--embed-extra", type=int, default=config.embed_extra,
    help="Set config.embed_extra before instantiating networks; must match "
         "what the model was trained with.",
)
args = parser.parse_args()
config.embed_extra = args.embed_extra

# ---------------- Model ----------------
net = SuspensionNetworks()
net.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
net.eval()
E = net.E
D_dec = net.D
print(f"loaded {args.model}  (embed_dim={config.embed_dim})")

# ---------------- Canonical trajectory ----------------
ic = np.array(args.ic) if args.ic is not None else LIMIT_CYCLE_IC
segments, jump_pairs = simulate_rimless_wheel(ic[0], ic[1], n_impacts=args.n_impacts)
print(f"simulated rimless rollout from IC={ic}: "
      f"{len(segments)} arcs, {len(jump_pairs)} jumps")

# ---------------- Build encoded continuous lift + tags ----------------
n_s = args.n_s
s_vals = np.linspace(0.0, 1.0, n_s + 2)[1:-1]  # interior only

pieces_enc = []
pieces_inp = []
tags = []  # ("arc", j) or ("bridge", j) per row, aligned with lift rows

with torch.no_grad():
    for j, seg in enumerate(segments):
        inp_arc = np.hstack([seg, np.zeros((len(seg), 1))])
        z_arc = E(torch.tensor(inp_arc, dtype=torch.float32)).numpy()
        pieces_enc.append(z_arc)
        pieces_inp.append(inp_arc)
        tags.extend([("arc", j)] * len(z_arc))

        if j < len(jump_pairs):
            xm, _ = jump_pairs[j]
            inp_br = np.hstack([np.tile(xm, (n_s, 1)), s_vals[:, None]])
            z_br = E(torch.tensor(inp_br, dtype=torch.float32)).numpy()
            pieces_enc.append(z_br)
            pieces_inp.append(inp_br)
            tags.extend([("bridge", j)] * n_s)

Z = np.concatenate(pieces_enc, axis=0)
X_in = np.concatenate(pieces_inp, axis=0)
assert Z.shape[0] == X_in.shape[0] == len(tags)
print(f"encoded lift: shape={Z.shape}")

base_name = f"continuous_lift{args.suffix}"

# ---------------- Position and tangent files ----------------
np.save(os.path.join(DATA_DIR, f"{base_name}.npy"), Z)
print(f"[pos npy]  {base_name}.npy  shape={Z.shape}")

dz = np.diff(Z, axis=0)
dz = np.vstack([dz, dz[-1:]])
tangent_norms_pre = np.linalg.norm(dz, axis=1)
tx = dz / np.maximum(tangent_norms_pre, 1e-12)[:, None]
tx_norms = np.linalg.norm(tx, axis=1)
assert np.allclose(tx_norms, 1.0, atol=1e-6), \
    f"tangent unit-norm check failed: {tx_norms.min()}..{tx_norms.max()}"

np.save(os.path.join(DATA_DIR, f"{base_name}_tangents.npy"), tx)
print(f"[tan npy]  {base_name}_tangents.npy  shape={tx.shape}")

np.savetxt(os.path.join(DATA_DIR, f"{base_name}_positions.csv"), Z, delimiter=" ")
np.savetxt(os.path.join(DATA_DIR, f"{base_name}_tangents.csv"), tx, delimiter=" ")
print(f"[pos csv]  {base_name}_positions.csv")
print(f"[tan csv]  {base_name}_tangents.csv")

# ================================================================
# PHASE A ENCODER-QUALITY DIAGNOSTICS
# ================================================================
report_lines = []

def _log(msg):
    print(msg)
    report_lines.append(msg)

_log("")
_log("=" * 60)
_log("PHASE A  |  RELAXED-SPACE ENCODER DIAGNOSTICS (rimless)")
_log("=" * 60)
_log(f"model: {args.model}")
_log(f"trajectory: {len(segments)} arcs, {len(jump_pairs)} jumps, n_s={n_s}")
_log(f"lift shape: {Z.shape}  (samples, latent_dim)")

# ---------------- 1. Reconstruction via decoder ----------------
_log("")
_log("[1] Reconstruction D(E(.)) vs input  (injectivity witness on sampled data)")
with torch.no_grad():
    x_rec = D_dec(torch.as_tensor(Z, dtype=torch.float32)).numpy()

arc_mask = np.array([t[0] == "arc" for t in tags])
br_mask = ~arc_mask

full_err = np.linalg.norm(x_rec - X_in, axis=1)
state_err = np.linalg.norm(x_rec[:, :config.state_dim] - X_in[:, :config.state_dim], axis=1)

def _row(lbl, arr):
    _log(f"  {lbl:<24}  mean={arr.mean():.3e}  median={np.median(arr):.3e}  "
         f"p95={np.percentile(arr, 95):.3e}  max={arr.max():.3e}")

_row("full err (arcs, s=0)", full_err[arc_mask])
_row("full err (bridges)", full_err[br_mask])
_row("state err (arcs)", state_err[arc_mask])
_row("state err (bridges)", state_err[br_mask])
_log("  note: the decoder reconstructs s freely (no forcing); bridge")
_log("        samples near s=0 or s=1 are masked during Phase II training")
_log("        (manuscript Section 4) so some residual near boundaries is expected.")

# ---------------- 2. Near-collision screen in latent ----------------
_log("")
_log("[2] Latent near-collisions from time-distant samples (collapse screen)")

from scipy.spatial.distance import pdist, squareform  # noqa: E402
d_lat = squareform(pdist(Z))
d_inp = squareform(pdist(X_in))
N = len(Z)
time_gap = 20
mask = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) > time_gap
d_lat_masked = np.where(mask, d_lat, np.inf)
nn_lat = d_lat_masked.min(axis=1)
nn_idx = d_lat_masked.argmin(axis=1)
nn_inp = d_inp[np.arange(N), nn_idx]
ratio = nn_lat / np.maximum(nn_inp, 1e-12)
_log(f"  min time-distant latent distance:  {nn_lat.min():.3e}")
_log(f"  median time-distant latent/input ratio: {np.median(ratio):.3f}")
_log(f"  p5 ratio:                          {np.percentile(ratio, 5):.3f}  (low = potential collapse)")
suspicious = (ratio < 0.25) & (nn_inp > 0.1)
_log(f"  pairs with ratio < 0.25 AND input_dist > 0.1: {int(suspicious.sum())}/{N}")
if suspicious.any():
    idx = np.where(suspicious)[0][:5]
    for i in idx:
        j = nn_idx[i]
        _log(
            f"    sample {i} {tags[i]} <-> {j} {tags[j]}: "
            f"d_lat={nn_lat[i]:.2e}  d_inp={nn_inp[i]:.2e}  ratio={ratio[i]:.2f}"
        )

# ---------------- 3. Tangent quality ----------------
_log("")
_log("[3] Finite-difference tangent quality")
_log(f"  pre-normalize norm: median={np.median(tangent_norms_pre):.3e}  "
     f"p5={np.percentile(tangent_norms_pre, 5):.3e}  "
     f"min={tangent_norms_pre.min():.3e}")
_log(f"  samples with pre-norm < 1e-6: {int((tangent_norms_pre < 1e-6).sum())} "
     f"(unit-normalization yields direction from noise at these points)")

# Visual junction kinks: angle between the last segment-internal forward step
# on one side and the first segment-internal forward step on the other. This
# is what shows up as a corner in the lift plot. (A naive forward-difference
# at the seam itself reads tiny because bridge samples right after an impact
# already point in the s-direction, masking the real change-of-direction.)
pre_jump_angles = []   # arc-internal -> bridge-internal at impact (s=0 seam)
post_jump_angles = []  # bridge-internal -> next-arc-internal at gluing (s=1 seam)

def _step(a, b):
    v = Z[b] - Z[a]
    return v / max(np.linalg.norm(v), 1e-12)

for k in range(1, len(tags) - 2):
    if tags[k][0] == tags[k + 1][0]:
        continue
    if tags[k - 1] != tags[k] or tags[k + 1] != tags[k + 2]:
        continue
    t_left  = _step(k - 1, k)
    t_right = _step(k + 1, k + 2)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(t_left, t_right), -1.0, 1.0))))
    if tags[k][0] == "arc" and tags[k + 1][0] == "bridge":
        pre_jump_angles.append(angle)
    elif tags[k][0] == "bridge" and tags[k + 1][0] == "arc":
        post_jump_angles.append(angle)

def _angle_summary(lbl, arr):
    if not arr:
        _log(f"  {lbl}: (none)")
        return
    a = np.array(arr)
    _log(f"  {lbl}: median={np.median(a):.1f} deg  max={a.max():.1f} deg  (n={len(a)})")

_angle_summary("pre-jump  (arc -> bridge), visual kink", pre_jump_angles)
_angle_summary("post-jump (bridge -> arc), visual kink", post_jump_angles)

# ---------------- 4. Pairwise bridge separation under DynamicDistance ----------------
_log("")
_log("[4] Bridge-to-bridge separation under DynamicDistance")
_log("    metric: max(||p-q||, C*||v-w||) with C = boxsize * sb_radius")

bridge_rows = {}
for k, t in enumerate(tags):
    if t[0] == "bridge":
        bridge_rows.setdefault(t[1], []).append(k)
bridge_rows = {k: np.array(v) for k, v in bridge_rows.items()}

arc_rows = {}
for k, t in enumerate(tags):
    if t[0] == "arc":
        arc_rows.setdefault(t[1], []).append(k)
arc_rows = {k: np.array(v) for k, v in arc_rows.items()}

def dyn_dist(A_pos, A_tan, B_pos, B_tan, C):
    dpos = np.linalg.norm(A_pos[:, None, :] - B_pos[None, :, :], axis=-1)
    dtan = np.linalg.norm(A_tan[:, None, :] - B_tan[None, :, :], axis=-1)
    return np.maximum(dpos, C * dtan)

for C in (0.05, 0.2, 0.5, 1.0):
    inter_min = np.inf
    for i in bridge_rows:
        for j in bridge_rows:
            if i >= j:
                continue
            Zi, Ti = Z[bridge_rows[i]], tx[bridge_rows[i]]
            Zj, Tj = Z[bridge_rows[j]], tx[bridge_rows[j]]
            d = dyn_dist(Zi, Ti, Zj, Tj, C).min()
            if d < inter_min:
                inter_min = d
    arc_steps = []
    for j, rows in arc_rows.items():
        if len(rows) < 2:
            continue
        Zs = Z[rows]
        Ts = tx[rows]
        dpos_steps = np.linalg.norm(np.diff(Zs, axis=0), axis=1)
        dtan_steps = np.linalg.norm(np.diff(Ts, axis=0), axis=1)
        arc_steps.extend(np.maximum(dpos_steps, C * dtan_steps).tolist())
    arc_steps = np.array(arc_steps)
    rho = inter_min / np.median(arc_steps) if len(arc_steps) else np.nan
    _log(
        f"  C={C:>4.2f}:  inter_bridge_min={inter_min:.3e}  "
        f"arc_step_median={np.median(arc_steps):.3e}  rho={rho:.2f}"
    )
_log("  note: rimless bridges are observations of the same canonical jump")
_log("        (the wheel-spoke impact); low inter-bridge distance is expected.")

# ---------------- Save report ----------------
report_path = os.path.join(DATA_DIR, f"report{args.suffix}_encoder_diagnostics.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
_log("")
_log(f"saved diagnostics report: {report_path}")
