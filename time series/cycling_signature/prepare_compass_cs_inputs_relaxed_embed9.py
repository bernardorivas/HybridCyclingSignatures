"""
Experimental variant of prepare_compass_cs_inputs_relaxed.py for embed9.

Identical logic except:
  - sets config.embed_extra = 4 before constructing SuspensionNetworks
    (so the state_dict from model_embed9.pt loads into a 9-D latent)
  - default model path points at runs/compass_gait/model_embed9.pt
  - default --suffix is _relaxed_embed9

Writes new canonical artifacts:
    data/compass_gait/continuous_lift_relaxed_embed9{,_positions,_tangents}.{npy,csv}
    data/compass_gait/report_relaxed_embed9_encoder_diagnostics.txt

Baseline files are not touched.
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
sys.path.insert(0, os.path.join(TS_ROOT, "compass gait"))

from config import config, SystemType  # noqa: E402
config.system_type = SystemType.COMPASS_GAIT
config.device = "cpu"
config.embed_extra = 4        # <-- embed9 marker
from networks import SuspensionNetworks  # noqa: E402
from simulate import simulate_compass_gait, LIMIT_CYCLE_IC  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data", "compass_gait")
os.makedirs(DATA_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_s", type=int, default=50)
parser.add_argument("--n_impacts", type=int, default=5)
parser.add_argument(
    "--model",
    default=os.path.join(REPO_ROOT, "runs", "compass_gait", "model_embed9.pt"),
)
parser.add_argument("--suffix", default="_relaxed_embed9")
args = parser.parse_args()

net = SuspensionNetworks()
net.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
net.eval()
E = net.E
D_dec = net.D
print(f"loaded {args.model}  (embed_dim={config.embed_dim})")

segments, jump_pairs = simulate_compass_gait(LIMIT_CYCLE_IC, n_impacts=args.n_impacts)
print(f"simulated compass rollout: {len(segments)} arcs, {len(jump_pairs)} jumps")

n_s = args.n_s
s_vals = np.linspace(0.0, 1.0, n_s + 2)[1:-1]

pieces_enc = []
pieces_inp = []
tags = []

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
print(f"encoded lift: shape={Z.shape}")

base_name = f"continuous_lift{args.suffix}"
np.save(os.path.join(DATA_DIR, f"{base_name}.npy"), Z)

dz = np.diff(Z, axis=0)
dz = np.vstack([dz, dz[-1:]])
tangent_norms_pre = np.linalg.norm(dz, axis=1)
tx = dz / np.maximum(tangent_norms_pre, 1e-12)[:, None]
assert np.allclose(np.linalg.norm(tx, axis=1), 1.0, atol=1e-6)

np.save(os.path.join(DATA_DIR, f"{base_name}_tangents.npy"), tx)
np.savetxt(os.path.join(DATA_DIR, f"{base_name}_positions.csv"), Z, delimiter=" ")
np.savetxt(os.path.join(DATA_DIR, f"{base_name}_tangents.csv"), tx, delimiter=" ")
print(f"wrote {base_name}.npy / _tangents.npy / _positions.csv / _tangents.csv")

# --- Diagnostics (same as baseline) ---
report_lines = []
def _log(msg):
    print(msg); report_lines.append(msg)

_log("")
_log("=" * 60)
_log(f"PHASE A  |  RELAXED-SPACE EMBED9 ENCODER DIAGNOSTICS (compass)")
_log("=" * 60)
_log(f"model: {args.model}")
_log(f"embed_dim: {config.embed_dim}")
_log(f"trajectory: {len(segments)} arcs, {len(jump_pairs)} jumps, n_s={n_s}")
_log(f"lift shape: {Z.shape}")

with torch.no_grad():
    # Proper round-trip: D(E(x)). In the baseline script this was written as
    # D_dec(X_in) directly; that only happened to work when embed_dim == n+1.
    x_rec = D_dec(torch.as_tensor(Z, dtype=torch.float32)).numpy()
arc_mask = np.array([t[0] == "arc" for t in tags])
br_mask = ~arc_mask
full_err = np.linalg.norm(x_rec - X_in, axis=1)
state_err = np.linalg.norm(x_rec[:, :config.state_dim] - X_in[:, :config.state_dim], axis=1)

def _row(lbl, arr):
    _log(f"  {lbl:<24}  mean={arr.mean():.3e}  median={np.median(arr):.3e}  "
         f"p95={np.percentile(arr, 95):.3e}  max={arr.max():.3e}")

_log("")
_log("[1] Reconstruction D(E(.)) vs input  (injectivity witness on sampled data)")
_row("full err (arcs, s=0)", full_err[arc_mask])
_row("full err (bridges)", full_err[br_mask])
_row("state err (arcs)", state_err[arc_mask])
_row("state err (bridges)", state_err[br_mask])

# ---------------- 2. Latent near-collision screen ----------------
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

pre_jump_angles, post_jump_angles = [], []
for k in range(len(tags) - 1):
    if tags[k] == tags[k + 1]:
        continue
    dot = float(np.clip(np.dot(tx[k], tx[k + 1]), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dot)))
    if tags[k][0] == "arc" and tags[k + 1][0] == "bridge":
        pre_jump_angles.append(angle)
    elif tags[k][0] == "bridge" and tags[k + 1][0] == "arc":
        post_jump_angles.append(angle)

def _angle_summary(lbl, arr):
    if not arr:
        _log(f"  {lbl}: (none)"); return
    a = np.array(arr)
    _log(f"  {lbl}: median={np.median(a):.1f} deg  max={a.max():.1f} deg  (n={len(a)})")

_angle_summary("pre-jump  (arc -> bridge)", pre_jump_angles)
_angle_summary("post-jump (bridge -> arc)", post_jump_angles)

# ---------------- 4. Bridge-to-bridge separation under David's DynamicDistance ----------------
_log("")
_log("[4] Bridge-to-bridge separation under David's DynamicDistance")
_log("    metric: max(||p-q||, C*||v-w||) with C = boxsize * sb_radius")

bridge_rows, arc_rows = {}, {}
for k, t in enumerate(tags):
    (bridge_rows if t[0] == "bridge" else arc_rows).setdefault(t[1], []).append(k)
bridge_rows = {k: np.array(v) for k, v in bridge_rows.items()}
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
        Zs, Ts = Z[rows], tx[rows]
        dpos_steps = np.linalg.norm(np.diff(Zs, axis=0), axis=1)
        dtan_steps = np.linalg.norm(np.diff(Ts, axis=0), axis=1)
        arc_steps.extend(np.maximum(dpos_steps, C * dtan_steps).tolist())
    arc_steps = np.array(arc_steps)
    rho = inter_min / np.median(arc_steps)
    _log(
        f"  C={C:>4.2f}:  inter_bridge_min={inter_min:.3e}  "
        f"arc_step_median={np.median(arc_steps):.3e}  rho={rho:.2f}"
    )

report_path = os.path.join(DATA_DIR, f"report{args.suffix}_encoder_diagnostics.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
_log("")
_log(f"saved: {report_path}")
