"""
Prepare cycling-signature pipeline inputs for the compass-gait analytic lift.

Deterministic: regenerates the 5-impact limit-cycle rollout from
LIMIT_CYCLE_IC, builds the analytic lift with a chosen bridge shape, and
writes position and unit-tangent arrays in both Python-native (.npy) and
Julia-interop (.csv) formats.

Usage:
    python "time series/cycling_signature/prepare_compass_cs_inputs.py"
    python "time series/cycling_signature/prepare_compass_cs_inputs.py" --bridge-shape quartic
    python "time series/cycling_signature/prepare_compass_cs_inputs.py" --bridge-shape quartic --base-path tangent_matched

Canonical outputs under data/compass_gait/ use the suffix convention:
    affine + parabolic         -> continuous_lift_analytic*
    affine + quartic           -> continuous_lift_analytic_eta2*
    tangent_matched + quartic  -> continuous_lift_analytic_tgt*
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TS_ROOT = os.path.join(REPO_ROOT, "time series")
sys.path.insert(0, TS_ROOT)
sys.path.insert(0, os.path.join(TS_ROOT, "compass gait"))

from simulate import simulate_compass_gait, LIMIT_CYCLE_IC, state_velocity
from data_construction import build_continuous_lift_analytic

DATA_DIR = os.path.join(REPO_ROOT, "data", "compass_gait")
os.makedirs(DATA_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bridge-shape",
    choices=("parabolic", "quartic"),
    default="parabolic",
    help="Analytic eta profile for the bridge lift.",
)
parser.add_argument(
    "--base-path",
    choices=("affine", "tangent_matched"),
    default="affine",
    help="Base-coordinate bridge geometry.",
)
parser.add_argument(
    "--tangent-scale",
    type=float,
    default=1.0,
    help="Chord-scaled endpoint tangent magnitude for tangent_matched bridges.",
)
parser.add_argument(
    "--suffix",
    default=None,
    help=(
        "Optional explicit artifact suffix (defaults: '' for affine+parabolic, "
        "'_eta2' for affine+quartic, '_tgt' for tangent_matched+quartic)."
    ),
)
args = parser.parse_args()

suffix = args.suffix
if suffix is None:
    suffix_map = {
        ("affine", "parabolic"): "",
        ("affine", "quartic"): "_eta2",
        ("tangent_matched", "quartic"): "_tgt",
    }
    suffix = suffix_map.get(
        (args.base_path, args.bridge_shape),
        f"_{args.base_path}_{args.bridge_shape}",
    )

base_name = f"continuous_lift_analytic{suffix}"

segments, jump_pairs = simulate_compass_gait(LIMIT_CYCLE_IC, n_impacts=5)
endpoint_tangents = None
if args.base_path == "tangent_matched":
    endpoint_tangents = [
        (state_velocity(x_minus), state_velocity(x_plus))
        for x_minus, x_plus in jump_pairs
    ]

Z = build_continuous_lift_analytic(
    segments,
    jump_pairs,
    c=1.0,
    n_s=50,
    bridge_shape=args.bridge_shape,
    base_path=args.base_path,
    endpoint_tangents=endpoint_tangents,
    tangent_scale=args.tangent_scale,
)

lift_npy = os.path.join(DATA_DIR, f"{base_name}.npy")
np.save(lift_npy, Z)
print(
    f"[pos npy]  wrote             {lift_npy}  shape={Z.shape}  "
    f"bridge_shape={args.bridge_shape}  base_path={args.base_path}"
)

assert Z.shape[1] == 5, f"Expected 5 columns (4 compass state + eta), got {Z.shape[1]}"

dz = np.diff(Z, axis=0)
dz = np.vstack([dz, dz[-1:]])
norms = np.linalg.norm(dz, axis=1, keepdims=True)
min_norm = float(norms.min())
tx = dz / np.maximum(norms, 1e-12)
tx_norms = np.linalg.norm(tx, axis=1)
assert np.allclose(tx_norms, 1.0, atol=1e-6), (
    f"Tangents not unit-norm: min={tx_norms.min()}, max={tx_norms.max()}"
)

tan_npy = os.path.join(DATA_DIR, f"{base_name}_tangents.npy")
np.save(tan_npy, tx)
print(f"[tan npy]  wrote             {tan_npy}  shape={tx.shape}")

pos_csv = os.path.join(DATA_DIR, f"{base_name}_positions.csv")
tan_csv = os.path.join(DATA_DIR, f"{base_name}_tangents.csv")
np.savetxt(pos_csv, Z, delimiter=" ")
np.savetxt(tan_csv, tx, delimiter=" ")
print(f"[pos csv]  wrote             {pos_csv}")
print(f"[tan csv]  wrote             {tan_csv}")

print()
print(f"samples={Z.shape[0]}, dim={Z.shape[1]}")
print(f"min pre-normalize tangent norm: {min_norm:.3e}")
print(f"unit-tangent norm check: min={tx_norms.min():.8f}, max={tx_norms.max():.8f}")
