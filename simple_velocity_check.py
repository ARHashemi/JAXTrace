#!/usr/bin/env python3
"""
Simple Velocity Check
=====================

Check what velocity values are actually in the loaded fields.
"""

import numpy as np
from pathlib import Path
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu

MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (120, 122)
VELOCITY_FIELD_NAME = 'Displacement'

print("Loading velocity sequence...")
node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
    MESH_BASE_PATH,
    MESH_FILE_PATTERN,
    VELOCITY_TIMESTEP_RANGE,
    field_name=VELOCITY_FIELD_NAME,
    verbose=False
)

print(f"\nVelocity sequence shape: {velocity_sequence.shape}")
print(f"  {velocity_sequence.shape[0]} timesteps")
print(f"  {velocity_sequence.shape[1]} nodes")
print(f"  {velocity_sequence.shape[2]} components (x, y, z)")

print(f"\nVelocity field statistics across all timesteps:")
vel_all = velocity_sequence.reshape(-1, 3)
vel_mag_all = np.linalg.norm(vel_all, axis=1)

print(f"  Mean |vel|: {vel_mag_all.mean():.6e}")
print(f"  Max |vel|: {vel_mag_all.max():.6e}")
print(f"  Min |vel|: {vel_mag_all.min():.6e}")
print(f"  Std |vel|: {vel_mag_all.std():.6e}")

# Sample some random nodes
n_samples = 10
np.random.seed(42)
sample_indices = np.random.choice(velocity_sequence.shape[1], n_samples, replace=False)

print(f"\n{n_samples} random node velocity magnitudes (timestep 0):")
for i, idx in enumerate(sample_indices):
    vel = velocity_sequence[0, idx]
    vel_mag = np.linalg.norm(vel)
    print(f"  Node {idx:7d}: |vel| = {vel_mag:.6e} m/s")

# Check if there's any suspicious scaling
DT = 0.0025
expected_disp_per_step = vel_mag_all.mean() * DT
print(f"\nExpected particle displacement per tracking step:")
print(f"  <|vel|> × dt = {vel_mag_all.mean():.6e} × {DT}")
print(f"  = {expected_disp_per_step:.6e} m = {expected_disp_per_step*1000:.6f} mm")

# From trajectory analysis, actual was ~0.017 mm/step
actual_disp_per_step = 0.016632e-3  # meters
ratio = actual_disp_per_step / expected_disp_per_step

print(f"\nActual displacement per step (from VTK analysis): {actual_disp_per_step:.6e} m")
print(f"Ratio (actual/expected): {ratio:.6f}")

if abs(ratio - 1.0) < 0.1:
    print(f"  ✅ Particles moving as expected")
else:
    print(f"  ⚠️  Particles moving {1/ratio:.1f}x SLOWER than expected!")
    print(f"  This suggests velocity is being scaled down by a factor of ~{1/ratio:.1f}")
