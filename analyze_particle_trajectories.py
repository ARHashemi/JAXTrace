#!/usr/bin/env python3
"""
Analyze Particle Trajectories from VTK Output
==============================================

Load exported VTK files and analyze actual particle displacement.
"""

import numpy as np
from pathlib import Path
import vtk
from vtk.util import numpy_support

# Configuration
OUTPUT_DIR = Path("./output/global_morton_timedep")
DT = 0.0025  # Tracking timestep
EXPORT_FREQUENCY = 10

print("="*80)
print("PARTICLE TRAJECTORY ANALYSIS")
print("="*80)

# Find VTK files
vtk_files = sorted(OUTPUT_DIR.glob("particles_step_*.vtu"))
if not vtk_files:
    print(f"\n❌ No VTK files found in {OUTPUT_DIR}")
    exit(1)

print(f"\nFound {len(vtk_files)} VTK files")
print(f"  First: {vtk_files[0].name}")
print(f"  Last: {vtk_files[-1].name}")

# Load first and last files
def load_vtk_positions(filepath):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    output = reader.GetOutput()
    positions = numpy_support.vtk_to_numpy(output.GetPoints().GetData())
    return positions.astype(np.float64)

print("\nLoading particle positions...")
pos_first = load_vtk_positions(vtk_files[0])
pos_last = load_vtk_positions(vtk_files[-1])

print(f"  First file: {len(pos_first):,} particles")
print(f"  Last file: {len(pos_last):,} particles")

# Extract step numbers
import re
step_first = int(re.search(r'step_(\d+)', vtk_files[0].name).group(1))
step_last = int(re.search(r'step_(\d+)', vtk_files[-1].name).group(1))

n_steps_simulated = step_last - step_first
actual_steps = n_steps_simulated  # Since export happens at these specific steps
time_elapsed = n_steps_simulated * DT

print(f"\nSimulation:")
print(f"  Step range: {step_first} to {step_last}")
print(f"  Steps simulated: {n_steps_simulated}")
print(f"  Time elapsed: {time_elapsed:.4f} s")
print(f"  dt: {DT} s")

# Assume same particles (same ordering)
if len(pos_first) != len(pos_last):
    print(f"\n⚠️  WARNING: Different number of particles!")
    print(f"     Using minimum count for analysis")
    n_particles = min(len(pos_first), len(pos_last))
    pos_first = pos_first[:n_particles]
    pos_last = pos_last[:n_particles]
else:
    n_particles = len(pos_first)

# Compute displacement
displacement = pos_last - pos_first
disp_mag = np.linalg.norm(displacement, axis=1)

print(f"\nParticle displacement statistics:")
print(f"  Mean |displacement|: {disp_mag.mean():.6e} m = {disp_mag.mean()*1000:.3f} mm")
print(f"  Max |displacement|: {disp_mag.max():.6e} m = {disp_mag.max()*1000:.3f} mm")
print(f"  Min |displacement|: {disp_mag.min():.6e} m = {disp_mag.min()*1000:.3f} mm")
print(f"  Std |displacement|: {disp_mag.std():.6e} m = {disp_mag.std()*1000:.3f} mm")

# Compute mean velocity from displacement
mean_velocity_from_disp = disp_mag.mean() / time_elapsed
print(f"\nMean velocity (from displacement):")
print(f"  v = displacement / time = {disp_mag.mean():.6e} / {time_elapsed:.4f}")
print(f"  v = {mean_velocity_from_disp:.6e} m/s")

# Expected displacement if velocity field mean is ~0.195 m/s
expected_vel_field_mean = 0.195  # From earlier tests
expected_disp = expected_vel_field_mean * time_elapsed

print(f"\nExpected displacement (if vel field mean = {expected_vel_field_mean} m/s):")
print(f"  displacement = vel × time = {expected_vel_field_mean} × {time_elapsed:.4f}")
print(f"  displacement = {expected_disp:.6e} m = {expected_disp*1000:.3f} mm")

# Compare
ratio = disp_mag.mean() / expected_disp
print(f"\nRatio:")
print(f"  Actual / Expected = {ratio:.3f}x")

if abs(ratio - 1.0) < 0.1:
    print(f"  ✅ Displacement matches expected (within 10%)")
elif ratio > 1.5:
    print(f"  ⚠️  Particles moved {ratio:.1f}x MORE than expected!")
elif ratio < 0.5:
    print(f"  ⚠️  Particles moved {ratio:.1f}x LESS than expected!")
else:
    print(f"  ⚠️  Displacement differs from expected by {abs(ratio-1.0)*100:.1f}%")

# Displacement per step
disp_per_step = disp_mag.mean() / n_steps_simulated
print(f"\nDisplacement per timestep:")
print(f"  Mean displacement / n_steps = {disp_mag.mean():.6e} / {n_steps_simulated}")
print(f"  = {disp_per_step:.6e} m = {disp_per_step*1000:.6f} mm/step")

expected_disp_per_step = expected_vel_field_mean * DT
print(f"\nExpected displacement per timestep:")
print(f"  vel × dt = {expected_vel_field_mean} × {DT}")
print(f"  = {expected_disp_per_step:.6e} m = {expected_disp_per_step*1000:.6f} mm/step")

ratio_per_step = disp_per_step / expected_disp_per_step
print(f"\nRatio per step:")
print(f"  Actual / Expected = {ratio_per_step:.3f}x")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if abs(ratio_per_step - 3.0) < 0.2:
    print(f"\n⚠️  CONFIRMED: Particles are moving ~3x faster than expected!")
    print(f"   This suggests there may be an issue with:")
    print(f"   1. Velocity field units or interpretation")
    print(f"   2. RK4 timestep application")
    print(f"   3. Time-dependent velocity indexing")
elif abs(ratio_per_step - 1.0) < 0.1:
    print(f"\n✅ Particles are moving as expected!")
else:
    print(f"\n⚠️  Particles are moving {ratio_per_step:.2f}x the expected amount")

print("="*80)
