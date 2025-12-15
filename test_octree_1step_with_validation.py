#!/usr/bin/env python3
"""
Single-step octree test with validation
- 10,000 particles
- 1 timestep
- Validation after initial search and after octree
- GPU/CPU/memory monitoring
"""
import time
import subprocess
import psutil
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Import production script components but don't execute time marching
with open('test_octree_production_1step.py', 'r') as f:
    production_code = f.read()

# Split into setup and time marching
parts = production_code.split('# TIME MARCHING')
setup_code = parts[0]

# Execute setup portion
exec(setup_code)

# Now we have all variables in scope: particle_data, mesh_gpu, velocity_field_gpu, etc.

print("="*80)
print("VALIDATION: INITIAL ASSIGNMENT")
print("="*80)

from jaxtrace.gpu.search.level0_cached import point_in_tet_jax

# Validate initial assignment
n_validate = min(1000, particle_data.n_active)
validate_indices = np.random.choice(particle_data.n_active, n_validate, replace=False)
n_correct_init = 0

for i in validate_indices:
    pos = particle_data.positions[i]
    elem_id = particle_data.element_ids[i]
    tet_nodes = node_positions[connectivity[elem_id]]
    is_inside = bool(np.array(point_in_tet_jax(jax.device_put(pos), jax.device_put(tet_nodes))))
    if is_inside:
        n_correct_init += 1

print(f"✓ Initial assignment validation: {n_correct_init}/{n_validate} correct ({100*n_correct_init/n_validate:.1f}%)")
print()

# Get GPU/CPU/memory before timestep
print("="*80)
print("SINGLE TIMESTEP WITH MONITORING")
print("="*80)

gpu_mem_before = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip())
cpu_percent_before = psutil.cpu_percent(interval=1)
ram_before = psutil.virtual_memory().used / 1024**3

print(f"GPU memory before: {gpu_mem_before} MB")
print(f"RAM before: {ram_before:.2f} GB")
print(f"CPU before: {cpu_percent_before:.1f}%")
print()

# Run single RK4 timestep (pass ParticleData object)
print("Running single timestep...")
t0_step = time.perf_counter()

particle_data_updated, stats = rk4_step_func(
    particle_data,
    velocity_field_gpu,
    DT,
    mesh_gpu,
    current_time=0.0
)

t_step = time.perf_counter() - t0_step

# Extract results
positions_final = particle_data_updated.positions
element_ids_final = particle_data_updated.element_ids

# Get GPU/CPU/memory after
gpu_mem_after = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True).stdout.strip())
cpu_percent_after = psutil.cpu_percent(interval=1)
ram_after = psutil.virtual_memory().used / 1024**3

print(f"✓ Timestep complete ({t_step:.4f} s)")
print(f"  Throughput: {particle_data.n_active/t_step:,.0f} p/s")
print()

# Validate after timestep
print("="*80)
print("VALIDATION: AFTER TIMESTEP (L0 + L1 3-hop + L2 Octree)")
print("="*80)

found_mask = element_ids_final >= 0
n_found_final = found_mask.sum()
n_lost = particle_data.n_active - n_found_final

print(f"Particle status:")
print(f"  Found: {n_found_final}/{particle_data.n_active} ({100*n_found_final/particle_data.n_active:.1f}%)")
print(f"  Lost: {n_lost}/{particle_data.n_active} ({100*n_lost/particle_data.n_active:.1f}%)")
print()

# Validate accuracy of found particles
if n_found_final > 0:
    n_validate_final = min(100, n_found_final)
    found_indices = np.where(found_mask)[0]
    validate_final_indices = np.random.choice(found_indices, n_validate_final, replace=False)

    n_correct_final = 0
    for idx in validate_final_indices:
        pos = positions_final[idx]
        elem_id = element_ids_final[idx]
        tet_nodes = node_positions[connectivity[elem_id]]
        is_inside = bool(np.array(point_in_tet_jax(jax.device_put(pos), jax.device_put(tet_nodes))))
        if is_inside:
            n_correct_final += 1

    accuracy_final = 100 * n_correct_final / n_validate_final
    print(f"✓ Final validation: {n_correct_final}/{n_validate_final} correct ({accuracy_final:.1f}%)")
else:
    accuracy_final = 0.0
    print("⚠ No particles found - cannot validate accuracy")

print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Performance:")
print(f"  Timestep time: {t_step:.4f} s")
print(f"  Throughput: {particle_data.n_active/t_step:,.0f} p/s")
print()
print("Accuracy:")
print(f"  Initial assignment: {100*n_correct_init/n_validate:.1f}% ({n_correct_init}/{n_validate} correct)")
print(f"  After timestep (L0+L1+L2): {accuracy_final:.1f}% ({n_correct_final if n_found_final > 0 else 0}/{n_validate_final if n_found_final > 0 else 0} correct)")
print()
print("Retention:")
print(f"  Particles retained: {n_found_final}/{particle_data.n_active} ({100*n_found_final/particle_data.n_active:.1f}%)")
print(f"  Particles lost: {n_lost}/{particle_data.n_active} ({100*n_lost/particle_data.n_active:.1f}%)")
print()
print("Resources:")
print(f"  GPU memory: {gpu_mem_before} MB → {gpu_mem_after} MB (Δ {gpu_mem_after-gpu_mem_before:+d} MB)")
print(f"  RAM: {ram_before:.2f} GB → {ram_after:.2f} GB (Δ {ram_after-ram_before:+.2f} GB)")
print(f"  CPU: {cpu_percent_before:.1f}% → {cpu_percent_after:.1f}%")
print()
print("="*80)
