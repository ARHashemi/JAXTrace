
# Comprehensive, Memory-Safe Flow for GPU-Native Particle Tracking in JAX

---

## Overview

This document outlines an optimal workflow for high-performance, GPU-native, JAX-based particle tracking on block/octree-unstructured meshes. The plan prevents memory explosions (OOM) by ensuring partitioned, blockwise processing at every stage, with careful host-device (RAM↔GPU) data movement. It synthesizes all critical lessons and architecture from our in-depth discussion.

---

## Flowchart (High-Level Pseudocode)

```
graph TD;
    A[Load & Analyze Mesh (CPU)]
    B[Block/Octree Partition & Connectivity (CPU)]
    C[Precompute/PAD block/element/etc. arrays (CPU)]
    D[Copy static arrays to GPU]
    E[Particle Seeding (CPU or GPU)]
    F[Assign Initial Element/block for particles (GPU, blockwise)]
    G[INITIALIZATION COMPLETE]
    H[Time Marching Loop (lax.scan/JIT, GPU)]
    H1[ Interpolate velocities for particles' element ]
    H2[ Runge-Kutta Step (positions updated) ]
    H3[ Assign particles to block (GPU-hash/octree) ]
    H4[ Partition particles by block (argsort/split) ]
    H5[ For each block (parallel/batch GPU): blockwise element search ]
    H6[ For misses: neighbor block search (batch GPU) ]
    H7[ [OPTIONAL]: Fallback search/reporting ]
    Z[Write Results/Diagnostics (RAM) or Continue Next Time Step]

    A-->B-->C-->D
    D-->E-->F-->G
    G-->H
    H-->H1-->H2-->H3-->H4-->H5-->H6-->H7-->Z
```

---

## **Step-by-Step Detailed Process**

### 1. **Mesh Loading & Analysis (CPU)**
- Load mesh geometry (node positions, element connectivity).
- Analyze element sizes/volumes/distribution for optimal block/octree partitioning.

### 2. **Build Blocks/Octrees & Connectivity (CPU)**
- Compute octree nodes or uniform spatial blocks.
- For each block, list elements intersecting that block (static, padded).
- For each element, store neighbor IDs (static, padded).

### 3. **Static Array Preprocessing (CPU)**
- **PAD all block element lists:** to max size per block (store `block_elem_counts` as needed).
- Store block bounds, neighbor block IDs, other mappings as static arrays.
- **Host memory:** All core static arrays now reside in RAM.

### 4. **Move Static Arrays to GPU**
- Transfer:
    - Node positions (`(n_nodes, 3)`)
    - Element connectivity (`(n_elements, 4)`)
    - All block element arrays (`(n_blocks, max_elem_per_block)`)
    - Neighbor/neighbor-block arrays, etc.
- These arrays are **read-only** on GPU for entire simulation.

### 5. **Particle Seeding (CPU or GPU)**
- Generate initial positions for all particles (e.g., uniform random in physical space).
- Optionally, generate on GPU if initial spatial distribution is large.
- **Retain on RAM until assignment.**

### 6. **Assign Initial Elements/Blocks to Particles (GPU, Blockwise)**
- Copy seed positions to GPU.
- Use a **GPU-patched search kernel (blockwise)**:
    - For each particle, assign block via hash/octree, only search block’s element array for host element.
- Store particle positions, block IDs, and element IDs on GPU as **scan carry** for time-marching.

---

## **Time-Marching: Main JAX-GPU Loop (on GPU, JIT/scan)**

### (Within each time step:)
1. **Interpolate Velocities (GPU):**
    - For every particle, gather element’s node indices and interpolate using particle’s (x, y, z).
    - Pure `vmap` over all particles, efficient memory access.

2. **RK4 Integration to Update Particle Positions (GPU):**
    - Runge-Kutta or other integrators, fully vectorized/JIT.
    - Output: new positions (on GPU).

3. **Assign New Block IDs (GPU):**
    - For all particles, map new position to block/octree node (hash, Morton, grid index).
    - Pure vectorized mapping on GPU.

4. **Partition Particles by Block (GPU):**
    - Sort or group indices by block ID (use `jnp.argsort` or segmented index).
    - Prepares block-based processing without large per-batch expansion.

5. **Blockwise Element Search for Each Block (On GPU, vmap/lax.map per block):**
    - For each nonempty block:
        - Search only/block’s element list for all local block particles (never global mesh).
        - All padded arrays; use mask and batch search (no OOM).
        - Write results to particle element IDs.

6. **Neighbor Block Fallback (GPU, Batched):**
    - For any “lost” particles (not found in current block), batch-search all 26 block neighbors' element arrays.
    - Use mask/vmap strategy, minimal memory.

7. **[Optional] Global/diagnostic fallback (rare, mostly for pathological misses):**
    - Only for small fraction/edge cases.
    - Mark as not found, gather for later debug if needed.

8. **[Loop or Exit]**
    - Write results, step, or continue depending on convergence/target step.

---

## **Variable Placement and Data Movement Table**

| Variable                           | Stage                           | RAM→GPU | GPU→RAM | Kept Where         | Why                              |
|-------------------------------------|----------------------------------|---------|---------|--------------------|-----------------------------------|
| Node positions, connectivity        | After mesh/load/prep             | Yes     | Never   | GPU (const)        | Read-only for interp/search       |
| Block/octree, neighbor arrays       | After block build                | Yes     | Never   | GPU (const)        | Static search structures          |
| Particle initial positions          | After seeding                    | Yes     | Rarely* | GPU (carry)        | Carried for interp/integration    |
| Particle block/element IDs          | After initial assignment; marching | Yes  | Rarely* | GPU (carry)        | Carried for search                |
| Block-element arrays                | After partition                  | Yes     | Never   | GPU (const)        | Required for search               |
| Interpolated velocities             | On-the-fly, GPU                  | -       | -       | ephemeral (GPU)    | Only used for next step           |
| Simulation results/output           | End/checkpoint/periodic          | No      | Yes     | RAM                | Only transfer for I/O/diagnostics |

(*) Copy back to RAM occasionally, e.g., for data dump/checkpointing.

---

## **Parallelization and JAX/GPU Patterns**

- **Mesh and search arrays:** Always static, preallocated, device-resident constants.
- **Particle data (`positions`, `block_ids`, `element_ids`):**  
  - Carried through `lax.scan` (or `lax.fori_loop` for explicit time steps), minimal, per-particle only.
- **Vectorization:** All per-particle routines are `vmap` or batch-chunked; per-block search is vmap over block’s local particles.
- **Blockwise batching:** Batches are never globally broadcasted over all mesh elements—partition one block at a time, fixed-sized arrays, safe for memory.

---

## **Summary of OOM Avoidance**

- **Never allocate any (N_particles, N_elements) or (N_particles, max_block_elements_global) arrays.**
- **Always restrict to (max_particles_in_block, max_block_elements_in_block) per kernel call.**
- **All search and interpolation use static padding/masking, and block-level or local arrays only.**
- **All expensive work is GPU-batched, never in serial or uncontrolled vmap-of-vmap.**

---

## **Conclusion**

This workflow enables:
- Immediate, scalable, robust memory safety for massive meshes and particle counts in JAX.
- Transparent GPU/host data management—no hidden transfers or OOM surprises.
- Best-practice scientific code for particle tracking, integrating spatial search, time integration, and parallel field interpolation.
- Readiness for further optimization, profiling, and hybrid distributed scaling if needed in the future.

---
