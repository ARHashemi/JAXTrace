# CPU vs GPU Algorithm Comparison

## Overview

This document compares the CPU and GPU implementations of particle tracking algorithms to identify differences and performance bottlenecks.

---

## 1. Point-in-Tetrahedron Test

### CPU Implementation (`jaxtrace/interpolation/element.py`)

```python
def point_in_tetrahedron(point, vertices):
    """
    Test if point is inside tetrahedron using barycentric coordinates.

    Algorithm:
    1. Set up 3×3 linear system: A @ lambda = (p - v0)
       where A = [v1-v0, v2-v0, v3-v0]
    2. Solve for barycentric coordinates λ₁, λ₂, λ₃
    3. Compute λ₀ = 1 - λ₁ - λ₂ - λ₃
    4. Check: all λᵢ ≥ -ε (with tolerance for boundary)

    Uses: np.linalg.solve (LAPACK DGESV on CPU)
    Precision: 64-bit float (double)
    """
    v0 = vertices[0]
    A = np.column_stack([
        vertices[1] - v0,
        vertices[2] - v0,
        vertices[3] - v0
    ])
    b = point - v0

    try:
        lambdas = np.linalg.solve(A, b)  # CPU: LAPACK DGESV
    except np.linalg.LinAlgError:
        return False

    lambda0 = 1.0 - np.sum(lambdas)
    epsilon = 1e-10

    return (lambda0 >= -epsilon and
            np.all(lambdas >= -epsilon) and
            np.sum(lambdas) <= 1.0 + epsilon)
```

**CPU Characteristics:**
- Uses LAPACK's DGESV (LU decomposition with partial pivoting)
- 64-bit double precision
- Exception handling for singular matrices
- Processes one point at a time
- Memory: O(1) per call

---

### GPU Implementation (`jaxtrace/gpu/kernels.py`)

```python
@jax.jit
def point_in_tetrahedron_jax(point: jnp.ndarray, vertices: jnp.ndarray) -> bool:
    """
    GPU version using JAX primitives.

    Algorithm:
    1. Same linear system setup as CPU
    2. Solve using jnp.linalg.solve (calls cuSolver DGESV)
    3. Same barycentric coordinate test

    Uses: jnp.linalg.solve (cuSolver on GPU)
    Precision: 32-bit float (default for GPU)

    NOTE: GPU uses 32-bit floats by default, CPU uses 64-bit.
          This causes minor differences in boundary cases.
    """
    v0 = vertices[0]
    A = jnp.column_stack([
        vertices[1] - v0,
        vertices[2] - v0,
        vertices[3] - v0
    ])
    b = point - v0

    try:
        lambdas = jnp.linalg.solve(A, b)  # GPU: cuSolver DGESV
    except Exception:
        return False

    lambda0 = 1.0 - jnp.sum(lambdas)
    epsilon = 1e-10

    return (lambda0 >= -epsilon) & jnp.all(lambdas >= -epsilon) & (jnp.sum(lambdas) <= 1.0 + epsilon)


@jax.jit
def point_in_tetrahedron_safe(point: jnp.ndarray, vertices: jnp.ndarray) -> bool:
    """
    Safe version that handles degenerate cases without exceptions.

    Differences from CPU:
    - Cannot use try/except in JAX (not compatible with JIT)
    - Checks matrix condition number first
    - Returns False for ill-conditioned matrices (cond > 1e6)

    This is GPU-specific: JAX requires pure functional code.
    """
    v0 = vertices[0]
    A = jnp.column_stack([
        vertices[1] - v0,
        vertices[2] - v0,
        vertices[3] - v0
    ])
    b = point - v0

    # GPU-SPECIFIC: Check condition number instead of try/except
    cond = jnp.linalg.cond(A)
    is_well_conditioned = cond < 1e6

    # Safe solve (returns garbage for singular, but we check cond)
    lambdas = jnp.linalg.solve(A, b)

    lambda0 = 1.0 - jnp.sum(lambdas)
    epsilon = 1e-10

    is_inside = (lambda0 >= -epsilon) & jnp.all(lambdas >= -epsilon) & (jnp.sum(lambdas) <= 1.0 + epsilon)

    return is_inside & is_well_conditioned
```

**GPU Characteristics:**
- Uses cuSolver's DGESV (GPU linear solver)
- 32-bit float precision (default)
- No exception handling (JAX limitation)
- Can process in batches with vmap
- Memory: O(n) for n parallel calls

**Key Differences:**
1. **Precision**: CPU uses 64-bit, GPU uses 32-bit → ~1% mismatch on boundaries
2. **Error handling**: CPU uses try/except, GPU uses condition number check
3. **Parallelization**: GPU can vmap over thousands of points simultaneously

---

## 2. Three-Tier Element Search

### CPU Implementation (`jaxtrace/gpu/search.py`)

```python
def find_containing_element_cpu(
    point: np.ndarray,
    cached_element_id: int,
    element_neighbors: np.ndarray,
    element_to_block: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray,
) -> int:
    """
    Three-tier search on CPU.

    Algorithm:

    Level 0 (Cached Element):
    - Check if point is in cached_element_id
    - If yes: return immediately (85-95% hit rate)
    - Cost: 1 point-in-element test

    Level 1 (Neighbor Elements):
    - Get neighbors from element_neighbors[cached_element_id]
    - Check each neighbor (up to 4 neighbors)
    - If found: return immediately (3-10% hit rate)
    - Cost: 1-4 point-in-element tests

    Level 2 (Block-Local Search):
    - Determine which block contains the point
    - Get ALL elements in that block
    - Check ONLY elements in that block (filtered list)
    - Return first match or -1
    - Cost: ~100 point-in-element tests (for ThreadedA)

    Returns: element_id or -1 if not found
    """

    # Level 0: Check cached element
    if cached_element_id >= 0:
        element_node_ids = connectivity[cached_element_id]
        vertices = positions[element_node_ids]

        if point_in_tetrahedron(point, vertices):
            return cached_element_id

    # Level 1: Check neighbors
    if cached_element_id >= 0:
        neighbors = element_neighbors[cached_element_id]

        for neighbor_id in neighbors:
            if neighbor_id < 0:  # No more neighbors
                break

            element_node_ids = connectivity[neighbor_id]
            vertices = positions[element_node_ids]

            if point_in_tetrahedron(point, vertices):
                return neighbor_id

    # Level 2: Block-local search
    # Determine block containing point
    block_id = position_to_block_id(point, domain_bounds, grid_size)

    if block_id < 0:  # Outside domain
        return -1

    # CPU: Filter elements to only those in this block
    # This creates a compact list of candidate elements
    block_elements = np.where(element_to_block == block_id)[0]

    # Check only elements in this block
    for elem_id in block_elements:
        element_node_ids = connectivity[elem_id]
        vertices = positions[element_node_ids]

        if point_in_tetrahedron(point, vertices):
            return elem_id

    return -1  # Not found
```

**CPU Level 2 Characteristics:**
- Uses `np.where` to create compact list of block elements
- Iterates only over valid elements (no dummy elements)
- Sequential loop (one element at a time)
- Early termination on first match
- Memory: O(k) where k = elements in block (~100 for ThreadedA)
- Time: O(k) with early termination

---

### GPU Implementation (Original - SLOW)

```python
@jax.jit
def search_block_elements_jax_ORIGINAL(
    point: jnp.ndarray,
    block_id: int,
    element_to_block: jnp.ndarray,
    positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    max_elements_to_check: int = 10000
) -> Tuple[bool, int]:
    """
    ORIGINAL IMPLEMENTATION (SLOW!)

    Level 2: Block-local search - PROBLEMATIC VERSION

    Algorithm:
    1. Get ALL block element IDs into array of size max_elements_to_check
    2. Pad with -1 if fewer elements (e.g., 100 real + 900 dummy)
    3. Use lax.scan to iterate through ALL max_elements_to_check elements
    4. Return first match

    PROBLEM: For ThreadedA mesh:
    - Block has ~100 real elements
    - max_elements_to_check = 1000
    - Creates array: [e0, e1, ..., e99, -1, -1, ..., -1]  (900 dummies!)
    - lax.scan processes ALL 1000 elements
    - 90% of work is checking dummy elements!

    For 10K particles:
    - 10,000 particles × 1,000 elements × point-in-element = 10M checks
    - 9M are wasted on dummy elements!
    - This causes 40× slowdown vs CPU
    """

    # Get block element IDs (padded to max_elements_to_check)
    block_element_ids = jnp.full(max_elements_to_check, -1, dtype=jnp.int32)

    # Fill in real element IDs
    # (simplified - actual code is more complex)
    valid_elements = jnp.where(element_to_block == block_id,
                                jnp.arange(len(element_to_block)),
                                -1)

    # Take first max_elements_to_check
    search_ids = valid_elements[:max_elements_to_check]

    # BOTTLENECK: lax.scan over ALL max_elements_to_check elements
    # Even though most are -1 (dummy), scan still processes them
    def check_element(carry, elem_id):
        found, result_id = carry

        # Skip if already found (but scan continues!)
        # Skip if elem_id is -1 (but scan continues!)
        is_valid = (elem_id >= 0) & (~found)

        # Get vertices
        vertices = positions[connectivity[elem_id]]

        # Check containment
        is_inside = point_in_tetrahedron_safe(point, vertices)

        new_found = found | (is_valid & is_inside)
        new_result = jnp.where(is_valid & is_inside, elem_id, result_id)

        return (new_found, new_result), None

    init_carry = (False, -1)
    (found, result_id), _ = jax.lax.scan(check_element, init_carry, search_ids)

    return found, result_id
```

**GPU Original Level 2 Characteristics (SLOW):**
- Creates sparse array with dummy elements
- lax.scan processes ALL elements (no early termination)
- 90% wasted computation on dummy elements
- Memory: O(max_elements_to_check) = 1000
- Time: O(max_elements_to_check) = 1000 checks per particle
- **Result: 40× slower than CPU!**

---

### GPU Implementation (New - OUT OF MEMORY)

```python
@jax.jit
def search_block_elements_jax_NEW(
    point: jnp.ndarray,
    block_id: int,
    element_to_block: jnp.ndarray,
    positions: jnp.ndarray,
    connectivity: jnp.ndarray,
    max_elements_to_check: int = 10000
) -> Tuple[bool, int]:
    """
    NEW IMPLEMENTATION (OUT OF MEMORY!)

    Algorithm:
    1. Check ALL elements in mesh (not just block)
    2. Filter in each check: only process if element is in block
    3. Use vmap for parallel execution
    4. Return first match

    PROBLEM: For ThreadedA mesh:
    - Total elements: 3,494,800
    - vmap creates 3.5M parallel operations
    - Each requires vertices (4 nodes × 3 coords = 48 bytes)
    - Total memory: 3.5M × 48 = 168 MB per batch
    - With intermediate arrays: ~4 GB!
    - GPU has only ~8 GB total
    - Out of memory error!

    This is WORSE than lax.scan approach!
    """

    # Get number of elements
    n_elements = element_to_block.shape[0]

    # Define check for single element
    def check_element(elem_id):
        # Check if element is in block
        in_block = element_to_block[elem_id] == block_id

        # Get vertices
        vertices = positions[connectivity[elem_id]]

        # Check containment (only if in block)
        is_inside = jnp.where(
            in_block,
            point_in_tetrahedron_safe(point, vertices),
            False
        )

        return in_block & is_inside, jnp.where(in_block & is_inside, elem_id, -1)

    # vmap over ALL elements (3.5M elements!)
    element_ids = jnp.arange(n_elements, dtype=jnp.int32)
    found_array, result_array = jax.vmap(check_element)(element_ids)

    # Find first match
    found_any = jnp.any(found_array)
    first_match_idx = jnp.argmax(found_array)
    result_id = result_array[first_match_idx]

    final_found = found_any & (block_id >= 0)
    final_result = jnp.where(final_found, result_id, -1)

    return final_found, final_result
```

**GPU New Level 2 Characteristics (OUT OF MEMORY):**
- Vectorizes over ALL mesh elements (3.5M)
- Massive memory allocation
- GPU OOM error
- **Result: Doesn't run at all!**

---

## 3. Batch Processing

### CPU Implementation

```python
def update_particle_element_ids(
    particles: ParticleData,
    element_neighbors: np.ndarray,
    element_to_block: np.ndarray,
    positions: np.ndarray,
    connectivity: np.ndarray,
) -> ParticleData:
    """
    CPU: Sequential processing of particles.

    Algorithm:
    - Loop over each particle (sequential)
    - For each: call find_containing_element_cpu
    - Update element_ids array

    Characteristics:
    - Single-threaded (no parallelization)
    - Low memory usage
    - Processes one particle at a time
    """
    n_particles = len(particles.positions)
    new_element_ids = np.empty(n_particles, dtype=np.int32)

    for i in range(n_particles):
        point = particles.positions[i]
        cached_id = particles.element_ids[i]

        new_element_ids[i] = find_containing_element_cpu(
            point, cached_id, element_neighbors,
            element_to_block, positions, connectivity
        )

    return ParticleData(
        positions=particles.positions,
        element_ids=new_element_ids,
        ...
    )
```

**CPU Batch Characteristics:**
- Sequential loop
- Memory: O(n_particles)
- Time: O(n_particles × search_cost)
- Single-threaded

---

### GPU Implementation

```python
# Define vectorized search function
find_containing_elements_batch = jax.jit(jax.vmap(
    find_containing_element_gpu,
    in_axes=(0, 0, 0, None, None, None, None)
))

def update_particle_elements(self, particles, batch_size=None):
    """
    GPU: Batch processing with vmap.

    Algorithm:
    - Transfer particles to GPU
    - Use vmap to process ALL particles in parallel
    - Optional batching if too many particles
    - Transfer results back to CPU

    Characteristics:
    - Massively parallel (1000s of particles simultaneously)
    - High memory usage
    - Transfer overhead (CPU ↔ GPU)
    """
    n_particles = len(particles.positions)

    # Transfer to GPU
    pos_gpu = jax.device_put(particles.positions, device=self.device)
    cached_ids_gpu = jax.device_put(particles.element_ids, device=self.device)
    block_ids_gpu = jax.device_put(block_ids, device=self.device)

    if batch_size is None:
        # Process all at once
        new_elem_gpu = find_containing_elements_batch(
            pos_gpu,
            cached_ids_gpu,
            block_ids_gpu,
            self.element_neighbors_gpu,
            self.element_to_block_gpu,
            self.positions_gpu,
            self.connectivity_gpu
        )
    else:
        # Process in batches
        results = []
        for i in range(0, n_particles, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_particles))
            batch_result = find_containing_elements_batch(
                pos_gpu[batch_slice],
                cached_ids_gpu[batch_slice],
                block_ids_gpu[batch_slice],
                self.element_neighbors_gpu,
                self.element_to_block_gpu,
                self.positions_gpu,
                self.connectivity_gpu
            )
            results.append(batch_result)
        new_elem_gpu = jnp.concatenate(results)

    # Transfer back to CPU
    new_elem_cpu = np.array(new_elem_gpu)

    return ParticleData(
        positions=particles.positions,
        element_ids=new_elem_cpu,
        ...
    )
```

**GPU Batch Characteristics:**
- Parallel processing (vmap over particles)
- High memory usage
- Transfer overhead
- Memory: O(n_particles × mesh_size) in worst case

---

## 4. Key Differences Summary

| Component | CPU | GPU (Current) | Impact |
|-----------|-----|---------------|--------|
| **Precision** | 64-bit float | 32-bit float | ~1% boundary mismatch |
| **Point-in-element** | Sequential, 1 at a time | Parallel (vmap) | GPU faster |
| **Level 0/1 search** | Sequential | Parallel | GPU faster |
| **Level 2 search** | Compact list, early exit | Sparse array OR full mesh | **GPU 40× slower!** |
| **Memory** | O(1) per particle | O(n_particles × search_space) | GPU OOM |
| **Error handling** | try/except | Condition number | Minor difference |
| **Parallelization** | Single-threaded | Thousands of threads | GPU advantage |

---

## 5. Root Cause Analysis

### Why GPU is Slower (Current State)

**For 10,000 particles on ThreadedA mesh:**

**CPU:**
- Level 0: 8,500 particles (85% hit) → 8,500 checks
- Level 1: 1,000 particles (10% hit) → 4,000 checks (avg 4 neighbors)
- Level 2: 500 particles (5% hit) → 50,000 checks (avg 100 elements/block)
- **Total: ~62,500 point-in-element tests**

**GPU (Original with lax.scan):**
- Level 0: 8,500 particles → 8,500 checks (parallel)
- Level 1: 1,000 particles → 4,000 checks (parallel)
- Level 2: 500 particles × 1,000 max_elements = **500,000 checks**
  - But only 500 × 100 = 50,000 are real elements
  - **450,000 wasted checks on dummy elements!**
- **Total: ~462,500 point-in-element tests (7× more than CPU!)**

**GPU (New with full vmap):**
- Level 0: 8,500 particles → 8,500 checks (parallel)
- Level 1: 1,000 particles → 4,000 checks (parallel)
- Level 2: 500 particles × 3,494,800 elements = **1.75 BILLION checks**
  - Memory: 500 × 3.5M × 48 bytes = **84 GB!**
- **Result: Out of memory, doesn't run**

---

## 6. What Needs to be Fixed

### Option 1: Compact Array with Limited Size
- Pre-filter block elements before scan
- Use fixed-size array (e.g., 200) based on actual block sizes
- Pad with -1 only if block has fewer elements
- For ThreadedA: ~100 real elements + 100 padding = 200 total

### Option 2: Block-Specific Element Lists (Phase 8/9)
- Pre-compute element lists per block
- Store as jagged array or hash table
- Level 2 search only iterates actual block elements
- This is what Phase 9 (hash octree) will implement

### Option 3: Hybrid CPU/GPU
- Use GPU for Level 0/1 (high hit rate, parallel)
- Fall back to CPU for Level 2 (rare, sequential)
- Only ~5% of particles need Level 2

### Option 4: Reduce max_elements_to_check
- Current: 1000 (way too high for ThreadedA)
- Analyze actual block sizes
- Use 90th percentile + buffer (e.g., 200 for ThreadedA)
- Still has dummy elements, but fewer

---

## 7. Recommended Fix (Immediate)

**Implement compact block element lists:**

```python
# Pre-process: Create compact element lists per block (CPU, once)
def build_block_element_lists(element_to_block, n_blocks, max_per_block=200):
    """Build compact arrays of elements per block."""
    block_elements = []
    block_counts = []

    for block_id in range(n_blocks):
        elements = np.where(element_to_block == block_id)[0]
        count = len(elements)

        # Pad to max_per_block
        if count > max_per_block:
            elements = elements[:max_per_block]
            count = max_per_block
        else:
            elements = np.pad(elements, (0, max_per_block - count),
                            constant_values=-1)

        block_elements.append(elements)
        block_counts.append(count)

    return np.array(block_elements), np.array(block_counts)

# Usage in Level 2 search (GPU)
@jax.jit
def search_block_elements_jax_FIXED(
    point: jnp.ndarray,
    block_id: int,
    block_elements: jnp.ndarray,  # [n_blocks, max_per_block]
    block_counts: jnp.ndarray,    # [n_blocks]
    positions: jnp.ndarray,
    connectivity: jnp.ndarray,
) -> Tuple[bool, int]:
    """Fixed version using pre-computed compact lists."""

    # Get element list for this block
    elements = block_elements[block_id]  # [max_per_block]
    count = block_counts[block_id]

    # vmap over fixed-size array (e.g., 200 instead of 1000)
    def check_element(elem_id):
        is_valid = elem_id >= 0
        vertices = positions[connectivity[jnp.where(is_valid, elem_id, 0)]]
        is_inside = jnp.where(is_valid,
                             point_in_tetrahedron_safe(point, vertices),
                             False)
        return is_inside, jnp.where(is_inside, elem_id, -1)

    found_array, result_array = jax.vmap(check_element)(elements)

    found_any = jnp.any(found_array)
    first_match = result_array[jnp.argmax(found_array)]

    return found_any, jnp.where(found_any, first_match, -1)
```

**Benefits:**
- Fixed memory usage: n_blocks × max_per_block (32 × 200 = 6,400 elements)
- No dummy elements beyond necessary padding
- For ThreadedA: 100 real + 100 padding = 200 (vs 1000 before)
- 5× reduction in Level 2 checks
- Should make GPU competitive with CPU

---

## 8. Expected Performance After Fix

**For 10,000 particles:**

- CPU: ~62,500 checks, ~0.7s
- GPU (fixed): ~112,500 checks (parallel), expected ~0.1-0.2s
- **Expected speedup: 3-7× faster than CPU**

The GPU advantage comes from:
1. Parallel Level 0/1 (8× speedup)
2. Parallel Level 2 within block (4× speedup)
3. Batch processing eliminates per-particle overhead

The GPU disadvantage (transfer overhead) is minimal for 10K particles (~10ms).
