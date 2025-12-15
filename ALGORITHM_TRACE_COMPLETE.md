# Complete Algorithm Trace: Octree Accuracy Test

This document traces the EXACT algorithm flow from test start to final search result, showing WHERE Morton codes are computed and HOW they relate to the octree.

## Overview of the Problem

**User's Hypothesis**: "Morton is hashing elements before octree, instead of hashing octree"

This trace will reveal:
1. WHEN Morton codes are computed (CPU build time vs GPU search time)
2. WHAT is being hashed (element centroids vs query positions)
3. HOW the octree structure relates to Morton codes
4. WHERE the mismatch occurs

---

## Test Flow: test_octree_accuracy.py

### Step 1: Load Mesh (Lines 121-131)

```python
node_positions, connectivity, velocity_field = load_mesh_from_pvtu(Path(MESH_PATH))
# Result: 
# - node_positions: (900658, 3) float32
# - connectivity: (3512279, 4) int32
```

**No Morton codes yet.**

---

### Step 2: Build Adaptive Octree (Lines 135-145)

```python
morton_struct = build_global_morton_octree(
    node_positions=node_positions,
    connectivity=connectivity,
    leaf_capacity=256,
    max_depth=21,
    verbose=False
)
```

**This is where Morton codes are FIRST computed.**

#### Step 2.1: Compute Element Centroids (CPU)

**File**: `jaxtrace/gpu/search/morton_octree_builder.py`  
**Function**: `build_global_morton_octree()` lines 138-149

```python
# 1. Compute element centroids
centroids = np.zeros((n_elements, 3), dtype=np.float32)
for i in range(n_elements):
    nodes = connectivity[i]
    centroid = node_positions[nodes].mean(axis=0)
    centroids[i] = centroid
```

**Result**: Element centroids for ALL 3.5M elements computed on CPU.

**Key Point**: Centroids are computed ONCE during octree build, NOT during search.

---

#### Step 2.2: Compute Bounding Box (CPU)

**Lines**: 151-159

```python
bbox_min = centroids.min(axis=0).astype(np.float32)
bbox_max = centroids.max(axis=0).astype(np.float32)

# Add epsilon
epsilon = 1e-6 * (bbox_max - bbox_min)
bbox_min -= epsilon
bbox_max += epsilon
```

**Result**: Global bounding box for Morton normalization.

**Key Point**: Same bbox used for BOTH element hashing (build) and position hashing (search).

---

#### Step 2.3: Compute Morton Codes for ALL Elements (CPU)

**Lines**: 161-179

```python
morton_codes = np.zeros(n_elements, dtype=np.uint64)

# Vectorized Morton encoding
normalized = (centroids - bbox_min) / (bbox_max - bbox_min)
normalized = np.clip(normalized, 0.0, 1.0)
grid_max = (2 ** max_depth) - 1
u = np.floor(normalized * grid_max).astype(np.uint32)

# Interleave bits (vectorized)
for i in range(21):
    morton_codes |= ((u[:, 0] >> i) & 1).astype(np.uint64) << (3*i + 0)
    morton_codes |= ((u[:, 1] >> i) & 1).astype(np.uint64) << (3*i + 1)
    morton_codes |= ((u[:, 2] >> i) & 1).astype(np.uint64) << (3*i + 2)
```

**Result**: `morton_codes[elem_id]` = 63-bit Morton code for element `elem_id`'s centroid.

**Key Point**: Morton codes are for ELEMENT CENTROIDS, not query positions.

**Bit Layout** (verified from HOT spec):
```
Bit 62-60: (x0, y0, z0) - coarsest level (octant)
Bit 59-57: (x1, y1, z1) - level 1
...
Bit 2-0:   (x20, y20, z20) - finest level
```

**MSB contains coarse octant, LSB contains fine details.**

---

#### Step 2.4: Sort Elements by Morton Code (CPU)

**Lines**: 181-186

```python
sort_indices = np.argsort(morton_codes)
morton_sorted = morton_codes[sort_indices]
elem_ids_sorted = np.arange(n_elements, dtype=np.int32)[sort_indices]
```

**Result**:
- `morton_sorted[i]`: i-th smallest Morton code
- `elem_ids_sorted[i]`: Element ID with i-th smallest Morton code

**Key Point**: Elements are now sorted by spatial proximity (Z-order).

---

#### Step 2.5: Build Adaptive Octree Leaves (CPU)

**Lines**: 188-200  
**Function**: `build_adaptive_octree_leaves()`

```python
def build_adaptive_octree_leaves(morton_sorted, elem_ids_sorted, leaf_capacity=256, max_depth=21):
    leaves = []
    
    def subdivide_node(start_idx, end_idx, morton_prefix, prefix_bits, depth):
        n_elements_node = end_idx - start_idx
        
        # Base case 1: small enough to be a leaf
        if n_elements_node <= leaf_capacity:
            leaf = OctreeLeaf(
                start_idx=start_idx,
                length=n_elements_node,
                morton_prefix=morton_prefix,  # <-- Prefix of ELEMENT Morton codes
                prefix_bits=prefix_bits
            )
            leaves.append(leaf)
            return
        
        # Base case 2: max depth
        if depth >= max_depth:
            # Force leaf
            ...
        
        # Recursive: subdivide into 8 octants
        octant_ranges = compute_octant_ranges(morton_sorted, start_idx, end_idx, morton_prefix, prefix_bits)
        
        for octant, octant_start, octant_end in octant_ranges:
            octant_prefix = (morton_prefix << 3) | octant
            subdivide_node(octant_start, octant_end, octant_prefix, prefix_bits + 3, depth + 1)
```

**Key Point**: Octree is built by SUBDIVIDING the sorted element list by Morton code prefixes.

**Example**:
- Leaf at depth 3 with prefix `0b101` contains all elements whose Morton codes start with `0b101xxx...`
- These elements are spatially in octant (1, 0, 1) at level 3

---

#### Step 2.6: Compute Octant Ranges (CPU)

**Function**: `compute_octant_ranges()` lines 52-96

```python
def compute_octant_ranges(morton_sorted, start_idx, end_idx, morton_prefix, prefix_bits):
    octant_ranges = []
    shift = 63 - (prefix_bits + 3)  # Align next 3 bits to MSB
    
    for octant in range(8):
        octant_prefix = (morton_prefix << 3) | octant
        
        # Binary search for range where (morton >> shift) == octant_prefix
        # Find first morton where (morton >> shift) >= octant_prefix
        left = start_idx
        right = end_idx
        while left < right:
            mid = (left + right) // 2
            morton_mid = morton_sorted[mid] >> shift
            if morton_mid < octant_prefix:
                left = mid + 1
            else:
                right = mid
        octant_start = left
        
        # Find first morton where (morton >> shift) > octant_prefix
        left = octant_start
        right = end_idx
        while left < right:
            mid = (left + right) // 2
            morton_mid = morton_sorted[mid] >> shift
            if morton_mid <= octant_prefix:
                left = mid + 1
            else:
                right = mid
        octant_end = left
        
        if octant_end > octant_start:
            octant_ranges.append((octant, octant_start, octant_end))
    
    return octant_ranges
```

**Key Insight**: 
- Uses `morton >> shift` to extract prefix bits
- Shift amount: `63 - (prefix_bits + 3)`
- **This extracts bits from MSB (most significant bits)**

**Example** (depth=0, prefix_bits=0):
```
shift = 63 - 3 = 60
morton >> 60 extracts top 3 bits (octant at root level)
```

---

#### Step 2.7: Build Prefix Table (CPU)

**Function**: `build_prefix_table()` lines 140-194

```python
def build_prefix_table(leaves, max_depth=21):
    # Find max prefix_bits among leaves
    max_prefix_bits = max(leaf.prefix_bits for leaf in leaves)
    
    # Choose table depth (depth where table size reasonable)
    for table_depth_bits in range(max_prefix_bits, 2, -3):
        table_size = 8 ** (table_depth_bits // 3)
        if table_size <= 1_000_000:
            break
    
    table_depth = table_depth_bits // 3  # Example: 6 levels = 18 bits
    table_size = 8 ** table_depth         # Example: 8^6 = 262,144 entries
    
    # Create table
    prefix_table = np.full(table_size, -1, dtype=np.int32)
    
    # Fill table
    for leaf_id, leaf in enumerate(leaves):
        leaf_depth = leaf.prefix_bits // 3
        
        if leaf_depth >= table_depth:
            # Leaf deeper than table: extract table_depth-bit prefix
            shift = leaf.prefix_bits - (table_depth * 3)
            prefix = leaf.morton_prefix >> shift
            prefix_table[prefix] = leaf_id
        else:
            # Leaf shallower than table: fill all descendant prefixes
            n_descendants = 8 ** (table_depth - leaf_depth)
            base_prefix = leaf.morton_prefix << ((table_depth - leaf_depth) * 3)
            for i in range(n_descendants):
                prefix = base_prefix + i
                prefix_table[prefix] = leaf_id
    
    return prefix_table, table_depth
```

**Result**: `prefix_table[prefix]` = leaf_id for 18-bit prefix

**Key Point**: Prefix table maps Morton CODE PREFIXES to leaf IDs.

**Example** (table_depth=6):
- If query position has Morton code `0xABCD1234567890AB`
- Extract top 18 bits: `prefix = 0xABCD1234567890AB >> 45`
- Lookup: `leaf_id = prefix_table[prefix]`

---

### Step 3: Upload to GPU (Lines 152-158)

```python
mesh_gpu_morton = upload_global_morton_to_gpu(
    morton_struct,
    connectivity,
    node_positions
)
```

**Function**: `upload_global_morton_to_gpu()` in `morton_global_search.py` lines 566-626

```python
def upload_global_morton_to_gpu(morton_struct, connectivity, node_positions):
    return MeshGPUGlobalMorton(
        connectivity=jax.device_put(connectivity),
        node_positions=jax.device_put(node_positions),
        
        elem_ids_sorted=jax.device_put(morton_struct.elem_ids_sorted),
        morton_sorted=jax.device_put(morton_struct.morton_sorted),
        leaf_start=jax.device_put(morton_struct.leaf_start),
        leaf_length=jax.device_put(morton_struct.leaf_length),
        
        prefix_table=jax.device_put(morton_struct.prefix_table),  # NEW
        table_depth=jnp.int32(morton_struct.table_depth),         # NEW
        
        bbox_min=jax.device_put(morton_struct.bbox_min),
        bbox_max=jax.device_put(morton_struct.bbox_max),
        max_depth=jnp.int32(morton_struct.max_depth),
        ...
    )
```

**Result**: All octree data now on GPU.

---

### Step 4: Compute Test Centroids (CPU)

**Test code** lines 175-183:

```python
# Randomly sample elements
test_elem_ids = np.random.randint(0, n_elements, size=N_TEST_PARTICLES)

# Compute centroids for sampled elements
centroids = np.zeros((N_TEST_PARTICLES, 3), dtype=np.float32)
for i, elem_id in enumerate(test_elem_ids):
    nodes = connectivity[elem_id]
    centroid = node_positions[nodes].mean(axis=0)
    centroids[i] = centroid
```

**Key Point**: Test centroids are computed FRESH, NOT using cached centroids from octree build.

**Question**: Are these centroids EXACTLY the same as octree build centroids?
- **Answer**: Should be, same formula `node_positions[nodes].mean(axis=0)`

---

### Step 5: Run L2 Search on Centroids (GPU)

**Test code** lines 189-195:

```python
centroids_gpu = jax.device_put(centroids)

found_elem_ids = jax.vmap(
    lambda p: search_L2_global_morton_single(p, mesh_gpu_morton, jnp.int32(L2_SEARCH_RADIUS))
)(centroids_gpu)
```

**This is where the SEARCH happens.**

---

#### Step 5.1: Search Function Entry

**Function**: `search_L2_global_morton_single()` in `morton_global_search.py` lines 476-558

```python
def search_L2_global_morton_single(
    pos: jax.Array,  # (3,) float32 - QUERY POSITION (centroid)
    mesh_gpu: MeshGPUGlobalMorton,
    search_radius: jnp.int32 = jnp.int32(1)
) -> jnp.int32:
    
    # Map position to leaf using appropriate method
    center_leaf_id = jnp.where(
        mesh_gpu.table_depth > 0,
        position_to_leaf_id_octree(pos, mesh_gpu),  # NEW: Octree lookup
        position_to_leaf_id(pos, mesh_gpu)          # OLD: Binary search
    )
    
    # Search center leaf
    elem_id = search_in_leaf_global(pos, center_leaf_id, mesh_gpu)
    
    # Search neighbors if not found
    ...
```

**Key Point**: This function computes Morton code for QUERY POSITION, not for elements.

---

#### Step 5.2: Position to Leaf ID (Octree Method)

**Function**: `position_to_leaf_id_octree()` lines 206-257

```python
def position_to_leaf_id_octree(pos: jax.Array, mesh_gpu: MeshGPUGlobalMorton) -> jnp.int32:
    # 1. Compute Morton code for QUERY position
    m = morton_encode_position_jax(
        pos,                    # QUERY POSITION (centroid from test)
        mesh_gpu.bbox_min,      # Same bbox as octree build
        mesh_gpu.bbox_max,
        mesh_gpu.max_depth      # Same max_depth (21)
    )
    
    # 2. Extract prefix bits
    table_depth_int = int(mesh_gpu.table_depth)  # 6
    prefix_bits_int = table_depth_int * 3        # 18
    shift_amount = 63 - prefix_bits_int          # 63 - 18 = 45
    
    # 3. Right-shift Morton code to extract prefix
    prefix = lax.shift_right_logical(m, jnp.uint64(shift_amount))
    prefix = prefix.astype(jnp.int32)
    
    # 4. Lookup prefix in table
    prefix = jnp.clip(prefix, 0, mesh_gpu.prefix_table.shape[0] - 1)
    leaf_id = mesh_gpu.prefix_table[prefix]
    
    return leaf_id
```

---

#### Step 5.3: Morton Encode Position (GPU)

**Function**: `morton_encode_position_jax()` lines 119-136

```python
def morton_encode_position_jax(
    pos: jax.Array,       # QUERY POSITION (centroid from test)
    bbox_min: jax.Array,
    bbox_max: jax.Array,
    max_depth: int
) -> jnp.uint64:
    # Normalize position to [0, 1]
    normalized = (pos - bbox_min) / (bbox_max - bbox_min)
    normalized = jnp.clip(normalized, 0.0, 1.0)
    
    # Map to integer grid [0, 2^21 - 1]
    grid_max = (2 ** max_depth) - 1
    u = jnp.floor(normalized * grid_max).astype(jnp.uint32)
    
    # Interleave bits
    return interleave_bits_3d_jax(u[0], u[1], u[2])
```

**Function**: `interleave_bits_3d_jax()` lines 61-109

```python
def interleave_bits_3d_jax(x: jnp.uint32, y: jnp.uint32, z: jnp.uint32) -> jnp.uint64:
    # Convert to uint64
    x = x.astype(jnp.uint64)
    y = y.astype(jnp.uint64)
    z = z.astype(jnp.uint64)
    
    # Expand x (position 0, 3, 6, 9, ...)
    x = (x | (x << 32)) & jnp.uint64(0x001f00000000ffff)
    x = (x | (x << 16)) & jnp.uint64(0x001f0000ff0000ff)
    x = (x | (x <<  8)) & jnp.uint64(0x100f00f00f00f00f)
    x = (x | (x <<  4)) & jnp.uint64(0x10c30c30c30c30c3)
    x = (x | (x <<  2)) & jnp.uint64(0x1249249249249249)
    
    # Expand y (position 1, 4, 7, 10, ...)
    y = ... # Same pattern
    
    # Expand z (position 2, 5, 8, 11, ...)
    z = ... # Same pattern
    
    # Combine: x at bit 0, y at bit 1, z at bit 2
    return x | (y << 1) | (z << 2)
```

**Result**: Morton code with bits interleaved as:
```
Bit 0: x0
Bit 1: y0
Bit 2: z0
Bit 3: x1
Bit 4: y1
Bit 5: z1
...
Bit 60: x20
Bit 61: y20
Bit 62: z20
```

**KEY OBSERVATION**: LSB contains coarse octant (x0, y0, z0), MSB contains fine details (x20, y20, z20).

**This is OPPOSITE of what I assumed!**

---

## THE BUG: Bit Order Mismatch

### Octree Build (compute_octant_ranges)

```python
shift = 63 - (prefix_bits + 3)
morton >> shift  # Extracts from MSB
```

**Example** (depth=0, extracting octant):
```
shift = 63 - 3 = 60
morton >> 60 extracts bits [62:60] = (x20, y20, z20) = FINEST level
```

**WRONG**: This extracts the FINEST level bits, not the coarsest octant!

### GPU Search (position_to_leaf_id_octree)

```python
shift_amount = 63 - prefix_bits_int  # 63 - 18 = 45
prefix = m >> shift_amount           # Extracts bits [62:45]
```

**Example** (table_depth=6, 18 bits):
```
shift = 63 - 18 = 45
morton >> 45 extracts bits [62:45] = (x20...x14, y20...y14, z20...z14)
```

**WRONG**: Extracts FINEST level bits, not coarsest!

---

## Correct Bit Order for Octree

### What We Need

For octree subdivision:
- Level 0 (root): Extract octant from bits [2:0] (x0, y0, z0) - COARSEST
- Level 1: Extract octant from bits [5:3] (x1, y1, z1)
- Level 2: Extract octant from bits [8:6] (x2, y2, z2)
- ...
- Level 20: Extract octant from bits [62:60] (x20, y20, z20) - FINEST

### Correct Formula for Octant Extraction

```python
# For depth D, extract 3 bits starting at position D*3
octant_bits_start = depth * 3
octant_bits_end = octant_bits_start + 3
octant = (morton >> octant_bits_start) & 0b111
```

**Example** (depth=0):
```python
octant = (morton >> 0) & 0b111  # Extract bits [2:0]
```

**Example** (depth=6, for prefix table):
```python
prefix_bits = 6 * 3  # 18 bits
prefix = morton & ((1 << 18) - 1)  # Extract bits [17:0] (mask)
```

---

## Correct Implementation

### Fix 1: compute_octant_ranges()

**OLD (WRONG)**:
```python
shift = 63 - (prefix_bits + 3)
octant_prefix = (morton_prefix << 3) | octant
morton_mid = morton_sorted[mid] >> shift
```

**NEW (CORRECT)**:
```python
# Prefix is at LSB, not MSB
# For depth D, we want bits [D*3+2:D*3] for next octant
octant_prefix = (morton_prefix << 3) | octant
octant_mask = (1 << (prefix_bits + 3)) - 1
morton_mid = morton_sorted[mid] & octant_mask
```

### Fix 2: build_prefix_table()

**OLD (WRONG)**:
```python
if leaf_depth >= table_depth:
    shift = leaf.prefix_bits - (table_depth * 3)
    prefix = leaf.morton_prefix >> shift
```

**NEW (CORRECT)**:
```python
if leaf_depth >= table_depth:
    # Extract LSB bits
    prefix_mask = (1 << (table_depth * 3)) - 1
    prefix = leaf.morton_prefix & prefix_mask
```

### Fix 3: position_to_leaf_id_octree()

**OLD (WRONG)**:
```python
shift_amount = 63 - prefix_bits_int
prefix = lax.shift_right_logical(m, jnp.uint64(shift_amount))
```

**NEW (CORRECT)**:
```python
# Extract LSB bits
prefix_mask = jnp.uint64((1 << prefix_bits_int) - 1)
prefix = m & prefix_mask
```

---

## Root Cause Summary

**Problem**: Algorithm assumes Morton code prefixes are at MSB (most significant bits), but they're actually at LSB (least significant bits).

**Why It Happened**: 
1. Morton encoding uses standard bit interleaving (x0 at bit 0, LSB)
2. Octree subdivision code assumed prefixes grow from MSB
3. Both build and search use wrong bit extraction (shift right from MSB instead of mask from LSB)
4. Mismatch is CONSISTENT (both wrong), so structure validates but lookups fail

**Evidence**:
- Octree builder validation passed (structure is self-consistent)
- Spatial coherence is good (2.55 ratio - leaves are spatially aligned)
- But lookups fail (10.8% success) because we're extracting wrong prefix bits
- When element IS found, it's correct (100% point-in-tet) - wrong leaf, but valid element

**Fix**: Change ALL shift-right operations to mask operations (extract from LSB, not MSB).

