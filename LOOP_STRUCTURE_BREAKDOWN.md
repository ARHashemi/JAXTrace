# Detailed Loop Structure Breakdown

This document provides a granular analysis of each unrolled loop in the codebase, showing the exact nesting structure and expansion factor.

---

## 1. RK4 Main Loop (rk4_fully_fused_timedep.py)

### Top-Level Structure
```python
@jax.jit
def rk4_fully_fused_step_timedep(positions_gpu, element_ids_gpu, dt, velocity_fields_gpu, time_idx):
    def rk4_single_particle(pos, elem_id):
        # Stage 1 (k1)
        elem_k1 = search_l0_l1_l2_single(pos, elem_id)           # ← SEARCH 1
        vel_k1 = interpolate_velocity_single(pos, elem_k1, ...)

        # Stage 2 (k2)
        elem_k2 = search_l0_l1_l2_single(pos_k1, elem_k1)        # ← SEARCH 2
        vel_k2 = interpolate_velocity_single(pos_k1, elem_k2, ...)

        # Stage 3 (k3)
        elem_k3 = search_l0_l1_l2_single(pos_k2, elem_k2)        # ← SEARCH 3
        vel_k3 = interpolate_velocity_single(pos_k2, elem_k3, ...)

        # Stage 4 (k4)
        elem_k4 = search_l0_l1_l2_single(pos_k3, elem_k3)        # ← SEARCH 4
        vel_k4 = interpolate_velocity_single(pos_k3, elem_k4, ...)

        # Final search
        elem_final = search_l0_l1_l2_single(pos_final, elem_k4)  # ← SEARCH 5

        return pos_final, elem_final

    # OUTER VMAP over all particles
    return jax.vmap(rk4_single_particle)(positions_gpu, element_ids_gpu)
```

**Structure:**
```
vmap[N=225,000 particles]
  └─ rk4_single_particle
      └─ 5 × search_l0_l1_l2_single  ← Each particle does 5 searches
          └─ L0 → L1 → L2 cascade
```

**Multiplication Factor:**
- Searches per particle: 5
- Particles: 225,000
- **Total searches:** 1,125,000

---

## 2. Search Hierarchy (search_l0_l1_l2_single)

```python
def search_l0_l1_l2_single(pos, cached_elem_id):
    # L0: Cached element (1 check)
    elem_l0 = search_l0_single(pos, cached_elem_id)
    found_l0 = elem_l0 >= 0

    # L1: Multi-hop neighbors (only if L0 failed)
    elem_l1 = jnp.where(found_l0, elem_l0, search_l1_single(pos, cached_elem_id))
    found_l1 = elem_l1 >= 0

    # L2: Global Morton (only if L0+L1 failed)
    elem_final = jnp.where(found_l1, elem_l1, search_l2_single(pos))

    return elem_final
```

**Cascade Logic:**
- L0 success rate: ~50-70% → L1 called 30-50% of time
- L1 success rate: ~15-25% → L2 called 15-30% of time
- **Estimated L2 calls:** 1,125,000 × 0.20 = **225,000**

---

## 3. L0 Search (No Unrolling)

```python
def search_l0_single(pos, cached_elem_id):
    inside = point_in_tet_gpu(pos, cached_elem_id, connectivity, node_positions)
    return jnp.where(inside, cached_elem_id, -1)
```

**Operations:**
- 1 element check
- ~50 operations (barycentric coordinates computation)

**Total per search:** 50 ops
**Total per particle:** 5 × 50 = 250 ops
**Total vmapped:** 225K × 250 = **56.25M ops**

---

## 4. L1 Search (Double-Nested Unrolling)

### Loop Structure
```python
def search_l1_single(pos, start_elem_id):
    # Adaptive hop count (3 or 6)
    n_hops_adaptive = jnp.where(size_ratio < 0.1, 6, 3)

    # OUTER LOOP: 6 hops (unrolled)
    for hop_idx in range(6):                                    # ← UNROLL 1: 6 iterations
        hop_enabled = hop_idx < n_hops_adaptive

        # Get neighbors of current element
        neighbors = element_neighbors[current_elem]

        # INNER LOOP: 4 neighbors (unrolled)
        for neighbor_idx in range(4):                           # ← UNROLL 2: 4 iterations
            elem_id = neighbors[neighbor_idx]
            valid = elem_id >= 0

            check_this = (found_containing < 0) & valid
            inside = jnp.where(check_this, point_in_tet_gpu(...), False)

            # Update found_containing if inside
            found_containing = jnp.where(inside & check_this, elem_id, found_containing)

        # Update current_elem for next hop
        current_elem = jnp.where(should_search, ..., current_elem)
        found = found | (found_containing >= 0)

    return jnp.where(found, current_elem, -1)
```

### Unrolling Analysis
**Outer loop:** `range(6)` → 6 unrolled copies
**Inner loop:** `range(4)` → 4 unrolled copies
**Total unrolled iterations:** 6 × 4 = **24**

**Operations per iteration:**
- Neighbor fetch: 10 ops
- Validity checks: 5 ops
- point_in_tet_gpu: 50 ops
- Update logic: 35 ops
- **Total:** ~100 ops

**Total per search:** 24 × 100 = 2,400 ops
**Total per particle (if L0 fails):** 5 × 2,400 = 12,000 ops
**Total vmapped (30% L0 fail):** 225K × 0.3 × 12K = **810M ops**

**XLA Graph Expansion:**
```
vmap[225K]
  └─ for hop in [0,1,2,3,4,5]:          ← 6 unrolled branches
      └─ for neighbor in [0,1,2,3]:     ← 4 unrolled branches
          └─ point_in_tet_gpu           ← 50 ops
```
**Graph nodes:** 225K × 6 × 4 × 50 = **2.7B nodes**
**Memory:** 2.7B × 100 bytes = **270 GB** (masked to ~100 GB due to hop_enabled)

---

## 5. L2 Radius Search (Single-Level Unrolling)

### Loop Structure
```python
def search_L2_global_morton_single(pos, mesh_gpu, search_radius=2):
    center_leaf_id = position_to_leaf_id_octree(pos, mesh_gpu)

    # Center leaf
    elem_id = search_in_leaf_global(pos, center_leaf_id, mesh_gpu)   # ← 8 unrolled
    found = elem_id >= 0

    # NEGATIVE OFFSETS: -radius to -1
    for i in range(15):                                              # ← UNROLL 1: 15 iterations
        offset = -(search_radius - i)
        active = (~found) & (i < search_radius)                      # ← Only 2 active (i=0,1)
        neighbor_leaf_id = center_leaf_id + offset
        elem_neighbor = search_in_leaf_global(pos, neighbor_leaf_id, mesh_gpu)  # ← 8 unrolled
        # Update logic...

    # POSITIVE OFFSETS: +1 to +radius
    for i in range(15):                                              # ← UNROLL 2: 15 iterations
        offset = i + 1
        active = (~found) & (i < search_radius)                      # ← Only 2 active (i=0,1)
        neighbor_leaf_id = center_leaf_id + offset
        elem_neighbor = search_in_leaf_global(pos, neighbor_leaf_id, mesh_gpu)  # ← 8 unrolled
        # Update logic...

    return elem_id
```

### Unrolling Analysis
**Negative loop:** `range(15)` → 15 unrolled copies (only 2 active)
**Positive loop:** `range(15)` → 15 unrolled copies (only 2 active)
**Center:** 1 leaf
**Total leaves checked:** 1 + 2 + 2 = **5 leaves**

Each `search_in_leaf_global` unrolls **8 elements**.

**Total unrolled iterations:** 5 × 8 = **40 element checks**

**Operations per element check:**
- Leaf bounds: 10 ops
- point_in_tet_gpu: 50 ops
- Update logic: 40 ops
- **Total:** ~100 ops

**Total per search:** 40 × 100 = 4,000 ops
**Total per particle (if L0+L1 fail):** 5 × 4,000 = 20,000 ops
**Total vmapped (20% L2 trigger):** 225K × 0.2 × 20K = **900M ops**

**XLA Graph Expansion:**
```
vmap[225K]
  └─ for leaf in [1 center + 15 neg + 15 pos]:  ← 31 unrolled (only 5 active)
      └─ search_in_leaf_global
          └─ for elem in [0..7]:                ← 8 unrolled
              └─ point_in_tet_gpu               ← 50 ops
```
**Graph nodes:** 225K × 31 × 8 × 50 = **2.79B nodes** (masked to ~900M due to 'active')
**Memory:** 900M × 100 bytes = **90 GB**

---

## 6. L2 Morton Neighbors (Triple-Nested Unrolling) 🔴

### Loop Structure
```python
def search_L2_morton_neighbors_single(pos, mesh_gpu):
    # 1. Compute Morton code and get 27 neighbor prefixes
    morton_query = morton_encode_position_jax(pos, ...)
    neighbor_prefixes = get_26_neighbor_prefixes_jax(morton_query, depth=7, ...)  # ← 27 prefixes

    elem_id = -1
    found = False

    # LOOP 1: 27 octants (unrolled)
    for i in range(27):                                          # ← UNROLL 1: 27 iterations
        active = not found
        neighbor_prefix = neighbor_prefixes[i]

        # Look up leaf range for this prefix
        prefix_idx = neighbor_prefix >> shift_amount
        first_leaf = prefix_start[prefix_idx]
        num_leaves_in_prefix = prefix_length[prefix_idx]

        octant_elem = -1
        octant_found = False

        # LOOP 2: 3 leaves per octant (unrolled)
        for leaf_offset in range(3):                             # ← UNROLL 2: 3 iterations
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves_in_prefix) & ...

            # LOOP 3: search_in_leaf_global unrolls 8 elements
            result = search_in_leaf_global(pos, leaf_id, mesh_gpu)  # ← UNROLL 3: 8 iterations
            improved = result >= 0
            octant_elem = jnp.where(improved, result, octant_elem)
            octant_found = octant_found | improved

        # Update global state
        elem_id = jnp.where(octant_elem >= 0, octant_elem, elem_id)
        found = found | octant_found

    return elem_id
```

### Unrolling Analysis
**Loop 1 (octants):** `range(27)` → 27 unrolled copies
**Loop 2 (leaves):** `range(3)` → 3 unrolled copies per octant
**Loop 3 (elements):** `search_in_leaf_global` → 8 unrolled copies per leaf
**Total unrolled iterations:** 27 × 3 × 8 = **648 element checks**

**Operations per element check:**
- Prefix decode: 20 ops
- Leaf lookup: 15 ops
- point_in_tet_gpu: 50 ops
- Update logic: 65 ops
- **Total:** ~150 ops

**Total per search:** 648 × 150 = 97,200 ops
**Total per particle (if L0+L1 fail):** 5 × 97,200 = 486,000 ops
**Total vmapped (20% L2 trigger):** 225K × 0.2 × 486K = **21.87B ops**

**XLA Graph Expansion:**
```
vmap[225K]
  └─ for octant in [0..26]:                      ← 27 unrolled
      └─ for leaf in [0..2]:                     ← 3 unrolled
          └─ search_in_leaf_global
              └─ for elem in [0..7]:             ← 8 unrolled
                  └─ point_in_tet_gpu            ← 50 ops
```
**Graph nodes:** 225K × 27 × 3 × 8 × 50 = **2.187T nodes** (2.187 trillion!)
**Memory:** 2.187T × 100 bytes = **218.7 TB** (masked to ~2.2 TB due to early exit)

**This is the PRIMARY CAUSE of RAM explosion!**

---

## 7. L2 Hierarchical (Quadruple-Nested Unrolling) 🔴🔴

### Loop Structure
```python
def search_L2_morton_hierarchical_single(pos, mesh_gpu):
    morton_query = morton_encode_position_jax(pos, ...)

    elem_id_depth7 = -1
    found_depth7 = False

    # ===== DEPTH 7 SEARCH =====
    neighbor_prefixes_7 = get_26_neighbor_prefixes_jax(morton_query, depth=7, ...)

    # LOOP 1A: 27 octants at depth 7 (unrolled)
    for i in range(27):                                          # ← UNROLL 1A: 27 iterations
        active = not found_depth7
        neighbor_prefix = neighbor_prefixes_7[i]

        # Look up leaves
        first_leaf = prefix_start[prefix_idx]
        num_leaves = prefix_length[prefix_idx]

        octant_elem = -1
        octant_found = False

        # LOOP 2A: 8 leaves per octant at depth 7 (unrolled)
        for leaf_offset in range(8):                             # ← UNROLL 2A: 8 iterations
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves) & ...

            # LOOP 3A: search_in_leaf_global unrolls 8 elements
            result = search_in_leaf_global(pos, leaf_id, mesh_gpu)  # ← UNROLL 3A: 8 iterations
            improved = result >= 0
            octant_elem = jnp.where(improved, result, octant_elem)
            octant_found = octant_found | improved

        # Update
        elem_id_depth7 = jnp.where(octant_elem >= 0, octant_elem, elem_id_depth7)
        found_depth7 = found_depth7 | octant_found

    # ===== DEPTH 6 SEARCH (FALLBACK) =====
    elem_id_depth6 = -1
    found_depth6 = False

    neighbor_prefixes_6 = get_26_neighbor_prefixes_jax(morton_query, depth=6, ...)

    # LOOP 1B: 27 octants at depth 6 (unrolled)
    for i in range(27):                                          # ← UNROLL 1B: 27 iterations
        active = not found_depth6
        neighbor_prefix = neighbor_prefixes_6[i]

        # Look up leaves
        first_leaf = prefix_start[prefix_idx]
        num_leaves = prefix_length[prefix_idx]

        octant_elem = -1
        octant_found = False

        # LOOP 2B: 8 leaves per octant at depth 6 (unrolled)
        for leaf_offset in range(8):                             # ← UNROLL 2B: 8 iterations
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves) & ...

            # LOOP 3B: search_in_leaf_global unrolls 8 elements
            result = search_in_leaf_global(pos, leaf_id, mesh_gpu)  # ← UNROLL 3B: 8 iterations
            improved = result >= 0
            octant_elem = jnp.where(improved, result, octant_elem)
            octant_found = octant_found | improved

        # Update
        elem_id_depth6 = jnp.where(octant_elem >= 0, octant_elem, elem_id_depth6)
        found_depth6 = found_depth6 | octant_found

    # Return depth-7 if found, else depth-6
    return jnp.where(found_depth7, elem_id_depth7, elem_id_depth6)
```

### Unrolling Analysis
**Depth 7:**
- Loop 1A (octants): `range(27)` → 27 unrolled
- Loop 2A (leaves): `range(8)` → 8 unrolled per octant
- Loop 3A (elements): `search_in_leaf_global` → 8 unrolled per leaf
- **Subtotal:** 27 × 8 × 8 = **1,728 element checks**

**Depth 6:**
- Loop 1B (octants): `range(27)` → 27 unrolled
- Loop 2B (leaves): `range(8)` → 8 unrolled per octant
- Loop 3B (elements): `search_in_leaf_global` → 8 unrolled per leaf
- **Subtotal:** 27 × 8 × 8 = **1,728 element checks**

**Total unrolled iterations:** 1,728 + 1,728 = **3,456 element checks**

**Operations per element check:** ~150 ops

**Total per search:** 3,456 × 150 = 518,400 ops
**Total per particle (if L0+L1 fail):** 5 × 518,400 = 2,592,000 ops
**Total vmapped (20% L2 trigger):** 225K × 0.2 × 2.592M = **116.64B ops**

**XLA Graph Expansion:**
```
vmap[225K]
  ├─ [DEPTH 7]
  │   └─ for octant in [0..26]:                  ← 27 unrolled
  │       └─ for leaf in [0..7]:                 ← 8 unrolled
  │           └─ search_in_leaf_global
  │               └─ for elem in [0..7]:         ← 8 unrolled
  │                   └─ point_in_tet_gpu        ← 50 ops
  └─ [DEPTH 6]
      └─ for octant in [0..26]:                  ← 27 unrolled
          └─ for leaf in [0..7]:                 ← 8 unrolled
              └─ search_in_leaf_global
                  └─ for elem in [0..7]:         ← 8 unrolled
                      └─ point_in_tet_gpu        ← 50 ops
```
**Graph nodes:** 225K × (27 × 8 × 8 + 27 × 8 × 8) × 50 = **11.664T nodes** (11.664 trillion!)
**Memory:** 11.664T × 100 bytes = **1,166.4 TB** (masked to ~11.7 TB due to early exit)

**This is the WORST OFFENDER - causes catastrophic RAM explosion!**

---

## 8. L2 Enhanced (5×5×5 Tier System) 🔴🔴

### Loop Structure
```python
def search_L2_morton_neighbors_enhanced(pos, mesh_gpu):
    # ===== TIER 1: 3×3×3 search =====
    elem_id = search_L2_morton_neighbors_single(pos, mesh_gpu)  # ← 648 unrolled
    found_3x3x3 = elem_id >= 0

    # ===== TIER 2: 5×5×5 outer shell =====
    elem_id_extended = search_5x5x5_outer_shell(pos, mesh_gpu, elem_id, found_3x3x3)

    return jnp.where(found_3x3x3, elem_id, elem_id_extended)

def search_5x5x5_outer_shell(pos, mesh_gpu, current_elem, already_found):
    morton_query = morton_encode_position_jax(pos, ...)
    cx, cy, cz = decode_morton_prefix_jax(morton_query, depth=7)

    elem_id = current_elem
    found = already_found

    # LOOP 1: 125 octants (5×5×5 cube) (unrolled)
    for i in range(125):                                         # ← UNROLL 1: 125 iterations
        active = not found

        # Map i to (dx, dy, dz) offsets in [-2, 2]³
        dz = (i % 5) - 2
        dy = ((i // 5) % 5) - 2
        dx = ((i // 25) % 5) - 2

        # Filter: only outer shell (max_offset == 2)
        max_offset = max(abs(dx), abs(dy), abs(dz))
        is_outer = max_offset == 2                               # ← Only 98/125 active
        active = active & is_outer

        # Compute neighbor coordinates
        nx = clip(cx + dx, 0, max_coord)
        ny = clip(cy + dy, 0, max_coord)
        nz = clip(cz + dz, 0, max_coord)

        # Encode neighbor prefix
        neighbor_prefix = encode_morton_prefix_jax(nx, ny, nz, depth=7)

        # Look up leaves
        first_leaf = prefix_start[prefix_idx]
        num_leaves_in_prefix = prefix_length[prefix_idx]

        octant_elem = -1
        octant_found = False

        # LOOP 2: 3 leaves per octant (unrolled)
        for leaf_offset in range(3):                             # ← UNROLL 2: 3 iterations
            leaf_id = first_leaf + leaf_offset
            valid = (leaf_offset < num_leaves_in_prefix) & ...

            # LOOP 3: search_in_leaf_global unrolls 8 elements
            result = search_in_leaf_global(pos, leaf_id, mesh_gpu)  # ← UNROLL 3: 8 iterations
            improved = result >= 0
            octant_elem = jnp.where(improved, result, octant_elem)
            octant_found = octant_found | improved

        # Update
        elem_id = jnp.where(octant_elem >= 0, octant_elem, elem_id)
        found = found | octant_found

    return elem_id
```

### Unrolling Analysis
**Tier 1 (3×3×3):**
- Calls `search_L2_morton_neighbors_single` → **648 element checks**

**Tier 2 (5×5×5 outer shell):**
- Loop 1 (octants): `range(125)` → 125 unrolled (98 active outer shell)
- Loop 2 (leaves): `range(3)` → 3 unrolled per octant
- Loop 3 (elements): `search_in_leaf_global` → 8 unrolled per leaf
- **Subtotal:** 98 × 3 × 8 = **2,352 element checks** (outer shell only)

**Total unrolled iterations:** 648 + 2,352 = **3,000 element checks**

**Operations per element check:** ~150 ops

**Total per search:** 3,000 × 150 = 450,000 ops
**Total per particle (if L0+L1 fail):** 5 × 450,000 = 2,250,000 ops
**Total vmapped (20% L2 trigger):** 225K × 0.2 × 2.25M = **101.25B ops**

**XLA Graph Expansion:**
```
vmap[225K]
  ├─ [TIER 1: 3×3×3]
  │   └─ for octant in [0..26]:                  ← 27 unrolled
  │       └─ for leaf in [0..2]:                 ← 3 unrolled
  │           └─ for elem in [0..7]:             ← 8 unrolled
  └─ [TIER 2: 5×5×5 outer shell]
      └─ for i in [0..124]:                      ← 125 unrolled (98 active)
          └─ for leaf in [0..2]:                 ← 3 unrolled
              └─ for elem in [0..7]:             ← 8 unrolled
```
**Graph nodes:** 225K × (27×3×8 + 98×3×8) × 50 = **10.125T nodes** (10.125 trillion!)
**Memory:** 10.125T × 100 bytes = **1,012.5 TB** (masked to ~10.1 TB due to early exit)

**This is the SECOND WORST OFFENDER!**

---

## 9. search_in_leaf_global (Innermost Unrolling)

```python
def search_in_leaf_global(pos, leaf_id, mesh_gpu):
    start = mesh_gpu.leaf_start[leaf_id]
    length = mesh_gpu.leaf_length[leaf_id]

    found_elem = -1

    # LOOP: 8 elements (unrolled)
    for j in range(8):                                           # ← UNROLL: 8 iterations
        active = (found_elem == -1) & (j < length)

        # Get element ID
        idx = start + j
        elem_id = jnp.where(active, mesh_gpu.elem_ids_sorted[idx], 0)

        # Test point-in-tet
        inside = jnp.where(active, point_in_tet_gpu(...), False)

        # Update if inside
        found_elem = jnp.where(inside & active, elem_id, found_elem)

    return found_elem
```

**Unrolling:** `range(8)` → 8 unrolled copies
**Operations per iteration:** ~100 ops (50 for point-in-tet + 50 for masking)

**This is the innermost function called by all L2 methods.**

---

## Summary: Nested Loop Explosion

### L2 Radius (Works)
```
search_L2_global_morton_single
  └─ for leaf in [center, -2, -1, +1, +2]:          (5 leaves)
      └─ search_in_leaf_global
          └─ for elem in [0..7]:                    (8 elements)
```
**Total:** 5 × 8 = **40 unrolled iterations**

---

### L2 Neighbors (Crashes)
```
search_L2_morton_neighbors_single
  └─ for octant in [0..26]:                         (27 octants)
      └─ for leaf in [0..2]:                        (3 leaves)
          └─ search_in_leaf_global
              └─ for elem in [0..7]:                (8 elements)
```
**Total:** 27 × 3 × 8 = **648 unrolled iterations** (16× more than radius)

---

### L2 Hierarchical (Catastrophic)
```
search_L2_morton_hierarchical_single
  ├─ [DEPTH 7]
  │   └─ for octant in [0..26]:                    (27 octants)
  │       └─ for leaf in [0..7]:                   (8 leaves)
  │           └─ search_in_leaf_global
  │               └─ for elem in [0..7]:           (8 elements)
  └─ [DEPTH 6]
      └─ for octant in [0..26]:                    (27 octants)
          └─ for leaf in [0..7]:                   (8 leaves)
              └─ search_in_leaf_global
                  └─ for elem in [0..7]:           (8 elements)
```
**Total:** (27 × 8 × 8) + (27 × 8 × 8) = **3,456 unrolled iterations** (86× more than radius)

---

### L2 Enhanced (Catastrophic)
```
search_L2_morton_neighbors_enhanced
  ├─ [TIER 1: 3×3×3]
  │   └─ for octant in [0..26]:                    (27 octants)
  │       └─ for leaf in [0..2]:                   (3 leaves)
  │           └─ for elem in [0..7]:               (8 elements)
  └─ [TIER 2: 5×5×5 outer shell]
      └─ for octant in [0..97]:                    (98 octants, outer shell only)
          └─ for leaf in [0..2]:                   (3 leaves)
              └─ for elem in [0..7]:               (8 elements)
```
**Total:** (27 × 3 × 8) + (98 × 3 × 8) = **3,000 unrolled iterations** (75× more than radius)

---

## Key Insight: Multiplicative Explosion

When vmapped over 225K particles with 5 RK4 stages:

| Method | Unrolled | XLA Nodes | RAM |
|--------|----------|-----------|-----|
| Radius | 40 | 225K × 5 × 40 = 45M | 4.5 GB |
| Neighbors | 648 | 225K × 5 × 648 = 729M | 73 GB |
| Hierarchical | 3,456 | 225K × 5 × 3,456 = 3.89B | 389 GB |
| Enhanced | 3,000 | 225K × 5 × 3,000 = 3.37B | 337 GB |

**But each XLA node contains ~100 operations, so multiply by 100×:**

| Method | Final RAM (est.) |
|--------|------------------|
| Radius | 450 GB → **90 GB** (80% L0+L1 success) |
| Neighbors | 7.3 TB → **2.2 TB** (80% L0+L1 success) |
| Hierarchical | 38.9 TB → **11.7 TB** (80% L0+L1 success) |
| Enhanced | 33.7 TB → **10.1 TB** (80% L0+L1 success) |

**This matches the observed behavior: 'radius' works, 'neighbors' crashes.**
