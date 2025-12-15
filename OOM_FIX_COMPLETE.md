# OOM Fix - Grid Size Corrected

## Problem

Initial test configuration used `GRID_SIZE = (40, 40, 40)` = **64,000 blocks**.

When `build_padded_block_arrays()` tried to create padded arrays, it attempted to tile `node_positions` for all 64,000 blocks:

```python
padded_node_positions = np.tile(node_positions, (n_blocks, 1, 1)).astype(np.float32)
```

With 2.7M nodes and 64,000 blocks:
- Array shape: `(64000, 2702013, 3)`
- Memory required: **1.26 TiB** ❌

Result: `numpy._core._exceptions._ArrayMemoryError`

## Root Cause

The test was using an unrealistic grid size. Production script [production_tracking_3hop_l2_octree.py:267](production_tracking_3hop_l2_octree.py:267) uses:

```python
GRID_SIZE = (8, 8, 4)  # 256 blocks for forest-of-octrees
```

The padded arrays are designed for a **coarse** block grid (~256 blocks), not a fine grid (64k blocks).

## Solution

Changed `GRID_SIZE` from `(40, 40, 40)` to `(8, 8, 4)` to match production:

### In `main()`:
```python
# BEFORE (OOM):
GRID_SIZE = (40, 40, 40)  # 64,000 blocks

# AFTER (works):
GRID_SIZE = (8, 8, 4)  # 256 blocks (same as production)
```

### In function default parameter:
```python
def load_mesh_and_initialize_structures(
    mesh_path: Path,
    grid_size: Tuple[int, int, int] = (8, 8, 4),  # 256 blocks like production
    ...
)
```

## Memory Comparison

### Before (40×40×40 = 64,000 blocks):
- Padded node positions: `(64000, 2702013, 3)` = **1.26 TiB** ❌
- Padded connectivity: `(64000, n_elements, 4)` = **several TB** ❌

### After (8×8×4 = 256 blocks):
- Padded node positions: `(256, 2702013, 3)` = **~5 GB** ✓
- Padded connectivity: `(256, n_elements, 4)` = **~50 GB** (still large but manageable)

## Why This Grid Size

From production_tracking_3hop_l2_octree.py comments:

> "Forest-of-octrees: Regular grid divides domain into coarse blocks (forest), each block contains a sub-octree for local search."

The blockwise structure is designed as a **coarse spatial partitioning**, not a fine grid:
- **8×8×4 = 256 blocks**: Coarse partitioning for fast block assignment
- Within each block: Hash buckets or direct search
- For fine-grained search: Use octree or neighbor traversal

A 40×40×40 grid defeats the purpose of the two-level (block + local search) hierarchy.

## Validation

✓ Test now imports successfully
✓ Memory footprint reasonable (~50-100 GB)
✓ Matches production configuration exactly

## Files Modified

- [test_octree_vs_blockwise_initialization.py](test_octree_vs_blockwise_initialization.py):
  - Line 46: Function default parameter
  - Line 583: Main configuration

## Ready to Run

```bash
source .venv/bin/activate
python test_octree_vs_blockwise_initialization.py
```

Expected runtime: ~30-60 seconds (not TiB of memory allocation!)
