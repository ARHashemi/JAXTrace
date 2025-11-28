# OLD Code Archive

This folder contains deprecated/superseded code from JAXTrace GPU implementation.

**Created**: 2025-11-14
**Purpose**: Prevent confusion by separating working implementations from failed/deprecated attempts

## Structure

- `search_v1_v2/`: V1 (working but slow) and V2 (OOM crashes) search implementations
- `tests/`: Test files for deprecated APIs
- `initial_attempts/`: Early prototypes and failed approaches

## What's Deprecated

### V2 JAX vmap (FAILED - OOM)
- **Files**: `multi_level_search_v2.py`, `initial_assignment_v2.py`
- **Status**: OOM crash on ThreadedA (tried to allocate 9.8GB)
- **Reason**: Full vectorization over all particles × full mesh is too memory-intensive
- **Replaced by**: Phase 2 batched block-wise architecture

### V1 Serial Python Loop (WORKING but SLOW)
- **Files**: `multi_level_search.py` (original, not moved)
- **Performance**: 188 p/s on ThreadedA (1,000 particles)
- **Status**: Still used as baseline reference
- **Reason for keeping**: Correctness reference, still faster than Phase 2 until Phase 2 is optimized

### Deprecated Test Files
- **test_batch_processor_small.py**: API mismatch with V5 PaddedArrays
- **test_phase2_integration.py**: Early integration attempt

## Current Working Code

**Location**: `jaxtrace/gpu/`

### Phase 2 Batched Block-Wise (IN PROGRESS)
- **batching/**: batch_processor.py, batch_config.py, memory_utils.py, validation.py
- **search/block_search.py**: JAX-native block-wise search kernels
- **Status**: ~80% complete, real mesh testing pending

### Multi-Level Search Infrastructure (WORKING)
- **search/level0_cached.py**: L0 cached element search
- **search/level1_neighbors.py**: L1 neighbor search
- **search/level2a_light.py**: L2a light block search
- **search/level2b_heavy.py**: L2b heavy block hash search
- **search/level3_neighbor_blocks.py**: L3 neighbor block search
- **search/hash_bucket.py**: Morton Z-order spatial hashing

### Forest Structure (COMPLETE)
- **forest/**: Block grid, element assignment, padded arrays (V5)
- **Status**: Production-ready, used by all search implementations

## Performance Comparison

| Implementation | Throughput | Status | Notes |
|----------------|------------|--------|-------|
| V1 Serial | 188 p/s | Working | Baseline reference |
| V2 vmap | OOM crash | Failed | Too memory-intensive |
| Phase 2 Batched | Not measured | In progress | Expected 500+ p/s after optimization |

## References

- [BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md](../docs/gpu/BATCHED_BLOCKWISE_ARCHITECTURE_REFINED.md)
- [PHASE1_IMPLEMENTATION_STATUS.md](../docs/gpu/PHASE1_IMPLEMENTATION_STATUS.md)
- [STATUS_REPORT_ON_BATCHED_BLOCKWISE_PLAN.md](../docs/gpu/STATUS_REPORT_ON_BATCHED_BLOCKWISE_PLAN.md)
