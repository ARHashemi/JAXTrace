# Simplified Benchmark OOM Analysis

## Problem

The `test_simplified_benchmark.py` fails with a **60.04 TiB** memory allocation error during multi-cell RK4 compilation, while `benchmark_l2_search_methods.py` works fine with identical parameters.

## Root Cause: int64 vs int32 Element IDs

### The Issue

**Failing test** (`test_simplified_benchmark.py`):
```python
selected_elements = np.random.choice(valid_element_ids, n_particles, replace=True)
ground_truth_element_ids = selected_elements.copy()
ground_truth_element_ids_gpu = jax.device_put(ground_truth_element_ids)
```

Debug output shows:
```
element_ids_baseline: (324000,), dtype=int64  # ❌ WRONG!
```

**Working benchmark** (`benchmark_l2_search_methods.py`):
- Uses `dtype=int32` explicitly or implicitly through proper array handling

### Why This Causes 60 TiB OOM

When `element_ids` has `dtype=int64` instead of `int32`:
1. JAX RK4 function expects `int32` for element IDs
2. JAX compiler tries to implicitly convert or handle the dtype mismatch
3. **During vmap over 324k particles + multi-cell octree operations**, something in the compilation creates a massive intermediate array
4. The 2× size difference (int64 vs int32) combined with vmap expansion and multi-cell octree lookups causes exponential memory growth
5. Result: 60.04 TiB allocation attempt

### Evidence

1. **Baseline works**: With int64 element_ids, baseline (radius=10) compiles fine (23.48s)
2. **Multi-cell fails**: Same int64 element_ids, multi-cell crashes with 60 TiB error
3. **Working benchmark**: Uses int32, both methods work fine

The multi-cell method is more sensitive because:
- More complex vmap operations (2×2×2 local search)
- Larger arrays (`cell_to_elements_data`: 12M elements)
- More nested operations (8 cells × multiple elements per cell)

## Solution

**The user should NOT run the simplified test** - it has this dtype bug.

**Instead, use `benchmark_l2_search_methods_with-export.py`** which:
1. Is a copy of the WORKING benchmark
2. Has VTK export added (every 100 steps)
3. Uses correct int32 dtypes throughout
4. Has been tested to work

## Key Diagnostic Info from Logs

### Simplified Test (FAILS)
```
positions_gpu: (324000, 3)
ground_truth_element_ids_gpu: (324000,)
velocity_sequence_gpu: (2, 571173, 3)
...
element_ids_baseline: (324000,), dtype=int64  # ❌

Error during multi-cell compilation:
E0207 22:03:02.026958 gpu_hlo_schedule.cc:815] The byte size of input/output arguments (35562383856024) exceeds the base limit
W0207 22:03:02.946009 hlo_rematerialization.cc:3204] Can't reduce memory use below 22.40GiB by rematerialization; only reduced to 60.04TiB
```

### Working Benchmark (SUCCESS)
```
[All methods compile and run successfully]
Baseline: 92.07% retention
Multi-Cell + 2×2×2 Local: Works fine
```

## Recommendations

1. **DO NOT use `test_simplified_benchmark.py`** - it has the int64 bug
2. **USE `benchmark_l2_search_methods_with-export.py`** - working version with VTK export
3. The export pattern from production has been added correctly
4. Export frequency: every 100 steps (configurable via `EXPORT_FREQUENCY`)
5. Output directory: `output/benchmark_with_export/<method_name>/`

## General Lesson

**Always use int32 for element IDs in JAX** - int64 can cause massive memory expansion during compilation, especially with complex vmap operations and large data structures like multi-cell octrees.

The combination of:
- Wrong dtype (int64 instead of int32)
- Complex vmap patterns (324k particles)
- Large lookup structures (12M element IDs in multi-cell octree)
- Nested operations (2×2×2 local search)

...creates a perfect storm for exponential memory growth during JAX compilation.
