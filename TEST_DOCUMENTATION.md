# Single-Particle Search Test Documentation

## Overview

Comprehensive test suite for validating single-particle search implementations against batch-level implementations.

**Test file:** [test_single_particle_search.py](test_single_particle_search.py)

## What the Test Validates

### 1. Correctness (Element-by-Element Comparison)

For each search level (L0, L1, L2, Fused):
- Compares batch implementation results vs single-particle implementation results
- Checks if results match using `jnp.allclose()`
- Reports number of mismatches (should be 0)
- Verifies hit rates match between implementations

**Pass criteria:** `match = True` and `mismatches = 0` for all levels

### 2. Accuracy (Point-in-Tet Validation)

For each search level:
- Randomly samples found particles (500-1000 particles)
- Validates each particle is actually inside reported element using `point_in_tet_jax`
- Calculates accuracy percentage

**Sample sizes:**
- L0: 500 particles from L0 hits
- L1: 500 particles from L1-only hits (excluding L0 hits)
- L2: 500 particles from L2-only hits (excluding L0+L1 hits)
- Fused: 1000 particles from all found particles

**Pass criteria:** Accuracy > 99.0% for all levels

### 3. Performance (Timing and Throughput)

For each search level:
- Measures batch implementation time
- Measures single-particle implementation time (with outer vmap)
- Calculates throughput (particles/second)
- Computes speedup ratio (batch time / single time)
- Tracks GPU memory usage (before/after each test)

**Metrics reported:**
- Time in milliseconds
- Throughput in particles/second
- Speedup multiplier
- GPU memory delta

### 4. Hit Rate Breakdown

Tracks multi-level search effectiveness:
- **L0 hits:** Particles found in cached element
- **L1-only hits:** Particles found by multi-hop search (excluding L0)
- **L2-only hits:** Particles found by octree (excluding L0+L1)
- **Cumulative coverage:** Percentage found at each level

## Test Output Structure

### Per-Level Output (L0, L1, L2)

```
================================================================================
TEST X: LX Search - Correctness, Accuracy, Performance
================================================================================

Correctness:
  Batch hits:    XXX/1000 (XX.X%)
  Single hits:   XXX/1000 (XX.X%)
  LX-only hits:  XX (excluding previous levels)
  Results match: True
  Mismatches:    0

Accuracy (point-in-tet validation, LX-only particles):
  Validated:     XXX particles
  Correct:       XXX/XXX (XX.X%)

Performance:
  Batch time:    XX.XX ms (XXX,XXX p/s)
  Single time:   XX.XX ms (XXX,XXX p/s)
  Speedup:       X.XX×
  GPU memory:    XXXX MB → XXXX MB (ΔXXX MB)
```

### Fused Search Output

```
================================================================================
TEST 4: Fused L0+L1+L2 Search - Complete Pipeline
================================================================================

Correctness:
  Fused hits:         XXX/1000 (XX.X%)
  Matches batch:      True
  Mismatches:         0

Accuracy (point-in-tet validation, all found particles):
  Validated:          1000 particles
  Correct:            XXX/1000 (XX.X%)

Performance:
  Fused time:         XX.XX ms (XXX,XXX p/s)
  GPU memory:         XXXX MB → XXXX MB (ΔXXX MB)
```

### Comprehensive Summary Tables

#### 1. Correctness Validation Table

```
Level        Batch Hits    Single Hits   Match     Mismatches
------------ ------------- ------------- --------- -----------
L0           XXX/1000      XXX/1000      True      0
L1           XXX/1000      XXX/1000      True      0
L2           XXX/1000      XXX/1000      True      0
Fused        XXX/1000      XXX/1000      True      0
```

#### 2. Accuracy Validation Table

```
Level        Validated     Correct       Accuracy
------------ ------------- ------------- ---------
L0           500           XXX           XX.X%
L1 (only)    500           XXX           XX.X%
L2 (only)    500           XXX           XX.X%
Fused (all)  1000          XXX           XX.X%
```

#### 3. Performance Comparison Table

```
Level        Batch Time    Single Time   Throughput (single)   Speedup
------------ ------------- ------------- --------------------- --------
L0           XX.XX ms      XX.XX ms      XXX,XXX p/s           X.XX×
L1           XX.XX ms      XX.XX ms      XXX,XXX p/s           X.XX×
L2           XX.XX ms      XX.XX ms      XXX,XXX p/s           X.XX×
Fused        N/A           XX.XX ms      XXX,XXX p/s           N/A
```

#### 4. Hit Rate Breakdown Table

```
Level        Hits          Miss Rate     Cumulative Coverage
------------ ------------- ------------- --------------------
L0           XXX           XX.X%         XX.X%
L1           XX            XX.X%         XX.X%
L2           XX            XX.X%         XX.X%
```

### Final Verdict

```
================================================================================
FINAL VERDICT
================================================================================
✓ ALL TESTS PASSED
  • Correctness: Single-particle implementations match batch versions
  • Accuracy: XX.X% point-in-tet validation
  • Performance: Speedups documented above

READY FOR INTEGRATION INTO RK4
```

OR

```
✗ TESTS FAILED
  Correctness issues:
    • LX results do not match batch implementation
  Accuracy issues:
    • LX accuracy: XX.X% (expected >99%)
```

## How to Run the Test

```bash
source .venv/bin/activate
python test_single_particle_search.py
```

**Note:** The test loads mesh data and octree from the production script, which may take 1-2 minutes to load.

## Test Configuration

**Particles:** 1,000 test particles
**Particle distribution:** Random positions in refined region
**Particle movement:** Small displacement (dt=0.001, velocity=0.01)
**L1 hops:** 5 hops (maximum neighborhood coverage)
**L2 octree depth:** 10 iterations
**Validation samples:**
- L0: 500 particles
- L1: 500 particles (L1-only)
- L2: 500 particles (L2-only)
- Fused: 1000 particles (all found)

## Expected Results

### Correctness
- All levels should match: `True`
- All mismatches: `0`

### Accuracy
- L0: >99% (typically 99.8-100%)
- L1: >99% (typically 99.5-100%)
- L2: >99% (typically 99.0-100%)
- Fused: >99% (overall accuracy)

### Performance
- Speedup: ~0.95-1.05× (essentially same performance)
  - This matches empirical findings from [test_jax_cond_early_exit.py](test_jax_cond_early_exit.py)
  - No performance improvement expected (JAX doesn't skip expensive branches)

### Hit Rates (Typical)
- L0: 85-95% (depends on time step size)
- L1: 8-14% (most particles that miss L0)
- L2: <1% (rare cases)
- Total coverage: >99%

## What the Test Does NOT Validate

1. **Early exit performance:** The test confirms single-particle implementations are correct, but does NOT expect performance improvements (as proven by [test_jax_cond_early_exit.py](test_jax_cond_early_exit.py))

2. **RK4 integration:** This only tests search functions in isolation, not integrated into RK4 time stepping

3. **Large-scale performance:** Test uses 1,000 particles; production uses 45,000-105,000 particles

4. **Memory scaling:** GPU memory tracking is included, but full memory profiling under load is not performed

## Pass/Fail Criteria

### PASS Requirements

All of the following must be true:
- ✓ L0 correctness: `l0_match = True`
- ✓ L1 correctness: `l1_match = True`
- ✓ L2 correctness: `l2_match = True`
- ✓ Fused correctness: `fused_match = True`
- ✓ L0 accuracy: `> 99.0%`
- ✓ L1 accuracy: `> 99.0%` (if L1-only particles exist)
- ✓ L2 accuracy: `> 99.0%` (if L2-only particles exist)
- ✓ Fused accuracy: `> 99.0%`

If all pass: "✓ ALL TESTS PASSED - READY FOR INTEGRATION INTO RK4"

### FAIL Scenarios

Any of the following cause failure:
- ✗ Any `match = False`
- ✗ Any `mismatches > 0`
- ✗ Any accuracy `<= 99.0%`

If failed: Detailed error report listing which tests failed and why

## Debugging Failed Tests

### If correctness fails (match = False):

1. Check element-by-element differences:
   ```python
   diff = element_ids_batch != element_ids_single
   print(f"Mismatches: {diff.sum()}")
   print(f"Mismatch indices: {jnp.where(diff)[0]}")
   ```

2. Examine specific mismatches:
   ```python
   idx = jnp.where(diff)[0][0]
   print(f"Batch: {element_ids_batch[idx]}")
   print(f"Single: {element_ids_single[idx]}")
   print(f"Position: {positions[idx]}")
   ```

### If accuracy fails (< 99%):

1. Check which particles are incorrectly assigned:
   ```python
   for i in incorrect_indices:
       pos = positions[i]
       elem_id = element_ids_single[i]
       print(f"Particle {i}: pos={pos}, elem={elem_id}")
       # Manually verify point-in-tet
   ```

2. Look for edge cases:
   - Particles on element boundaries
   - Degenerate tetrahedra
   - Numerical precision issues

### If performance is unexpectedly slow:

1. Check GPU memory:
   ```bash
   nvidia-smi
   ```

2. Check if JIT compilation is working:
   - First run should be slow (compilation)
   - Subsequent runs should be fast

3. Profile with JAX profiler:
   ```python
   jax.profiler.start_trace("/tmp/jax-trace")
   # Run test
   jax.profiler.stop_trace()
   ```

## Integration Next Steps

After tests pass:

1. **Implement velocity interpolation single-particle:**
   ```python
   def interpolate_velocity_single_particle(
       position, element_id, connectivity, node_positions, velocity_field
   ) -> velocity  # (3,)
   ```

2. **Implement single-particle RK4:**
   ```python
   def rk4_single_particle(
       position, element_id, dt, mesh_data, velocity_field
   ) -> (position_new, element_id_new)
   ```

3. **Test RK4 integration:**
   - Compare single-particle RK4 with batch RK4
   - Verify particle trajectories match
   - Measure performance

4. **Production deployment:**
   - Run on full 45k-105k particle load
   - Benchmark against current implementation
   - Validate retention rates and accuracy

## References

- [jaxtrace/gpu/search/single_particle_search.py](jaxtrace/gpu/search/single_particle_search.py) - Single-particle implementations
- [SINGLE_PARTICLE_IMPLEMENTATION_COMPLETE.md](SINGLE_PARTICLE_IMPLEMENTATION_COMPLETE.md) - Implementation documentation
- [test_jax_cond_early_exit.py](test_jax_cond_early_exit.py) - Empirical proof that lax.cond doesn't provide early exit
- [ARCHITECTURE_DECISION_FINAL.md](ARCHITECTURE_DECISION_FINAL.md) - Architecture analysis and performance expectations
