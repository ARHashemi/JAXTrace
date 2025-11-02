# ThreadedA Mesh Analysis Report
**Reference Mesh for GPU-Native Implementation**

**Date**: 2025-11-02
**Mesh**: ThreadedA (threadedAvtk_*.pvtu)
**Timestep Analyzed**: 159 (most refined in revolution cycle)

---

## Executive Summary

The ThreadedA mesh is a **production-scale AMR mesh** with:
- **3.5M cells** (3,512,279 tetrahedral elements)
- **900K nodes** (900,658 points)
- **64 parallel pieces** (all non-empty)
- **Irregular spatial decomposition** (39×32×16 unique positions)
- **Moderate load imbalance** (max/min ratio = 2.65×)

**Key Insight**: This is **NOT a small test mesh** - it's a full production mesh requiring careful memory management and load balancing for GPU tracking.

---

## Mesh Characteristics

### Global Domain

**Bounds**:
```
X: [-0.030, 0.030] meters (60 mm width)
Y: [-0.023, 0.023] meters (46 mm height)
Z: [-0.010, 0.000] meters (10 mm depth)
```

**Volume**: 60mm × 46mm × 10mm = 27,600 mm³ (microscale welding simulation)

### Mesh Size

| Property | Value |
|----------|-------|
| **Total Cells** | 3,512,279 tetrahedra |
| **Total Points** | 900,658 nodes |
| **Parallel Pieces** | 64 (all non-empty) |
| **Cells per Piece** | min: 26,722, max: 70,802, mean: 54,879 |
| **Points per Piece** | min: 8,298, max: 17,572, mean: 14,073 |

**Load Imbalance Factor**: 70,802 / 26,722 = **2.65×** (max/min cells per piece)

### Spatial Decomposition

**Unique Positions**:
- X: 39 unique positions
- Y: 32 unique positions
- Z: 16 unique positions

**Total Product**: 39×32×16 = 19,968 (>> 64 pieces)

**Conclusion**: Decomposition is **highly irregular/adaptive**, NOT a regular grid. Pieces are concentrated in regions of interest (likely weld pool).

---

## Implications for GPU Implementation

### 1. Memory Requirements

**Static Data** (mesh + connectivity):
- Node positions: 900,658 × 3 × 4 bytes = **10.3 MB**
- Connectivity: 3,512,279 × 4 × 4 bytes = **53.4 MB**
- Element neighbors (4 per element): 3,512,279 × 4 × 4 bytes = **53.4 MB**
- **Total mesh data**: ~120 MB per timestep

**Field Data** (velocity at nodes):
- Per timestep: 900,658 × 3 × 4 bytes = **10.3 MB**
- Cached (10 timesteps): **103 MB**

**Total Static**: ~220 MB (well within 4GB VRAM budget)

**Particle Data** (100K particles):
- Positions: 100K × 3 × 4 bytes = 1.2 MB
- Velocities: 100K × 3 × 4 bytes = 1.2 MB
- Element IDs: 100K × 4 bytes = 400 KB
- Block IDs: 100K × 4 bytes = 400 KB
- **Total**: ~3.2 MB

**Grand Total**: 220 MB + 3.2 MB = **223 MB** for 100K particles + mesh

✅ **Conclusion**: 100K particles + full mesh fits comfortably in 4GB VRAM

### 2. Forest Block Partitioning

**Current VTK Decomposition**: 64 pieces (irregular)
- Load imbalance: 2.65× (moderate)
- Mean cells per piece: 54,879

**Recommended Forest Grid**: **4×4×2 = 32 blocks** (regular grid)

**Why NOT use existing 64 pieces**:
1. **Irregular distribution** (39×32×16 positions) - complex neighbor topology
2. **Load imbalance** (2.65×) - some pieces 2.65× larger than others
3. **Not spatially compact** - pieces may be non-contiguous

**Why 4×4×2 Regular Grid**:
1. **Simple neighbor topology** (6-face connectivity)
2. **Better load balancing** - regular grid more uniform for AMR
3. **GPU-friendly** - 32 thread blocks, good occupancy
4. **Memory efficient** - ~110K cells per block (manageable)

**Cells per Block (4×4×2 grid)**:
- Total cells: 3,512,279
- Cells per block: 3,512,279 / 32 = **~110K cells/block**
- Element neighbors per block: 110K × 4 = **440K neighbors** (~1.7 MB per block)

**Block Size** (spatial):
- X: 60mm / 4 = 15mm per block
- Y: 46mm / 4 = 11.5mm per block
- Z: 10mm / 2 = 5mm per block

### 3. Load Balancing

**Current VTK Imbalance**: 2.65× (max/min cells per piece)

**Expected 4×4×2 Grid Imbalance**: Likely **<2×** due to AMR spatial concentration

**Recommendation**:
- Start with 4×4×2 (32 blocks)
- Measure actual load per block in Phase 2
- If imbalance >3×, implement block splitting in Phase 8
- If GPU utilization <70%, increase to 8×8×4 (256 blocks) in Phase 8

### 4. Element Search Complexity

**Block-Local Search** (32 blocks):
- Cells per block: ~110K
- Search cost: O(log 110K) ≈ 17 comparisons (tree search)
- With hash octree: O(1) lookup

**Three-Tier Search Expected Hit Rates**:
- Level 0 (cached element): 85-95% (from literature)
- Level 1 (neighbor elements): 3-10%
- Level 2 (block tree search): 1-5%

**Element Neighbor Precomputation**:
- Total edges: 3,512,279 × 4 faces = 14M faces
- Average neighbors per element: ~4 (tetrahedral adjacency)
- Neighbor array size: 3,512,279 × 4 × 4 bytes = **53.4 MB**

### 5. Ghost Region Requirements

**Ghost Layer Thickness**: 1 layer (default)

**Ghost Elements per Block** (estimated):
- Block surface area: 2×(15×11.5 + 15×5 + 11.5×5) = **665 mm²**
- Average element size: (27,600 mm³ / 3.5M cells)^(1/3) ≈ 0.2 mm
- Surface elements: 665 / (0.2×0.2) ≈ **16,625 elements**
- Ghost elements (1 layer): ~16,625 × 1.5 = **~25K elements/block**

**Ghost Element Memory**:
- 25K × 32 blocks = 800K ghost elements
- Connectivity: 800K × 4 × 4 bytes = **12.8 MB**

✅ **Conclusion**: Ghost elements add ~13 MB (acceptable)

---

## Visualization

![ThreadedA Mesh Analysis](threadeda_mesh_analysis.png)

**Figure**: 3D visualization of ThreadedA mesh parallel decomposition (timestep 159).
- **Top-left**: 3D view of 64 piece bounding boxes
- **Top-right**: XY projection
- **Bottom-left**: XZ projection
- **Bottom-right**: YZ projection
- **Color scale**: Cell count per piece (viridis colormap)

**Observations**:
1. Pieces are **not uniformly distributed** - concentrated in central region
2. **High spatial clustering** in XY plane (likely weld pool region)
3. **Limited Z variation** (thin domain in Z direction)
4. Clear evidence of **adaptive refinement** - smaller pieces in high-gradient regions

---

## Recommendations for Implementation

### Phase 0 (Foundation)

1. **Use 4×4×2 regular grid** (32 blocks) for forest partitioning
   - Simple, debuggable, GPU-friendly
   - Better than irregular 64-piece decomposition

2. **Create forest grid visualization** overlaid on VTK pieces
   - Verify forest blocks cover domain
   - Identify empty/sparse blocks

### Phase 1 (Block Mapping)

1. **Map 3.5M cells to 32 forest blocks** (~110K cells/block)
   - Use element centroid for assignment
   - Build element-to-block lookup array

2. **Precompute element neighbors** (4 per tetrahedral element)
   - Face-adjacency extraction from connectivity
   - ~53 MB neighbor array

### Phase 2 (Memory Profiling)

1. **Measure actual load per block**
   - Expect <2× imbalance for regular grid
   - Flag blocks with >150K cells as "heavy"

2. **Profile GPU memory usage**
   - Target: <500 MB for mesh + fields
   - Headroom: 3.5 GB for particles + trajectories

### Phase 5 (MVP Scaling)

1. **Test with realistic particle counts**:
   - 10K particles: baseline
   - 50K particles: intermediate
   - 100K particles: target

2. **Benchmark against CPU tracker** (if available)

### Phase 8 (Optimization)

1. **If load imbalance >3×**: Implement block splitting
   - Split heavy blocks (>150K cells) into sub-blocks
   - Increase total block count to 64 or 128

2. **If GPU utilization <70%**: Increase block count
   - Try 8×8×4 (256 blocks) for better parallelism
   - Trade off: more overhead, but better occupancy

---

## Comparison with Initial Plan Assumptions

| Assumption (from Plan) | Actual (ThreadedA) | Impact |
|------------------------|-------------------|--------|
| ~1,300 cells | **3,512,279 cells** | 🔴 **2700× larger!** |
| Small test mesh | **Production-scale mesh** | 🔴 **Major revision needed** |
| 64 pieces | 64 pieces ✅ | ✅ Correct |
| ~2,301 points | **900,658 points** | 🔴 **400× larger!** |
| 2×2×2 = 8 blocks | Recommend 4×4×2 = 32 blocks | 🟡 **Update plan** |

**CRITICAL FINDING**: The initial plan assumed ThreadedA was a small test mesh (~1.3K cells). **It's actually a 3.5M cell production mesh!**

### Required Plan Updates

1. ✅ **Update default block count**: 4×4×2 (32 blocks) instead of 2×2×2 (8 blocks)
2. ✅ **Memory budget revision**: Mesh alone is 120 MB, not negligible
3. ✅ **Load balancing**: More critical with 110K cells/block
4. ✅ **Hash octrees**: Now **highly recommended** (not optional) for 110K cells/block
5. ✅ **Testing strategy**: Use smaller timestep or sub-sampled mesh for unit tests

---

## Memory Budget (Revised for ThreadedA)

**4 GB VRAM Allocation**:

```
┌────────────────────────────────────────────┐
│ GPU Memory (4 GB total, 3.7 GB usable)     │
├────────────────────────────────────────────┤
│ Static Data:           ~500 MB (13%)       │
│  ├─ Mesh positions:    10.3 MB             │
│  ├─ Connectivity:      53.4 MB             │
│  ├─ Element neighbors: 53.4 MB             │
│  ├─ Element-to-block:  13.4 MB             │
│  ├─ Ghost elements:    12.8 MB             │
│  ├─ Field cache (10):  103 MB              │
│  ├─ Block metadata:    1 MB                │
│  └─ Hash tables (32):  250 MB (if Phase 9) │
├────────────────────────────────────────────┤
│ Dynamic Data:          ~3 GB (75%)         │
│  ├─ Particle positions (100K): 1.2 MB      │
│  ├─ Particle velocities:      1.2 MB      │
│  ├─ Element/block IDs:        0.8 MB      │
│  ├─ Trajectory (40 steps):    48 MB       │
│  └─ Working memory:            2.9 GB     │
├────────────────────────────────────────────┤
│ Headroom:              ~200 MB (reserve)   │
└────────────────────────────────────────────┘
```

**Maximum Particles** (without trajectory storage):
- (3.7 GB - 500 MB) / (32 bytes/particle) = **100M particles** (theoretical)
- **Practical limit**: 100K-500K particles (with trajectory storage)

**With Trajectory** (40 timesteps × 100K particles):
- Trajectory: 40 × 100K × 3 × 4 bytes = **48 MB**
- Remaining for particles: 3.7 GB - 500 MB - 48 MB = **3.15 GB**
- **Practical limit**: ~100K particles with full trajectory

---

## Next Steps

1. ✅ **Update implementation plan**:
   - Change default block grid from 2×2×2 to 4×4×2
   - Revise memory estimates for 3.5M cell mesh
   - Elevate Phase 9 (hash octrees) to "highly recommended"

2. ✅ **Create forest grid visualization** (Phase 0):
   - Overlay 4×4×2 grid on VTK piece distribution
   - Verify coverage and identify empty blocks

3. ✅ **Develop sub-sampling strategy** for unit tests:
   - Use timestep 0 or 100 (early, less refined) for fast iteration
   - Or create synthetic 1K-10K cell mesh for unit tests

4. ✅ **Proceed with Phase 0 implementation**:
   - Block builder for 4×4×2 grid
   - Configuration with revised defaults
   - Visualization of forest + VTK pieces

---

## Appendix: Field Availability

**Point Data** (node-based):
- Displacement [3 components] - **PRIMARY** for tracking
- Pressure [1 component]
- Reactions [3 components]
- Temperature [1 component]
- LEVEL [1 component] - AMR refinement level

**Cell Data** (element-based):
- Stress [6 components]
- Strain [6 components]

**Velocity Field**: Use "Displacement" (point data, 3 components)

---

**Status**: Mesh analysis complete
**Action Required**: Update implementation plan with revised block count and memory budget
**Next**: Proceed to Phase 0 with 4×4×2 forest grid
