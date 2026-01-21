Particle loss in your GPU Morton pipeline is consistent with the exact failure mode you describe: **single-assignment of each tetrahedron to one leaf (via centroid Morton code) + bounded leaf-neighborhood search** can miss the true containing element when (a) the element’s spatial support crosses leaf/octant boundaries and (b) the containing point lies in a different leaf band than the centroid’s leaf. Your documents describe this correctly and also correctly note that CPU octrees often avoid this by **multi-inserting** a cell into every intersected bucket/octant, which guarantees completeness at the cost of redundancy. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

That said, a few parts of your analysis are either too charitable to the current approach (it’s “an octree” but not “octree-aware” in the sense you want), or too quick to blame floating-point tolerance as the “most likely” root cause without first proving the geometric miss-rate. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)

## 1) Is your method “octree-aware Morton traversal” like BLD13?

Not really; it is closer to a **linear octree / SFC bucketization** than the “pass through octants” style shown for voxel octrees in BLD13. [graphics.cs.kuleuven](https://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/BLD14OCCSVO_paper.pdf)

- BLD13’s Figure-1 intuition (Morton order corresponding to a traversal through octants) is about constructing and traversing a **hierarchical voxel octree** where primitives/voxels are naturally attached to nodes by spatial containment, and the traversal is inherently about visiting spatial nodes. [graphics.cs.kuleuven](https://graphics.cs.kuleuven.be/publications/BLD13OCCSVO/)
- Your runtime search does not traverse octree nodes; it maps a point to a leaf (via prefix table) and then does a **band search in leaf-index space** (±radius leaves in the *sorted Morton list*). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)
- That leaf-band is not equivalent to “neighbor octants” in 3D; your own document explains why consecutive Morton leaves can be far apart and why mixed-depth leaves break “radius along Morton” as a spatial neighborhood notion. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)

So: it’s fair to call your structure a **linear octree with prefix-table acceleration**, but it is not an “octant neighbor traversal” method in the geometric sense; it is an **SFC-ordered candidate enumeration** method. [refubium.fu-berlin](https://refubium.fu-berlin.de/bitstream/handle/fub188/6956/2_kap2.pdf?sequence=3&isAllowed=y)

## 2) Should you assign elements to leaves using bounding boxes instead of centroids?

Yes—*if the goal is 100% completeness*, centroid-only assignment is the wrong invariant. Your analysis is right that CPU locators (e.g., bucket/octree-based) achieve completeness by inserting each cell into all spatial bins it overlaps. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

However, doing “full VTK-style multi-insert into all intersected leaves” is not the only option, and in GPU context it’s usually too expensive in bandwidth unless carefully bounded. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

Better options than pure centroid:

### Option A (most standard): multi-insert by AABB overlap, but at a **coarse depth**
Insert each tet into buckets at a *coarse* uniform depth \(D\) (e.g., 6–8), not into your fine adaptive leaves. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)
- Each tet’s AABB spans only a few coarse buckets in most cases, bounding duplication.
- Query: point → coarse bucket → test that bucket’s candidate list.  
This is basically “uniform grid / coarse octree bins + exact point-in-tet”, a very common GPU-friendly compromise. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

### Option B: store a **short “halo list”** per leaf
Instead of inserting elements into many leaves, expand each leaf’s candidate set by unioning neighbor-leaf element lists (e.g., the 26 spatial neighbors at the same depth, or “parent + siblings” across depths). This targets exactly your “element spans boundaries” problem while bounding redundancy. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)
Your current leaf-band approach is *not* this, because it uses Morton adjacency not spatial adjacency. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)

### Option C: keep centroid assignment but fix the *search invariant*
If you keep centroid-only storage, you must guarantee that your fallback search radius covers the maximum possible centroid-to-point leaf displacement for any point inside any element. That bound can be huge on graded adaptive meshes, which is why this approach tends to be fragile. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

## 3) Can you exploit the intrinsic octree nature of the Kuhn adaptive mesh?

Yes, and this is the most promising direction if you want a principled “no element spans leaves” guarantee.

Your mesh is described as Kuhn tetrahedra on an octree-adaptive Cartesian refinement, with edges aligned to X/Y/Z. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/9c86d3a8-3d38-43f3-8c06-7207a10fbf2a/diagnose_aa_tolerance.log)
If the mesh truly comes from a forest-of-octrees / 2:1 balanced adaptive grid, then there exists a natural parent cell (“octant/cube cell”) that each tetrahedron lives in, and tetrahedra do not cross that parent cell. In that case:

- Build your search index over the **mesh’s own octants/cells**, not over element centroids. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)
- Store each tet under its parent octant id (and tet-local id).  
Then point location becomes:
1) point → octant id via integer quantization (or Morton) at the *mesh’s refinement level*,  
2) test only the 4–6 Kuhn tets inside that octant.

This yields the property you want: **no tet crosses an octant**, so no “element extension between leaves” at all (assuming the parent octants match mesh generation). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)

This is also the cleanest way to turn “Morton/octree” from being a generic spatial heuristic into a mesh-aligned exact structure.

The caveat: this requires you to have (or reconstruct) the octree cell ownership for each tet (cell coordinates + level). If you only have an unstructured list of tetrahedra, reconstructing cell ids robustly can be nontrivial but is feasible given axis alignment and Kuhn structure. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/9c86d3a8-3d38-43f3-8c06-7207a10fbf2a/diagnose_aa_tolerance.log)

## 4) Critical review of your document’s proposed causes/solutions

Two main critiques:

### A) “Most likely root cause is float32 tolerance” is not yet proven
You argue the missing 2–4% is likely tolerance (1e-10 too tight) and should be 1e-6. It’s true that 1e-10 is far below float32’s practical error budget and will reject boundary points. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)
But your own document also gives a very plausible geometric miss case (centroid leaf too far beyond radius), which can easily dominate when dt is large and mesh grading is extreme. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

Before committing to the tolerance narrative, separate them experimentally:
- Log “not found” points and test them on CPU with a robust locator. If CPU finds an element, compare the element’s centroid leaf distance to the query leaf—this directly measures the geometric miss mechanism you described. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)
- Also check if the CPU-found element would have passed your GPU point-in-tet with a looser tolerance (that isolates tolerance failures). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

### B) The “radius along Morton leaves” concept is structurally weak on adaptive depth
Your explanation that larger radius doesn’t necessarily help due to Morton discontinuities and mixed-depth ordering is correct. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)
That implies that solutions that merely “increase radius schedule” are not conceptually satisfying (and you said you don’t like them)—and you’re right not to like them, because they don’t fix the invariant.

A real fix changes at least one invariant:
- candidates are complete for the region (multi-insert / halo), or
- the search neighborhood is spatial (octant neighbors), not Morton-adjacent, or
- the index matches the mesh’s intrinsic octree (cell-based lookup).

## Direct answers

1) **Is it like BLD13 Fig.1?**  
It uses Morton and octant prefixes, but the runtime search is not octant traversal; it’s leaf-band enumeration on the Morton-sorted list, which is a different beast. [graphics.cs.kuleuven](https://graphics.cs.kuleuven.be/publications/BLD14OCCSVO/BLD14OCCSVO_paper.pdf)

2) **Bounding boxes instead of centroids?**  
Yes for correctness. Full multi-insert into all intersected fine leaves is expensive; prefer coarse-depth binning or halo/neighbor expansion. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/f21359ca-0aa0-4162-befa-3b9880383921/GPU_OCTREE_POINT_LOCATION_COMPREHENSIVE_ANALYSIS.md)

3) **Match octants with mesh’s intrinsic octree?**  
Yes—index by the mesh’s parent octants/cells and store only the few Kuhn tets per cell; then no element spans leaves by construction. This is the most principled way to get CPU-like completeness without VTK-style explosion. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/9c86d3a8-3d38-43f3-8c06-7207a10fbf2a/diagnose_aa_tolerance.log)

4) **You dislike some suggested fixes**  
That’s reasonable: “increase radius,” “reduce dt,” or “accept 98%” are operational patches, not invariant fixes. The invariant fixes are: (i) multi-insert at some level, (ii) halo lists, or (iii) mesh-intrinsic octree cell lookup. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70091334/01c5cb8e-f377-4611-ae52-b57024e85512/MORTON_SEARCH_EXPLAINED.md)

Plain question to proceed: is the mesh generator able to provide per-tet parent octant coordinates/level (or can it output them)? If yes, the intrinsic-octree lookup becomes straightforward.
