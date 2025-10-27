# GPU JAX Octree Element Search Strategy for AMR Tetrahedral Meshes  

This document consolidates the full design for a GPU‑native, JAX‑friendly, memory‑efficient element search in adaptive tetrahedral meshes using a two‑level octree. It details the data layout, traversal, candidate testing, JAX/XLA considerations, chunking to control compilation memory, integration steps to remove the redundant monolithic octree, and practical parameters with diagrams and pseudocode derived from your current system and measurements.  

## 1) Executive summary  

- Use the existing two‑level hierarchy: shared coarse octree for levels 0–5 and per‑timestep fine octree for levels 6–12, with element IDs stored at leaves only, and reuse identical fine structures across timesteps (97.5% reuse in your dataset).  
- Eliminate the redundant “third” monolithic octree from the legacy interpolator; it duplicates what the coarse+fine hierarchy already provides and consumes 5–8 GB RAM for <5% speed benefit.  
- Implement stackless, fixed‑shape traversal in JAX to a leaf, then evaluate a fixed‑size candidate list per leaf (K=32) without loops; select the first “inside” tetrahedron via a vectorized predicate. This avoids dynamic indexing in nested loops, the documented root cause of XLA compilation memory explosions.  
- Wrap the element search kernel in a chunked interface to cap XLA compile‑time memory, e.g., batches of 50–200 particles depending on your GPU; 100‑particle chunks brought compile memory to ~1.5 GB in your tests and fit under a 3 GB limit.  

## 2) Background and constraints  

- Mesh scale: up to 780,922 nodes and 3,048,900 tetrahedra per timestep.  
- Two‑level octree structure:  
  - Coarse (static, all timesteps): ~3,105 nodes, levels 0–5, ~0.5 MB.  
  - Fine (per timestep, reused): ~0.5 MB per unique structure; in your case, only 1 unique across 40 timesteps (97.5% reuse), total ~0.5 MB.  
- Legacy “third octree”: monolithic overlap/centroid hybrid with 483,261 nodes; ~5–8 GB runtime memory, ~25 GB with pure overlap; redundant with coarse+fine search capability.  
- JAX/XLA issue: first compiled interpolation call tried to allocate 7.68 GiB for 500 particles due to dynamic indexing patterns inside nested loops; runtime data otherwise matched expectations.  

## 3) Data model and SoA arrays (GPU‑friendly)  

Keep device arrays contiguous and static‑shaped. Reuse your existing shapes and caps where possible.  

- Coarse octree (static across timesteps)  
  - centers: float32 [Nc, 3]  
  - sizes or halfsizes: float32 [Nc] or [Nc, 3]  
  - levels: int32 [Nc]  
  - children: int32 [Nc, 8] with −1 indicating no child  
  - node_element_lists: int32 [Nc, K], K=32 cap  
  - node_element_counts: int32 [Nc]  
  - is_leaf: bool [Nc] or inferred from children/counts  
  - Memory today: ~0.54 MB with K=32.  
- Fine octree (per‑timestep, reused)  
  - Same arrays as coarse, but only for levels 6–12  
  - node_parents: int32 [Nf] for connectivity  
  - node_element_lists: int32 [Nf, K], K=32 cap  
  - node_element_counts: int32 [Nf]  
  - Memory per unique structure: ~0.51 MB; 97.5% reuse yields ~0.51 MB total.  
- Coarse→fine mapping  
  - coarse_leaf_to_fine_root: int32 [Nc], mapping a coarse leaf to the starting fine node for that leaf and timestep, enabling direct continuation of traversal at level 6.  
- Leaf→element fixed mapping (optional normalization)  
  - If you prefer a global flattened leaf index space, store leaf2elem_ids: int32 [L, K] and leaf2elem_mask: bool [L, K]. Your current per‑node fixed arrays already match this design (K=32), so this step is mostly organizational.  
- Element‑level precomputations  
  - connectivity: int32 [Ne, 4]  
  - positions: float32 [Nn, 3]  
  - Optional culling: element AABBs min/max: float32 [Ne, 3] each; already present in the legacy octree build and easy to compute once for direct mode.  
  - Optional algebra: per‑tet barycentric transform matrices or precomputed face planes to reduce per‑query work; not strictly required by your current code, but beneficial for vectorized, loop‑free testing.  

## 4) Traversal: stackless, fixed‑depth state machine  

Implement iterative descent using integer node IDs and octant bits computed from sign(p − center). Do this twice per query: once in the shared coarse tree, then continue in the fine tree for the active timestep.  

- Octant selection:  
  - bit_x = p.x ≥ c.x, bit_y = p.y ≥ c.y, bit_z = p.z ≥ c.z  
  - octant = bit_x + 2·bit_y + 4·bit_z  
- Transition:  
  - child = children[node_id, octant]  
  - if child == −1 or is_leaf[node_id], stop; else node_id = child  
- Depth limits:  
  - Coarse: up to level 5  
  - Fine: continue to level 12 (or until leaf criteria)  
- Measured traversal work:  
  - ~6 node checks (coarse) + ~7 (fine) = ~13 per query on your data.  

ASCII traversal sketch  
- Inputs: point p  
- State: node_id, level, done  
- Loop: compute octant(p, center[node]); next = children[node, octant]; stop if leaf or −1; else node ← next; level++  
- Exit: node_id is a leaf in the relevant tree  

## 5) Candidate evaluation: loop‑free, vectorized  

At the leaf, evaluate up to K=32 tetrahedra in one vectorized pass. This eliminates nested loops and dynamic indexing that caused XLA to allocate 7.68 GiB in your tests.  

- Gather candidates  
  - ids = node_element_lists[leaf_id]  → int32 [K]  
  - mask = arange(count) < count or a stored node_element_mask → bool [K]  
- Optional cull via AABB  
  - aabb_ok = all(amin[ids] ≤ p ≤ amax[ids], axis=1) → bool [K]  
- Point‑in‑tet test  
  - Either vectorized face‑plane tests or a vectorized barycentric transform for all K candidates.  
  - Compute inside flags and combine with mask and aabb_ok.  
- Pick first hit  
  - Convert inside flags to a masked index and use argmin to select the first true, else return −1.  

Why this matters  
- Your 7.68 GiB compile attempt arose from XLA expanding nested loops with dynamic gathers into worst‑case graphs. One fixed‑size gather per query, followed by pure arithmetic on K entries, yields stable, small HLO graphs.  

## 6) JAX implementation details  

Use static shapes, SoA arrays, and lax.while_loop or equivalent JAX control flow. Avoid Python control flow inside @jit sections.  

ASCII compute flow  
- p → traverse_coarse → coarse_leaf → map to fine_root  
- p → traverse_fine(fine_root) → fine_leaf  
- ids[K] → gather element data → vector inside tests → first_true → elem_id  

Pseudocode outline (search_one)  
- leaf = traverse(coarse) then traverse(fine)  
- ids = leaf2elem_ids[leaf]; mask = leaf2elem_mask[leaf]  
- aabb_ok = inside aabb for ids  
- bary = affine transform or face‑plane tests for ids  
- inside = combined mask  
- elem_id = first_true(ids, inside), default −1  

Static limits  
- MAX_DEPTH: constants for coarse and fine  
- K=32: matches your node_element_lists shape and tuning  

## 7) Chunking to cap XLA compilation memory  

Compile and run the kernel for small batches; concatenate results. Your measurements show that processing 100 particles per batch drops compile‑time memory from 7.68 GiB to ~1.5 GiB and fits under a 3 GB GPU limit.  

- Motivation: XLA materializes conservative worst‑case buffers with vmap across large N; chunking limits this per‑compile memory while keeping compute amortized across batches.  
- Suggested defaults by GPU memory  
  - 2 GB: chunk_size=50  
  - 3 GB: chunk_size=100  
  - 4 GB: chunk_size=150  
  - 6 GB: chunk_size=200.  
- Performance expectations  
  - First batch compiles (~10 s), subsequent batches run fast (0.01–0.1 s each)  
  - For 45k particles at chunk_size=100: ~33 s total interpolation time as measured guidance.  

Implementation hint  
- Use your planned create_jax_direct_interpolator_chunked wrapper, pad only the last batch to maintain shapes, slice results back, and concatenate.  

## 8) Remove the redundant monolithic octree  

Refactor to route element search through the existing coarse+fine octrees; the third octree is architecturally redundant, causes massive duplication of element references, and consumes gigabytes for negligible speed benefit.  

Two practical options  
- Option 1 (recommended): Use SharedOctree in the interpolator  
  - Within SharedOctreeFEMTimeSeriesField.sample_at_positions, traverse coarse→fine to a leaf, fetch candidates, and test elements, removing dependence on the legacy OctreeMeshOptimized structure.  
- Option 2: Skip base class octree construction  
  - Bypass the base class that instantiates the legacy octree and own the interpolation via the shared octree structure directly.  

Why this pays off  
- Memory: 5–8 GB → ~1 MB by removing duplication and only keeping coarse+fine structures with leaf‑only element storage.  
- Performance: number of node checks/candidates is effectively the same; the monolithic tree saves ~1 traversal step on average, an immaterial difference for particle workloads.  

## 9) Full reference pseudocode  

Below is a condensed, JAX‑friendly outline consistent with your chunking plan and two‑level octree architecture.  

Schematic data  
- Octree arrays: centers[N,3], sizes[N], children[N,8], is_leaf[N], node_element_lists[N,K], node_element_counts[N]  
- Mapping: coarse_leaf_to_fine_root[Nc]  
- Elements: connectivity[Ne,4], positions[Nn,3], optional aabb_min/max[Ne,3]  

Traversal descender (stackless)  
- state = (node, depth, done)  
- while not done and depth < MAX_DEPTH:  
  - c = centers[node]  
  - octant = bits(p ≥ c)  
  - child = children[node, octant]  
  - leaf_here = is_leaf[node] or (child < 0)  
  - node = leaf_here ? node : child  
  - depth += 1  
  - done = leaf_here  
- return node  

Element test (vectorized over K)  
- ids = node_element_lists[leaf]  
- valid = arange(K) < node_element_counts[leaf]  
- aabb_ok = all(aabb_min[ids] ≤ p ≤ aabb_max[ids], axis=1)  
- inside = vectorized point‑in‑tet test for ids  
- inside = inside & valid & aabb_ok  
- first = argmin(where(inside, arange(K), K + arange(K)))  
- elem_id = where(inside[first], ids[first], −1)  

Chunked batch wrapper  
- For i in range(0, N, chunk_size):  
  - pad last chunk to fixed size  
  - jit compile base kernel once per chunk_size  
  - run, unpad, append  
- concat results  

## 10) Diagrams  

High‑level pipeline  
- positions[N,3] → traverse coarse → coarse_leaf  
- → map coarse_leaf → fine_root → traverse fine → fine_leaf  
- → fetch node_element_lists[fine_leaf][K], counts[fine_leaf]  
- → gather tet data for ids → vectorized tests → first_true → elem_id  

Two‑level structure  
- Coarse (0–5): ~3,105 nodes, static, leaf‑only element lists, ~0.5 MB  
- Fine (6–12): reused per timestep, leaf‑only element lists, ~0.5 MB total in your case  

Monolithic legacy (to remove)  
- 483,261 nodes, 374,927 leaves, ~5–8 GB, marginal traversal savings, heavy duplication under overlap assignment, unnecessary because coarse+fine already provide candidate elements.  

## 11) Practical parameters  

- K (max elements per leaf): 32; split nodes if exceeded to keep candidate vectors small and kernels compact.  
- MAX_DEPTH: coarse up to 5, fine up to 12; compile‑time constants stabilize shapes.  
- chunk_size: 50–200 depending on GPU memory; 100 yielded ~1.5 GB compile memory in your tests.  
- Timestep cache: keep a small cache of per‑timestep velocity/mesh to amortize I/O; your system already does this effectively.  

## 12) Pitfalls and mitigations  

- Avoid nested dynamic indexing in loops  
  - Replace per‑candidate loops with a single K‑wide vectorized pass to prevent XLA from creating worst‑case graphs and huge buffers.  
- Avoid element duplication across octree levels  
  - Store elements at leaves only; the monolithic overlap‑based approach compounded duplication across levels and caused multi‑GB footprints.  
- Keep shapes static in @jit sections  
  - Ensure MAX_DEPTH and K are constants; use lax.while_loop and masks, not Python control flow, inside traced code paths.  
- Control compile memory with chunking  
  - Use your chunked interpolator wrapper; this brought compile memory to ~1.5 GB for chunk_size=100.  

## 13) Integration plan and effort  

- Short term  
  - Implement the chunked wrapper and use it in SharedOctreeFEMTimeSeriesField; expose chunk_size in config; your plan and diffs are already drafted.  
- Medium term  
  - Replace monolithic octree usage with shared coarse+fine traversal in the interpolator; Option 1 or 2 as outlined; element search uses existing node_element_lists and counts from the shared/fine trees.  
- Expected gains  
  - Memory: 5–8 GB to ~1 MB for spatial structures; JAX compile memory to ~1.5 GB with chunking; total runtime footprint aligns with your measurements and expectations.  

## 14) What each octree node stores  

- Internal node  
  - center, size (or min/max), children[8], level  
- Leaf node  
  - node_element_lists[leaf, K] with node_element_counts[leaf]  
- Global mapping  
  - coarse_leaf_to_fine_root for per‑timestep fine traversal continuation  
- Why leaf‑only storage  
  - Prevents exponential duplication and keeps memory ~1 MB at your scales, versus the GBs seen in the monolithic approach.  

## 15) Why this matches your measurements  

- Two‑level traversal  
  - ~6 coarse + ~7 fine = ~13 node checks per query; ~11 candidates per leaf on average.  
- Memory  
  - Coarse ~0.54 MB + Fine ~0.51 MB (one unique) = ~1.04 MB total, confirmed by your logs.  
- Redundant monolithic octree  
  - 483k nodes, ~5–8 GB, previously 28M nodes and ~25 GB with pure overlap; negligible speed advantage; root cause of prior OOMs.  
- JAX compile memory  
  - 7.68 GiB observed at 500 particles for the looped dynamic‑index implementation; chunking to 100 particles expected ~1.5 GB compile memory and works under a 3 GB limit.  

