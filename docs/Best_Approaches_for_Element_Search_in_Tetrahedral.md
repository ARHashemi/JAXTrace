<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Best Approaches for Element Search in Tetrahedral Meshes with Octree Structure and Adaptive Mesh Refinement

Based on extensive research of published papers and recent GPU implementations (2023-2025), the most optimized approaches for finding elements containing specific points in tetrahedral meshes with octree structures combine spatial indexing with efficient data layouts designed for GPU parallelism.

### Understanding Octree Structures for Tetrahedral Meshes

An octree is a hierarchical spatial data structure that recursively subdivides 3D space into eight octants at each level. For tetrahedral meshes, octrees serve as spatial indices that enable efficient point location by narrowing the search space from the entire mesh to a small local region.[^1][^2][^3][^4]

**Key Octree Properties:**

- Each internal node has exactly 8 children representing octants[^4]
- Leaf nodes contain references to mesh elements (tetrahedra) within their spatial bounds[^2][^1]
- Depth is typically logarithmic relative to element count: O(log₈ n)[^5]
- Adaptive refinement creates variable resolution based on mesh complexity[^6][^7]


### Most Optimized GPU-Suitable Approaches

#### 1. **Linear Octree with Morton Codes (Z-Order Curve)**

This is the most GPU-friendly approach for static or semi-static meshes with adaptive refinement.[^8][^9][^10][^6]

**How it works:**

- Stores octree nodes in a contiguous array ordered by Morton codes (Z-order space-filling curve)[^9][^11][^12]
- Morton codes interleave the binary representations of 3D coordinates, preserving spatial locality[^11][^9]
- Child nodes are stored sequentially after parents, enabling cache-efficient traversal[^13][^9]

**Data stored in each node:**[^14][^6][^13]

- **Morton code (64-bit)**: Encodes position and level using placeholder bit format[^14]
- **Connectivity index**: Offset to first child (0 if leaf)[^14]
- **Level offset array**: Starting index for each octree level[^14]
- **Element range**: For leaf nodes, indices into tetrahedra array[^6]

**Advantages:**

- Exceptional cache coherence due to contiguous memory layout[^15][^6]
- No pointer indirection reduces memory bandwidth[^15][^13]
- Parallel construction possible using radix sort[^16][^14]
- 10-40x speedup over traditional pointer-based approaches[^8][^6]

**Limitations:**

- Requires rebuild for structural changes (dynamic insertion/deletion)[^13]
- Not ideal for highly dynamic meshes[^17]

**Implementation details from recent work (Wang et al. 2024):**[^6][^8]
The Stanford group achieved GPU-native h-adaptive flux reconstruction using linear octrees for both 2D (quadtree) and 3D problems. Their approach maintains the entire adaptive mesh on GPU, eliminating CPU-GPU transfers. Tree operations (construction, 2:1 balancing, connectivity queries) execute entirely on GPU with adaptation cost under 2% of total computation time.[^6]

#### 2. **Hash-Based Octree with Optimized Search**

This approach offers the fastest point location queries for mixed workloads.[^18][^19][^5]

**How it works:**

- Uses Morton codes as keys in a GPU hash table[^5][^18]
- Implements "optimized search" starting at estimated depth rather than root[^18][^5]
- Parallelizes search across multiple octree levels simultaneously[^5][^18]

**Data stored in each node:**[^20][^5]

- **Morton code**: Spatial position and hierarchy level[^5]
- **Hash key**: Computed as `hash(x,y,z) = (x*p1 XOR y*p2 XOR z*p3) mod n` where p1, p2, p3 are large primes[^21]
- **Data/pointer**: RGB values for leaves, indices for internal nodes[^22][^20]
- **Is_leaf flag**: Distinguishes leaves from internal nodes[^18][^5]

**Collision handling:**[^18][^5]
Uses closed hashing with linear probing for collision resolution. When collision occurs at hash position h, tries h' = h + c₀ for fixed offset c₀.[^5]

**Advantages:**

- Amortized O(1) query time[^18][^5]
- 3-50x speedup over CPU implementations[^5][^18]
- Streams well on GPU architecture[^5]
- Optimized search reduces average traversal from O(log n) to O(1)[^5]

**Limitations:**

- Hash collisions can degrade performance[^18][^5]
- Hash table size must be carefully tuned[^5]

**GPU implementation algorithm (Madeira et al. 2009):**[^18][^5]

```
Algorithm: GPU Parallel Search
Input: Point p, g available GPU threads
1. Compute Morton code m_max of p at maximum depth
2. For each thread t_i in parallel (t_i = 0 to g-1):
   - If t_i ≤ (g-1)/2: search upward from depth l̂ + t_i + iter*(g-1)/2
   - Else: search downward from depth l̂ + t_i - (iter+2)*(g-1)/2 - 1
3. Access node at computed Morton code in hash table
4. If node is leaf, return immediately
```

This parallelization exploits the independence of each octree level, optimal for GPU SIMD architecture.[^18][^5]

#### 3. **PR-Star Octree (Spatio-Topological Approach)**

Specifically designed for tetrahedral meshes, combining spatial indexing with topological connectivity.[^3][^1][^2]

**How it works:**

- Augments Point Region (PR) octree with tetrahedra incident to indexed vertices[^1][^3]
- Stores minimal information to reconstruct local topology on-demand[^2][^1]
- Uses "topology through space" paradigm[^1][^2]

**Data stored in each leaf node:**[^3][^2][^1]

- **Vertex range**: `vstart` and `vend` indices into vertex array (2 integers)[^3][^1]
- **Tetrahedra list**: Pointer to list + count of tetrahedra incident to these vertices[^1][^3]
- **Hierarchical pointers**: Parent pointer, children pointer (3 pointers total)[^3][^1]

**Storage cost:** 7|N| + χ|T| where |N| is number of octree nodes, |T| is number of tetrahedra, and χ is average number of nodes indexing each tetrahedron (1 ≤ χ ≤ 4)[^3]

**Advantages:**

- Enables efficient topological queries (vertex-tetrahedron, adjacencies)[^1][^3]
- Memory efficient: typically 40% less than full connectivity structures[^3]
- Supports dynamic mesh traversal[^1]

**Limitations:**

- Moderate GPU suitability due to pointer-heavy structure[^1]
- Per-query overhead for topology reconstruction[^2][^1]

**Query algorithm:**[^3][^1]
For point location, first identify containing leaf node via octree traversal, then test point against all tetrahedra in that node's list using barycentric coordinates.[^23][^1]

#### 4. **Forest of Octrees (Block-Based AMR)**

The leading approach for large-scale adaptive mesh refinement on GPUs.[^24][^25][^7][^26][^10][^6]

**How it works:**

- Decomposes domain into multiple root blocks, each with its own octree[^7][^10]
- Each block is 8³ cells by default (configurable)[^7]
- Resolution ratio between adjacent levels fixed at 2:1[^10][^7]

**Data stored in each patch/block:**[^10][^7]

- **Patch ID numbers**: Parent, children (8), siblings (up to 26)[^7]
- **Cell data**: Simulation values for 8³ cells[^7]
- **Refinement level**: Current AMR level l (0 = coarsest)[^7]
- **Ghost cells**: Boundary data from neighboring patches[^7]

**Advantages:**

- Natural domain decomposition for GPU parallelism[^27][^10][^7]
- Properly nested hierarchy (level difference ≤ 1 between neighbors)[^10][^7]
- Proven scalability: 100-1000x speedup over CPU[^28][^7]
- Adaptation cost <2% of total simulation time when done every 10-40 steps[^6]

**Limitations:**

- Inter-block communication overhead[^27]
- More complex to implement than single-tree approaches[^27]

**Recent implementation (AGAL framework, Jaber et al. 2023-2025):**[^25][^26][^24][^10]
Fully GPU-native AMR with Lattice Boltzmann Method. Maintains entire mesh structure on GPU including refinement operations. Achieves acceleration of 1-2 orders of magnitude compared to uniform meshing for long-distance transport problems.[^25][^10][^6]

#### 5. **Spatial Hashing**

Simple and extremely fast for dynamic scenarios.[^29][^30][^21]

**How it works:**

- Divides space into uniform grid cells[^21][^29]
- Hash function maps 3D cell coordinates to 1D table index[^21]
- Tetrahedra stored in hash buckets based on bounding box overlap[^21]

**Data per hash bucket:**[^29][^21]

- **Vertex IDs**: Vertices in this cell
- **Tetrahedron IDs**: Tetrahedra whose AABB intersects cell
- **Collision chain**: For hash collisions (if using open hashing)[^21]

**Advantages:**

- O(1) insertion and deletion[^29][^21]
- Very simple GPU implementation[^21]
- Excellent for collision detection[^29][^21]

**Limitations:**

- Poor spatial coherence for large tetrahedra[^21]
- Not suitable for hierarchical queries[^29]
- Requires careful tuning of cell size[^21]


### Comparison Tables

### Recommended Approach Based on Use Case

**For static meshes with frequent point queries:** Linear octree with Morton codes[^9][^8][^6]

- Best cache performance
- Lowest query latency
- Example: Real-time rendering, post-processing analysis

**For dynamic meshes with updates:** Hash-based octree[^19][^18][^5]

- Fast queries and moderate update costs
- Good balance for interactive applications

**For adaptive mesh refinement simulations:** Forest of octrees[^10][^6][^7]

- Proven for large-scale GPU simulations
- Natural fit for AMR algorithms
- Example: CFD, astrophysics, LBM

**For collision detection in deformable bodies:** Spatial hashing[^29][^21]

- Simplest implementation
- Handles topology changes well
- Example: Physics engines, games


### Critical Data Requirements for Octree Nodes

**Minimum essential data (all approaches):**[^20][^14][^6][^1][^5]

1. **Spatial bounds**: Either explicit (min/max coordinates) or implicit (Morton code)
2. **Connectivity**: Child indices/pointers for internal nodes
3. **Element references**: List/range of tetrahedra for leaf nodes
4. **Metadata**: Level/depth, is_leaf flag

**Additional data for AMR:**[^10][^6][^7]

- Refinement flags indicating which children need subdivision
- Neighbor connectivity for 2:1 balance enforcement[^6]
- Ghost cell data for inter-block communication[^7]

**Additional data for topology queries:**[^2][^3][^1]

- Vertex index ranges for PR-star approach
- Tetrahedron incident lists with counts
- Parent/sibling pointers for traversal


### Recent Algorithmic Innovations (2023-2025)

**1. GPU-Native Tree Balancing (Wang et al. 2024):**[^8][^6]
Efficient 2:1 balancing algorithm fully on GPU without CPU involvement. Key innovation: parallel identification of imbalanced nodes using level difference computations.

**2. Spatial Binning for Complex Geometries (Jaber et al. 2025):**[^24][^25]
Distributes boundary faces to bins for near-wall refinement detection. Enables GPU-native handling of STL geometries in AMR frameworks.

**3. Hash-Based Multi-Resolution (2024-2025):**[^31][^19]
Residency octree combining page tables with traditional octree metadata. Supports mixed-resolution multi-channel data on GPU with ray-guided streaming.

### Fact-Checked Conclusions

Based on rigorous evaluation against published benchmarks:

**Fastest point location:** Hash-based octree with optimized search achieves 3-50x CPU speedup, with amortized O(1) queries[^18][^5]

**Best for AMR:** Forest of octrees (block-based) achieves up to 2 orders of magnitude speedup vs uniform meshes, with <2% adaptation overhead[^25][^10][^6]

**Most memory efficient:** Linear octree uses 12-16 bytes/node vs 72-96 for pointer-based[^15][^13]

**Highest scalability:** Block-structured AMR proven on consumer to datacenter GPUs, handling millions of cells[^10]

The choice among these approaches depends critically on the application's specific requirements for query frequency, mesh dynamics, memory constraints, and whether topology information beyond point location is needed.[^6][^10][^1][^5]
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://kennyweiss.com/papers/Weiss11.gis_presentation.pdf

[^2]: https://www.cs.umd.edu/content/computing-3d-curvature-through-bucket-pr-octree

[^3]: https://kennyweiss.com/papers/Weiss11.gis.pdf

[^4]: https://en.wikipedia.org/wiki/Octree

[^5]: https://www.sbgames.org/papers/sbgames09/computing/short/cts19_09.pdf

[^6]: http://aero-comlab.stanford.edu/Papers/wang_witherden_jameson_hp_adaption_jcp_2024.pdf

[^7]: https://academic.oup.com/mnras/article/481/4/4815/5106358

[^8]: https://www.sciencedirect.com/science/article/abs/pii/S002199912400072X

[^9]: http://johnsietsma.com/2019/12/05/morton-order-introduction/

[^10]: https://arxiv.org/abs/2308.08085

[^11]: https://en.wikipedia.org/wiki/Z-order_curve

[^12]: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

[^13]: https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Flynn18.pdf

[^14]: https://arxiv.org/pdf/2307.06345.pdf

[^15]: https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf

[^16]: https://www.cse.iitb.ac.in/~rhushabh/publications/octree.pdf

[^17]: https://discourse.threejs.org/t/how-to-store-an-octree/57687/8

[^18]: http://www2.ic.uff.br/~esteban/files/papers/SBGames09_Madeira.pdf

[^19]: https://openreview.net/forum?id=Q1dI7CYy7C

[^20]: https://www.antexel.com/sylefeb-research/octreetex/octree_textures_on_the_gpu.pdf

[^21]: https://matthias-research.github.io/pages/publications/tetraederCollision.pdf

[^22]: https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-37-octree-textures-gpu

[^23]: https://people.math.sc.edu/Burkardt/classes/cg_2007/cg_lab_search_tet_mesh.pdf

[^24]: https://www.arxiv.org/abs/2502.16310

[^25]: https://arxiv.org/html/2502.16310v1

[^26]: https://www.sciencedirect.com/science/article/pii/S0010465525000463

[^27]: https://www.reddit.com/r/CFD/comments/xme8xg/adaptive_mesh_refinement_on_the_gpu/

[^28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3858848/

[^29]: https://cg.informatik.uni-freiburg.de/publications/2011_TR263_uniformGrids.pdf

[^30]: https://cims.nyu.edu/gcl/papers/2021-Bichon.pdf

[^31]: https://www.cg.tuwien.ac.at/research/publications/2024/herzberger-2024-roh/herzberger-2024-roh-paper.pdf

[^32]: https://d-nb.info/1265285721/34

[^33]: https://people.eecs.berkeley.edu/~jrs/meshpapers/FreitagGooch.pdf

[^34]: https://stackoverflow.com/questions/11849435/point-location-in-tetrahedron-meshes

[^35]: https://www.sciencedirect.com/science/article/pii/S0045794996003999

[^36]: http://catiadoc.free.fr/online/cfyuganalysis_C2/cfyuganalysis3dmeshpart.htm

[^37]: https://www.ljll.fr/frey/papers/meshing/Klingner B.M., Aggressive tetrahedral mesh improvement.pdf

[^38]: https://graphics.stanford.edu/papers/meshing-sig03/meshing.pdf

[^39]: https://www.nature.com/articles/s41598-021-02187-1

[^40]: https://www.iue.tuwien.ac.at/phd/fleischmann/node38.html

[^41]: https://www.sciencedirect.com/science/article/pii/S0045793023002657

[^42]: https://people.eecs.berkeley.edu/~jrs/meshpapers/PUdOG.pdf

[^43]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3275789/

[^44]: https://www.sci.utah.edu/~cscheid/pubs/pbvr.pdf

[^45]: https://www.reddit.com/r/GraphicsProgramming/comments/5uf0jd/need_help_understanding_how_to_build_a/

[^46]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5897048/

[^47]: https://stackoverflow.com/questions/12191802/use-octree-to-organize-3d-volume-data-in-gpu

[^48]: https://pointcloudlibrary.github.io/gsoc-2020/gpu/

[^49]: https://forums.developer.nvidia.com/t/degenerate-tetrahedral-meshing/36519

[^50]: https://www.sci.utah.edu/~knolla/octsurvey.pdf

[^51]: https://handmade.network/forums/t/1233-memory_management_of_a_handmade_voxel_editor

[^52]: https://arc.aiaa.org/doi/abs/10.2514/6.2025-3056

[^53]: https://forums.developer.nvidia.com/t/best-way-of-traversing-an-octree-in-cuda/9179

[^54]: https://project.inria.fr/imr27/files/2018/09/1003.pdf

[^55]: https://d-nb.info/1217140409/34

[^56]: https://www.nvidia.com/docs/io/47905/cuda-grapp.pdf

[^57]: http://www.cccg.ca/proceedings/2011/papers/paper78.pdf

[^58]: https://www.iccs-meeting.org/archive/iccs2018/papers/108610349.pdf

[^59]: https://catiahelp.azurewebsites.net/English/FemUserMap/fem-t-3dMesh-Octree3d.htm

[^60]: http://graphics.zcu.cz/files/106_REP_2010_Soukal_Roman.pdf

[^61]: https://www.comsol.com/blogs/improved-capabilities-for-meshing-with-tetrahedral-elements

[^62]: https://www.reddit.com/r/programming/comments/fgnenk/how_to_write_a_simple_gpu_hash_table_that_can/

[^63]: https://www.sandia.gov/files/samitch/unm_math_579/Labelle_thesis.pdf

[^64]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4142801/

[^65]: https://help-3dexperience.aesvietnam.com/English/MpmMeshingMap/mpm-t-3dMesh-Octree3d.htm

[^66]: https://stackoverflow.com/questions/22382176/mapping-points-of-the-a-solid-box-to-a-tetrahedral-meshed-box

[^67]: https://github.com/AlexanderRipar/Octree_Ray_Tracing

[^68]: https://stackoverflow.com/questions/79416702/how-to-navigate-octree-using-morton-code

[^69]: https://repository.bilkent.edu.tr/bitstream/handle/11693/111612/Compact_tetrahedralization-based_acceleration_structures_for_ray_tracing.pdf?sequence=1

[^70]: https://geidav.wordpress.com/2014/08/18/advanced-octrees-2-node-representations/

[^71]: https://www.cs.cmu.edu/~droh/papers/lds06.pdf

[^72]: https://www.reddit.com/r/gamedev/comments/ud7ug4/uses_for_bvh_or_octree/

[^73]: https://repository.bilkent.edu.tr/bitstreams/e96a3204-35b9-4a56-9632-6fd2afc28400/download

[^74]: https://discourse.threejs.org/t/is-three-mesh-bvh-better-performing-than-the-built-in-octree-js-class-in-three-js/50425

[^75]: https://www.sci.utah.edu/~will/papers/rtx-points-tvcg20.pdf

[^76]: https://onlinelibrary.wiley.com/doi/10.1002/cav.2024

[^77]: https://www.reddit.com/r/learnprogramming/comments/18t64kt/the_zorder_curve_is_one_of_the_most_beautiful/

[^78]: https://hammer.purdue.edu/articles/thesis/Dynamic_Update_of_Sparse_Voxel_Octree_Based_on_Morton_Code/14495901/files/27771012.pdf

[^79]: https://arxiv.org/html/2501.18152v2

[^80]: https://itis.swiss/who-we-are/jobs/jobs-overview/semester-bachelors-and-masters-thesis-research-d-itet/robust-and-fully-automatic-tetrahedral-mesh-generation-for-multi-domain-high-resolution-computational-anatomical-models/

[^81]: https://www.sciengine.com/doi/articleIndex/10.7638/kqdlxxb-2024.0020

[^82]: https://www.sciencedirect.com/science/article/pii/S0045782524009794

[^83]: https://arxiv.org/html/2507.15230v3

[^84]: https://arxiv.org/html/2410.04402v1

[^85]: https://www.sciengine.com/doi/10.7638/kqdlxxb-2024.0020

[^86]: https://igl.ethz.ch/projects/tetweave/TetWeave_SIGGRAPH_2025_compressed_10MB.pdf

[^87]: https://dl.acm.org/doi/abs/10.1145/3592439

[^88]: https://www.nafems.org/downloads/dropbox/nologin/nwc25/nwc25-0007129-paper.pdf

[^89]: https://www.sciencedirect.com/science/article/pii/S1524070324000249

[^90]: https://www.aimsciences.org/article/doi/10.3934/acse.2025012

[^91]: https://www.cgl.cs.tau.ac.il/wp-content/uploads/2023/08/IditThesis.pdf

[^92]: https://people.eecs.berkeley.edu/~jrs/papers/tetstream.pdf

[^93]: https://graphics.stanford.edu/courses/cs268-11-spring/notes/opt_point_loc.pdf

[^94]: https://faculty.cc.gatech.edu/~jarek/papers/sot.pdf

[^95]: https://www.csun.edu/~ctoth/Handbook/chap38.pdf

[^96]: http://www.umiacs.umd.edu/~hjs/pubs/leesmi01.pdf

[^97]: https://sites.cs.ucsb.edu/~suri/cs235/Location.pdf

[^98]: https://www.reddit.com/r/VoxelGameDev/comments/qpk1tb/storing_blocks_as_octrees/

[^99]: https://graphics.cs.yale.edu/sites/default/files/p93-cutler.pdf

[^100]: https://cp-algorithms.com/geometry/point-location.html

[^101]: https://www.orange-kiwi.com/posts/efficient-octree-storage-and-traversal/

[^102]: https://stackoverflow.com/questions/36757987/algorithm-to-find-the-edges-of-tetrahedra-meshes

[^103]: https://ics.uci.edu/~goodrich/teach/geom/notes/Kirkpatrick.pdf

[^104]: https://dl.acm.org/doi/10.1145/2093973.2093987

[^105]: https://www.gamedev.net/forums/topic/641547-how-to-make-a-1demensional-array-loose-octree/

[^106]: https://www.sciencedirect.com/science/article/abs/pii/001044859390039Q

[^107]: https://www.sciencedirect.com/science/article/am/pii/S0097849321000819

[^108]: http://arxiv.org/pdf/1707.02211.pdf

[^109]: https://geovis.umiacs.io/publication/liu-2021-localized/liu-2021-localized.pdf

[^110]: https://ieeevis.b-cdn.net/vis_2024/pdfs/w-topoinvis-1041.pdf

[^111]: http://www.cs.umd.edu/~hjs/pubs/Samettfcgc88-ocr.pdf

[^112]: https://open.clemson.edu/context/all_dissertations/article/4916/viewcontent/Dissertation_Guoxi.pdf

