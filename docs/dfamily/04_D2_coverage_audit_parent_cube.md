# Octree coverage audit — `D2_results`

Final step 1500; 6,735 particles with `ElementID < 0` out of 288,000.

Auditing a random sample of 150 (seed 42).

Mesh: `A2_34.pvtu` (pattern `A2_{timestep}.pvtu`, timestep 34)

Mesh: 1,921,303 nodes, 10,807,677 elements (881,568 duplicate nodes merged)

Octree (`parent_cube`): 5,986,797 cells, 14.2 elem/cell, max 95

Levels present: [8, 9, 10, 11, 12, 13, 14, 15] (canonical cell size per level, as the search uses)

## Result

| outcome | n | % | meaning |
|---|---:|---:|---|
| no containing element | 0 | 0.0 | genuinely outside the mesh — not a search bug |
| host in a VISITED cell | 150 | 100.0 | **SEARCH defect** — it should have been found |
| host only in NON-visited cells | 0 | 0.0 | **REGISTRATION COVERAGE defect** |
| host registered in NO cell | 0 | 0.0 | element dropped from the octree |

**Verdict:** The containing element IS in a cell the search visits, so the registration is adequate and the defect is in the SEARCH itself — traversal, level selection, or the point-in-tet tolerance.

Examples:

- particle 251677: true host element 10349623, in a VISITED cell
- particle 261832: true host element 3171517, in a VISITED cell
- particle 262391: true host element 3171098, in a VISITED cell
- particle 276111: true host element 3171810, in a VISITED cell
- particle 266831: true host element 3172099, in a VISITED cell

_Caveat: the audit rebuilds the octree from the mesh at a single timestep. If the mesh moves between timesteps, a particle may have been lost against a different mesh state than the one audited here._
