# Frozen-particle diagnostic — `D2_results`

Steps found: 151  (first 0, last 1500)

## 1. How many stop, and when

| population | count | % |
|---|---:|---:|
| dead on arrival (step 0->100) | 0 | 0.00 |
| froze later | 6,585 | 2.29 |
| frozen at final step | 6,585 | 2.29 |
| total particles | 288,000 | 100.00 |

## 2. Do frozen particles still have a host element?

_ElementID<0 is read as a failed host lookup. That reading assumes the run had no inlet wall; with `--inlet-wall`, pending-entry particles are forced to -1 while still outside the mesh bbox._

| group | n | ElementID < 0 (search failed) | ElementID >= 0 (velocity was zero) |
|---|---:|---:|---:|
| froze later | 6,585 | 6,585 (100.0%) | 0 (0.0%) |
| frozen (all) | 6,585 | 6,585 (100.0%) | 0 (0.0%) |
| still moving | 281,415 | 150 (0.1%) | 281,265 (99.9%) |

**Verdict for the late freezers:** 100.0% have no host element.
Dominated by **a failed host lookup** — the particles have
no host element, so the velocity field and level set are
exonerated. That does NOT yet say the search kernel is at
fault: the search can only find what the octree registers.

Separate the two causes before acting:

- `scripts/dfamily/registration_sweep.sh <CASE> --only aabb,parent_cube` — if `aabb` eliminates the loss,
  the cause is registration COVERAGE, not the search.
- `scripts/dfamily/audit_octree_coverage.py` — asks, per
  lost particle, whether the element that truly contains
  it is registered in a cell the 3x3x3 search visits.

The paper's Table T2 is the prior here: `aabb` is 100.00%
correct on all 10 cohort meshes while `centroid` drops to
99.64%, and the D meshes are 50-59% non-Kuhn. Widening
`ENHANCED_SEARCH_BAND` / `L0_SKIP_BAND` / `L2_NEIGHBORHOOD`
cannot find an element absent from the cells searched, so
reach for those only once coverage is ruled out.

## 3. Where they stop

Tool region (`LEVEL < 0`) from `A2_34.pvtu` (8 pieces sampled):

| depth (mm) | tool radius (mm) | nodes |
|---|---:|---:|
| -6 .. -4 | 2.49 | 14,345 |
| -4 .. -2 | 2.49 | 1,390 |
| -2 .. 0 | 6.73 | 15,110 |
| 0 .. 2 | 6.73 | 49,864 |

Late freezers by depth:

| depth (mm) | n | median r (mm) | inside tool |
|---|---:|---:|---:|
| -4 .. -2 | 662 | 8.60 | 0 (0.0%) |
| -2 .. 0 | 5,923 | 8.19 | 19 (0.3%) |

Radial histogram of late freezers (1 mm bins):

```
     6- 7 mm      86 #
     7- 8 mm   2,046 ########################
     8- 9 mm   4,251 ##################################################
     9-10 mm     202 ##
```

