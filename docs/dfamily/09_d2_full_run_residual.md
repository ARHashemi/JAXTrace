# D2 full run after the lattice fix: what is fixed, and what is not

**Run:** D2, 8000 steps, 288,000 particles, velocity 34..199,
`FAILED_SUBSTAGE=last_valid_vel`, workstation RTX 5090, 13 Oct 2026.
Exit code 0, 8,525 s wall.
Output: `/scratch/users/ali/outputs/D2_latticefix_20261013_100727_2156709`.

---

## 1. What the lattice fix achieved

| | baseline | this run |
|---|---|---|
| initial assignment | 17,481 unassigned | **288,000 / 288,000, zero unassigned** |
| first loss | step 670 | step 900 |
| throughput | ~0.5 step/s | **1.6 step/s** |
| final lost | — | 11,724 of 288,000 (**4.07%**) |

The lattice defect itself is independently verified as eliminated: on the same
2,000 previously-lost particles, all three mismatch categories measure zero and
90.4% are now hosted ([`07_root_cause.md`](07_root_cause.md) §5.2). Initial
assignment placing all 288,000 particles is consistent with that.

**But 4.07% are still lost, and they are a different mechanism.**

---

## 2. The residual is a second, distinct mechanism

Measured on the final state. Each line is a separate test.

| test | result | reading |
|---|---|---|
| brute-force containment, 1,500 sampled | **1,244 (82.9%) HAVE a containing element**; 256 have none | the majority are real search failures, not domain exits |
| position vs **true** mesh bbox | **11,724 of 11,724 inside** (0 beyond any wall) | they did not leave the domain |
| `escaped_ever` flag | set on all 11,724 | **the flag is unreliable** — it latches on any axis; do not use it to classify |
| inside the tool (`LEVEL < 0`) | **0.0%** of lost (vs 0.6% of survivors) | **not** the level-set velocity mask |
| nearest-node `LEVEL`, median | **lost +0.00029** vs survivors +0.01089 (37x smaller) | they sit *on the tool surface*, just outside it |
| radial position | r 5–9 mm: **78%** | the shoulder/pin surface band |
| axial position | z −1…+1.4 mm: **80%** | at and just above the workpiece top |
| temperature | **lost 0.0**, survivors 321.6 | they never sampled a valid field |
| loss rate at step 8000 | ~40 per 50 steps, still accruing | not a settling transient |

### What this rules out

- **The lattice bug** — fixed and verified separately; zero mismatched cells.
- **Domain escape** — every particle is inside the true bbox. The `escaped_ever`
  flag says otherwise and is wrong; it latches and tests any axis. This is the
  second time that flag has produced a misleading read, so treat it as
  diagnostic noise.
- **The level-set / tool mask** — 0.0% of lost particles are inside the tool.
- **The seed-box bug** — this run seeded with `grid`, not `box-frac`, and all
  288,000 were assigned at step 0.

### What it points to

A thin boundary-layer failure **at the tool surface**: particles hugging the
shoulder and pin, where the mesh is most refined, elements are sliver-shaped,
and the point-in-tet test is marginal. The near-zero `LEVEL` median is the
sharpest signal — these particles are within one element of the tool wall.

Candidate causes, in the order worth testing:

1. **`POINT_IN_TET_TOLERANCE` (1e-6, float32)** against slivers at the wall.
   Earlier sampling found 2 of 2,000 failures at min barycentric −3.3e-04,
   which a 1e-6 tolerance rejects. If the wall elements are worse, this alone
   could account for it.
2. **Boundary projection** (`on`, tol 1e-6) pushing a particle marginally
   outside its host without re-locating it.
3. **The rotating tool sweeping past a particle**: the mesh is static but the
   `LEVEL` field moves, so a particle can be left in a cell whose elements
   belong to the region the tool just vacated.

---

## 3. Consequence for the full D-family run

**Do not launch the full D-family production run yet.** It would reproduce this
same unexplained 4% across all five cases, at ~2.4 h of mesh loading each. The
cheap next step is to re-run D2 over a few hundred steps with a widened
`POINT_IN_TET_TOLERANCE` and boundary projection off, seeding an annulus at
r 5–9 mm / z −1…+1.4 mm where the losses concentrate, and compare loss counts.

The lattice fix is nonetheless a clear net gain and should ship: zero
unassigned at seeding, 3.2x throughput, and the losses it did cause are gone.

---

## 4. Reproducing these measurements

Logs and the final state are under
`/scratch/users/ali/outputs/D2_latticefix_20261013_100727_2156709/`, with the
run script preserved at `/scratch/users/ali/d2_fullrun/run_d2_full.sh`
(a copy of the case's `run_jaxtrace_ws.sh` with three changes: `MESH_SUBDIR=""`,
`OUTPUT_TARGET=scratch`, `FAILED_SUBSTAGE=last_valid_vel`). Nothing was written
into the shared case folder.
