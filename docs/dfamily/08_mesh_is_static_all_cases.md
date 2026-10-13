# All 20 PinShapes cases have a static mesh

**Measured, not assumed.** LUMI job 22775342, 13 Oct 2026, 1 min 44 s.

## The question

The tool rotates through many revolutions over a run. If the mesh moved with
it, the octree would have to be rebuilt — or at least re-registered — on every
loaded velocity step, which would dominate the wall clock and would change how
the D-family fix has to be deployed.

## The answer

| | |
|---|---|
| cases checked | **20** |
| static | **20** |
| moving | **0** |
| inconclusive | **0** |

So **one search structure is valid for an entire run**: the grid, the Morton
codes *and* the element registration can all be built once. Nothing needs to
cycle per revolution.

This is the Eulerian picture: the mesh is a fixed grid and the tool rotates
*through* it. Only the velocity field and the level-set field change per
timestep.

## Per-case detail

Each row reports the velocity range from that case's own `run_jaxtrace.sh`, the
size of one mesh piece, and how many timesteps were compared.

| family | case | velocity steps | n steps | points/cells per piece | verdict |
|---|---|---|---:|---|---|
| A-FlatsVariations | A1 | 34..199 | 166 | 13,248 / 62,845 | STATIC (5 compared) |
| A-FlatsVariations | A2 | 34..199 | 166 | 20,087 / 99,294 | STATIC (5 compared) |
| A-FlatsVariations | A2old | 34..199 | 166 | 15,769 / 76,008 | STATIC (5 compared) |
| A-FlatsVariations | A3 | 34..199 | 166 | 16,824 / 82,940 | STATIC (5 compared) |
| A-FlatsVariations | A3old | 34..199 | 166 | 16,819 / 78,250 | STATIC (5 compared) |
| A-FlatsVariations | A4 | 34..199 | 166 | 14,693 / 72,510 | STATIC (5 compared) |
| A-FlatsVariations | A4old | 34..199 | 166 | 14,920 / 69,383 | STATIC (5 compared) |
| B-FluteVariations | B1 | 34..199 | 166 | 17,887 / 83,587 | STATIC (5 compared) |
| B-FluteVariations | **B2** | **2..201** | **200** | 19,867 / 90,558 | STATIC (5 compared) |
| B-FluteVariations | B3 | 34..199 | 166 | 18,958 / 87,195 | STATIC (5 compared) |
| B-FluteVariations | B4 | 34..199 | 166 | 20,677 / 95,535 | STATIC (5 compared) |
| C-ThreadsVariations | C1 | 34..199 | 166 | 17,312 / 76,886 | STATIC (5 compared) |
| C-ThreadsVariations | C2 | 34..199 | 166 | 16,896 / 74,528 | STATIC (5 compared) |
| C-ThreadsVariations | C3 | 34..199 | 166 | 16,116 / 78,829 | STATIC (5 compared) |
| C-ThreadsVariations | C4 | 34..199 | 166 | 16,644 / 81,395 | STATIC (5 compared) |
| D-ConcavityTilt | D1 | 34..199 | 166 | 15,704 / 73,765 | STATIC (5 compared) |
| D-ConcavityTilt | D2 | 34..199 | 166 | 23,166 / 107,991 | STATIC (5 compared) |
| D-ConcavityTilt | D2old | 34..199 | 166 | 17,077 / 82,173 | STATIC (5 compared) |
| D-ConcavityTilt | D3 | 34..199 | 166 | 18,710 / 92,261 | STATIC (5 compared) |
| D-ConcavityTilt | D4 | 34..199 | 166 | 16,120 / 79,063 | STATIC (5 compared) |

**B2 is the one case that loads a different velocity range** (2..201, 200
steps, not 34..199/166). Worth remembering when sizing a B2 run — see
[`lumi-mesh-loading-dominates`] in the project notes: snapshot loading, not the
GPU, sets the wall clock.

## How it was checked, and the bug that made the first attempt lie

The survey reads each case's `run_jaxtrace.sh` for its `MESH_DIR`, file
pattern and velocity range, then loads the mesh geometry at several timesteps
spread across that range and compares node coordinates byte-for-byte.

**The first submission (job 22775201) reported "ALL cases static" with
`piece 0p/0c` — zero points and zero cells read.** It was deriving each piece's
filename by string arithmetic on the parent `.pvtu` name:

```python
prefix = name[:-(len(str(vs)) + 6)]      # drops a character too many
```

which lost the trailing underscore, and it assumed pieces number from `_1` when
they start at `_0`. Every candidate filename missed, nothing was read, and
"no differences found" was reported as STATIC.

Two changes make the result trustworthy:

1. piece paths are resolved from the parent file's own
   `<Piece Source="..."/>` entries instead of being reconstructed, and
2. the script **refuses to report STATIC** unless it actually read ≥ 2
   timesteps with a non-zero point count.

The per-piece point and cell counts in the table above are the evidence that
this run read real geometry. A `0p/0c` row in any future survey means the
loader missed, not that the mesh is empty.

## Consequence for the D-family fix

The lattice fix in [`07_root_cause.md`](07_root_cause.md) is build-time only.
Since the build happens once per run for every case, the fix costs one extra
sweep over the elements per run — not per step.
