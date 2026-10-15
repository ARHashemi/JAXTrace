# Two ideas explained from scratch

Written because the roadmap uses two terms without defining them:
*"reducing the static trip count"* and *"exact tet-vs-box registration"*.
Both are explained here with no prior context assumed, including what is
proven, what is not, and whether each is still viable.

---

# Part 1 — "Reducing the static trip count"

## 1.1 What a "trip count" is

A loop's **trip count** is simply how many times it runs. In the search
kernel the level loop is:

```python
n_levels = max_level - min_level + 1        # 8 on D2
jax.lax.fori_loop(0, n_levels, try_level, ...)
```

so its trip count is 8: the search examines octree levels 15, 14, 13, 12, 11,
10, 9, 8 for **every particle on every RK4 substage**.

## 1.2 What "static" means here, and why it matters

JAX compiles the kernel once, ahead of time, for the GPU. At that moment the
loop bound must be one of two things:

- **static** — an ordinary Python integer, known while compiling. XLA can then
  unroll the loop, schedule the instructions, and keep everything in registers.
- **traced** — a value that only exists at run time (an array entry, say).
  XLA must then emit a real loop with a condition it re-evaluates each pass.

`n_levels` is computed on the host before compiling, so it is **static**. That
is good for speed, but it also means **the number 8 is baked into the compiled
program**. Nothing at run time can make it 3.

## 1.3 Why the loop cannot stop early (the bit that surprises people)

The natural idea is: *once a particle's host is found, stop looping.* The code
appears to do this:

```python
should_search = (found_elem < 0) & (cell_idx >= 0)   # "only if not found yet"
```

But look at what that flag actually does — it selects the **result**:

```python
is_inside  = point_in_tet(...)                       # ALWAYS computed
new_found  = jnp.where(valid & is_inside, elem_id, found)   # then masked
```

The expensive part (`point_in_tet`, ~120 floating-point operations plus four
memory gathers) **runs regardless**. The mask only decides whether the answer
is kept. On a GPU all lanes execute the same instruction stream, so there is
nothing to skip.

Measured, 200,000 lanes, 8 iterations, varying where the host is found:

| | time |
|---|---:|
| every iteration does real work | 0.476 ms |
| masked, host found on iteration 0 | 0.455 ms |
| masked, host found on iteration 7 | 0.534 ms |

**Identical within noise.** Finding the answer immediately costs the same as
finding it last. This is why reordering the levels was rejected: the order
changes which iteration *finds* the host, and that has no effect on cost.

`lax.cond` does not help either — measured at 0.648 ms against `jnp.where`'s
0.645 ms, because under `vmap` a conditional becomes a select, i.e. masking
again.

And a genuinely dynamic loop (`lax.while_loop`, exit when done) is **worse**:
`vmap` has no per-lane control flow, so it becomes one loop for the whole
batch that must run until the *last* lane finishes. With 200,000 lanes where
just one needs 64 iterations and the rest need 1, the cost is 3.83 ms — the
same as if *all* lanes needed 64, and 21× the 0.18 ms of a uniform batch.

## 1.4 So the only lever is the number itself

If the work cannot be skipped at run time, the remaining option is to compile
a kernel with a **smaller static number**. Those iterations then do not exist
at all:

| trip count | time | |
|---:|---:|---|
| 8 (current) | 0.645 ms | — |
| 2 | 0.202 ms | **3.20× faster** |

This is real, because nothing is masked — there is simply less program.

## 1.5 How a smaller number could be used — and why it fails

A reasonable objection: *you cannot put a condition between two loops, because
on a GPU both branches execute anyway.* That is correct for the obvious
design, and it is worth being precise about why.

**Design A — masked branch inside one kernel. Does not work.**

```python
fast = fori_loop(0, 1, body, pos)      # level 8 only
full = fori_loop(0, 8, body, pos)      # all levels
out  = jnp.where(fast_found, fast, full)
```

Every lane runs *both* loops, so the cost is 1 + 8 = **9**, worse than the 8
it replaces. This is the same masking problem as §1.3.

**Design B — two separate kernel launches with compaction between.**

```python
elem = fast_kernel(pos)                      # launch 1: 1 level, ALL N
idx  = nonzero(elem < 0)                     # compaction, host-visible size
elem = elem.at[idx].set(full_kernel(pos[idx]))   # launch 2: 8 levels, M << N
```

Here there is **no GPU branch at all**. Each kernel is straight-line code; the
two differ only in how many particles they are launched over. Total work is
`N×1 + M×8` instead of `N×8`, which on paper breaks even at a miss fraction of
7/8 and wins below it.

**Measured: Design B is also a dead end.**

| miss fraction | one 8-level kernel | two-launch | speedup |
|---:|---:|---:|---:|
| 0% | 0.658 ms | 0.326 ms | 2.02× |
| 1% | 0.477 ms | **2.154 ms** | **0.22×** |
| 5% | 0.466 ms | 2.310 ms | 0.20× |
| 20% | 0.469 ms | 2.246 ms | 0.21× |
| 50% | 0.483 ms | 2.236 ms | 0.22× |
| 100% | 0.480 ms | 2.310 ms | 0.21× |

It wins only at a 0% miss fraction — the degenerate case where launch 2 never
happens. At **any** non-zero miss rate it is ~**5× slower**, and the cost is
*flat* from 1% to 100%, which identifies the culprit: not the extra level
work, but the **compaction itself**. `nonzero` produces a dynamically shaped
array, which cannot live inside a jitted function, so it forces a device-host
synchronisation every time — about 1.8 ms, against the ~0.47 ms that the
entire 8-level search costs.

In a real run that sync would occur five times per RK4 step (k1–k4 plus the
final assignment) across 8000 steps — 40,000 synchronisations. Impractical by
a wide margin.

## 1.6 Verdict: NOT viable

| design | why it fails |
|---|---|
| stop the loop early | masking selects the result; the work runs anyway (§1.3) |
| `lax.cond` per lane | lowers to a select under `vmap`; identical cost |
| `lax.while_loop` | `vmap` runs it to the batch maximum; one slow lane costs as much as all of them |
| masked two-loop branch | both loops execute: 1 + 8 > 8 |
| **two separate launches** | **compaction sync (~1.8 ms) exceeds the entire search cost (~0.47 ms)** |

Reducing the static trip count is the only mechanism that genuinely removes
work, and there is no way to *select* a smaller trip count per particle
without paying more than it saves. **This line of attack is closed.** The
level loop stays at 8.

The one thing that would help is making the mesh have fewer octree levels in
the first place — a mesh-generation decision, outside this codebase.

---

# Part 2 — "Exact tet-vs-box registration"

## 2.1 What registration is

The octree is a lookup table: *cell → which elements might contain a point in
this cell*. Building it means deciding, for each tetrahedron, which cells to
list it in. Listing too few loses particles; listing too many makes every
search slower.

There are two methods in the code:

- **parent cube** (Kuhn elements): the tetrahedron has axis-aligned edges and
  sits inside one identifiable cube, so it is listed in exactly **1** cell.
- **AABB overlap** (non-Kuhn elements, ~50% of D2): the tetrahedron has no
  such cube, so the builder takes its **axis-aligned bounding box** — the
  smallest box containing all four vertices — and lists the element in
  **every cell that box touches**.

## 2.2 Why AABB overlap over-registers

A bounding box is a crude stand-in for a tetrahedron. In 3-D a tetrahedron
fills at most **1/6** of its bounding box — think of a corner cut off a cube:
the box includes a great deal of empty space the tet never reaches.

```
        AABB (what gets registered)        tetrahedron (what is really there)
        ┌───────────────┐                  ┌
        │▓▓▓▓▓▓░░░░░░░░░│                  │▓▓▓▓▓▓
        │▓▓▓▓░░░░░░░░░░░│                  │▓▓▓▓
        │▓▓░░░░░░░░░░░░░│                  │▓▓
        └───────────────┘                  └
         ▓ = tet     ░ = empty, but still registered
```

Every ░ cell is a wasted entry: the search will load that element and run a
point-in-tet test that can never succeed for any point in that cell.

## 2.3 What "exact" means

Replace *"does the bounding box touch this cell?"* with *"does the tetrahedron
itself touch this cell?"*. The computation used is the **separating-axis
theorem** (SAT): two convex shapes do not overlap if and only if some axis
exists on which their projections do not overlap. For a tet versus an
axis-aligned box it suffices to test 13 axes:

- 3 box face normals (x, y, z);
- 4 tetrahedron face normals;
- 6 cross products of box edge directions with tet edge directions.

If any axis separates them, the tet does not reach that cell and the entry is
dropped. If none does, they genuinely overlap and the entry is kept.

This is exact — not a tolerance, not an approximation. A cell is dropped only
when the tetrahedron **provably** does not intersect it.

## 2.4 Why this makes the search faster

The search cost is `n_levels × 27 cells × MAX_ELEMS_PER_CELL`, and the last
factor is set by the **fullest cell in the mesh** (the loop is static, §1.2).
Removing entries lowers that maximum. Measured on the production structure, by
re-testing every element of 4,000 sampled over-full cells against that cell's
real box:

| | before | after |
|---|---:|---:|
| occupancy of over-full cells | mean 47.7, median 48, max 72 | **mean 34.0, median 35, max 56** |
| entries kept | — | **71.3%** |
| **static bound** | **79** | **56** |
| tests per query | 17,064 | **12,096 → 1.41×** |
| CSR entries | 85.1 M | ~61 M |

## 2.5 The coverage worry — resolved

**The worry was right to raise**, and it was the untested gap. Dropping a cell
the tetrahedron *does* touch would make particles there unfindable: the same
failure mode as the cell-overflow bug.

### What is measured

`scripts/dfamily/test_tetbox_coverage.py` takes positions with a known host,
emulates the real 3×3×3 search under **both** registrations, and counts
positions found **before but not after**. Barycentres are the safest possible
probe (deep inside an element, far from any boundary), so two adversarial
modes were added: `face` pushes the point 99.9% of the way toward a vertex,
and `cellface` snaps it onto a cell-boundary plane and forces it through
float32 — the case where rounding could put it in a different cell than the
build used.

| mode | n | current | pruned | **lost** |
|---|---:|---:|---:|---:|
| barycentre (interior) | 5,000 | 100.00% | 100.00% | **0** |
| face (near element face) | 4,000 | 100.00% | 100.00% | **0** |
| cellface (on a cell plane, float32) | 4,000 | 99.92% | 99.92% | **0** |
| **total** | **13,000** | | | **0** |

The `cellface` row is the informative one: 3 of 4,000 fail under **both**
registrations, so those are pre-existing search limits (a point nudged onto a
cell plane can land outside its own element), **not** losses caused by
pruning. Pruning tracked the baseline exactly, including on its failures.

### Why it holds in general, not just on the sample

Sampling 13,000 of 10.8M elements bounds the loss rate but does not prove
zero. There is a stronger argument:

For the search to locate a particle at *p* via element *e*, *e* must be listed
in some cell the 3×3×3 scan visits. Now, *p* lies in **some** cell `C₀` at
*e*'s own level, and `base = floor(p / pitch)` **is** `C₀`'s grid index — so
`C₀` is the *centre* cell of the neighbourhood, always scanned. And since *p*
is inside both *e* and `C₀`, the tetrahedron genuinely intersects `C₀`, which
is exactly the kind of pair the exact test **keeps**.

So whatever an offset cell would have found, the centre cell finds too. The
26 offset cells are a bonus that pruning may thin, but they are never *needed*
for a particle whose host is *e*.

Two scope conditions make this sound rather than hand-waving:

1. **It applies only to the AABB path**, which is exhaustive over intersected
   cells — pruning keeps precisely the intersected ones. **Kuhn elements are
   not pruned at all** (parent-cube registers one cell by centroid and is
   untouched by this change), so their coverage is trivially unchanged. This
   matters because for Kuhn elements the offset cells *are* load-bearing — the
   centroid cell need not be *p*'s cell — which is why the 3×3×3 exists.
2. **It requires the search lattice to match the registration lattice**,
   i.e. `floor(p / pitch)` at search time equals the index `C₀` was registered
   with. That is exactly the property the earlier lattice fix established, and
   it is guarded by `scripts/dfamily/test_lattice_invariant.py`.

### Verdict: coverage is intact; adopt it

The argument covers the general case and 13,000 adversarial probes agree with
it, including on the three positions the current search already cannot locate.

**Residual gaps, for honesty rather than as blockers:**

- all probes are on **D2**; D1 (max occupancy 115, different non-Kuhn mix) is
  unmeasured, so run the coverage test there before a D1 production run;
- the probes are **static**, whereas production advects particles over 8000
  steps with a cached element hint feeding L0/L1. The argument above is
  hint-independent (it concerns what is *listed* in a cell, not how the search
  arrives), but an end-to-end loss comparison on a short run is the natural
  final check;
- the probe computes the SAT test in float64 and only the position in float32;
  the production builder would run it in whatever precision the build uses.

---

# Summary

| | status | gain | what is left to prove |
|---|---|---|---|
| **reduce the static trip count** (two-tier kernel) | **REJECTED** | — | nothing: measured 0.22× because the compaction sync costs ~4× the whole search |
| **exact tet-vs-box registration** | **VIABLE — adopt** | **1.41× measured**, −28% memory | nothing blocking: 0 losses over 13,000 adversarial probes, plus a general argument; re-run the coverage test on D1 before a D1 production run |

Neither is a correctness fix; both are optimisations on top of the shipped
cell-overflow fix, which already recovers 93% of the lost particles. If the
LUMI baseline run is acceptable as it stands, neither is needed.
