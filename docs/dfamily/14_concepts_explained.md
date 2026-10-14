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

## 1.5 The catch, and why it is not a free win

Levels cannot just be dropped. All eight are populated on D2, and a particle
whose host sits at a level the kernel never visits is **lost** — precisely the
class of bug this whole investigation was about.

So a smaller trip count is only safe as a **two-tier search**:

1. a **fast kernel** compiled with few levels, run on all particles;
2. a **full kernel** with all 8 levels, run only on the particles the fast
   kernel failed to locate.

Correctness is preserved because tier 2 is the current kernel, unchanged. The
gain depends entirely on how many particles tier 1 resolves: if 99% are
handled by a 1-level kernel, the average cost approaches one level instead of
eight.

## 1.6 Is it viable? Promising, not yet proven

Measuring 20,000 positions from the **production seed grid** against the real
D2 octree:

```
levels walked, in order : [15, 14, 13, 12, 11, 10, 9, 8]
host level              : level 8 -> 100.00%
iteration found at      : iteration 7 of 8 -> 100.00%
```

Every seeded particle is hosted at level 8 — the **last** level tried. So a
1-level fast kernel at level 8 would resolve all of them in one iteration
instead of eight.

That looks like ~8×, but one honest caveat stands in the way. The seed box is
`x = −19.9 … −11.1 mm`, entirely **behind** the tool: no seed starts within
r ≤ 9 mm. Level 8 (3.9 mm cells) tiles the coarse background filling most of
the domain *volume*; level 14 (5.88 M cells) tiles only the small refined
region near the tool. So the measurement describes particles **at seeding**,
and the particles that matter are the ones that later reach the tool — where
hosts sit at fine levels, and a level-8 fast tier would miss them, pushing
exactly the interesting particles onto the slow path.

**Verdict: viable in principle, decided by one cheap measurement** — the host-
level histogram *during* a run rather than at seeding. If most particle-steps
stay coarse, this is the largest available win (up to ~8×). If the
distribution spreads once particles enter the refined zone, the two-tier
scheme mostly adds overhead. That measurement is step 1 of the roadmap.

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

## 2.5 The coverage worry — is it real?

**The worry is exactly right to raise**, and it was the untested gap. Dropping
a cell the tetrahedron *does* touch would make particles in that cell
unfindable: the same failure mode as the cell-overflow bug.

What had been tested before this document:

| check | result |
|---|---|
| SAT correctness on 7 hand-built cases | PASS, including the AABB corner a tet misses (correctly separate) and a box touching only one vertex (correctly intersecting) |
| elements left with **zero** cells on real D2 | **0** |
| occupancy reduction | mean 47.7 → 34.0 |

But "no element is orphaned" is **weaker** than "no particle becomes
unfindable", and only the second matters. A test for the stronger property is
`scripts/dfamily/test_tetbox_coverage.py`: for thousands of positions with a
known host, it emulates the real search under both registrations and counts
positions found **before but not after**. That number must be zero.

### Why it should be zero in principle

If a particle sits at position *p* inside element *e*, and the search finds it
in cell *C*, then *p* is inside both *e* and *C* — so *e* genuinely intersects
*C*. That is precisely the kind of pair the exact test **keeps**. Pruning only
removes pairs where the tetrahedron never reaches the cell at all, and such a
cell could never have held a particle hosted by that element.

The argument is sound, but two things could still break it in practice, which
is why it is measured rather than asserted:

1. the lattice the search divides by must match the lattice the cell was
   registered on — the earlier D-family defect was exactly this mismatch;
2. float32 rounding at a cell face could place *p* in a neighbouring cell from
   the one used at build time.

**Verdict: viable, with the decisive coverage test now written and running.**
If it reports any non-zero loss, the change must not be adopted. If it reports
zero, the 1.41× is bankable at low risk.

---

# Summary

| | status | gain | what is left to prove |
|---|---|---|---|
| **reduce the static trip count** (two-tier kernel) | promising | up to ~8× if the common case is one level | the in-run host-level histogram; seeding data alone is not enough |
| **exact tet-vs-box registration** | viable | **1.41× measured**, −28% memory | the search-coverage test must report zero losses |

Neither is a correctness fix; both are optimisations on top of the shipped
cell-overflow fix, which already recovers 93% of the lost particles. If the
LUMI baseline run is acceptable as it stands, neither is needed.
