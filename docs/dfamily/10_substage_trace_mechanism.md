# The D2 residual: a latched dead state, and a real search failure behind it

**Run:** D2, 80,000 particles seeded in the measured loss band
(r 2.6–10.2 mm, z −4.8…0 mm), 1,200 steps, velocity 34..60,
`FAILED_SUBSTAGE=last_valid_vel`, `--trace-substages` every 25 steps.
Workstation RTX 5090, 13 Oct 2026. Final lost 3,288 of 80,000 (**4.11%**),
reproducing the full run's 4.07% on a 10× smaller, 7× faster run.

---

## 1. What the trace shows

110,997 logged particle-steps had a failed substage — but only **3,302
distinct particles**, i.e. ~34 failures each.

| measurement | value | meaning |
|---|---|---|
| all four stages failed together | **110,737 (99.8%)** | not a single-stage excursion |
| only k1 / only k2 / only k3 failed | **0 / 0 / 0** | no isolated early-stage failure exists |
| first failing stage = k1 | **110,737 (99.8%)** | the particle enters the step already lost |
| final search also failed | **110,901 (99.9%)** | the step never recovers |
| intra-record substage movement | **0.000000 mm** (110,737 of 110,997) | the particle does not move at all |
| particles failing more than once | **3,259 of 3,302 (98.7%)**, max 49 of 49 logged steps | the state persists |

## 2. The mechanism

It is a **self-sustaining dead state**, not an in-flight escape:

```
k1 searches at pos with hint elem_id   -> fails
  vel_k1 = 0        (k1 has no predecessor, so last_valid_vel cannot help it)
  pos_k1 = pos                          <- no movement
k2 searches at pos_k1 == pos           -> fails for the same reason
k3, k4 likewise
  pos_final = pos + (dt/6)(0+0+0+0) = pos
  elem_final = search(pos)             -> fails again
```

The particle is frozen at the position where it first failed and re-fails
every step for the rest of the run. That is why all four stages fail together,
why no isolated k1/k2/k3 failure exists, why the stride is exactly zero, and
why 98.7% of the particles repeat.

**This also explains the earlier puzzle.** 82.8% of "lost" particles were
findable at their exported final position because the exported position *is*
the position where they froze, and most freezes are marginal: a different code
path (brute force, looser tolerance, float64) finds a host there even though
the production search does not.

## 3. What the freeze positions actually are

Brute-force containment at 600 of the 3,302 freeze positions:

| | n | share |
|---|---:|---:|
| **HAVE a containing element** | **561** | **93.5%** |
| no containing element | 39 | 6.5% |

Freeze positions: r 1.62–13.65 mm (median 6.13), z −4.57…+0.26 mm
(median −0.24) — inside the bulk, not at the free surface. By `LEVEL`: 88.7%
in valid material, 10.5% above the free surface, 0.8% inside the tool, so the
level-set mask is not the cause.

### Profile of the host elements (375 hosted freezes)

| property | value | against the mesh |
|---|---|---|
| non-Kuhn | **93.1%** (349/375) | ~50–59% mesh-wide → **1.7× enriched** |
| registered in >1 cell (Pass 2 AABB) | **90.4%**, median 8 cells, max 27 | — |
| registered in exactly 1 cell (Pass 1) | 9.6% | — |
| longest axis | **z in 93.3%** (x 4.3%, y 2.4%) | — |
| extent anisotropy max/min | median **1.198** | the same dz/dx = 1.1972 as the original bug |
| own `cell_size` / level median | x 1.0000, y 1.0000, z 1.0106 | the first patch's lattice equalisation holds |
| levels | 14 (366), 13 (5), 12 (3), 11 (1) | all inside the walked range |

So they are **not** elongated on a new axis, and the per-level pitch the first
patch introduced is **not** mismatched at these sites.

## 4. The elimination chain — every search component succeeds here

This is the part that changes the diagnosis. Each line is a separate
measurement at the freeze positions.

| candidate | measurement | verdict |
|---|---|---|
| lattice / registration | index delta to the nearest cell holding the host: **median 0, p90 0, max 1**; 100% within the 3×3×3 scan | ruled out |
| wider reach (5×5×5, neighbour-cell registration) | **0.0%** of cases have delta ≥ 2 | **cannot help** |
| cell binary lookup | needs ⌈log2(5,982,836)⌉ = 23 iterations, `max_iters = 25`; lexicographic (morton, level) compare correct | ruled out |
| host present in the cell's element list | **227/234 (97.0%)** | ruled out |
| point-in-tet tolerance | production float32 Skala **accepts 234/234**; float64 min-barycentric median **3.9e-02**, min 7.7e-07, all within 1e-6 | ruled out |
| float32 precision | **0** rejections | ruled out |
| degeneracy guard | relative, `|det| < 1e-12·L³`; at these scales `(L²)^1.5 ≈ 1.03e-13`, far above float32 underflow | ruled out |
| **production L2 replayed standalone** | **finds a host at 380/400 (95.0%)** | **ruled out** |

The last row is decisive: the exact production search function, at production
precision, succeeds at 95% of the positions where the run returned −1 and
latched permanently.

**Therefore the failure is not reproducible from (position, mesh).** It
depended on the kernel's state at that moment. An earlier claim in this
investigation that these were "a real search failure inside valid mesh" was
too strong: a host exists *and* every production component finds it.

**Consequence for the planned A/B arms.** `--point-in-tet-tol`,
`--l2-neighborhood 5` and `--enhanced-search-band` are all ruled out by the
table above, so running them would cost workstation time for a guaranteed null
result.

**The open hypothesis** is the integration of L0/L1/L2 inside the fused vmapped
kernel rather than any component of it — consistent with the standing project
note that deeply nested `lax.cond` under `jax.vmap` in the full RK4 graph
produces compilation artifacts, where single-particle replay finds everything
and the vmapped kernel loses particles. That is being tested directly
(L2 serial vs L2 vmapped vs full hierarchy serial vs full hierarchy fused),
not inferred.

## 5. Two independent fixes follow

**(a) Stop the latch.** `RK4_CARRY_LAST_VALID_VEL = True` (already implemented,
default off) gives k1 the previous step's velocity instead of zero, so a
particle that misses once keeps moving and gets another chance to be found
rather than freezing forever. This converts a permanent loss into a transient
one. It does not fix the miss.

**(b) Find out why the fused kernel misses what the search finds.** Not a
reach or tolerance problem — §4 rules those out by measurement. The remaining
suspect is how L0/L1/L2 are sequenced under `jnp.where` inside the vmapped RK4
graph. Until that test returns, no search knob should be changed.

The latch fix (a) is worth applying regardless: it converts a permanent loss
into a transient one even if the underlying miss remains, and it costs nothing
when the flag is off.

Both should be measured before the full D-family production run, since the
4.11% reproduces in 12 minutes on a band-seeded 80k run.
