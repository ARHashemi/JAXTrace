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

## 3. The defect behind the freeze

Brute-force containment test at 600 of the 3,302 freeze positions:

| | n | share |
|---|---:|---:|
| **HAVE a containing element** | **561** | **93.5%** |
| no containing element | 39 | 6.5% |

For the 39 with no host, the best min-barycentric is a median of −6.3e-04 —
genuinely outside, and far beyond any tolerance worth setting (0 of them are
within 1e-6; 5 within 1e-4).

**So 93.5% of the freezes are a real point-location failure inside valid
mesh.** Freeze positions: r 1.62–13.65 mm (median 6.13), z −4.57…+0.26 mm
(median −0.24) — inside the bulk, not at the free surface.

Where the failing substage lands, classified against the `LEVEL` field:

| location | n | share |
|---|---:|---:|
| **in material, `LEVEL ≥ 0`** | **98,431** | **88.7%** |
| above the free surface (z>0) | 11,704 | 10.5% |
| inside the tool (`LEVEL < 0`) | 862 | 0.8% |

The tool interior is 0.8%, so the level-set mask is not the cause.

## 4. What this rules in and out

| candidate | verdict |
|---|---|
| lattice / registration (the fixed bug) | **not this** — registration measured 100% consistent with the lookup pitch, index delta ≤ 1 everywhere |
| point-in-tet tolerance | **not the main cause** — the no-host cases sit at −6.3e-04, and the 93.5% with a host are a search failure, not a predicate one |
| float32 precision | **no** — measured zero difference |
| level-set / tool mask | **no** — 0.8% inside the tool |
| free-surface escape | **minor** — 10.5% |
| **the search missing a host that exists** | **the defect**, 93.5% of freezes |
| **k1's zero fallback latching the state** | **the amplifier** — turns one miss into a permanent freeze |

## 5. Two independent fixes follow

**(a) Stop the latch.** `RK4_CARRY_LAST_VALID_VEL = True` (already implemented,
default off) gives k1 the previous step's velocity instead of zero, so a
particle that misses once keeps moving and gets another chance to be found
rather than freezing forever. This converts a permanent loss into a transient
one. It does not fix the miss.

**(b) Find the missing host.** 93.5% of freeze positions are inside an element
the 3×3×3 search does not reach. Registration is provably correct at these
sites, so the remaining candidates are the search's *reach* rather than its
indexing — `--l2-neighborhood 5`, `--enhanced-search-band` near the tool, or
`--registration aabb` for wider coverage. The A/B harness
[`ab_test_tool_surface.sh`](../../scripts/dfamily/ab_test_tool_surface.sh)
tests these one knob at a time.

Both should be measured before the full D-family production run, since the
4.11% reproduces in 12 minutes on a band-seeded 80k run.
