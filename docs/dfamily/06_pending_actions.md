# Pending actions for the D-family fix rollout

Decisions already taken, to be applied when the D2 fix is confirmed and the
full runs are prepared. Recorded here so they are not lost between sessions.

## 1. Switch the failed-substage policy to `last_valid_vel`

**Decision (user, 2026-10-13): use `last_valid_vel`, not `zero_vel`.**

In `run_jaxtrace.sh` for every case:

```sh
FAILED_SUBSTAGE=last_valid_vel      # was: zero_vel
```

### Why

RK4 combines the four stages unconditionally
(`rk4_fully_fused_timedep.py:771`):

```python
pos_final = pos + (dt / 6.0) * (vel_k1 + 2.0*vel_k2 + 2.0*vel_k3 + vel_k4)
```

and a failed substage search returns zero velocity for that term alone
(`rk4_fully_fused_timedep.py:569`):

```python
return jnp.where(valid, vel, jnp.zeros(3, ...))
```

So with `zero_vel` a particle whose k2 search fails still advances by
`(dt/6)(k1 + 2*k3 + k4)`. It is **not** frozen — it integrates with a
partially zeroed velocity, which silently biases the trajectory toward
stagnation. `last_valid_vel` instead substitutes the previous stage's
velocity (`vel_k2 = where(elem_k2 >= 0, vel_k2, vel_k1)`), preserving
continuity, which is the physically defensible choice.

Only `skip_step` ever freezes a particle: it discards the whole step when
*any* substage fails (`any_failed` at line 782).

### Scope

- D2 on the workstation: apply for the confirmation run.
- All cases on LUMI: apply when the LUMI repo is updated with the fixed code.
  **Needs the user's permission first** — LUMI is copy-only until then.

## 2. Registration-lattice fix (implemented in full, pending confirmation)

`mesh_aligned_octree_parent_cube.py` now derives the per-level lookup pitch in
a classification sweep that runs **before either pass registers anything**, and
Pass 1, the Pass 2 AABB path and the legacy centroid path all index on it. The
pitch is `median(cell sizes at that level)`, derived exactly as
`upload_mesh_aligned_octree_to_gpu` derives `level_cell_sizes`, so registration
and lookup agree by construction.

This took two steps. Aligning Pass 2 alone (the first version of this fix)
recovered 1,164 of 2,000 previously-lost particles, and all 644 reproducible
failures that remained sat on mismatched **Pass 1 (Kuhn)** cells — 0 on
non-Kuhn, 0 on matching — which is what localised the remainder and prompted
the pass-0 restructure. Full account in
[`07_root_cause.md`](07_root_cause.md) §5.

Regression guard: `scripts/dfamily/test_lattice_invariant.py`, no case data
needed, FAILs on the pre-fix code (66.67% of cells unreachable, max index error
32) and PASSes after, while the isotropic mesh passes either way.

Evidence it is the root cause, measured on D2 (job 22661472 output, 2,000
sampled lost particles, real search re-run to confirm reproducibility):

| group | n | share |
|---|---:|---:|
| reproduce as lost | 858 | |
| → no containing element | 192 | not a bug |
| → host on a **mismatched** lattice | **666** | **100.0%** |
| → host on a **matching** lattice | **0** | **0.0%** |

Index error at the **particle's own position**: median 2, p90 2, `Δ ≤ 1` is
0.0%. The 3×3×3 neighbourhood reaches only ±1, so the host cell is never
tested. (At cell *centres* `Δ ≤ 1` was 99.1% — centre probing is a lower bound,
which is why the two figures appeared to disagree.)

## 3. Still unattributed

1,142 of the 2,000 sampled lost particles do **not** reproduce when the search
is re-run at their exported position. That is expected and not a second bug:
with `zero_vel` the exported position is the *result* of a step containing a
failed substage, not the position where the search failed, and RK4 substage
positions are never exported. Confirming them would need substage-position
export. Worth revisiting only if losses persist after the fix.

## 4. Verified non-issues — do not re-investigate

| candidate | finding |
|---|---|
| mesh changes during tool revolution | all 166 timesteps byte-identical; the mesh is Eulerian, so octree-once is valid. Extended to **all 20 cases** by LUMI job 22775342 — see [`08_mesh_is_static_all_cases.md`](08_mesh_is_static_all_cases.md) |
| Morton mis-mapping to far cells | 0 of 2,000 codes inconsistent with the stored grid index |
| unregistered hosts | 0 — the orphan-drop problem is fixed in the dev repo |
| levels outside the walked range | 0 |
| point-in-tet predicate | 2 of 2,000, at min barycentric −3.3e-04, far beyond any tolerance fix |
| `MAX_ELEMS_PER_CELL` truncation | not applicable: the `_where` variant tracking uses tests `n_elems_in_cell`, unbounded |
