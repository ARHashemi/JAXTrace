#!/usr/bin/env python3
"""
Point-in-Tet Method Mismatch Analysis

Tests whether the 'inverse' method (precomputed M_inv in float32) gives different
answers than the 'current' and 'skala' methods (computed on-the-fly from float32
node_positions) for particles near element boundaries.

PRECISION CHAIN ANALYSIS:
========================
  1. Mesh loader:     node_positions loaded as float64
  2. Deduplication:   preserves float64
  3. GPU upload:      jax.device_put() → float32 (JAX default)
  4. Precompute M_inv:
     - node_positions passed as float64 to precompute_inverse_matrices()
     - M = column_stack([p1-p0, p2-p0, p3-p0])  ← float64 arithmetic
     - M_inv = np.linalg.inv(M).astype(float32)  ← TRUNCATED to float32
     - p0_array[elem_id] = p0.astype(float32)    ← TRUNCATED to float32
  5. GPU point_in_tet_inverse:
     - M_inv[elem_id] is float32, p0[elem_id] is float32
     - pos is float32 (from GPU)
     - local = pos - p0  ← float32 - float32
     - bary = M_inv @ local  ← float32 matmul
  6. GPU point_in_tet_current (or skala):
     - connectivity[elem_id] → node indices
     - node_positions[nodes] → float32 (GPU array)
     - All arithmetic in float32

POTENTIAL MISMATCH:
  The 'inverse' method stores M_inv precomputed from float64 then truncated to float32.
  The 'current/skala' methods compute everything on-the-fly in float32.
  Because matrix inversion is numerically sensitive, the float32 M_inv may not
  exactly correspond to what you'd get computing M and inverting in float32.

  More critically: the 'inverse' method stores p0 separately, while 'current/skala'
  look up node_positions[nodes[0]]. If node_positions was originally float64 and
  p0_array was created via float64→float32 truncation, the p0 values should match
  the GPU node_positions (also float32). BUT if deduplication changes node ordering,
  or if there's any mismatch in which node is "p0", the barycentric coords differ.

THIS SCRIPT:
  1. Loads the actual mesh
  2. Seeds particles at element boundaries (face centers, edge midpoints)
  3. Runs all 3 PIT methods on each particle
  4. Reports disagreements where one method says "inside" and another says "outside"
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'

import numpy as np
import jax
import jax.numpy as jnp
import time
from pathlib import Path

# Mesh loading
from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes

# Point-in-tet methods
from jaxtrace.gpu.search.point_in_tet_methods import (
    point_in_tet_current,
    point_in_tet_skala,
    set_inverse_matrices_gpu,
)
from jaxtrace.gpu.search.point_in_tet_inverse import (
    precompute_inverse_matrices,
    point_in_tet_inverse,
)

# =============================================================================
# Configuration
# =============================================================================
MESH_BASE_PATH = Path("/home/arhashemi/Workspace/welding/Edgar/FLA/post/0eule")
MESH_FILE_PATTERN = "featurelessAvtk_{timestep}.pvtu"
VELOCITY_TIMESTEP_RANGE = (158, 159)
VELOCITY_FIELD_NAME = 'Displacement'

N_ELEMENTS_TO_TEST = 50000  # Test this many elements
N_BOUNDARY_POINTS_PER_ELEM = 10  # Points per element (faces + edges)

# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("Point-in-Tet Method Mismatch Analysis")
    print("=" * 80)

    # ── Load mesh ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading mesh...")
    t0 = time.time()
    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH, file_pattern=MESH_FILE_PATTERN,
        timestep_range=VELOCITY_TIMESTEP_RANGE, field_name=VELOCITY_FIELD_NAME,
        verbose=False,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  node_positions dtype: {node_positions.dtype}")  # Should be float64

    print("  Deduplicating...")
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    print(f"  Removed {n_dup:,} duplicates → {node_positions.shape[0]:,} nodes")
    print(f"  node_positions dtype after dedup: {node_positions.dtype}")

    n_elements = connectivity.shape[0]
    print(f"  Elements: {n_elements:,}")

    # ── Precompute inverse matrices ──────────────────────────────────────────
    print("\n[2/5] Precomputing inverse matrices...")
    print(f"  Input node_positions dtype: {node_positions.dtype}")
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)
    print(f"  M_inv dtype: {M_inv_array.dtype}, p0 dtype: {p0_array.dtype}")

    # ── Check p0 consistency ─────────────────────────────────────────────────
    print("\n[3/5] Checking p0 consistency (precomputed vs GPU node_positions)...")

    # p0_array was computed as: node_positions[connectivity[elem][0]].astype(float32)
    # GPU will have: node_positions.astype(float32) (when uploaded via jax.device_put)
    node_positions_f32 = node_positions.astype(np.float32)
    p0_from_gpu = node_positions_f32[connectivity[:, 0]]  # What GPU would compute

    p0_diff = np.abs(p0_array - p0_from_gpu)
    max_p0_diff = p0_diff.max()
    n_p0_mismatch = (p0_diff > 0).any(axis=1).sum()
    print(f"  Max p0 difference: {max_p0_diff:.2e}")
    print(f"  Elements with ANY p0 difference: {n_p0_mismatch:,}/{n_elements:,}")

    if max_p0_diff > 0:
        print(f"  ⚠ p0 MISMATCH detected!")
        print(f"    This means the 'inverse' method uses a DIFFERENT reference vertex")
        print(f"    than what the GPU sees in node_positions.")
        print(f"    Cause: float64→float32 truncation path differs:")
        print(f"      inverse: float64_node_pos → float32 (direct truncation)")
        print(f"      GPU:     float64_node_pos → float32 (via jax.device_put)")
        # Show examples
        mismatch_elems = np.where(p0_diff.any(axis=1))[0][:5]
        for eid in mismatch_elems:
            print(f"    elem {eid}: p0_precomp={p0_array[eid]}, p0_gpu={p0_from_gpu[eid]}, diff={p0_diff[eid]}")
    else:
        print(f"  ✓ p0 arrays are identical (float64→float32 consistent)")

    # ── Check M_inv consistency ──────────────────────────────────────────────
    print("\n  Checking M_inv precision loss...")
    # Recompute M_inv purely in float32 (what GPU would do if computing on-the-fly)
    n_check = min(10000, n_elements)
    max_bary_diff = 0.0
    max_bary_diff_elem = -1
    n_sign_disagree = 0

    # Generate a test point at the centroid of each element (should be well inside)
    # and also at face centers (boundary)
    np.random.seed(42)
    test_elems = np.random.choice(n_elements, n_check, replace=False)

    for eid in test_elems:
        nodes_idx = connectivity[eid]
        # Float32 vertices (what GPU has)
        verts_f32 = node_positions_f32[nodes_idx]  # (4, 3) float32

        # Method A: inverse (precomputed from float64, stored as float32)
        M_inv = M_inv_array[eid]  # (3,3) float32
        p0_inv = p0_array[eid]    # (3,) float32

        # Method B: on-the-fly from float32 (what current/skala would use)
        p0_fly = verts_f32[0]
        M_fly = np.column_stack([
            verts_f32[1] - verts_f32[0],
            verts_f32[2] - verts_f32[0],
            verts_f32[3] - verts_f32[0],
        ])  # float32

        # Test at face center (average of 3 vertices — on the boundary)
        for face in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
            face_center = verts_f32[list(face)].mean(axis=0)

            # Inverse method bary
            local_inv = face_center - p0_inv
            bary_inv = M_inv @ local_inv
            b0_inv = 1.0 - bary_inv.sum()
            bary_full_inv = np.array([b0_inv, bary_inv[0], bary_inv[1], bary_inv[2]])

            # On-the-fly bary (same math as 'current' method but in numpy)
            local_fly = face_center - p0_fly
            try:
                M_fly_inv = np.linalg.inv(M_fly.astype(np.float64)).astype(np.float32)
            except np.linalg.LinAlgError:
                continue
            bary_fly = M_fly_inv @ local_fly
            b0_fly = 1.0 - bary_fly.sum()
            bary_full_fly = np.array([b0_fly, bary_fly[0], bary_fly[1], bary_fly[2]])

            diff = np.abs(bary_full_inv - bary_full_fly)
            max_d = diff.max()
            if max_d > max_bary_diff:
                max_bary_diff = max_d
                max_bary_diff_elem = eid

            # Check sign disagreement (one says inside, other says outside)
            tol = 1e-6
            inside_inv = np.all(bary_full_inv >= -tol)
            inside_fly = np.all(bary_full_fly >= -tol)
            if inside_inv != inside_fly:
                n_sign_disagree += 1

    print(f"  Checked {n_check} elements × 4 faces = {n_check*4} boundary points")
    print(f"  Max barycentric coord difference: {max_bary_diff:.2e} (elem {max_bary_diff_elem})")
    print(f"  Sign disagreements (one inside, other outside): {n_sign_disagree}")

    # ── GPU comparison ───────────────────────────────────────────────────────
    print("\n[4/5] GPU comparison: inverse vs current vs skala...")

    # Upload to GPU
    node_positions_gpu = jax.device_put(node_positions_f32)
    connectivity_gpu = jax.device_put(connectivity)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)

    # Set module-level for dispatcher
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    # Generate test positions:
    # 1. Face centers (on boundary — most sensitive)
    # 2. Edge midpoints (on boundary)
    # 3. Slightly perturbed face centers (just inside/outside)
    n_test_elems = min(N_ELEMENTS_TO_TEST, n_elements)
    test_elems = np.random.choice(n_elements, n_test_elems, replace=False)

    print(f"  Testing {n_test_elems} elements...")

    # Prepare batch test
    positions_list = []
    elem_ids_list = []
    point_types = []

    for eid in test_elems:
        nodes_idx = connectivity[eid]
        verts = node_positions_f32[nodes_idx]  # (4, 3)

        # Face centers (4 faces per tet, should have one bary coord = 0)
        for face in [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]:
            fc = verts[list(face)].mean(axis=0)
            positions_list.append(fc)
            elem_ids_list.append(eid)
            point_types.append('face_center')

        # Edge midpoints (6 edges per tet, should have two bary coords = 0)
        for e0, e1 in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
            mp = 0.5 * (verts[e0] + verts[e1])
            positions_list.append(mp)
            elem_ids_list.append(eid)
            point_types.append('edge_midpoint')

    positions_np = np.array(positions_list, dtype=np.float32)
    elem_ids_np = np.array(elem_ids_list, dtype=np.int32)
    print(f"  Generated {len(positions_np):,} test positions "
          f"({n_test_elems} elems × 10 boundary points)")

    # Run all 3 methods on GPU
    positions_gpu = jax.device_put(positions_np)
    elem_ids_gpu = jax.device_put(elem_ids_np)

    @jax.jit
    def test_inverse(pos, eid):
        return point_in_tet_inverse(pos, eid, M_inv_gpu, p0_gpu, tolerance=1e-6)

    @jax.jit
    def test_current(pos, eid):
        return point_in_tet_current(pos, eid, connectivity_gpu, node_positions_gpu)

    @jax.jit
    def test_skala(pos, eid):
        return point_in_tet_skala(pos, eid, connectivity_gpu, node_positions_gpu)

    # Vectorize
    inverse_results = jax.vmap(test_inverse)(positions_gpu, elem_ids_gpu)
    current_results = jax.vmap(test_current)(positions_gpu, elem_ids_gpu)
    skala_results = jax.vmap(test_skala)(positions_gpu, elem_ids_gpu)

    # Block and convert
    inverse_np = np.array(inverse_results)
    current_np = np.array(current_results)
    skala_np = np.array(skala_results)

    # Comparison
    n_total = len(positions_np)
    agree_all = (inverse_np == current_np) & (inverse_np == skala_np)
    n_agree = agree_all.sum()
    n_disagree = n_total - n_agree

    inv_vs_cur = (inverse_np != current_np)
    inv_vs_ska = (inverse_np != skala_np)
    cur_vs_ska = (current_np != skala_np)

    print(f"\n  Results ({n_total:,} boundary points):")
    print(f"    All 3 agree:           {n_agree:,} ({100*n_agree/n_total:.2f}%)")
    print(f"    Any disagreement:      {n_disagree:,} ({100*n_disagree/n_total:.4f}%)")
    print(f"    inverse ≠ current:     {inv_vs_cur.sum():,}")
    print(f"    inverse ≠ skala:       {inv_vs_ska.sum():,}")
    print(f"    current ≠ skala:       {cur_vs_ska.sum():,}")

    # Breakdown by point type
    point_types_np = np.array(point_types)
    for pt in ['face_center', 'edge_midpoint']:
        mask = point_types_np == pt
        n_pt = mask.sum()
        n_dis_pt = (~agree_all[mask]).sum()
        print(f"\n    {pt}: {n_dis_pt}/{n_pt} disagree ({100*n_dis_pt/n_pt:.4f}%)")

        # Among disagreements, who says inside vs outside?
        if n_dis_pt > 0:
            dis_mask = mask & ~agree_all
            inv_inside = inverse_np[dis_mask].sum()
            cur_inside = current_np[dis_mask].sum()
            ska_inside = skala_np[dis_mask].sum()
            print(f"      inverse says inside: {inv_inside}/{n_dis_pt}")
            print(f"      current says inside: {cur_inside}/{n_dis_pt}")
            print(f"      skala says inside:   {ska_inside}/{n_dis_pt}")

    # ── Show disagreement examples ───────────────────────────────────────────
    print(f"\n[5/5] Disagreement examples...")
    disagree_idx = np.where(~agree_all)[0]
    n_show = min(30, len(disagree_idx))
    if n_show > 0:
        print(f"\n  Showing first {n_show} disagreements:")
        print(f"  {'Idx':>6s}  {'ElemID':>8s}  {'Type':>14s}  {'inverse':>8s}  {'current':>8s}  {'skala':>8s}  Position")
        for i in range(n_show):
            idx = disagree_idx[i]
            pos = positions_np[idx]
            print(f"  {idx:>6d}  {elem_ids_np[idx]:>8d}  {point_types[idx]:>14s}  "
                  f"{'IN' if inverse_np[idx] else 'OUT':>8s}  "
                  f"{'IN' if current_np[idx] else 'OUT':>8s}  "
                  f"{'IN' if skala_np[idx] else 'OUT':>8s}  "
                  f"({pos[0]:+.8f},{pos[1]:+.8f},{pos[2]:+.8f})")

    # ── Critical: What fraction of the mesh is affected? ─────────────────────
    if n_disagree > 0:
        # For each disagreement, check which direction the mismatch goes
        # inverse=True but current=False → inverse is MORE PERMISSIVE
        # inverse=False but current=True → inverse is MORE RESTRICTIVE
        inv_more_permissive = (inverse_np & ~current_np).sum()
        inv_more_restrictive = (~inverse_np & current_np).sum()

        print(f"\n  CRITICAL FINDING:")
        print(f"    inverse MORE PERMISSIVE (says inside, current says outside): {inv_more_permissive}")
        print(f"    inverse MORE RESTRICTIVE (says outside, current says inside): {inv_more_restrictive}")

        if inv_more_restrictive > inv_more_permissive:
            print(f"\n  → The 'inverse' method REJECTS positions that 'current' ACCEPTS.")
            print(f"    This means particles at element boundaries may fail the")
            print(f"    point-in-tet test during RK4 sub-steps, even though they ARE")
            print(f"    geometrically inside. This DIRECTLY causes particle loss.")
            print(f"    FIX: Switch to 'current' or 'skala' method, or increase tolerance.")
        elif inv_more_permissive > inv_more_restrictive:
            print(f"\n  → The 'inverse' method ACCEPTS positions that 'current' REJECTS.")
            print(f"    This means the inverse method is more lenient, not more restrictive.")
            print(f"    The particle loss cause is likely elsewhere.")
    else:
        print(f"\n  ✓ All methods agree perfectly on {n_total:,} boundary points.")
        print(f"    The point-in-tet method is NOT the cause of particle loss.")

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
