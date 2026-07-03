"""
Gradient recovery + higher-order velocity reconstruction for P1 tetrahedral
meshes.

Implements the pipeline documented in docs/gradient_recovery_pipeline.md:

    Step 1: raw per-element gradient tensor (piecewise constant on P1 tets)
    Step 2: patch map (elements sharing each node)
    Step 3: per-node SPR fit (linear polynomial least-squares over the patch)
    Step 4: reassemble as a nodal C0 gradient field
    Step 5: build a smooth velocity reconstruction per element from
            (nodal velocities) + (recovered nodal gradients)
    Step 6: RK4 samples the reconstruction (done in the kernel; this module
            only produces the reconstruction arrays)

Step 5 supports two methods today ('centroid_taylor' and 'vertex_taylor')
and is designed to accept more (e.g. 'hct_cubic' for a full Hsieh-
Clough-Tocher tetrahedral macro-element) as they are implemented.

centroid_taylor (Step 5.a):
    For each element e we precompute
      x_c  = centroid of e                              (3,)
      v_c  = P1-interpolated nodal velocity at x_c      (3,)
      G_c  = P1-interpolated recovered gradient at x_c  (3, 3)
    and evaluate at query point p as
      v(p) = v_c + G_c @ (p - x_c)
    Exactly reproduces the raw nodal velocities in the limit of a linear
    field, and adds first-order correction from the recovered gradient in
    smooth regions where the raw P1 gradient has jumps across element
    boundaries. For P1 elements, v_c is the mean of the 4 nodal velocities.
    Simple: one 3x3 matvec per query, no P1 blend, no per-vertex gather.

vertex_taylor (Step 5.b):
    For each element, evaluate a per-vertex Taylor expansion and blend
    them with the P1 shape functions:
      v(p) = sum_a  N_a(p) * ( v_a + G_a @ (p - x_a) )
    where N_a(p) are the barycentric weights of p in the element and
    (v_a, x_a, G_a) are the vertex velocity, position, and recovered
    nodal gradient. Exact at nodes: v(x_a) = v_a for every a. Uses the
    recovered gradient AT THE SAMPLING VERTEX rather than a frozen
    centroid value, so smooth fields with strong gradient variation
    inside an element are reconstructed with less bias than
    centroid_taylor. Costs 4 matvecs + 4 barycentric weights per query
    but no per-element precompute of a centroid tensor.

Both methods reduce to raw P1 nodal interpolation when the recovered
gradients Ga match the FEM's own piecewise-constant gradient (i.e.
when SPR yields no correction). In smoother regions they diverge from
raw P1 as the recovered gradient absorbs the P1 discontinuities.

Everything here runs on the CPU (NumPy). We do the least-squares fits
with numpy.linalg.lstsq / solve; for typical FSW meshes (10^5-10^6 elements,
~10^5 nodes) the whole pipeline is dominated by the SPR loop and completes
in seconds. Vectorising across nodes is a follow-up if profiles justify it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GradientRecoveryData:
    """Output of gradient recovery + velocity reconstruction precompute.

    The ``method`` field picks which set of reconstruction arrays the
    RK4 kernel should read. Fields not used by the chosen method are
    still populated (for diagnostics and to make swapping cheap), but
    the kernel only touches the ones its evaluator needs.

    Attributes
    ----------
    method : str
        Reconstruction method name. Currently 'centroid_taylor' or
        'vertex_taylor'. See module docstring for the formulas.
    element_centroid : (n_elements, 3) float
        Centroid position of every element. Used by centroid_taylor.
    element_v_centroid : (n_elements, 3) float
        P1 velocity at the element centroid (== mean of 4 nodal velocities
        for a P1 tetrahedron). Used by centroid_taylor.
    element_gradient : (n_elements, 3, 3) float
        Recovered velocity gradient tensor at the element centroid.
        gradient[e, i, j] = ∂ u_i / ∂ x_j  at centroid of element e.
        Used by centroid_taylor.
    nodal_gradient : (n_nodes, 3, 3) float
        Recovered nodal gradient tensor (Step 4 output). Used by
        vertex_taylor (gathered per element via connectivity in the
        kernel). Kept as diagnostic for centroid_taylor.
    raw_element_gradient : (n_elements, 3, 3) float
        Raw per-element gradient (Step 1 output). Kept for diagnostics.
    stage_times : dict[str, float]
        Wall-clock breakdown by pipeline stage. Used by run_tracking.py's
        stage_times reporter.
    """
    method: str
    element_centroid: np.ndarray
    element_v_centroid: np.ndarray
    element_gradient: np.ndarray
    nodal_gradient: np.ndarray
    raw_element_gradient: np.ndarray
    stage_times: dict


# ---------------------------------------------------------------------------
# Step 1: raw element gradient
# ---------------------------------------------------------------------------

def compute_element_gradients(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    node_velocities: np.ndarray,
) -> np.ndarray:
    """Per-element full 3×3 velocity gradient tensor for P1 tetrahedra.

    For a linear tet with vertices p0..p3 and nodal velocities v0..v3,
    the velocity field inside the element is
        u(x) = sum_a  N_a(x)  v_a
    with grad(N_a) constant. So grad_u is a constant 3×3 tensor:
        grad_u_ij = sum_a  v_a_i  (grad N_a)_j
    We compute grad N_a from the standard formula:
        [grad N_1; grad N_2; grad N_3] = J^{-T}
        grad N_0 = -sum_{a=1..3} grad N_a
    where J = [p1-p0 | p2-p0 | p3-p0] is the reference-to-physical Jacobian.

    Args
    ----
    node_positions : (n_nodes, 3) float
    connectivity   : (n_elements, 4) int
    node_velocities: (n_nodes, 3) float

    Returns
    -------
    (n_elements, 3, 3) float — gradient[e, i, j] = ∂ u_i / ∂ x_j
    """
    n_elements = connectivity.shape[0]
    pos = node_positions.astype(np.float64)
    vel = node_velocities.astype(np.float64)

    # Vectorised over elements.
    p = pos[connectivity]              # (n_elements, 4, 3)
    v = vel[connectivity]              # (n_elements, 4, 3)
    p0 = p[:, 0]                       # (n_elements, 3)
    # J columns = (p1-p0, p2-p0, p3-p0)
    J = np.stack([p[:, 1] - p0, p[:, 2] - p0, p[:, 3] - p0], axis=2)  # (n_elements, 3, 3)
    # We need grad N_a for a=0..3, each shape (n_elements, 3). Use the identity:
    #   [grad N_1 grad N_2 grad N_3]^T = J^{-1}
    # so grad N_a for a=1..3 are the rows of J^{-1}, and grad N_0 = -sum(grad N_1..3)
    J_inv = np.linalg.inv(J)                # (n_elements, 3, 3)
    grad_N123 = J_inv                       # rows are grad N_1, grad N_2, grad N_3
    grad_N0 = -np.sum(grad_N123, axis=1, keepdims=True)  # (n_elements, 1, 3)
    grad_N = np.concatenate([grad_N0, grad_N123], axis=1)  # (n_elements, 4, 3)

    # grad_u[e, i, j] = sum_a v[e, a, i] * grad_N[e, a, j]
    # -> einsum
    grad_u = np.einsum("eai,eaj->eij", v, grad_N)
    return grad_u.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: node -> element patch
# ---------------------------------------------------------------------------

def build_node_patches(
    n_nodes: int,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style node-to-element map.

    Returns
    -------
    patch_offsets : (n_nodes + 1,) int64
    patch_elements: (total_incidence,) int64
        Elements adjacent to node i occupy the slice
        patch_elements[patch_offsets[i]:patch_offsets[i + 1]].
    """
    n_elements = connectivity.shape[0]
    conn_flat = connectivity.astype(np.int64).ravel()   # (n_elements * 4,)
    elem_ids = np.repeat(np.arange(n_elements, dtype=np.int64), 4)  # (n_elements * 4,)
    # Sort by node id so we can build offsets.
    order = np.argsort(conn_flat, kind="stable")
    conn_sorted = conn_flat[order]
    elem_sorted = elem_ids[order]
    # patch_offsets[i] = index of the first entry with node id == i.
    patch_offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    counts = np.bincount(conn_sorted, minlength=n_nodes)
    np.cumsum(counts, out=patch_offsets[1:])
    return patch_offsets, elem_sorted


# ---------------------------------------------------------------------------
# Step 3+4: SPR fit at each node
# ---------------------------------------------------------------------------

def spr_recover_nodal_gradients(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    element_gradient: np.ndarray,
    patch_offsets: np.ndarray,
    patch_elements: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Recover a nodal gradient by fitting a local linear polynomial to
    the patch of raw element gradients around each node.

    Step 3: for each node i, and for each of the 9 gradient components
    (or fewer if we only recover the symmetric part), fit
        sigma*(x) = a0 + a1 * x + a2 * y + a3 * z
    to the m patch samples (m = number of elements around node i), taking
    the sample location to be each element's centroid, then evaluate at
    the node position.

    Step 4 is trivial once we have the recovered nodal values — they form
    a piecewise-linear C0 field via the same shape functions as the
    velocity itself.

    Args
    ----
    node_positions : (n_nodes, 3) float
    connectivity   : (n_elements, 4) int
    element_gradient: (n_elements, 3, 3) float
        Raw per-element gradient tensor (Step 1 output).
    patch_offsets, patch_elements: node-to-element CSR map (Step 2 output).

    Returns
    -------
    (n_nodes, 3, 3) float — recovered nodal gradient tensor.
    """
    n_nodes = node_positions.shape[0]
    node_positions = node_positions.astype(np.float64)
    element_gradient_f64 = element_gradient.astype(np.float64)
    # Precompute element centroids.
    p = node_positions[connectivity]  # (n_elements, 4, 3)
    centroids = p.mean(axis=1)        # (n_elements, 3)

    recovered = np.zeros((n_nodes, 3, 3), dtype=np.float64)
    n_deg_small = 0     # count of small patches falling back to average

    # Precompute the transpose in a form suitable for gathering: the
    # per-element gradient tensor is (3, 3). We can flatten to (n_elements, 9)
    # and fit each of the 9 components independently in one shot per patch
    # via a single lstsq call with 9 right-hand sides.
    Ge = element_gradient_f64.reshape(len(element_gradient_f64), 9)  # (n_elements, 9)

    t_start = time.time()
    for i in range(n_nodes):
        lo = int(patch_offsets[i])
        hi = int(patch_offsets[i + 1])
        m = hi - lo
        if m == 0:
            # Orphan node (shouldn't happen after dedup); leave zero.
            continue
        patch_elem_ids = patch_elements[lo:hi]
        X = centroids[patch_elem_ids]          # (m, 3)
        S = Ge[patch_elem_ids]                 # (m, 9)

        # Fewer than 4 samples: linear polynomial in 3D is undetermined.
        # Fall back to the arithmetic mean of the patch, which is what
        # standard SPR references recommend for small patches at the
        # domain boundary.
        if m < 4:
            n_deg_small += 1
            recovered_at_node = S.mean(axis=0)
        else:
            # Centre the sample points at the node position. The idea
            # is to make the constant column orthogonal to the position
            # columns so the fit at the node reduces to a0. This works
            # perfectly for interior patches but fails at CORNER and
            # EDGE nodes of symmetric meshes (Kuhn tetrahedralisation,
            # regular hex grids), where the m incident element
            # centroids are all offset from the node along a common
            # direction n̂. Then the constant column [1,...,1] is
            # linearly dependent on the linear terms Xc @ n̂, and the
            # full design [1 | Xc] becomes numerically singular even
            # though Xc alone is rank 3.
            #
            # Explicit condition-number check on the FULL design matrix
            # is the right diagnostic; falling back to the patch mean
            # gives the exact answer for a constant field and a stable
            # estimator otherwise.
            xi = node_positions[i]
            Xc = X - xi                                # (m, 3), centred
            A = np.hstack([np.ones((m, 1)), Xc])       # (m, 4)
            # np.linalg.cond returns np.inf for singular matrices; the
            # cutoff 1e10 is generous — a well-conditioned SPR patch
            # typically has cond(A) < 100.
            cond_A = np.linalg.cond(A)
            if not np.isfinite(cond_A) or cond_A > 1e10:
                n_deg_small += 1
                recovered_at_node = S.mean(axis=0)
            else:
                coeffs, *_ = np.linalg.lstsq(A, S, rcond=None)
                # After centring, the fit at the node position is a0.
                recovered_at_node = coeffs[0]          # (9,)
        recovered[i] = recovered_at_node.reshape(3, 3)

    if verbose:
        t_spr = time.time() - t_start
        print(f"  SPR: {n_nodes:,} nodes, patch size avg "
              f"{len(patch_elements) / max(n_nodes, 1):.1f}, "
              f"{n_deg_small:,} small patches "
              f"(< 4 samples, used mean fallback)  [{t_spr:.1f}s]")
    return recovered.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5: velocity reconstruction per element (Taylor form)
# ---------------------------------------------------------------------------

def build_taylor_reconstruction(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    node_velocities: np.ndarray,
    nodal_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-element Taylor reconstruction (v_centroid, G_centroid, x_centroid).

    Args
    ----
    node_positions : (n_nodes, 3) float
    connectivity   : (n_elements, 4) int
    node_velocities: (n_nodes, 3) float
        Nodal velocity field the RK4 kernel would otherwise interpolate
        directly.
    nodal_gradient : (n_nodes, 3, 3) float
        Recovered nodal gradient tensor from Step 4.

    Returns
    -------
    centroid   : (n_elements, 3)
    v_centroid : (n_elements, 3)  — mean of the 4 node velocities
    G_centroid : (n_elements, 3, 3) — mean of the 4 recovered nodal gradients
    """
    p = node_positions[connectivity]        # (n_elements, 4, 3)
    v = node_velocities[connectivity]       # (n_elements, 4, 3)
    G = nodal_gradient[connectivity]        # (n_elements, 4, 3, 3)

    centroid = p.mean(axis=1)               # (n_elements, 3)
    v_c = v.mean(axis=1)                    # (n_elements, 3)
    G_c = G.mean(axis=1)                    # (n_elements, 3, 3)

    return (centroid.astype(np.float32),
            v_c.astype(np.float32),
            G_c.astype(np.float32))


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

_SUPPORTED_METHODS = ("centroid_taylor", "vertex_taylor")

# Legacy alias so 'taylor' from earlier configs keeps working. If either
# 'taylor' or 'centroid_taylor' is requested, the output arrays are
# byte-identical.
_METHOD_ALIASES = {"taylor": "centroid_taylor"}


def build_recovery(
    node_positions: np.ndarray,
    connectivity: np.ndarray,
    node_velocities: np.ndarray,
    method: str = "centroid_taylor",
    verbose: bool = True,
) -> GradientRecoveryData:
    """Run the full Steps 1-5 pipeline for a single steady velocity field.

    Args
    ----
    node_positions : (n_nodes, 3) float
    connectivity   : (n_elements, 4) int
    node_velocities: (n_nodes, 3) float
        The velocity field we want to reconstruct — same values the raw
        P1 interp would use inside the RK4 kernel.
    method : str
        Reconstruction method. Currently 'centroid_taylor' (default,
        one 3x3 matvec per query) or 'vertex_taylor' (per-vertex Taylor
        expansion blended by P1 shape functions). See module docstring
        for the exact formulas. The legacy name 'taylor' is accepted
        as an alias for 'centroid_taylor'.
    verbose : bool
        If True, print per-stage wall time and simple statistics.
    """
    # Resolve legacy method aliases.
    method = _METHOD_ALIASES.get(method, method)
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"unknown recovery method '{method}'; "
            f"expected one of {_SUPPORTED_METHODS} "
            f"(aliases: {list(_METHOD_ALIASES.keys())})"
        )

    n_nodes = node_positions.shape[0]
    n_elements = connectivity.shape[0]
    stage_times: dict[str, float] = {}

    if verbose:
        print(f"[gradient-recovery] method='{method}' "
              f"n_nodes={n_nodes:,}, n_elements={n_elements:,}")

    # Step 1
    t0 = time.time()
    element_gradient = compute_element_gradients(
        node_positions, connectivity, node_velocities,
    )
    stage_times["1_element_gradient"] = time.time() - t0
    if verbose:
        eg_norm = np.linalg.norm(element_gradient.reshape(n_elements, 9), axis=1)
        print(f"  Step 1 (element gradient): "
              f"|grad_u|_F  min={eg_norm.min():.3e}  "
              f"mean={eg_norm.mean():.3e}  max={eg_norm.max():.3e}  "
              f"[{stage_times['1_element_gradient']:.1f}s]")

    # Step 2
    t0 = time.time()
    patch_offsets, patch_elements = build_node_patches(n_nodes, connectivity)
    stage_times["2_patches"] = time.time() - t0
    if verbose:
        avg_patch = len(patch_elements) / max(n_nodes, 1)
        print(f"  Step 2 (patches): total incidence {len(patch_elements):,}, "
              f"avg patch size {avg_patch:.1f}  "
              f"[{stage_times['2_patches']:.1f}s]")

    # Step 3 + 4 (SPR yields the recovered nodal gradients; Step 4 is
    # implicit — the recovered nodal field is a piecewise-linear C0
    # field once interpreted through the standard shape functions).
    t0 = time.time()
    nodal_gradient = spr_recover_nodal_gradients(
        node_positions, connectivity, element_gradient,
        patch_offsets, patch_elements, verbose=verbose,
    )
    stage_times["3_spr"] = time.time() - t0
    if verbose:
        ng_norm = np.linalg.norm(nodal_gradient.reshape(n_nodes, 9), axis=1)
        print(f"  Step 3+4 (SPR + reassemble): "
              f"|grad_u*|_F  min={ng_norm.min():.3e}  "
              f"mean={ng_norm.mean():.3e}  max={ng_norm.max():.3e}  "
              f"[{stage_times['3_spr']:.1f}s]")

    # Step 5. Both currently supported methods share the centroid-array
    # precompute — vertex_taylor uses only nodal_gradient for its
    # evaluator, but centroid_* arrays are useful diagnostics and
    # allow a user to swap methods post-facto without rebuilding.
    t0 = time.time()
    centroid, v_c, G_c = build_taylor_reconstruction(
        node_positions, connectivity, node_velocities, nodal_gradient,
    )
    stage_times["5_reconstruction"] = time.time() - t0
    if verbose:
        if method == "centroid_taylor":
            print(f"  Step 5 (centroid_taylor): "
                  f"element_centroid, v_centroid, G_centroid built  "
                  f"[{stage_times['5_reconstruction']:.1f}s]")
        elif method == "vertex_taylor":
            print(f"  Step 5 (vertex_taylor): "
                  f"nodal_gradient will be gathered per-element in the "
                  f"kernel (no extra precompute)  "
                  f"[{stage_times['5_reconstruction']:.1f}s]")
        total = sum(stage_times.values())
        print(f"[gradient-recovery] total wall time: {total:.1f}s")

    return GradientRecoveryData(
        method=method,
        element_centroid=centroid,
        element_v_centroid=v_c,
        element_gradient=G_c,
        nodal_gradient=nodal_gradient,
        raw_element_gradient=element_gradient,
        stage_times=stage_times,
    )
