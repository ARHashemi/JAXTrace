#!/usr/bin/env python3
"""
L2 Search Accuracy & Performance Benchmark

Comprehensive benchmark for MALMO (Mesh-Aligned Multi-Level Octree) point
location vs. Morton-band baseline.  Produces all data needed for the paper:

  1. Found rate & accuracy across perturbation levels and position types
  2. Mean PIT (point-in-tet) tests per query
  3. Level distribution: which octree level resolves each query
  4. Scalability: throughput vs. batch size (N_p sweep)
  5. Mesh diagnostics: non-Kuhn element count, memory footprint
  6. Detailed performance metrics: queries/s, PIT tests/s, GPU utilisation

All timing uses JAX block_until_ready() + warmup runs for accurate GPU measurement.
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('JAX_PLATFORMS', 'cuda,rocm,cpu')

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jaxtrace.config as config
import jax
import jax.numpy as jnp

from jaxtrace.gpu.mesh_loader_timedep import load_velocity_sequence_from_pvtu
from jaxtrace.gpu.mesh_deduplication import deduplicate_nodes
from jaxtrace.gpu.search.mesh_aligned_octree_vertex_multi import extract_octree_cells_vertex_multi
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.search.mesh_aligned_morton_builder import build_mesh_aligned_morton_structure
from jaxtrace.gpu.search.mesh_aligned_morton_search import (
    upload_mesh_aligned_morton_to_gpu,
    search_L2_mesh_aligned_morton_single,
)
from jaxtrace.gpu.search.aa_detection import precompute_aa_metadata, precompute_element_vertices
from jaxtrace.gpu.search.point_in_tet_methods import set_corrected_metadata, set_inverse_matrices_gpu
from jaxtrace.gpu.search.point_in_tet_inverse import precompute_inverse_matrices
from jaxtrace.gpu.search.mesh_aligned_point_location import (
    search_mesh_aligned_octree_1x1x1_where,
    search_mesh_aligned_octree_multi_local_where,
    search_mesh_aligned_octree_5x5x5_where,
    search_mesh_aligned_octree_3x3x3_with_stats,
)
from jaxtrace.gpu.search.mesh_aligned_octree_single_cell import extract_octree_cells_single


# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="L2 Search Accuracy & Performance Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=Path("/media/arhashemi/HDD2TB/Workspace/welding/Edgar/FLA/post"),
        help="Base input directory containing 0eule/ (mesh)",
    )
    parser.add_argument(
        "--mesh-subdir", type=str, default="0eule",
        help="Subdirectory within --input containing mesh PVTU files",
    )
    parser.add_argument(
        "--mesh-pattern", type=str, default="cylA_{timestep}.pvtu",
        help="Mesh PVTU file pattern with {timestep} placeholder",
    )
    parser.add_argument(
        "--vel-range", type=int, nargs=2, default=[159, 159], metavar=("START", "END"),
        help="Velocity timestep range for mesh loading",
    )
    parser.add_argument(
        "--vel-field", type=str, default="Displacement",
        help="Velocity field name in PVTU",
    )
    parser.add_argument(
        "--n-particles", type=int, default=10000,
        help="Number of test particles per seeding scenario",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50000,
        help="Batch size for vmapped search (reduce if OOM)",
    )
    parser.add_argument(
        "--point-in-tet-tol", type=float, default=1e-6,
        help="Point-in-tet containment tolerance",
    )
    # --- Seeding region (fraction of mesh bounding box per axis) ---
    parser.add_argument(
        "--seed-x", type=float, nargs=2, default=[0.0, 1.0], metavar=("MIN", "MAX"),
        help="Seeding region as fraction of mesh X extent (0.0=min, 1.0=max)",
    )
    parser.add_argument(
        "--seed-y", type=float, nargs=2, default=[0.0, 1.0], metavar=("MIN", "MAX"),
        help="Seeding region as fraction of mesh Y extent",
    )
    parser.add_argument(
        "--seed-z", type=float, nargs=2, default=[0.0, 1.0], metavar=("MIN", "MAX"),
        help="Seeding region as fraction of mesh Z extent",
    )
    # --- Perturbation factors ---
    parser.add_argument(
        "--perturbations", type=float, nargs="+", default=[0.0],
        help="List of perturbation scale factors (multiples of element size)",
    )
    # --- Intra-element accuracy test ---
    parser.add_argument(
        "--position-types", type=str, nargs="+",
        default=['centroid', 'random', 'near_face', 'near_edge', 'near_vertex'],
        choices=['centroid', 'random', 'near_face', 'near_edge', 'near_vertex'],
        help="Intra-element position types for accuracy test.",
    )
    # --- Timing ---
    parser.add_argument(
        "--warmup-runs", type=int, default=2,
        help="Number of warmup runs before timing (excludes JIT compilation)",
    )
    parser.add_argument(
        "--timing-runs", type=int, default=5,
        help="Number of timed runs for statistics (min/mean/max reported)",
    )
    # --- Scalability ---
    parser.add_argument(
        "--scalability", action="store_true",
        help="Run N_p scalability sweep (1k, 2k, 5k, 10k, 20k, 50k, 100k)",
    )
    parser.add_argument(
        "--scalability-sizes", type=int, nargs="+",
        default=[1000, 2000, 5000, 10000, 20000, 50000, 100000],
        help="Particle counts for scalability sweep",
    )
    # --- Feature toggles ---
    parser.add_argument(
        "--skip-intra", action="store_true",
        help="Skip intra-element accuracy test (faster run)",
    )
    parser.add_argument(
        "--skip-failure-analysis", action="store_true",
        help="Skip 1x1x1 failure analysis (faster run)",
    )
    return parser.parse_args()


# =============================================================================
# Search wrappers
# =============================================================================

def search_3x3x3_batch(positions_gpu, octree_gpu, batch_size=50000):
    """Search using 3x3x3 mesh-aligned octree (production L2)."""
    n = positions_gpu.shape[0]
    max_tests = jnp.int32(600)

    @jax.jit
    def _batch(pos_batch):
        def single(pos):
            elem_id, n_tests = search_mesh_aligned_octree_multi_local_where(
                pos, octree_gpu, max_tests=max_tests
            )
            return elem_id, n_tests
        return jax.vmap(single)(pos_batch)

    all_eids = np.full(n, -1, dtype=np.int32)
    all_tests = np.zeros(n, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        eids, tests = _batch(positions_gpu[start:end])
        eids = jax.block_until_ready(eids)
        all_eids[start:end] = np.array(eids, dtype=np.int32)
        all_tests[start:end] = np.array(tests, dtype=np.int32)

    return all_eids, all_tests


def search_3x3x3_with_stats_batch(positions_gpu, octree_gpu, batch_size=50000):
    """Search using 3x3x3 with extended stats (elem_id, n_tests, found_level)."""
    n = positions_gpu.shape[0]
    max_tests = jnp.int32(600)

    @jax.jit
    def _batch(pos_batch):
        def single(pos):
            return search_mesh_aligned_octree_3x3x3_with_stats(
                pos, octree_gpu, max_tests=max_tests
            )
        return jax.vmap(single)(pos_batch)

    all_eids = np.full(n, -1, dtype=np.int32)
    all_tests = np.zeros(n, dtype=np.int32)
    all_levels = np.full(n, -1, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        eids, tests, levels = _batch(positions_gpu[start:end])
        eids = jax.block_until_ready(eids)
        all_eids[start:end] = np.array(eids, dtype=np.int32)
        all_tests[start:end] = np.array(tests, dtype=np.int32)
        all_levels[start:end] = np.array(levels, dtype=np.int32)

    return all_eids, all_tests, all_levels


def search_5x5x5_batch(positions_gpu, octree_gpu, batch_size=50000):
    """Search using 5x5x5 mesh-aligned octree."""
    n = positions_gpu.shape[0]
    max_tests = jnp.int32(1500)

    @jax.jit
    def _batch(pos_batch):
        def single(pos):
            elem_id, n_tests = search_mesh_aligned_octree_5x5x5_where(
                pos, octree_gpu, max_tests=max_tests
            )
            return elem_id, n_tests
        return jax.vmap(single)(pos_batch)

    all_eids = np.full(n, -1, dtype=np.int32)
    all_tests = np.zeros(n, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        eids, tests = _batch(positions_gpu[start:end])
        eids = jax.block_until_ready(eids)
        all_eids[start:end] = np.array(eids, dtype=np.int32)
        all_tests[start:end] = np.array(tests, dtype=np.int32)

    return all_eids, all_tests


def search_1x1x1_batch(positions_gpu, octree_gpu, batch_size=50000):
    """Search using 1x1x1 mesh-aligned octree (center cell only)."""
    n = positions_gpu.shape[0]
    max_tests = jnp.int32(150)

    @jax.jit
    def _batch(pos_batch):
        def single(pos):
            elem_id, n_tests = search_mesh_aligned_octree_1x1x1_where(
                pos, octree_gpu, max_tests=max_tests
            )
            return elem_id, n_tests
        return jax.vmap(single)(pos_batch)

    all_eids = np.full(n, -1, dtype=np.int32)
    all_tests = np.zeros(n, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        eids, tests = _batch(positions_gpu[start:end])
        eids = jax.block_until_ready(eids)
        all_eids[start:end] = np.array(eids, dtype=np.int32)
        all_tests[start:end] = np.array(tests, dtype=np.int32)

    return all_eids, all_tests


def search_radius_batch(positions_gpu, morton_gpu, search_radius, batch_size=50000):
    """Search using mesh-aligned Morton radius search."""
    n = positions_gpu.shape[0]
    radius_jax = jnp.int32(search_radius)

    @jax.jit
    def _batch(pos_batch):
        def single(pos):
            elem_id = search_L2_mesh_aligned_morton_single(
                pos, morton_gpu, search_radius=radius_jax
            )
            return elem_id
        return jax.vmap(single)(pos_batch)

    all_eids = np.full(n, -1, dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        eids = _batch(positions_gpu[start:end])
        eids = jax.block_until_ready(eids)
        all_eids[start:end] = np.array(eids, dtype=np.int32)

    return all_eids


# =============================================================================
# Timed search: warmup + multiple runs with block_until_ready
# =============================================================================

def timed_search(search_fn, positions_gpu, n_warmup, n_runs):
    """
    Run search_fn with warmup and precise GPU timing.

    Args:
        search_fn: callable(positions_gpu) -> (eids, ...) or eids
        positions_gpu: JAX device array
        n_warmup: warmup iterations (includes JIT)
        n_runs: timed iterations

    Returns:
        result: output from last run
        times: list of n_runs wall-clock seconds
    """
    # Warmup (includes JIT compilation on first call)
    for _ in range(n_warmup):
        result = search_fn(positions_gpu)
        if isinstance(result, tuple):
            jax.block_until_ready(result[0])
        else:
            jax.block_until_ready(result)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = search_fn(positions_gpu)
        if isinstance(result, tuple):
            jax.block_until_ready(result[0])
        else:
            jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    return result, times


# =============================================================================
# Seeding
# =============================================================================

def generate_particles(connectivity, node_positions, n_particles, perturbation_scale_factor,
                       rng, element_sizes, valid_element_ids=None):
    """
    Generate particles seeded from random element centroids with perturbation.

    Args:
        perturbation_scale_factor: Multiplier on per-element characteristic size.
            0.0 = exact centroids, 1.0 = perturbation ~ 1 element size, etc.
        valid_element_ids: Optional array of element indices to seed from.

    Returns:
        positions: (n_particles, 3) float64
        source_elements: (n_particles,) int32
    """
    if valid_element_ids is None:
        n_choices = connectivity.shape[0]
        source_elements = rng.integers(0, n_choices, size=n_particles).astype(np.int32)
    else:
        source_elements = rng.choice(valid_element_ids, size=n_particles, replace=True).astype(np.int32)

    verts = node_positions[connectivity[source_elements]]  # (n_particles, 4, 3)
    centroids = verts.mean(axis=1)

    if perturbation_scale_factor == 0.0:
        return centroids, source_elements

    sizes = element_sizes[source_elements]
    perturbation = rng.standard_normal((n_particles, 3)) * (sizes[:, None] * perturbation_scale_factor)
    return centroids + perturbation, source_elements


def generate_intra_element_particles(connectivity, node_positions, n_particles, rng,
                                     valid_element_ids=None, position_type='random'):
    """
    Generate particles at KNOWN positions INSIDE elements using barycentric coordinates.
    Ground truth element is guaranteed.

    Args:
        position_type: 'centroid', 'random', 'near_face', 'near_edge', 'near_vertex'

    Returns:
        positions: (n_particles, 3) float64
        source_elements: (n_particles,) int32
    """
    if valid_element_ids is None:
        n_choices = connectivity.shape[0]
        source_elements = rng.integers(0, n_choices, size=n_particles).astype(np.int32)
    else:
        source_elements = rng.choice(valid_element_ids, size=n_particles, replace=True).astype(np.int32)

    verts = node_positions[connectivity[source_elements]]  # (n_particles, 4, 3)

    if position_type == 'centroid':
        positions = verts.mean(axis=1)

    elif position_type == 'random':
        u = np.sort(rng.random((n_particles, 3)), axis=1)
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        lam[:, 0] = u[:, 0]
        lam[:, 1] = u[:, 1] - u[:, 0]
        lam[:, 2] = u[:, 2] - u[:, 1]
        lam[:, 3] = 1.0 - u[:, 2]
        positions = np.einsum('ni,nid->nd', lam, verts)

    elif position_type == 'near_face':
        eps = 0.02
        small_idx = rng.integers(0, 4, size=n_particles)
        u = np.sort(rng.random((n_particles, 2)), axis=1)
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        remaining = 1.0 - eps
        for i in range(n_particles):
            si = small_idx[i]
            lam[i, si] = eps
            others = [j for j in range(4) if j != si]
            lam[i, others[0]] = u[i, 0] * remaining
            lam[i, others[1]] = (u[i, 1] - u[i, 0]) * remaining
            lam[i, others[2]] = (1.0 - u[i, 1]) * remaining
        positions = np.einsum('ni,nid->nd', lam, verts)

    elif position_type == 'near_edge':
        eps = 0.02
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        for i in range(n_particles):
            small = rng.choice(4, size=2, replace=False)
            lam[i, small[0]] = eps * rng.random()
            lam[i, small[1]] = eps * rng.random()
            remaining = 1.0 - lam[i, small[0]] - lam[i, small[1]]
            others = [j for j in range(4) if j not in small]
            split = rng.random()
            lam[i, others[0]] = split * remaining
            lam[i, others[1]] = (1.0 - split) * remaining
        positions = np.einsum('ni,nid->nd', lam, verts)

    elif position_type == 'near_vertex':
        eps = 0.02
        lam = np.zeros((n_particles, 4), dtype=np.float64)
        big_idx = rng.integers(0, 4, size=n_particles)
        for i in range(n_particles):
            bi = big_idx[i]
            total_small = 0.0
            for j in range(4):
                if j != bi:
                    val = eps * rng.random()
                    lam[i, j] = val
                    total_small += val
            lam[i, bi] = 1.0 - total_small
        positions = np.einsum('ni,nid->nd', lam, verts)

    else:
        raise ValueError(f"Unknown position_type: {position_type}")

    return positions, source_elements


def compute_element_sizes(connectivity, node_positions):
    """Compute characteristic size (minimum edge length) per element."""
    v = node_positions[connectivity]  # (n_elem, 4, 3)
    n_elem = v.shape[0]

    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    edge_lengths = np.zeros((n_elem, 6), dtype=np.float64)
    for i, (a, b) in enumerate(edges):
        edge_lengths[:, i] = np.linalg.norm(v[:, a] - v[:, b], axis=1)

    return edge_lengths.min(axis=1)


# =============================================================================
# Formatting helpers
# =============================================================================

def fmt_time_range(times):
    """Format a list of times as 'min--max' with 2 decimal places."""
    return f"{min(times):.2f}--{max(times):.2f}"


def fmt_time_stats(times):
    """Format times as 'mean ± std (min--max)'."""
    arr = np.array(times)
    return f"{arr.mean():.3f} ± {arr.std():.3f} ({arr.min():.3f}--{arr.max():.3f})"


def print_gpu_info():
    """Print GPU device information."""
    devices = jax.devices()
    for d in devices:
        print(f"  Device: {d}")
        if hasattr(d, 'device_kind'):
            print(f"    Kind: {d.device_kind}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 90)
    print("L2 Search Accuracy & Performance Benchmark")
    print("=" * 90)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print_gpu_info()
    print(f"Float dtype: {config.FLOAT_DTYPE_NP}")
    print(f"Warmup runs: {args.warmup_runs}, Timing runs: {args.timing_runs}")
    print()

    # ---- Load mesh ----
    MESH_BASE_PATH = args.input / args.mesh_subdir
    print(f"[1/5] Loading mesh from {MESH_BASE_PATH} ...")
    t0 = time.time()

    node_positions, connectivity, velocity_sequence = load_velocity_sequence_from_pvtu(
        base_path=MESH_BASE_PATH,
        file_pattern=args.mesh_pattern,
        timestep_range=tuple(args.vel_range),
        field_name=args.vel_field,
        verbose=False,
    )
    node_positions, connectivity, n_dup, velocity_sequence = deduplicate_nodes(
        node_positions, connectivity, velocity_sequence=velocity_sequence, verbose=False
    )
    connectivity = connectivity.astype(np.int32)

    n_elements = connectivity.shape[0]
    n_nodes = node_positions.shape[0]
    print(f"  Elements: {n_elements:,}, Nodes: {n_nodes:,} (removed {n_dup:,} duplicates)")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ---- Precompute metadata ----
    print(f"\n[2/5] Precomputing metadata...")
    t0 = time.time()

    config.POINT_IN_TET_METHOD = 'inverse'
    config.POINT_IN_TET_TOLERANCE = args.point_in_tet_tol

    aa_metadata = precompute_aa_metadata(connectivity, node_positions, verbose=False)
    element_vertices = precompute_element_vertices(connectivity, node_positions, verbose=False)
    M_inv_array, p0_array = precompute_inverse_matrices(connectivity, node_positions)

    element_sizes = compute_element_sizes(connectivity, node_positions)
    print(f"  Element sizes: min={element_sizes.min():.2e}, "
          f"mean={element_sizes.mean():.2e}, max={element_sizes.max():.2e}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Build structures & upload to GPU ----
    print(f"\n[3/5] Building search structures and uploading to GPU...")
    t0 = time.time()

    # Mesh-aligned multi-cell octree (for 3x3x3 and 5x5x5)
    mesh_octree_cells_multi = extract_octree_cells_vertex_multi(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    print(f"  Multi-cell octree: {mesh_octree_cells_multi.n_cells:,} cells, "
          f"{mesh_octree_cells_multi.elements_per_cell_mean:.1f} elem/cell, "
          f"{mesh_octree_cells_multi.cells_per_element_mean:.1f} cells/elem")

    # --- Non-Kuhn element count ---
    cells_per_elem = np.diff(mesh_octree_cells_multi.element_to_cells_offsets)
    n_unregistered = int(np.sum(cells_per_elem == 0))
    n_registered = n_elements - n_unregistered
    print(f"  Registered elements: {n_registered:,} / {n_elements:,} "
          f"({100*n_registered/n_elements:.2f}%)")
    if n_unregistered > 0:
        print(f"  Unregistered (non-Kuhn, no Kuhn neighbor): {n_unregistered:,}")

    octree_gpu = upload_mesh_aligned_octree_to_gpu(
        connectivity, node_positions, mesh_octree_cells_multi, verbose=False
    )

    # Mesh-aligned Morton structure (for radius search)
    mesh_octree_cells_single = extract_octree_cells_single(
        node_positions, connectivity, tolerance=1e-6, verbose=False
    )
    morton_struct = build_mesh_aligned_morton_structure(
        node_positions, connectivity, mesh_octree_cells=mesh_octree_cells_single, verbose=False
    )
    morton_gpu = upload_mesh_aligned_morton_to_gpu(
        node_positions, connectivity, morton_struct, verbose=False
    )
    print(f"  Morton structure: {morton_struct.n_cells:,} cells, "
          f"{morton_struct.elements_per_cell_mean:.1f} elem/cell")

    # --- Level distribution of octree cells ---
    cell_levels = mesh_octree_cells_multi.cell_levels
    unique_levels, level_counts = np.unique(cell_levels, return_counts=True)
    print(f"\n  Octree level distribution:")
    for lev, cnt in zip(unique_levels, level_counts):
        print(f"    Level {lev:2d}: {cnt:>8,} cells")

    # Upload point-in-tet data
    aa_metadata_gpu_obj = type(aa_metadata)(
        base_vertex_indices=jax.device_put(aa_metadata.base_vertex_indices),
        base_vertices=jax.device_put(aa_metadata.base_vertices),
        inv_edge_lengths=jax.device_put(aa_metadata.inv_edge_lengths),
        axis_indices=jax.device_put(aa_metadata.axis_indices),
        is_axis_aligned=jax.device_put(aa_metadata.is_axis_aligned),
    )
    element_vertices_gpu = jax.device_put(element_vertices)
    M_inv_gpu = jax.device_put(M_inv_array)
    p0_gpu = jax.device_put(p0_array)
    set_corrected_metadata(aa_metadata_gpu_obj, element_vertices_gpu)
    set_inverse_matrices_gpu(M_inv_gpu, p0_gpu)

    # --- Memory footprint ---
    morton_codes_bytes = mesh_octree_cells_multi.cell_morton_codes.nbytes
    levels_bytes = mesh_octree_cells_multi.cell_levels.nbytes
    csr_offsets_bytes = mesh_octree_cells_multi.cell_to_elements_offsets.nbytes
    csr_data_bytes = mesh_octree_cells_multi.cell_to_elements_data.nbytes
    grid_indices_bytes = mesh_octree_cells_multi.cell_grid_indices.nbytes
    cell_sizes_bytes = mesh_octree_cells_multi.cell_sizes.nbytes
    total_octree_bytes = (morton_codes_bytes + levels_bytes + csr_offsets_bytes +
                          csr_data_bytes + grid_indices_bytes + cell_sizes_bytes)
    hot_working_set = morton_codes_bytes + levels_bytes

    print(f"\n  GPU memory footprint (octree):")
    print(f"    cell_morton_codes: {morton_codes_bytes / 1e6:.1f} MB")
    print(f"    cell_levels:      {levels_bytes / 1e6:.1f} MB")
    print(f"    CSR offsets:      {csr_offsets_bytes / 1e6:.1f} MB")
    print(f"    CSR data:         {csr_data_bytes / 1e6:.1f} MB")
    print(f"    Hot working set:  {hot_working_set / 1e6:.1f} MB (codes + levels)")
    print(f"    Total octree:     {total_octree_bytes / 1e6:.1f} MB")

    print(f"  Done in {time.time() - t0:.1f}s")

    # ---- Filter seeding region ----
    domain_min = node_positions.min(axis=0)
    domain_max = node_positions.max(axis=0)
    domain_extent = domain_max - domain_min

    seed_bounds_min = domain_min + np.array([args.seed_x[0], args.seed_y[0], args.seed_z[0]]) * domain_extent
    seed_bounds_max = domain_min + np.array([args.seed_x[1], args.seed_y[1], args.seed_z[1]]) * domain_extent

    all_centroids = node_positions[connectivity].mean(axis=1)
    in_seed_region = np.all(
        (all_centroids >= seed_bounds_min) & (all_centroids <= seed_bounds_max), axis=1
    )
    valid_element_ids = np.where(in_seed_region)[0]

    print(f"\n  Seeding region:")
    print(f"    X: [{seed_bounds_min[0]:.6f}, {seed_bounds_max[0]:.6f}] "
          f"(fraction [{args.seed_x[0]:.2f}, {args.seed_x[1]:.2f}])")
    print(f"    Y: [{seed_bounds_min[1]:.6f}, {seed_bounds_max[1]:.6f}] "
          f"(fraction [{args.seed_y[0]:.2f}, {args.seed_y[1]:.2f}])")
    print(f"    Z: [{seed_bounds_min[2]:.6f}, {seed_bounds_max[2]:.6f}] "
          f"(fraction [{args.seed_z[0]:.2f}, {args.seed_z[1]:.2f}])")
    print(f"    Valid elements: {len(valid_element_ids):,} / {n_elements:,} "
          f"({100*len(valid_element_ids)/n_elements:.1f}%)")

    # ========================================================================
    # [4/5] ACCURACY BENCHMARK
    # ========================================================================
    print(f"\n[4/5] Running L2 accuracy benchmark...")
    print(f"  Particles per scenario: {args.n_particles:,}")
    print(f"  Batch size: {args.batch_size:,}")
    print(f"  Point-in-tet tolerance: {args.point_in_tet_tol:.0e}")
    print()

    perturbation_factors = args.perturbations

    l2_methods = [
        ('radius r=2',  'radius',  2),
        ('radius r=10', 'radius', 10),
        ('1x1x1',       '1x1x1', None),
        ('3x3x3',       '3x3x3', None),
        ('5x5x5',       '5x5x5', None),
    ]

    rng = np.random.default_rng(args.seed)

    # Pre-generate all particle sets
    print("  Generating particle sets...")
    particle_sets = {}
    for pf in perturbation_factors:
        positions, source_elems = generate_particles(
            connectivity, node_positions, args.n_particles, pf, rng, element_sizes,
            valid_element_ids=valid_element_ids,
        )
        particle_sets[pf] = (positions, source_elems)
        in_domain = np.all(
            (positions >= domain_min) & (positions <= domain_max), axis=1
        )
        mean_disp = np.linalg.norm(
            positions - node_positions[connectivity[source_elems]].mean(axis=1), axis=1
        ).mean()
        print(f"    perturbation={pf:.1f}x: mean_displacement={mean_disp:.2e}, "
              f"in_mesh_bbox={in_domain.sum()}/{args.n_particles}")
    print()

    in_bbox_sets = {}
    for pf in perturbation_factors:
        positions, _ = particle_sets[pf]
        in_bbox_sets[pf] = np.all(
            (positions >= domain_min) & (positions <= domain_max), axis=1
        )

    results = {}
    n_warmup = args.warmup_runs
    n_runs = args.timing_runs

    for method_name, method_type, radius in l2_methods:
        results[method_name] = {}
        print(f"  --- {method_name} ---")

        for pf in perturbation_factors:
            positions, source_elems = particle_sets[pf]
            in_bbox = in_bbox_sets[pf]
            positions_gpu = jax.device_put(positions.astype(config.FLOAT_DTYPE_NP))

            # Define search function
            if method_type == '1x1x1':
                search_fn = lambda p: search_1x1x1_batch(p, octree_gpu, args.batch_size)
            elif method_type == '3x3x3':
                search_fn = lambda p: search_3x3x3_batch(p, octree_gpu, args.batch_size)
            elif method_type == '5x5x5':
                search_fn = lambda p: search_5x5x5_batch(p, octree_gpu, args.batch_size)
            elif method_type == 'radius':
                _r = radius  # capture
                search_fn = lambda p, _r=_r: search_radius_batch(p, morton_gpu, _r, args.batch_size)
            else:
                raise ValueError(method_type)

            # Timed search
            raw_result, times = timed_search(search_fn, positions_gpu, n_warmup, n_runs)

            # Extract results
            if method_type == 'radius':
                found_eids = raw_result
                mean_tests = float('nan')
            else:
                found_eids, tests = raw_result
                mean_tests = float(tests.mean())

            found_mask = found_eids >= 0
            n_found = int(found_mask.sum())
            n_in_bbox = int(in_bbox.sum())

            not_found = ~found_mask
            n_outside_bbox = int((not_found & ~in_bbox).sum())
            n_search_fail = int((not_found & in_bbox).sum())

            n_correct = int(np.sum(found_eids[found_mask] == source_elems[found_mask]))
            n_wrong = n_found - n_correct
            pct_correct_of_found = 100 * n_correct / n_found if n_found > 0 else 0.0

            # Performance metrics
            n_p = args.n_particles
            t_mean = np.mean(times)
            queries_per_sec = n_p / t_mean if t_mean > 0 else 0
            pit_per_sec = (n_p * mean_tests) / t_mean if (t_mean > 0 and not np.isnan(mean_tests)) else float('nan')

            results[method_name][pf] = {
                'n_found': n_found,
                'n_in_bbox': n_in_bbox,
                'n_outside_bbox': n_outside_bbox,
                'n_search_fail': n_search_fail,
                'n_correct': n_correct,
                'n_wrong': n_wrong,
                'pct_correct_of_found': pct_correct_of_found,
                'mean_tests': mean_tests,
                'times': times,
                'queries_per_sec': queries_per_sec,
                'pit_per_sec': pit_per_sec,
            }

            pct_found = 100 * n_found / n_p
            tests_str = f", mean_PIT={mean_tests:.1f}" if not np.isnan(mean_tests) else ""
            perf_str = f", {queries_per_sec:.0f} queries/s"
            print(f"    perturb={pf:.1f}x: found={n_found:,} ({pct_found:.2f}%), "
                  f"correct_elem={n_correct:,}/{n_found:,} ({pct_correct_of_found:.1f}%), "
                  f"search_fail={n_search_fail}, "
                  f"time={fmt_time_stats(times)}s{tests_str}{perf_str}")

        print()

    # ---- Summary tables ----

    header = f"{'Method':<16s}"
    for pf in perturbation_factors:
        header += f"  {pf:.1f}x"
        header += " " * max(0, 8 - len(f"{pf:.1f}x"))

    print()
    print("=" * 90)
    print("SUMMARY: Percentage of particles FOUND (assigned to any element)")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for method_name, _, _ in l2_methods:
        row = f"{method_name:<16s}"
        for pf in perturbation_factors:
            pct = 100 * results[method_name][pf]['n_found'] / args.n_particles
            row += f"  {pct:7.2f}%"
        print(row)

    print()
    print("=" * 90)
    print("ACCURACY: Among FOUND particles, % matching source element")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for method_name, _, _ in l2_methods:
        row = f"{method_name:<16s}"
        for pf in perturbation_factors:
            pct = results[method_name][pf]['pct_correct_of_found']
            row += f"  {pct:7.1f}%"
        print(row)

    print()
    print("=" * 90)
    print("UNFOUND ANALYSIS: Search failures (inside bbox but not found)")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for method_name, _, _ in l2_methods:
        row = f"{method_name:<16s}"
        for pf in perturbation_factors:
            n_fail = results[method_name][pf]['n_search_fail']
            row += f"  {n_fail:>7d}"
        print(row)

    print()
    print("=" * 90)
    print("TIMING: mean ± std (min--max) seconds per batch")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for method_name, _, _ in l2_methods:
        row = f"{method_name:<16s}"
        for pf in perturbation_factors:
            row += f"  {fmt_time_stats(results[method_name][pf]['times'])}"
        print(row)

    print()
    print("=" * 90)
    print("MEAN PIT TESTS PER QUERY (octree methods only; radius uses fixed 256 bound)")
    print("=" * 90)
    print(header)
    print("-" * 90)
    for method_name, _, _ in l2_methods:
        row = f"{method_name:<16s}"
        for pf in perturbation_factors:
            mt = results[method_name][pf]['mean_tests']
            if np.isnan(mt):
                row += f"      n/a"
            else:
                row += f"  {mt:7.1f}"
        print(row)

    print()
    print("=" * 90)
    print("PERFORMANCE: Queries/s and PIT tests/s (mean over timed runs)")
    print("=" * 90)
    print(f"{'Method':<16s}  {'Queries/s':>12s}  {'PIT tests/s':>14s}  {'Mean time (s)':>14s}  {'Relative':>8s}")
    print("-" * 90)
    # Use first perturbation factor for the performance summary
    pf0 = perturbation_factors[0]
    ref_time = np.mean(results['1x1x1'][pf0]['times'])
    for method_name, _, _ in l2_methods:
        r = results[method_name][pf0]
        t_mean = np.mean(r['times'])
        qps = r['queries_per_sec']
        pps = r['pit_per_sec']
        rel = t_mean / ref_time if ref_time > 0 else 0
        pps_str = f"{pps:.0f}" if not np.isnan(pps) else "n/a"
        print(f"{method_name:<16s}  {qps:>12,.0f}  {pps_str:>14s}  {t_mean:>14.4f}  {rel:>7.2f}x")

    # ========================================================================
    # LEVEL DISTRIBUTION (3×3×3 with stats)
    # ========================================================================
    print()
    print("=" * 90)
    print("LEVEL DISTRIBUTION: Which octree level resolves each query (3x3x3 method)")
    print("=" * 90)

    # Use centroid particles (perturbation=0) for level analysis
    pf_for_levels = perturbation_factors[0]
    positions_lvl, _ = particle_sets[pf_for_levels]
    positions_lvl_gpu = jax.device_put(positions_lvl.astype(config.FLOAT_DTYPE_NP))

    eids_lvl, tests_lvl, levels_lvl = search_3x3x3_with_stats_batch(
        positions_lvl_gpu, octree_gpu, args.batch_size
    )

    found_mask_lvl = eids_lvl >= 0
    found_levels = levels_lvl[found_mask_lvl]

    if len(found_levels) > 0:
        level_vals, level_cnts = np.unique(found_levels, return_counts=True)
        total_found = len(found_levels)
        print(f"\n  Particles: {args.n_particles:,}, Found: {total_found:,}, "
              f"Not found: {args.n_particles - total_found:,}")
        print(f"\n  {'Level':>6s}  {'Count':>8s}  {'Fraction':>10s}  {'Cumulative':>10s}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*10}")
        cumulative = 0
        for lev, cnt in zip(level_vals, level_cnts):
            cumulative += cnt
            print(f"  {lev:>6d}  {cnt:>8,}  {100*cnt/total_found:>9.2f}%  {100*cumulative/total_found:>9.2f}%")

        # Statistics
        print(f"\n  Mean resolving level: {found_levels.mean():.2f}")
        print(f"  Median resolving level: {np.median(found_levels):.0f}")
        print(f"  Mode resolving level: {level_vals[np.argmax(level_cnts)]}")
    else:
        print("  No particles found (cannot compute level distribution)")

    # ========================================================================
    # INTRA-ELEMENT ACCURACY TEST
    # ========================================================================
    if not args.skip_intra:
        print()
        print("=" * 90)
        print("INTRA-ELEMENT ACCURACY TEST")
        print("  Particles placed inside elements via barycentric coordinates.")
        print("  Ground truth element is GUARANTEED — any failure is a real error.")
        print("=" * 90)

        position_types = args.position_types
        rng_intra = np.random.default_rng(args.seed + 1000)

        intra_sets = {}
        for pt in position_types:
            positions, source_elems = generate_intra_element_particles(
                connectivity, node_positions, args.n_particles, rng_intra,
                valid_element_ids=valid_element_ids, position_type=pt,
            )
            intra_sets[pt] = (positions, source_elems)
            print(f"  {pt:>12s}: generated {args.n_particles:,} particles")
        print()

        intra_methods = [
            ('radius r=2',  'radius',  2),
            ('radius r=10', 'radius', 10),
            ('1x1x1',       '1x1x1', None),
            ('3x3x3',       '3x3x3', None),
            ('5x5x5',       '5x5x5', None),
        ]

        intra_results = {}
        for method_name, method_type, radius in intra_methods:
            intra_results[method_name] = {}
            print(f"  --- {method_name} ---")

            for pt in position_types:
                positions, source_elems = intra_sets[pt]
                positions_gpu = jax.device_put(positions.astype(config.FLOAT_DTYPE_NP))

                if method_type == '1x1x1':
                    search_fn = lambda p: search_1x1x1_batch(p, octree_gpu, args.batch_size)
                elif method_type == '3x3x3':
                    search_fn = lambda p: search_3x3x3_batch(p, octree_gpu, args.batch_size)
                elif method_type == '5x5x5':
                    search_fn = lambda p: search_5x5x5_batch(p, octree_gpu, args.batch_size)
                elif method_type == 'radius':
                    _r = radius
                    search_fn = lambda p, _r=_r: search_radius_batch(p, morton_gpu, _r, args.batch_size)
                else:
                    raise ValueError(method_type)

                raw_result, times = timed_search(search_fn, positions_gpu, n_warmup, n_runs)

                if method_type == 'radius':
                    found_eids = raw_result
                else:
                    found_eids, _ = raw_result

                found_mask = found_eids >= 0
                n_found = int(found_mask.sum())
                n_not_found = args.n_particles - n_found
                n_correct = int(np.sum(found_eids == source_elems))
                n_wrong_elem = n_found - n_correct

                intra_results[method_name][pt] = {
                    'n_found': n_found,
                    'n_not_found': n_not_found,
                    'n_correct': n_correct,
                    'n_wrong_elem': n_wrong_elem,
                    'times': times,
                }

                print(f"    {pt:>12s}: found={n_found:,}/{args.n_particles:,}, "
                      f"correct_elem={n_correct:,}, wrong_elem={n_wrong_elem:,}, "
                      f"NOT_FOUND={n_not_found:,}, time={fmt_time_stats(times)}s")

            print()

        # Intra-element summary tables
        intra_header = f"{'Method':<10s}"
        for pt in position_types:
            intra_header += f"  {pt:>12s}"

        print("=" * 90)
        print("INTRA-ELEMENT: Found rate (should be 100.00% for all)")
        print("=" * 90)
        print(intra_header)
        print("-" * 90)
        for method_name, _, _ in intra_methods:
            row = f"{method_name:<10s}"
            for pt in position_types:
                pct = 100 * intra_results[method_name][pt]['n_found'] / args.n_particles
                row += f"  {pct:11.2f}%"
            print(row)

        print()
        print("=" * 90)
        print("INTRA-ELEMENT: Correct element rate (should be 100.00% for all)")
        print("=" * 90)
        print(intra_header)
        print("-" * 90)
        for method_name, _, _ in intra_methods:
            row = f"{method_name:<10s}"
            for pt in position_types:
                pct = 100 * intra_results[method_name][pt]['n_correct'] / args.n_particles
                row += f"  {pct:11.2f}%"
            print(row)

    # ========================================================================
    # 1×1×1 FAILURE ANALYSIS
    # ========================================================================
    if not args.skip_failure_analysis and not args.skip_intra:
        if '1x1x1' in intra_results and '3x3x3' in intra_results:
            print()
            print("=" * 90)
            print("1x1x1 FAILURE ANALYSIS")
            print("  Why does 1x1x1 miss ~50% of particles that 3x3x3 finds?")
            print("=" * 90)

            from jaxtrace.gpu.search.mesh_aligned_octree import encode_morton_3d

            analysis_pt = 'centroid' if 'centroid' in intra_sets else position_types[0]
            positions_ana, source_elems_ana = intra_sets[analysis_pt]

            positions_gpu_ana = jax.device_put(positions_ana.astype(config.FLOAT_DTYPE_NP))
            eids_1x1, _ = search_1x1x1_batch(positions_gpu_ana, octree_gpu, args.batch_size)
            eids_3x3, _ = search_3x3x3_batch(positions_gpu_ana, octree_gpu, args.batch_size)

            missed_by_1x1 = (eids_1x1 < 0) & (eids_3x3 >= 0)
            n_missed = int(missed_by_1x1.sum())
            print(f"\n  Position type: {analysis_pt}")
            print(f"  Missed by 1x1x1 but found by 3x3x3: {n_missed}")

            if n_missed > 0:
                elem_to_cells_offsets = mesh_octree_cells_multi.element_to_cells_offsets
                elem_to_cells_data = mesh_octree_cells_multi.element_to_cells_data
                cell_morton_codes_cpu = mesh_octree_cells_multi.cell_morton_codes
                cell_levels_cpu = mesh_octree_cells_multi.cell_levels

                unique_levels_cpu = np.unique(cell_levels_cpu)
                max_level = int(np.max(unique_levels_cpu))
                level_cell_sizes_cpu = np.zeros((max_level + 1, 3), dtype=np.float64)
                cell_sizes_cpu = mesh_octree_cells_multi.cell_sizes
                for lev in unique_levels_cpu:
                    level_mask = cell_levels_cpu == lev
                    level_cell_sizes_cpu[lev] = cell_sizes_cpu[level_mask][0]

                morton_offset = 1 << 19
                morton_max_coord = 1 << 20

                cell_lookup = {}
                for cidx in range(len(cell_morton_codes_cpu)):
                    key = (int(cell_morton_codes_cpu[cidx]), int(cell_levels_cpu[cidx]))
                    cell_lookup[key] = cidx

                miss_indices = np.where(missed_by_1x1)[0]
                n_analyze = min(n_missed, 5000)

                n_elem_not_in_center = 0
                n_center_cell_missing = 0
                n_elem_in_center = 0
                neighbor_offsets_hist = {}

                for idx in miss_indices[:n_analyze]:
                    pos = positions_ana[idx]
                    true_elem = int(eids_3x3[idx])

                    e_start = elem_to_cells_offsets[true_elem]
                    e_end = elem_to_cells_offsets[true_elem + 1]
                    elem_cell_ids = set(int(elem_to_cells_data[j]) for j in range(e_start, e_end))

                    found_in_center_any_level = False
                    found_in_neighbor_any_level = False

                    for lev in unique_levels_cpu:
                        lev = int(lev)
                        cs = level_cell_sizes_cpu[lev]
                        if cs[0] == 0:
                            continue

                        gi = int(np.floor(pos[0] / cs[0]))
                        gj = int(np.floor(pos[1] / cs[1]))
                        gk = int(np.floor(pos[2] / cs[2]))

                        gi_m = max(0, min(gi + morton_offset, morton_max_coord - 1))
                        gj_m = max(0, min(gj + morton_offset, morton_max_coord - 1))
                        gk_m = max(0, min(gk + morton_offset, morton_max_coord - 1))

                        center_morton = encode_morton_3d(gi_m, gj_m, gk_m)
                        center_key = (center_morton, lev)
                        center_cidx = cell_lookup.get(center_key, -1)

                        if center_cidx >= 0 and center_cidx in elem_cell_ids:
                            found_in_center_any_level = True
                            break

                        if not found_in_neighbor_any_level:
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    for dk in range(-1, 2):
                                        if di == 0 and dj == 0 and dk == 0:
                                            continue
                                        ni = max(0, min(gi + di + morton_offset, morton_max_coord - 1))
                                        nj = max(0, min(gj + dj + morton_offset, morton_max_coord - 1))
                                        nk = max(0, min(gk + dk + morton_offset, morton_max_coord - 1))
                                        nb_morton = encode_morton_3d(ni, nj, nk)
                                        nb_key = (nb_morton, lev)
                                        nb_cidx = cell_lookup.get(nb_key, -1)
                                        if nb_cidx >= 0 and nb_cidx in elem_cell_ids:
                                            found_in_neighbor_any_level = True
                                            offset_key = (di, dj, dk)
                                            neighbor_offsets_hist[offset_key] = \
                                                neighbor_offsets_hist.get(offset_key, 0) + 1

                    if found_in_center_any_level:
                        n_elem_in_center += 1
                    elif found_in_neighbor_any_level:
                        n_elem_not_in_center += 1
                    else:
                        n_center_cell_missing += 1

                print(f"\n  Analyzed {n_analyze} failures:")
                print(f"    Element NOT in center cell, found in neighbor:  {n_elem_not_in_center} "
                      f"({100*n_elem_not_in_center/n_analyze:.1f}%)")
                print(f"    Element IN center cell (level/search issue):    {n_elem_in_center} "
                      f"({100*n_elem_in_center/n_analyze:.1f}%)")
                print(f"    No matching cell at any level (edge case):      {n_center_cell_missing} "
                      f"({100*n_center_cell_missing/n_analyze:.1f}%)")

                if neighbor_offsets_hist:
                    print(f"\n  Neighbor offset distribution (where 3x3x3 finds the element):")
                    sorted_offsets = sorted(neighbor_offsets_hist.items(), key=lambda x: -x[1])
                    total_neighbor = sum(v for _, v in sorted_offsets)
                    for offset, count in sorted_offsets[:15]:
                        di, dj, dk = offset
                        n_nonzero = sum(1 for d in offset if d != 0)
                        if n_nonzero == 1:
                            offset_type = "face-adjacent"
                        elif n_nonzero == 2:
                            offset_type = "edge-adjacent"
                        else:
                            offset_type = "corner-adjacent"
                        print(f"      ({di:+d},{dj:+d},{dk:+d}) {offset_type:16s}: "
                              f"{count:5d} ({100*count/total_neighbor:.1f}%)")

                    face_adj = sum(v for (di, dj, dk), v in sorted_offsets
                                   if sum(1 for d in (di, dj, dk) if d != 0) == 1)
                    edge_adj = sum(v for (di, dj, dk), v in sorted_offsets
                                   if sum(1 for d in (di, dj, dk) if d != 0) == 2)
                    corner_adj = sum(v for (di, dj, dk), v in sorted_offsets
                                     if sum(1 for d in (di, dj, dk) if d != 0) == 3)
                    print(f"\n  Summary by adjacency type:")
                    print(f"    Face-adjacent  (6 cells):  {face_adj:5d} ({100*face_adj/total_neighbor:.1f}%)")
                    print(f"    Edge-adjacent (12 cells):  {edge_adj:5d} ({100*edge_adj/total_neighbor:.1f}%)")
                    print(f"    Corner-adjacent (8 cells): {corner_adj:5d} ({100*corner_adj/total_neighbor:.1f}%)")

    # ========================================================================
    # [5/5] SCALABILITY SWEEP
    # ========================================================================
    if args.scalability:
        print()
        print("=" * 90)
        print("[5/5] SCALABILITY SWEEP: Throughput vs. batch size (3x3x3 method)")
        print("=" * 90)
        print(f"  Warmup runs: {n_warmup}, Timing runs: {n_runs}")
        print()

        scalability_sizes = args.scalability_sizes
        scalability_results = []

        for n_p in scalability_sizes:
            # Generate particles for this batch size
            rng_scale = np.random.default_rng(args.seed + n_p)
            positions_s, _ = generate_particles(
                connectivity, node_positions, n_p, 0.0, rng_scale, element_sizes,
                valid_element_ids=valid_element_ids,
            )
            positions_s_gpu = jax.device_put(positions_s.astype(config.FLOAT_DTYPE_NP))

            # Search with timing
            batch_sz = min(args.batch_size, n_p)

            def search_fn(p, _bs=batch_sz):
                return search_3x3x3_batch(p, octree_gpu, _bs)

            raw_result, times = timed_search(search_fn, positions_s_gpu, n_warmup, n_runs)

            found_eids, tests = raw_result
            n_found = int((found_eids >= 0).sum())
            mean_pit = float(tests.mean())
            t_mean = np.mean(times)
            t_min = np.min(times)
            t_max = np.max(times)
            qps = n_p / t_mean
            pit_s = (n_p * mean_pit) / t_mean
            us_per_query = 1e6 * t_mean / n_p

            scalability_results.append({
                'n_p': n_p,
                'n_found': n_found,
                'mean_pit': mean_pit,
                't_mean': t_mean,
                't_min': t_min,
                't_max': t_max,
                'qps': qps,
                'pit_s': pit_s,
                'us_per_query': us_per_query,
            })

            print(f"  N_p={n_p:>8,}: found={n_found:,}/{n_p:,}, "
                  f"time={fmt_time_stats(times)}s, "
                  f"{qps:,.0f} queries/s, "
                  f"{us_per_query:.1f} us/query, "
                  f"mean_PIT={mean_pit:.1f}")

        # Summary table
        print()
        print(f"  {'N_p':>8s}  {'Time (s)':>10s}  {'Queries/s':>12s}  {'us/query':>10s}  "
              f"{'PIT tests/s':>14s}  {'Mean PIT':>9s}  {'Found':>8s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*9}  {'-'*8}")
        for r in scalability_results:
            found_pct = 100 * r['n_found'] / r['n_p']
            print(f"  {r['n_p']:>8,}  {r['t_mean']:>10.4f}  {r['qps']:>12,.0f}  "
                  f"{r['us_per_query']:>10.1f}  {r['pit_s']:>14,.0f}  "
                  f"{r['mean_pit']:>9.1f}  {found_pct:>7.1f}%")

    print()
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
