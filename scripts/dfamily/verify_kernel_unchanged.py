#!/usr/bin/env python3
"""Prove the production RK4 kernel is unchanged by the substage-trace feature.

Compiles the production (no-stats, no-trace) step function and fingerprints
its OPTIMIZED HLO. Run once with the feature present and once against a source
without it; the fingerprints must match, or "zero cost when tracing is off" is
false.

Source line numbers and columns are stripped before hashing: adding lines to
the file shifts the debug metadata of every instruction below them without
changing a single operation. Everything else -- instruction list, fusion
count, shapes -- is compared verbatim.

Measured 2026-10-13 on this change: 54,693 tokens and 1,009 fusions both
before and after, and identical hashes after stripping source locations.

Usage:  HLO_OUT=/tmp/hlo_x.txt python3 verify_kernel_unchanged.py
"""
import os, sys, hashlib
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, ".")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "scripts/dfamily")
from test_lattice_invariant import build_mesh
from jaxtrace.gpu.search.mesh_aligned_octree_parent_cube import extract_octree_cells_parent_cube
from jaxtrace.gpu.search.mesh_aligned_octree_gpu import upload_mesh_aligned_octree_to_gpu
from jaxtrace.gpu.tracking.rk4_fully_fused_timedep import create_rk4_fully_fused_timedep_with_stats
from jaxtrace import config
# production D2 settings: mesh-aligned octree with the _where multi-local search
config.L2_SEARCH_METHOD = 'mesh_aligned_octree'

s = 1/16
npos, conn = build_mesh([(s, s, s)], z0=0.0)
conn = conn.astype(np.int32)
cells = extract_octree_cells_parent_cube(npos, conn, tolerance=1e-6, verbose=False)
oct_gpu = upload_mesh_aligned_octree_to_gpu(conn, npos, cells, verbose=False)

n_el = conn.shape[0]
nbrs = jnp.full((n_el, 4), -1, dtype=jnp.int32)
vols = jnp.ones(n_el, dtype=jnp.float32)

step, step_stats = create_rk4_fully_fused_timedep_with_stats(
    mesh_gpu_connectivity=jnp.asarray(conn),
    mesh_gpu_node_positions=jnp.asarray(npos, dtype=jnp.float32),
    mesh_gpu_element_neighbors=nbrs,
    mesh_gpu_element_volumes=vols,
    mesh_gpu_global_morton=None,
    mesh_aligned_octree=oct_gpu,
    mesh_aligned_octree_use_multi_local=True,
    mesh_aligned_octree_use_where=True,

)

P = jnp.asarray(np.array([[0.1, 0.1, 0.02]]), dtype=jnp.float32)
E = jnp.asarray(np.array([0]), dtype=jnp.int32)
VF = jnp.zeros((1, npos.shape[0], 3), dtype=jnp.float32)

low = jax.jit(step).lower(P, E, 1e-4, VF, 0)
txt = low.compile().as_text()
# strip names/metadata that legitimately vary run to run
import re
norm = re.sub(r'metadata=\{[^}]*\}', '', txt)
# Strip source locations: added lines shift these without changing any op.
norm = re.sub(r'(end_)?line=\d+', 'L', norm)
norm = re.sub(r'(end_)?column=\d+', 'C', norm)
norm = re.sub(r'source_file="[^"]*"', 'F', norm)
# This harness's own path appears in the HLO; normalise it so running the
# script from two different locations does not look like a kernel change.
norm = re.sub(r'"[^"]*(hlo_keep|verify_kernel_unchanged)\.py"', 'HARNESS', norm)
norm = re.sub(r'\s+', ' ', norm)
open(os.environ.get("HLO_OUT","/tmp/hlo_dump.txt"),"w").write(norm)
print("HLO_SHA256", hashlib.sha256(norm.encode()).hexdigest())
print("HLO_BYTES", len(norm))
print("N_FUSIONS", norm.count("fusion"))
