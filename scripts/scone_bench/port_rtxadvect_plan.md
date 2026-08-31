# RTXAdvect port plan — 4-hour timeboxed attempt

**Target**: RTXAdvect (Wang et al. 2022, CPC) compiling on CUDA 13.3 +
OptiX 9.1 + RTX 5090 (Blackwell / sm_120).

**Timebox**: half a day. If not compiling + running by end of the
box, abandon and fall back to OLBVH (Kohek 2020) in JAX.

## What's broken

The bundled `owl/` submodule is an ~2020 snapshot of the OptiX
Wrappers Library. RTXAdvect's own top-level code (~1000 lines)
doesn't call OptiX APIs directly — everything goes through OWL.
Errors from the initial build:

1. `owl::common::vec2ui / vec3ui / vec4ui / vec2ul / vec3ul / vec4ul`
   typedefs missing → OWL naming changed
2. `cudaMemAdviseSetPreferredLocation` takes `cudaMemLocation` not
   `int` in CUDA 12+
3. `divRoundUp(uint32_t,uint32_t)` ambiguous (int32 vs int64
   overloads) → C++ standard changed, needs explicit cast
4. `OptixPipelineLinkOptions.overrideUsesMotionBlur` removed in OptiX 8
5. `OptixBuildInput.aabbArray` renamed (customPrimitiveArray) in OptiX 8
6. `optixModuleCreateFromPTX` → `optixModuleCreate` (OptiX 8+)

## Strategy: swap OWL submodule for NVIDIA/OWL current

**Rationale**: All 6 errors are in the bundled `owl/` code, not in
RTXAdvect proper. NVIDIA/OWL is maintained, supports CUDA 12+, and
has API-level shims for OptiX 7-8 (they carry backwards compat
inside their own headers). If OptiX 9 is the sticking point, we can
downgrade to OptiX 8.1 — which is what NVIDIA/OWL targets, and which
still supports Blackwell (Blackwell = sm_120 landed in CUDA 12.4,
not OptiX-specific).

## Step-by-step (workstation time)

### Phase 1 · Try NVIDIA/OWL swap (target: 60 min)

```bash
cd /flash/shared/jax/JAXTrace/3rdParty/RTXAdvect
# Back up the old OWL just in case
mv owl owl.orig-2020
# Clone the current NVIDIA/OWL as a submodule replacement
git clone --depth 1 https://github.com/NVIDIA/OWL.git owl
# Point RTXAdvect's CMakeLists at the new OWL path (if needed;
# usually the CMake already looks in ./owl/)
```

Then rerun `setup_rtxadvect.sh`. Expected outcomes:

- **Best case**: NVIDIA/OWL builds cleanly under OptiX 9.1, and
  RTXAdvect's own code (`cudaParticleAdvection.cpp`,
  `sample.cu`) compiles because OWL's API surface for downstream
  users has been kept stable. **Continue to Phase 3.**
- **Likely case**: NVIDIA/OWL builds cleanly but its API has
  drifted from the 2020 version RTXAdvect calls. Fix the small
  RTXAdvect surface: `owl::vec3f` etc. may have moved namespaces.
  **Go to Phase 2.**
- **Worst case**: NVIDIA/OWL requires OptiX 7.x + specific
  variables. Abandon.

### Phase 2 · Fix the RTXAdvect-side calls (target: 90 min)

Only touch RTXAdvect's own source (not the OWL submodule). Common
migrations:
- `owl::common::vec3f` → `owl::vec3f` (namespace flattened in newer OWL)
- `owlContextCreate(devs)` may want an explicit `enableValidation`
  argument now
- Include-path changes: `<owl/owl.h>` still works, but samples
  reorganized

### Phase 3 · Fix the OptiX 9 → 8 fallback (target: 30 min)

If OptiX 9 still causes problems in NVIDIA/OWL:
```bash
# Download OptiX 8.1 SDK from https://developer.nvidia.com/optix
# Install to /flash/shared/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64
# Rerun setup with:
OPTIX_ROOT=/flash/shared/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64 \
  scripts/scone_bench/setup_rtxadvect.sh
```

### Phase 4 · Smoke test (target: 30 min)

Once compiling, run the tool's own sample:
```bash
cd /flash/shared/jax/JAXTrace/3rdParty/RTXAdvect/build
./cudaParticleAdvection --help
```

Then a tiny mesh test:
```bash
scripts/scone_bench/vtu_to_rtxadvect.py \
    --vtu /flash/shared/jax/JAXTrace/kim_meshes_vtu/FinalFuelPinTet63.vtu \
    --out-prefix /tmp/tet63
./cudaParticleAdvection \
    --num-particles 100 --num-steps 1 \
    --input_mesh /tmp/tet63.verts.dat /tmp/tet63.cells.dat \
    --input_tet_velocity_field /tmp/tet63.zero_velocity.dat \
    -dt 1e-3 \
    --seeding-box <bbox from tet63.meta.json>
```

### Phase 5 · Sweep (target: 30 min setup + run)

If smoke passes, launch `sweep_rtxadvect.sh 100000 1` and let it
run against the same Kim + FSW cohort.

## Go / no-go gates

- After Phase 1 (60 min): if RTXAdvect still doesn't compile → **STOP**, jump to OLBVH plan.
- After Phase 2 (150 min total): if compile errors persist → **STOP**, jump to OLBVH plan.
- After Phase 4 (240 min total, half-day boundary): if the binary
  runs but produces wrong results → **STOP**, jump to OLBVH plan.

## Fallback: OLBVH (Kohek et al. 2020) in JAX — 3-5 days

Rationale for choosing OLBVH:

- Peer-reviewed (*The Visual Computer* 36(11):2317-2329, 2020)
- Textbook LBVH construction via Karras 2012 GPU parallel build
- No third-party dependency — pure Python/JAX
- Runs on our existing infrastructure
- Reviewer cannot say "you implemented our method wrong" — the
  algorithm is Kohek's, but the implementation is entirely ours,
  benchmarked apples-to-apples with MALMO on the same
  hardware/framework

Deliverable: `jaxtrace/gpu/search/olbvh.py` with the same interface
as MALMO's `search_mesh_aligned_octree_multi_local_where`. Sweeps
the same 9 Kim meshes + FSW paper mesh, results merge into
`joint_summary.md`.

## Why not try to write our own OptiX code?

Even if we wrote from scratch, we'd hit the same OptiX 9 API
migration Wang's code hit — and it'd take a month, not a week. The
right level of abstraction is JAX+CUDA (which is what MALMO already
is), and the peer-reviewed algorithm at that level is OLBVH.
