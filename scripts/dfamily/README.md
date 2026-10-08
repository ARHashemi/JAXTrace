# D-family diagnostic tooling

Two scripts for the "particles stop near the tool in D-ConcavityTilt" issue.
Background and measurements: `docs/dfamily/01_diagnosis.md`.

## The question they answer

When a particle stops moving, is it because

- the **search** could not find its host element (`ElementID < 0`), or
- the search worked and the **velocity it was given was zero**
  (`ElementID >= 0`) — e.g. the level set marked it as inside the tool?

The two need opposite fixes, so this must be settled before tuning anything.

## 1. `make_diag_run.sh` — prepare the run

```bash
scripts/dfamily/make_diag_run.sh \
    /scratch/project_465002752/lorenzgl/Cases/PinShapes/D-ConcavityTilt/D2.gid \
    1500 10
```

Copies the case's own `run_jaxtrace.sh` into
`/scratch/<proj>/<user>/dfamily/<case>_diag/` and changes exactly eight
things: `EXPORT_ELEMENT_IDS=1`, `OUTPUT_TARGET=scratch`, `N_STEPS`,
`EXPORT_FREQ`, `AUTO_DETECT_CASE=0`, `INPUT`, `RUN_TAG`, plus a shorter
SLURM time limit and job name.

**The colleague's case folder is never written to.** Review with the `diff`
the script prints, then `sbatch run_jaxtrace_diag.sh`.

1500 steps is enough: in the existing D2 output 18% of particles are frozen
before step 100 and the late-freeze population is well established by ~1000.

## 2. `analyze_frozen_particles.py` — read the result

```bash
python scripts/dfamily/analyze_frozen_particles.py <run_dir> \
    --level-set <case>.gid/post/<MESH>_34.pvtu \
    --out docs/dfamily/02_D2_frozen_report.md
```

Reports, as a markdown table:

1. how many particles are dead-on-arrival vs freeze later,
2. **for each group, the ElementID split** — the decisive test,
3. where they stop, by depth and radius, optionally against the tool region
   read from the `LEVEL` field.

Runs without `--level-set`, and without ElementID in the files (it then says
so rather than guessing). Needs VTK, so on LUMI run it inside the same
singularity image used for tracking.

### Reading the verdict

- **>70% of late freezers have `ElementID < 0`** → search failure. Act on
  `ENHANCED_SEARCH_BAND`, `L0_SKIP_BAND`, `L2_NEIGHBORHOOD`.
- **<30%** → the host is found and the velocity is zero. Search settings will
  not help; look at `LEVELSET_MODE` and the velocity field.
- **in between** → both are active; use the depth/radius table to separate
  them (in D2 the shoulder region and the deep pin region behave differently).

## Note on mesh file prefixes

The mesh PVTU prefix is **not** always the case name: `D2.gid/post` contains
`A2_*.vtu`, and `D1.gid/post` contains `A1_*.vtu`. Check with `ls` before
passing `--level-set`.
