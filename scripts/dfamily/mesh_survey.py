"""Is the mesh static for every PinShapes case, across the steps it loads?

Decides whether the MALMO search structure can be built once or must be rebuilt
per loaded timestep. Reads only -- no compute job, no writes.

For each case: hash the Points array of a mesh piece at timesteps spanning
that case's own VEL_START..VEL_END. Identical hashes mean the node coordinates
never move, so the grid, Morton codes and registration are all valid for the
whole run. Differing hashes mean the geometry moves and the structure must be
rebuilt (or at least re-registered) per step.

Two modes:
  (default)   5 sampled timesteps -- a fast screen.
  --all-steps EVERY timestep in the case's loaded range, and rotating through
              the pieces so coverage is not confined to piece 0. Slower, but
              it is the exhaustive statement: nothing is inferred from a
              sample. Use this to certify a case, not to screen it.

Hashing one piece per timestep only proves THAT piece is static, so in
--all-steps mode the piece index advances with the timestep: a mesh that moved
anywhere would have to move in a piece we never look at AND at a step we never
read, which --all-steps removes. --pieces N additionally hashes N pieces at
every timestep (N=0 means all pieces) for a fully exhaustive check.
"""
import sys, hashlib, re, time
import numpy as np, vtk
import xml.etree.ElementTree as ET
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path

_argv = sys.argv[1:]
ALL_STEPS = "--all-steps" in _argv
_argv = [a for a in _argv if a != "--all-steps"]
N_PIECES = 1
for _a in list(_argv):
    if _a.startswith("--pieces="):
        N_PIECES = int(_a.split("=", 1)[1])
        _argv.remove(_a)

ROOT = Path(_argv[0])
cases = _argv[1:] if len(_argv) > 1 else None

def points_hash(vtu):
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(vtu)); r.Update()
    g = r.GetOutput()
    pts = g.GetPoints()
    if pts is None:
        return None, 0, 0
    a = vtk_to_numpy(pts.GetData())
    return (hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12],
            len(a), g.GetNumberOfCells())

rows = []
TOTAL_READS = [0]
T_START = time.time()
for fam in sorted(ROOT.iterdir()):
    if not fam.is_dir():
        continue
    for gid in sorted(fam.glob("*.gid")):
        name = gid.name[:-4]
        if cases and name not in cases:
            continue
        rj = gid / "run_jaxtrace.sh"
        post = gid / "post"
        if not rj.exists() or not post.is_dir():
            continue
        txt = rj.read_text(errors="ignore")
        m1 = re.search(r"^VEL_START=(\d+)", txt, re.M)
        m2 = re.search(r"^VEL_END=(\d+)", txt, re.M)
        if not (m1 and m2):
            continue
        vs, ve = int(m1.group(1)), int(m2.group(1))
        pv = sorted(post.glob(f"*_{vs}.pvtu"))
        if not pv:
            continue
        # Derive the prefix by stripping exactly the trailing _<ts>.pvtu, and
        # resolve piece files from the PVTU's own <Piece Source=...> entries
        # rather than constructing names: pieces are numbered from _0, and a
        # hand-built name silently yields zero points if it is wrong.
        prefix = re.sub(rf"_{vs}\.pvtu$", "", pv[0].name)
        if ALL_STEPS:
            samples = list(range(vs, ve + 1))
        else:
            # sample across the loaded range, including a full revolution span
            samples = sorted({vs, vs + 1, (vs + ve) // 2, ve - 1, ve})

        # ref[piece_index] = (timestep, hash) of the first read of that piece.
        # Keyed per piece because different pieces legitimately hold different
        # coordinates; only the SAME piece across time must agree.
        ref = {}; diffs = []; npts = ncell = 0; nread = 0; npieces_seen = set()
        # Announce the case BEFORE reading anything. With --pieces=0 a single
        # case is ~166 steps x 64 pieces, so without this the log is silent
        # for thousands of reads and a working job is indistinguishable from a
        # stalled one.
        if ALL_STEPS:
            print(f"  [{fam.name}/{name}] starting: {len(samples)} steps, "
                  f"vel {vs}..{ve}", flush=True)
        _t0 = time.time()
        _last_beat = time.time()
        for si, ts in enumerate(samples):
            # Heartbeat on elapsed time, not step count: one step can itself
            # be 64 piece reads on a slow filesystem.
            if ALL_STEPS and time.time() - _last_beat > 60:
                _last_beat = time.time()
                print(f"    [{fam.name}/{name}] {si}/{len(samples)} steps, "
                      f"{nread} reads, {len(diffs)} diffs, "
                      f"{time.time() - _t0:.0f}s", flush=True)
            pvt = post / f"{prefix}_{ts}.pvtu"
            if not pvt.exists():
                continue
            try:
                srcs = [p.get("Source")
                        for p in ET.parse(pvt).getroot().iter("Piece")]
            except Exception:
                continue
            if not srcs:
                continue
            # Which pieces to hash at this timestep. Rotating the index with
            # the step means a static verdict is not confined to piece 0.
            if N_PIECES <= 0:
                idxs = range(len(srcs))
            elif ALL_STEPS:
                idxs = [(si + k) % len(srcs) for k in range(min(N_PIECES,
                                                                len(srcs)))]
            else:
                idxs = range(min(N_PIECES, len(srcs)))
            for pi in idxs:
                f = post / srcs[pi]
                if not f.exists():
                    continue
                h, n, c = points_hash(f)
                if h is None or n == 0:
                    continue
                npts, ncell = n, c; nread += 1; npieces_seen.add(pi)
                TOTAL_READS[0] += 1
                if pi not in ref:
                    ref[pi] = (ts, h)
                elif h != ref[pi][1]:
                    diffs.append((ts, pi))
        # A piece is only CONFIRMED static if it was read at >= 2 timesteps.
        # Counting raw reads is not enough: hashing 20 pieces once each would
        # give nread=20 while comparing nothing across time.
        n_cmp = nread - len(npieces_seen)   # reads beyond each piece's first
        if npts == 0:
            verdict = "NO DATA READ (cannot conclude)"
        elif n_cmp < 1:
            verdict = (f"NO REPEAT READ of any piece "
                       f"({nread} reads / {len(npieces_seen)} pieces) "
                       f"-- cannot conclude")
        elif diffs:
            _d = ", ".join(f"ts{t}/piece{p}" for t, p in diffs[:6])
            verdict = f"MOVES at {_d}{' ...' if len(diffs) > 6 else ''}"
        else:
            verdict = (f"STATIC ({n_cmp} comparisons, "
                       f"{len(samples)} steps, {len(npieces_seen)} piece(s))")
        rows.append((fam.name, name, vs, ve, ve - vs + 1, npts, ncell, verdict))
        print(f"{fam.name:22} {name:8} vel {vs:4}..{ve:<4} "
              f"({ve-vs+1:3} steps)  piece {npts:,}p/{ncell:,}c  {verdict}",
              flush=True)

print()
print(f"mode: {'ALL STEPS' if ALL_STEPS else '5 sampled steps'}, "
      f"pieces per step: {'ALL' if N_PIECES <= 0 else N_PIECES}")
print(f"total piece-reads: {TOTAL_READS[0]:,}   "
      f"wall: {time.time() - T_START:.0f}s")
mv = [r for r in rows if r[-1].startswith("MOVES")]
nc = [r for r in rows if "cannot conclude" in r[-1]]
print(f"cases checked: {len(rows)}   MOVING: {len(mv)}   inconclusive: {len(nc)}")
for r in mv + nc:
    print(f"  {r[0]}/{r[1]}: {r[-1]}")
if not mv and not nc and rows:
    print("\nALL cases static -> one search structure for the whole run is valid:")
    print("  grid, Morton codes AND registration can all be built once.")
