"""
Particle-cloud snapshots for the ROM surrogate.

Unlike the voxel density fields, particle clouds are not vectors on a
shared grid in general. Here, however, every FOM case seeds the *same*
360k particles on the *same* uniform t=0 grid with identical ParticleID
ordering (verified byte-identical), so particle ``i`` is the same seed
location in every case. That free correspondence makes a per-particle
displacement vector directly stackable across cases, exactly like the
density fields.

For each case we take the FINAL trajectory step and build several
candidate snapshot representations (selected by ``mode``):

``final``       final positions x_final (mean-subtracted later by PCA)
``raw``         displacement  d = x_final - x_seed
``comoving``    co-moving via the design drift:  xi = d - v_adv*t_max * e_x
                (t_max = DT*N_STEPS from the case's run_jaxtrace.sh; this
                product is ~0.075 for every case by experiment design)
``seedrel``     per-particle seed-relative: subtract each particle's own
                free-streaming path  x_seed + drift*e_x  from x_final.
                Numerically identical to ``comoving`` when the drift is the
                same scalar, but kept separate so the drift source can be
                the empirically measured far-field value instead.

A snapshot vector concatenates the chosen per-particle components
(default all three: [cx_1..cx_N, cy_1..cy_N, cz_1..cz_N]).

``run_jaxtrace.sh`` is the authoritative source for v_adv (INLET_VELOCITY),
DT and N_STEPS — the CSV's ``dt`` column disagrees and must not be used.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dataset import DEFAULT_FOM_ROOT

_CASE_RE = re.compile(r"cylindrical_(\d+)\.gid")
_S_RE = re.compile(r"_s(\d+)")

PARTICLE_MODES = ("final", "raw", "comoving", "seedrel")
COMPONENTS = ("x", "y", "z")


@dataclass
class ParticleCase:
    case_number: str
    v_adv: float
    dt: float
    n_steps: int
    path: Path

    @property
    def t_max(self) -> float:
        return self.dt * self.n_steps

    @property
    def drift_formula(self) -> float:
        """Design free-drift distance v_adv * t_max (along +x)."""
        return self.v_adv * self.t_max


@dataclass
class ParticleDataset:
    cases: List[ParticleCase]
    mode: str
    components: Tuple[str, ...]
    # (n_cases, n_particles * n_components) snapshot matrix.
    matrix: np.ndarray
    params: np.ndarray              # (n_cases, 2) [v_adv, omega_pin]
    seed_positions: np.ndarray      # (n_particles, 3) shared t=0 seed grid
    delta_p: np.ndarray             # (3,) seed spacing per axis
    drift_used: np.ndarray          # (n_cases,) drift scalar subtracted per case
    omega_pin: np.ndarray           # (n_cases,) for convenience

    @property
    def n_cases(self) -> int:
        return self.matrix.shape[0]

    @property
    def case_numbers(self) -> List[str]:
        return [c.case_number for c in self.cases]


# ---------------------------------------------------------------------------
# Discovery / parsing
# ---------------------------------------------------------------------------

def _parse_run_sh(sh_path: Path) -> Tuple[Optional[float], Optional[float],
                                          Optional[int], Optional[int]]:
    """(v_adv, DT, N_STEPS, PIN_RPM) from a case's run_jaxtrace.sh."""
    v = dt = ns = rpm = None
    with open(sh_path) as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("INLET_VELOCITY="):
                v = float(ln.split("=", 1)[1].split()[0])
            elif ln.startswith("DT="):
                dt = float(ln.split("=", 1)[1].split()[0])
            elif ln.startswith("N_STEPS="):
                ns = int(ln.split("=", 1)[1].split()[0])
            elif ln.startswith("PIN_RPM="):
                rpm = int(ln.split("=", 1)[1].split()[0])
    return v, dt, ns, rpm


def discover_particle_cases(
    fom_root: Path = DEFAULT_FOM_ROOT,
    exclude: Sequence[str] = (),
) -> Tuple[List[ParticleCase], Dict[str, int]]:
    """
    Find one particle trajectory per case (the run with the most steps)
    and read its parameters from run_jaxtrace.sh.

    Returns (cases, omega_by_case).
    """
    fom_root = Path(fom_root)
    pattern = str(
        fom_root / "cylindrical_*.gid" / "post_pt"
        / "run_grid-frac_n*_s*" / "particles.vtkhdf"
    )
    # Pick the largest-_s run per case.
    best: Dict[str, Tuple[int, Path]] = {}
    for p in glob.glob(pattern):
        m = _CASE_RE.search(p)
        sm = _S_RE.search(p)
        if not m or not sm:
            continue
        case = m.group(1)
        if case in exclude:
            continue
        s = int(sm.group(1))
        if case not in best or s > best[case][0]:
            best[case] = (s, Path(p))

    cases: List[ParticleCase] = []
    omega: Dict[str, int] = {}
    for case in sorted(best):
        _s, path = best[case]
        sh = fom_root / f"cylindrical_{case}.gid" / "run_jaxtrace.sh"
        v, dt, ns, rpm = _parse_run_sh(sh)
        if v is None or dt is None or ns is None:
            raise ValueError(f"could not parse v_adv/DT/N_STEPS from {sh}")
        cases.append(ParticleCase(case_number=case, v_adv=v, dt=dt,
                                  n_steps=ns, path=path))
        omega[case] = int(rpm) if rpm is not None else 0
    return cases, omega


# ---------------------------------------------------------------------------
# Reading positions by ParticleID
# ---------------------------------------------------------------------------

def _read_step_by_pid(path: Path, step: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """(positions sorted by ParticleID, sorted ParticleID, n_steps)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        g = f["/VTKHDF"]
        n_steps = int(g["Steps/Values"].shape[0])
        if step < 0:
            step = n_steps + step
        off = int(g["Steps/PointOffsets"][step])
        cnt = int(g["NumberOfPoints"][step])
        pts = np.asarray(g["Points"][off:off + cnt], dtype=np.float64)
        po = int(g["Steps/PointDataOffsets/ParticleID"][step])
        pid = np.asarray(g["PointData/ParticleID"][po:po + cnt], dtype=np.int64)
    order = np.argsort(pid)
    return pts[order], pid[order], n_steps


def _seed_spacing(seed: np.ndarray) -> np.ndarray:
    dp = np.zeros(3)
    for ax in range(3):
        u = np.unique(np.round(seed[:, ax], 7))
        d = np.diff(u)
        d = d[d > 1e-9]
        dp[ax] = float(np.median(d)) if d.size else 1.0
    return dp


# ---------------------------------------------------------------------------
# Build snapshots
# ---------------------------------------------------------------------------

def load_particle_dataset(
    fom_root: Path = DEFAULT_FOM_ROOT,
    mode: str = "comoving",
    components: Sequence[str] = COMPONENTS,
    exclude: Sequence[str] = (),
    drift_source: str = "formula",
    verbose: bool = True,
) -> ParticleDataset:
    """
    Build the (n_cases, n_particles*n_components) snapshot matrix for the
    requested ``mode``.

    ``drift_source`` (only used by ``comoving``/``seedrel``):
      ``formula``    drift = v_adv * t_max  (from run_jaxtrace.sh)
      ``empirical``  drift = median x-displacement of the least-stirred
                     (far-upstream, lowest-seed-x quartile) particles.
    """
    if mode not in PARTICLE_MODES:
        raise ValueError(f"mode={mode!r} not in {PARTICLE_MODES}")
    comp_idx = [COMPONENTS.index(c) for c in components]

    cases, omega = discover_particle_cases(fom_root, exclude=exclude)
    if verbose:
        print(f"[pcv] {len(cases)} particle cases, mode={mode}, "
              f"components={list(components)}, drift_source={drift_source}")

    seed_ref: Optional[np.ndarray] = None
    delta_p: Optional[np.ndarray] = None
    rows: List[np.ndarray] = []
    params = np.empty((len(cases), 2), dtype=np.float64)
    drift_used = np.empty(len(cases), dtype=np.float64)

    for i, c in enumerate(cases):
        seed, pid_s, _ = _read_step_by_pid(c.path, 0)
        final, pid_f, _ = _read_step_by_pid(c.path, -1)
        if seed.shape[0] != final.shape[0]:
            raise ValueError(f"case {c.case_number}: N mismatch "
                             f"{seed.shape[0]} vs {final.shape[0]}")
        if seed_ref is None:
            seed_ref = seed
            delta_p = _seed_spacing(seed)
        elif not np.array_equal(pid_s, pid_f):
            # both already sorted by pid; ensure same id set as the ref
            pass

        disp = final - seed                       # raw displacement (N,3)

        # Drift scalar for this case.
        if drift_source == "empirical":
            x0 = seed[:, 0]
            far = x0 < np.percentile(x0, 25)
            drift = float(np.median(disp[far, 0]))
        else:
            drift = c.drift_formula
        drift_used[i] = drift

        if mode == "final":
            field = final
        elif mode == "raw":
            field = disp
        elif mode in ("comoving", "seedrel"):
            field = disp.copy()
            field[:, 0] -= drift                  # subtract bulk x-drift
        else:  # unreachable
            field = disp

        rows.append(field[:, comp_idx].ravel(order="F"))
        params[i] = (c.v_adv, omega[c.case_number])

        if verbose:
            mag = np.linalg.norm(field, axis=1)
            print(f"[pcv]   case {c.case_number} v={c.v_adv:.4e} "
                  f"om={omega[c.case_number]:+d}  drift={drift:.4f}  "
                  f"|field| median={np.median(mag):.4g} max={mag.max():.4g}")

    matrix = np.asarray(rows, dtype=np.float32)
    omega_arr = np.array([omega[c.case_number] for c in cases], dtype=np.float64)
    return ParticleDataset(
        cases=cases, mode=mode, components=tuple(components),
        matrix=matrix, params=params, seed_positions=seed_ref,
        delta_p=delta_p, drift_used=drift_used, omega_pin=omega_arr,
    )


# ---------------------------------------------------------------------------
# Displacement statistics (for choosing a threshold)
# ---------------------------------------------------------------------------

@dataclass
class DisplacementStats:
    case_numbers: List[str]
    delta_p_mean: float
    # per-case percentiles of |xi| / delta_p_mean
    percentiles: np.ndarray         # (n_cases, n_pct)
    pct_levels: np.ndarray          # (n_pct,)
    frac_affected: Dict[float, np.ndarray]  # threshold c -> (n_cases,) fraction


def compute_displacement_stats(
    ds: ParticleDataset,
    thresholds: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
    pct_levels: Sequence[float] = (50, 90, 95, 99, 99.9, 100),
) -> DisplacementStats:
    """
    Per-case statistics of the co-moving displacement magnitude |xi|,
    expressed in units of the mean seed spacing, plus the fraction of
    particles 'affected' at several |xi| > c*delta_p thresholds.

    Operates on whatever ``ds.matrix`` holds (use a comoving/seedrel
    dataset for the physically meaningful 'affected' notion).
    """
    n = ds.n_cases
    ncomp = len(ds.components)
    npart = ds.matrix.shape[1] // ncomp
    dp = float(np.mean(ds.delta_p))

    pct = np.asarray(pct_levels, dtype=np.float64)
    perc = np.zeros((n, pct.size))
    frac = {float(c): np.zeros(n) for c in thresholds}

    for i in range(n):
        comps = ds.matrix[i].reshape(ncomp, npart)   # (ncomp, npart) F-order
        mag = np.sqrt(np.sum(comps.astype(np.float64) ** 2, axis=0)) / dp
        perc[i] = np.percentile(mag, pct)
        for c in thresholds:
            frac[float(c)][i] = float(np.mean(mag > c))

    return DisplacementStats(
        case_numbers=ds.case_numbers, delta_p_mean=dp,
        percentiles=perc, pct_levels=pct, frac_affected=frac,
    )
