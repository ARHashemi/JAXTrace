"""
Data-driven reduced-order modelling (ROM) for the FSW density surrogate.

This package builds a surrogate that maps the two process scalars
``(v_adv, omega_pin)`` to a full 3D mean-density field, starting with the
simplest linear dimension-reduction: PCA / POD via the SVD of the
snapshot matrix.

Submodules
----------
``dataset``
    Discover the FOM cases, read each ``mean_density`` snapshot from its
    VTKHDF ImageData file, and resample every snapshot onto a single
    common reference grid so the snapshots can be stacked into one
    matrix.
``pca``
    Mean-centred SVD of the snapshot matrix, singular-value spectra,
    cumulative energy coverage, and projection/reconstruction helpers.
"""

from .dataset import (
    CaseSnapshot,
    ReferenceGrid,
    SnapshotDataset,
    discover_cases,
    load_dataset,
)
from .pca import NORMALIZE_CHOICES, PCAResult, fit_pca
from .cv import LOOCVResult, REGRESSORS, loocv
from .features import FEATURE_TRANSFORMS
from .active_subspace import ActiveSubspaceResult, active_subspace
from .first_stage import FirstStageCoeffs, load_first_stage_coeffs
from .manifold import ManifoldReport, analyze_manifold
from .particles import (
    PARTICLE_MODES,
    ParticleDataset,
    compute_displacement_stats,
    load_particle_dataset,
)

__all__ = [
    "CaseSnapshot",
    "ReferenceGrid",
    "SnapshotDataset",
    "discover_cases",
    "load_dataset",
    "PCAResult",
    "fit_pca",
    "NORMALIZE_CHOICES",
    "LOOCVResult",
    "REGRESSORS",
    "loocv",
    "PARTICLE_MODES",
    "ParticleDataset",
    "load_particle_dataset",
    "compute_displacement_stats",
    "FEATURE_TRANSFORMS",
    "ActiveSubspaceResult",
    "active_subspace",
    "FirstStageCoeffs",
    "load_first_stage_coeffs",
    "ManifoldReport",
    "analyze_manifold",
]
