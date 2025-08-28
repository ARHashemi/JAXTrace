from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from .base import TimeDependentField, GridMeta
from .structured import StructuredGridSampler
from .unstructured import UnstructuredField, UnstructuredMesh, ElementType


ArrayLike = Union[jnp.ndarray, "numpy.ndarray"]


@dataclass
class _DeviceRing:
    """
    Small device ring buffer holding up to 3 time slices on device memory.
    Keeps indices to identify which time step is cached and replaces oldest on update,
    matching the idea of reloading fields periodically and limiting cache growth[^3,^4,^5,^6].
    """
    max_slices: int = 3

    def __post_init__(self):
        self._keys = []     # time indices
        self._bufs = []     # device arrays

    def get(self, t_idx: int, loader: Callable[[], ArrayLike]) -> jnp.ndarray:
        if t_idx in self._keys:
            i = self._keys.index(t_idx)
            return self._bufs[i]
        # load to device
        host_arr = loader()
        dev = jax.device_put(host_arr)
        # insert/replace
        if len(self._keys) < self.max_slices:
            self._keys.append(t_idx)
            self._bufs.append(dev)
        else:
            # replace the oldest
            self._keys.pop(0)
            self._bufs.pop(0)
            self._keys.append(t_idx)
            self._bufs.append(dev)
        return dev

    def clear(self):
        self._keys.clear()
        self._bufs.clear()


class TimeSeriesField(TimeDependentField):
    """
    Generic time-series field with temporal interpolation and a pluggable spatial sampler.

    Usage:
      - Structured grids via from_dataset(..., grid_type="structured")
      - Unstructured meshes: from_unstructured(...)

    Temporal sampling follows the same approach as your RK methods: get value at t by
    blending adjacent slices, then spatially sampling at x[^17,^8].
    """

    def __init__(
        self,
        num_slices: int,
        loader: Callable[[int], ArrayLike],
        sampler: Union[StructuredGridSampler, UnstructuredField],
        bounds_arr: Optional[jnp.ndarray] = None,
        cache_device_slices: int = 3
    ):
        self.num_slices = int(num_slices)
        self.loader = loader
        self.sampler = sampler
        self._bounds = bounds_arr
        self._ring = _DeviceRing(max_slices=cache_device_slices)

    @classmethod
    def from_dataset(
        cls,
        ds,
        grid_type: str = "structured",
        cache_bytes: Optional[int] = None,  # reserved
    ):
        """
        Build a TimeSeriesField from a dataset (VTK/HDF5) opened via io.registry.open_dataset.

        For structured grids, we infer GridMeta from ds.grid_meta() when available.
        """
        n = len(ds)

        if grid_type == "structured":
            # Values per slice: (Nx,Ny,Nz,C)
            meta = ds.grid_meta() if hasattr(ds, "grid_meta") else None
            if meta is None:
                raise ValueError("Structured dataset must provide grid_meta()")
            meta_j = GridMeta(
                origin=jnp.asarray(meta.origin),
                spacing=jnp.asarray(meta.spacing),
                shape=meta.shape,
                bounds=jnp.asarray(meta.bounds),
            )
            sampler = StructuredGridSampler(meta=meta_j)

            def loader(i: int):
                return ds.load_slice(i)  # numpy array, moved to device on demand

            return cls(
                num_slices=n,
                loader=loader,
                sampler=sampler,
                bounds_arr=meta_j.bounds,
            )
        else:
            raise ValueError("For unstructured meshes, use from_unstructured(...)")

    @classmethod
    def from_unstructured(
        cls,
        nodes: ArrayLike,
        elements: Optional[ArrayLike],
        etype: str,
        order: int,
        slice_loader: Callable[[int], ArrayLike],
        num_slices: int,
        knn_k: Optional[int] = None,
        strict_inside: bool = False,
        bounds: Optional[ArrayLike] = None,
        cache_device_slices: int = 3,
    ):
        """
        Create a time-series unstructured field.

        - nodes: (Pn,3)
        - elements: (M,k) or None
        - slice_loader(i): returns nodal values at slice i as (Pn,C)
        """
        mesh = UnstructuredMesh(
            nodes=jnp.asarray(nodes),
            elements=None if elements is None else jnp.asarray(elements, dtype=jnp.int32),
            etype=ElementType(etype),
            order=int(order),
        )
        sampler = UnstructuredField(mesh=mesh, knn_k=knn_k, strict_inside=strict_inside)
        b = None
        if bounds is not None:
            b = jnp.asarray(bounds)
        else:
            lo = jnp.min(mesh.nodes, axis=0)
            hi = jnp.max(mesh.nodes, axis=0)
            b = jnp.stack([lo, hi], axis=0)
        return cls(
            num_slices=int(num_slices),
            loader=slice_loader,
            sampler=sampler,
            bounds_arr=b,
            cache_device_slices=cache_device_slices
        )

    def bounds(self) -> jnp.ndarray:
        return self._bounds

    def _temporal_blend(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Linearly blend values from floor/ceil time slices at point x, following the same
        temporal interpolation approach used in your integration functions[^17,^8].
        """
        # map t in [0, num_slices-1]
        tclamp = jnp.clip(t, 0.0, float(self.num_slices - 1))
        i0 = jnp.floor(tclamp).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, self.num_slices - 1)
        w = tclamp - i0.astype(tclamp.dtype)

        v0 = self._ring.get(int(i0), lambda: self.loader(int(i0)))
        v1 = self._ring.get(int(i1), lambda: self.loader(int(i1)))

        # Spatial sample for each slice, then linear blend in time
        s0 = self.sampler.sample_given_values(x, v0)
        s1 = self.sampler.sample_given_values(x, v1)
        return (1.0 - w) * s0 + w * s1

    @jax.jit
    def sample_t(self, x: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Sample the field at positions x and scalar time t.
        """
        return self._temporal_blend(x, t)

    def prefetch(self, t_indices):
        """
        Proactively stage slices on device. This mirrors your previous pattern of
        periodically reloading velocity fields and managing a small cache[^3,^4,^5,^6].
        """
        for ti in t_indices:
            _ = self._ring.get(int(ti), lambda: self.loader(int(ti)))

    def clear_device_cache(self):
        self._ring.clear()
