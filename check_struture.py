import os

expected = {
    "jaxtrace/__init__.py",
    "jaxtrace/io/__init__.py",
    "jaxtrace/io/vtk_io.py",
    "jaxtrace/io/hdf5_io.py",
    "jaxtrace/io/registry.py",
    "jaxtrace/fields/__init__.py",
    "jaxtrace/fields/base.py",
    "jaxtrace/fields/structured.py",
    "jaxtrace/fields/time_series.py",
    "jaxtrace/integrators/__init__.py",
    "jaxtrace/integrators/base.py",
    "jaxtrace/integrators/euler.py",
    "jaxtrace/integrators/rk2.py",
    "jaxtrace/integrators/rk4.py",
    "jaxtrace/tracking/__init__.py",
    "jaxtrace/tracking/particles.py",
    "jaxtrace/tracking/boundary.py",
    "jaxtrace/tracking/seeding.py",
    "jaxtrace/tracking/tracker.py",
    "jaxtrace/density/__init__.py",
    "jaxtrace/density/kernels.py",
    "jaxtrace/density/neighbors.py",
    "jaxtrace/density/kde.py",
    "jaxtrace/density/sph.py",
    "jaxtrace/visualization/__init__.py",
    "jaxtrace/visualization/static.py",
    "jaxtrace/visualization/dynamic.py",
    "jaxtrace/visualization/export_viz.py",
    "jaxtrace/utils/__init__.py",
    "jaxtrace/utils/jax_utils.py",
    "jaxtrace/utils/spatial.py",
    "jaxtrace/utils/logging.py",
    "jaxtrace/utils/random.py",
}

present = set()
for root, _, files in os.walk("jaxtrace"):
    for f in files:
        present.add(os.path.join(root, f).replace("\\", "/"))

missing = sorted(expected - present)
extras = sorted(present - expected)

print("Missing:", missing)
print("Unexpected extras:", extras)


from jaxtrace.utils import jax_utils, spatial, random  

# JAX guards  
_ = jax_utils.JAX_AVAILABLE  

# maybe_jit fallback works  
f = lambda x: x + 1  
g = jax_utils.maybe_jit(f, enable=False)  
assert g(2) == 3  

# spatial hashing  
import numpy as np  
pts = np.random.rand(5, 3)  
keys, cells = spatial.grid_hash(pts, cell_size=0.1)  
assert keys.shape == (5,) and cells.shape == (5, 3) 

