import importlib

mods = [
    "jaxtrace",
    "jaxtrace.io", "jaxtrace.io.vtk_io", "jaxtrace.io.hdf5_io", "jaxtrace.io.registry",
    "jaxtrace.fields", "jaxtrace.fields.base", "jaxtrace.fields.structured", "jaxtrace.fields.time_series",
    "jaxtrace.integrators", "jaxtrace.integrators.base", "jaxtrace.integrators.euler",
    "jaxtrace.integrators.rk2", "jaxtrace.integrators.rk4",
    "jaxtrace.tracking", "jaxtrace.tracking.particles", "jaxtrace.tracking.boundary",
    "jaxtrace.tracking.seeding", "jaxtrace.tracking.tracker",
    "jaxtrace.density", "jaxtrace.density.kernels", "jaxtrace.density.neighbors",
    "jaxtrace.density.kde", "jaxtrace.density.sph",
    "jaxtrace.visualization", "jaxtrace.visualization.static",
    "jaxtrace.visualization.dynamic", "jaxtrace.visualization.export_viz",
    "jaxtrace.utils", "jaxtrace.utils.jax_utils", "jaxtrace.utils.spatial",
    "jaxtrace.utils.logging", "jaxtrace.utils.random",
]

for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        print("Failed:", m, "->", repr(e))


