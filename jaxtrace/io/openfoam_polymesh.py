"""
OpenFOAM polyMesh <-> VTK bidirectional bridge for the SCONE / AMLG-Patch benchmark.

Purpose
-------
The Kim / Ravoisin / Parks (2026) adaptive Patch-Search reference implementation
lives inside SCONE (a Monte Carlo neutron transport code, Cambridge Nuclear).
SCONE consumes meshes exclusively in OpenFOAM polyMesh ASCII format. To
benchmark MALMO against AMLG-Patch on the *same* meshes on the *same*
workstation, we need to round-trip meshes between the JAXTrace VTK stack
(VTU / PVTU / VTKHDF) and OpenFOAM polyMesh.

The bridge is bidirectional:

- ``read_polymesh_to_vtu(polymesh_dir, vtu_path)`` : consumes an OpenFOAM
  polyMesh directory (typically bundled with SCONE at
  ``IntegrationTestFiles/Geometry/Meshes/OpenFOAM/<name>/``) and writes a
  VTU that JAXTrace can load via ``jaxtrace.io.vtk_reader``.

- ``write_vtu_to_polymesh(vtu_path, polymesh_dir)`` : converts a JAXTrace VTU
  (tet or arbitrary polyhedral) into an OpenFOAM polyMesh directory that
  SCONE's ``OpenFOAMMesh`` reader accepts. Enforces the OpenFOAM
  "internal faces first + owner <= neighbour" ordering convention.

Format reference
----------------
- OpenFOAM v12 User Guide, 4.2 "Basic input/output file format"
  <https://doc.cfd.direct/openfoam/user-guide-v12/basic-file-format>
- OpenFOAM v12 User Guide, 4.1 "Mesh description"
- SCONE mesh reader: Geometry/Meshes/UnstructuredMeshes/OpenFOAMMesh_class.f90

polyMesh contents (all ASCII files inside one directory):
- ``points``    : N vertices, each ``(x y z)`` on one line, 0-indexed
- ``faces``     : M faces, each ``k(v0 v1 .. v{k-1})`` on one line; k is the
                   polygonal face size (3 for tri, 4 for quad, ...)
- ``owner``     : M face -> owner cell index (0-indexed)
- ``neighbour`` : M_internal face -> neighbour cell index (only internal faces,
                   which appear first; owner < neighbour convention)
- ``cellZones`` : optional, material zone assignment (SCONE requires it be
                   present but a single "internalMesh" zone covering all cells
                   is acceptable for our benchmarks)

Each file starts with an OpenFOAM ``FoamFile`` header block that we emit and
skip on read. The count of entries appears on a line by itself between the
header and the opening ``(``.

Notes
-----
- Uses only ``numpy`` + ``vtk`` (matches the rest of jaxtrace.io — no meshio,
  no pyvista dependency).
- Tet-first: the ``faces_from_tet_connectivity`` helper builds owner/neighbour
  arrays with the OpenFOAM internal-faces-first convention. For general
  polyhedral output from a VTU we fall back to face extraction from
  ``vtkGeometryFilter`` cell faces.
- All arrays that touch SCONE are int32 or float64 (SCONE's ``defReal`` is
  double precision).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# FoamFile header emission and parsing
# ---------------------------------------------------------------------------

_FOAM_HEADER_TEMPLATE = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Version:  12  JAXTrace bridge
    \\\\  /    A nd           |
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {klass};
    {note}location    "constant/polyMesh";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _foam_header(klass: str, obj: str, note: Optional[str] = None) -> str:
    note_line = f'note        "{note}";\n    ' if note else ""
    return _FOAM_HEADER_TEMPLATE.format(klass=klass, obj=obj, note=note_line)


def _read_foam_list(path: Path) -> List[str]:
    """
    Read an OpenFOAM ASCII list file and return the content lines that are
    inside the top-level ``(...)`` block (one entry per line, header stripped,
    count line stripped).
    """
    text = path.read_text()
    # Strip the block-comment header (up to the closing */)
    end_hdr = text.find("*/")
    if end_hdr != -1:
        text = text[end_hdr + 2 :]

    # Strip the FoamFile{...} dictionary (find first '{' after cleaning)
    brace_open = text.find("{")
    if brace_open != -1:
        depth = 0
        i = brace_open
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    text = text[i + 1 :]
                    break
            i += 1

    # Strip C++ line comments
    lines = []
    for raw in text.splitlines():
        idx = raw.find("//")
        if idx != -1:
            raw = raw[:idx]
        if raw.strip():
            lines.append(raw.rstrip())

    # First non-blank line is the count; then '(', then entries, then ')'.
    if not lines:
        raise ValueError(f"{path}: empty polyMesh file")
    try:
        _count = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"{path}: expected entry count on first non-blank line, got {lines[0]!r}")

    # Find opening paren
    try:
        open_idx = next(i for i, ln in enumerate(lines[1:], start=1) if ln.strip() == "(")
    except StopIteration:
        raise ValueError(f"{path}: no '(' found after count line")
    # Find matching closing paren (last ')' in the file)
    close_idx = None
    for i in range(len(lines) - 1, open_idx, -1):
        if lines[i].strip() == ")":
            close_idx = i
            break
    if close_idx is None:
        raise ValueError(f"{path}: no closing ')' found")

    return lines[open_idx + 1 : close_idx]


def _parse_points(entry_lines: List[str]) -> np.ndarray:
    """Parse ``(x y z)`` per line into an (N, 3) float64 array."""
    out = np.empty((len(entry_lines), 3), dtype=np.float64)
    for i, ln in enumerate(entry_lines):
        s = ln.strip().lstrip("(").rstrip(")")
        parts = s.split()
        if len(parts) != 3:
            raise ValueError(f"points entry {i}: expected 3 coords, got {ln!r}")
        out[i, 0] = float(parts[0])
        out[i, 1] = float(parts[1])
        out[i, 2] = float(parts[2])
    return out


def _parse_faces(entry_lines: List[str]) -> List[np.ndarray]:
    """Parse ``k(v0 v1 .. v{k-1})`` per line into a list of int32 arrays."""
    faces: List[np.ndarray] = []
    for i, ln in enumerate(entry_lines):
        s = ln.strip()
        # k(v0 v1 ...); tolerate whitespace variants
        open_p = s.find("(")
        close_p = s.rfind(")")
        if open_p < 0 or close_p < 0:
            raise ValueError(f"faces entry {i}: expected 'k(...)' got {ln!r}")
        k = int(s[:open_p].strip())
        vals = s[open_p + 1 : close_p].split()
        if len(vals) != k:
            raise ValueError(
                f"faces entry {i}: header says k={k} but got {len(vals)} vertices"
            )
        faces.append(np.asarray([int(v) for v in vals], dtype=np.int32))
    return faces


def _parse_int_list(entry_lines: List[str]) -> np.ndarray:
    """Parse one integer per line into an int32 array."""
    out = np.empty(len(entry_lines), dtype=np.int32)
    for i, ln in enumerate(entry_lines):
        out[i] = int(ln.strip())
    return out


# ---------------------------------------------------------------------------
# polyMesh -> in-memory
# ---------------------------------------------------------------------------


def read_polymesh(polymesh_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Read a full polyMesh directory into numpy arrays.

    Parameters
    ----------
    polymesh_dir : path
        Directory containing at least ``points``, ``faces``, ``owner``, and
        (if any internal faces exist) ``neighbour``.

    Returns
    -------
    dict with keys:
        points     : (nPoints, 3) float64
        faces      : list[int32 array]  (variable-length per face)
        owner      : (nFaces,) int32
        neighbour  : (nInternalFaces,) int32
        nCells     : int
        nInternalFaces : int
    """
    p = Path(polymesh_dir)
    if not p.is_dir():
        raise ValueError(f"{polymesh_dir!r} is not a directory")

    points = _parse_points(_read_foam_list(p / "points"))
    faces = _parse_faces(_read_foam_list(p / "faces"))
    owner = _parse_int_list(_read_foam_list(p / "owner"))
    if len(owner) != len(faces):
        raise ValueError(
            f"owner ({len(owner)}) and faces ({len(faces)}) length mismatch in {polymesh_dir}"
        )

    n_internal = 0
    neighbour = np.empty(0, dtype=np.int32)
    if (p / "neighbour").exists():
        neighbour = _parse_int_list(_read_foam_list(p / "neighbour"))
        n_internal = int(len(neighbour))

    n_cells = int(max(owner.max() if owner.size else -1, neighbour.max() if neighbour.size else -1) + 1)

    return {
        "points": points,
        "faces": faces,
        "owner": owner,
        "neighbour": neighbour,
        "nCells": n_cells,
        "nInternalFaces": n_internal,
    }


# ---------------------------------------------------------------------------
# polyMesh -> VTU (via cell reconstruction)
# ---------------------------------------------------------------------------


def _cell_vertex_sets(mesh: Dict[str, np.ndarray]) -> List[set]:
    """
    Group face-vertex sets by their owning cell. Each cell's vertex set is
    the union of its faces' vertex sets.
    """
    faces = mesh["faces"]
    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    n_cells = mesh["nCells"]

    cell_verts: List[set] = [set() for _ in range(n_cells)]
    for f_idx, fv in enumerate(faces):
        c_own = int(owner[f_idx])
        cell_verts[c_own].update(int(v) for v in fv)
        if f_idx < len(neighbour):
            c_nbr = int(neighbour[f_idx])
            cell_verts[c_nbr].update(int(v) for v in fv)
    return cell_verts


def _cell_face_sets(mesh: Dict[str, np.ndarray]) -> List[List[int]]:
    """For each cell return the list of face indices adjacent to it."""
    n_cells = mesh["nCells"]
    faces_of: List[List[int]] = [[] for _ in range(n_cells)]
    owner = mesh["owner"]
    neighbour = mesh["neighbour"]
    for f_idx in range(len(mesh["faces"])):
        faces_of[int(owner[f_idx])].append(f_idx)
        if f_idx < len(neighbour):
            faces_of[int(neighbour[f_idx])].append(f_idx)
    return faces_of


def polymesh_to_vtk_unstructured(mesh: Dict[str, np.ndarray]):
    """
    Convert an in-memory polyMesh dict to a vtkUnstructuredGrid.

    Cells with exactly 4 vertices and 4 triangular faces are emitted as
    ``VTK_TETRA``. All other cells are emitted as ``VTK_POLYHEDRON``.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required for polymesh_to_vtk_unstructured")

    ug = vtk.vtkUnstructuredGrid()
    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    pts.SetNumberOfPoints(len(mesh["points"]))
    for i, xyz in enumerate(mesh["points"]):
        pts.SetPoint(i, float(xyz[0]), float(xyz[1]), float(xyz[2]))
    ug.SetPoints(pts)

    faces_all = mesh["faces"]
    faces_of_cell = _cell_face_sets(mesh)
    verts_of_cell = _cell_vertex_sets(mesh)

    for cid in range(mesh["nCells"]):
        cell_faces = faces_of_cell[cid]
        cell_verts_sorted = sorted(verts_of_cell[cid])
        n_v = len(cell_verts_sorted)
        n_f = len(cell_faces)

        is_tet = (
            n_v == 4
            and n_f == 4
            and all(len(faces_all[f]) == 3 for f in cell_faces)
        )
        if is_tet:
            id_list = vtk.vtkIdList()
            id_list.SetNumberOfIds(4)
            for j, v in enumerate(cell_verts_sorted):
                id_list.SetId(j, int(v))
            ug.InsertNextCell(vtk.VTK_TETRA, id_list)
        else:
            # VTK_POLYHEDRON expects face stream: [nFaces, nPts0, p0..., nPts1, p1..., ...]
            face_stream = vtk.vtkIdList()
            face_stream.InsertNextId(n_f)
            for f in cell_faces:
                fv = faces_all[f]
                face_stream.InsertNextId(len(fv))
                for v in fv:
                    face_stream.InsertNextId(int(v))
            pt_ids = vtk.vtkIdList()
            pt_ids.SetNumberOfIds(n_v)
            for j, v in enumerate(cell_verts_sorted):
                pt_ids.SetId(j, int(v))
            ug.InsertNextCell(vtk.VTK_POLYHEDRON, face_stream)

    return ug


def read_polymesh_to_vtu(
    polymesh_dir: Union[str, Path],
    vtu_path: Union[str, Path],
) -> Dict[str, int]:
    """
    Convert an OpenFOAM polyMesh directory to a .vtu file.

    Returns a small dict of counts for logging.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required for read_polymesh_to_vtu")
    mesh = read_polymesh(polymesh_dir)
    ug = polymesh_to_vtk_unstructured(mesh)

    vtu_path = Path(vtu_path)
    vtu_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(vtu_path))
    writer.SetInputData(ug)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    writer.Write()

    return {
        "nPoints": len(mesh["points"]),
        "nCells": mesh["nCells"],
        "nFaces": len(mesh["faces"]),
        "nInternalFaces": mesh["nInternalFaces"],
    }


# ---------------------------------------------------------------------------
# VTU -> polyMesh
# ---------------------------------------------------------------------------


def _canonical_face_key(vertex_ids: np.ndarray) -> Tuple[int, ...]:
    """Sorted tuple of vertex ids used as a dictionary key for face dedup."""
    return tuple(sorted(int(v) for v in vertex_ids))


def _extract_cell_faces(ug) -> List[Dict[str, object]]:
    """
    Return a list of per-cell dicts, each holding the cell's polygonal faces
    as **oriented** vertex-id sequences. Supports VTK_TETRA and VTK_POLYHEDRON
    directly, and falls back to ``vtkGeometryFilter`` cell-face extraction for
    other convex cell types.

    For a tet, the four faces are emitted with outward-consistent winding
    following VTK's standard tet-face convention (indices (0,2,1),(0,1,3),
    (0,3,2),(1,2,3) into the cell's local vertex list).
    """
    n_cells = ug.GetNumberOfCells()
    out: List[Dict[str, object]] = []

    # VTK's outward-facing tet face indexing (local vertex indices 0..3)
    _TET_FACES_LOCAL = [
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    ]

    for cid in range(n_cells):
        cell = ug.GetCell(cid)
        ct = cell.GetCellType()
        pt_ids = vtk.vtkIdList()
        ug.GetCellPoints(cid, pt_ids)
        vlist = [int(pt_ids.GetId(j)) for j in range(pt_ids.GetNumberOfIds())]

        cell_faces: List[np.ndarray] = []
        if ct == vtk.VTK_TETRA and len(vlist) == 4:
            for tri in _TET_FACES_LOCAL:
                cell_faces.append(np.asarray([vlist[tri[0]], vlist[tri[1]], vlist[tri[2]]], dtype=np.int32))

        elif ct == vtk.VTK_POLYHEDRON:
            # Face stream: [nFaces, nPts0, p0..., nPts1, p1..., ...]
            face_stream = vtk.vtkIdList()
            ug.GetFaceStream(cid, face_stream)
            idx = 0
            n_f = int(face_stream.GetId(idx))
            idx += 1
            for _ in range(n_f):
                k = int(face_stream.GetId(idx))
                idx += 1
                verts = np.empty(k, dtype=np.int32)
                for j in range(k):
                    verts[j] = int(face_stream.GetId(idx))
                    idx += 1
                cell_faces.append(verts)

        else:
            # Fallback: iterate over cell.GetNumberOfFaces() for VTK_HEXAHEDRON,
            # VTK_WEDGE, VTK_PYRAMID, VTK_VOXEL, VTK_PENTAGONAL_PRISM, etc.
            n_faces = cell.GetNumberOfFaces()
            for fi in range(n_faces):
                face = cell.GetFace(fi)
                face_pt_ids = face.GetPointIds()
                k = face_pt_ids.GetNumberOfIds()
                verts = np.empty(k, dtype=np.int32)
                for j in range(k):
                    verts[j] = int(face_pt_ids.GetId(j))
                cell_faces.append(verts)

        out.append({"cell_id": cid, "faces": cell_faces})

    return out


def _build_face_topology(cell_face_data: List[Dict[str, object]]) -> Dict[str, object]:
    """
    From per-cell oriented face lists build the OpenFOAM face table with the
    "internal faces first + owner < neighbour" convention.

    Returns
    -------
    dict with:
        faces      : list of int32 arrays (each vertex-id sequence, using the
                     winding from the owner side)
        owner      : (nFaces,) int32
        neighbour  : (nInternalFaces,) int32
        nInternalFaces : int
        nCells     : int
    """
    face_map: Dict[Tuple[int, ...], Dict[str, object]] = {}
    for entry in cell_face_data:
        cid = int(entry["cell_id"])
        for fverts in entry["faces"]:
            key = _canonical_face_key(fverts)
            if key not in face_map:
                # First cell to touch this face becomes the (tentative) owner.
                face_map[key] = {
                    "owner_cell": cid,
                    "owner_verts": np.asarray(fverts, dtype=np.int32),
                    "neighbour_cell": -1,
                }
            else:
                rec = face_map[key]
                if rec["neighbour_cell"] != -1:
                    raise ValueError(
                        f"Face {key} touched by three cells "
                        f"({rec['owner_cell']}, {rec['neighbour_cell']}, {cid}) — mesh is non-manifold"
                    )
                rec["neighbour_cell"] = cid

    n_cells = 1 + max(int(e["cell_id"]) for e in cell_face_data)
    internal, boundary = [], []
    for rec in face_map.values():
        (internal if rec["neighbour_cell"] != -1 else boundary).append(rec)

    # Enforce owner < neighbour on internal faces; flip winding if we have to swap.
    for rec in internal:
        if rec["owner_cell"] > rec["neighbour_cell"]:
            rec["owner_cell"], rec["neighbour_cell"] = rec["neighbour_cell"], rec["owner_cell"]
            rec["owner_verts"] = rec["owner_verts"][::-1].copy()

    # OpenFOAM upper-triangular ordering: sort by (owner, neighbour) for internal,
    # then by (owner) for boundary. This is what SCONE's reader implicitly expects.
    internal.sort(key=lambda r: (r["owner_cell"], r["neighbour_cell"]))
    boundary.sort(key=lambda r: r["owner_cell"])

    all_faces = internal + boundary
    faces = [rec["owner_verts"] for rec in all_faces]
    owner = np.asarray([rec["owner_cell"] for rec in all_faces], dtype=np.int32)
    neighbour = np.asarray([rec["neighbour_cell"] for rec in internal], dtype=np.int32)

    return {
        "faces": faces,
        "owner": owner,
        "neighbour": neighbour,
        "nInternalFaces": len(internal),
        "nCells": n_cells,
    }


def _write_polymesh_files(
    polymesh_dir: Path,
    points: np.ndarray,
    faces: List[np.ndarray],
    owner: np.ndarray,
    neighbour: np.ndarray,
    n_internal: int,
    n_cells: int,
) -> None:
    polymesh_dir.mkdir(parents=True, exist_ok=True)
    n_points = int(points.shape[0])
    n_faces = len(faces)
    note = f"nPoints:{n_points}  nCells:{n_cells}  nFaces:{n_faces}  nInternalFaces:{n_internal}"

    # points
    with open(polymesh_dir / "points", "w") as fh:
        fh.write(_foam_header("vectorField", "points"))
        fh.write(f"\n{n_points}\n(\n")
        for xyz in points:
            fh.write(f"({xyz[0]:.15g} {xyz[1]:.15g} {xyz[2]:.15g})\n")
        fh.write(")\n")

    # faces
    with open(polymesh_dir / "faces", "w") as fh:
        fh.write(_foam_header("faceList", "faces"))
        fh.write(f"\n{n_faces}\n(\n")
        for fv in faces:
            inner = " ".join(str(int(v)) for v in fv)
            fh.write(f"{len(fv)}({inner})\n")
        fh.write(")\n")

    # owner
    with open(polymesh_dir / "owner", "w") as fh:
        fh.write(_foam_header("labelList", "owner", note=note))
        fh.write(f"\n{n_faces}\n(\n")
        for v in owner:
            fh.write(f"{int(v)}\n")
        fh.write(")\n")

    # neighbour
    with open(polymesh_dir / "neighbour", "w") as fh:
        fh.write(_foam_header("labelList", "neighbour", note=note))
        fh.write(f"\n{n_internal}\n(\n")
        for v in neighbour[:n_internal]:
            fh.write(f"{int(v)}\n")
        fh.write(")\n")

    # cellZones — single all-cell zone named "internalMesh".
    # Format matches Kim et al.'s bundled polyMesh files (multi-line): SCONE's
    # reader uses a fixed 100-char buffer, so we MUST write one label per line
    # rather than a single-line List<label> N(...) which would truncate on
    # large zones.
    with open(polymesh_dir / "cellZones", "w") as fh:
        fh.write(_foam_header("regIOobject", "cellZones"))
        fh.write("\n1\n(\ninternalMesh\n{\n    type cellZone;\n")
        fh.write("cellLabels      List<label>;\n")
        fh.write(f"{n_cells}\n(\n")
        for i in range(n_cells):
            fh.write(f"{i}\n")
        fh.write(")\n;\n}\n)\n")


def write_vtu_to_polymesh(
    vtu_path: Union[str, Path],
    polymesh_dir: Union[str, Path],
) -> Dict[str, int]:
    """
    Read a .vtu (or .vtp / .pvtu — anything vtkXMLGenericDataObjectReader can
    open into an unstructured grid) and write an OpenFOAM polyMesh directory
    that SCONE's ``OpenFOAMMesh`` reader accepts.
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required for write_vtu_to_polymesh")

    vtu_path = Path(vtu_path)
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(str(vtu_path))
    reader.Update()
    ds = reader.GetOutput()
    if not ds.IsA("vtkUnstructuredGrid"):
        # Attempt to cast common multi-block or polydata cases into an ug
        raise ValueError(f"{vtu_path}: expected vtkUnstructuredGrid, got {ds.GetClassName()}")

    n_points = ds.GetNumberOfPoints()
    n_cells = ds.GetNumberOfCells()
    if n_points == 0 or n_cells == 0:
        raise ValueError(f"{vtu_path}: empty mesh (nPoints={n_points}, nCells={n_cells})")

    points = np.empty((n_points, 3), dtype=np.float64)
    for i in range(n_points):
        points[i] = ds.GetPoint(i)

    cell_face_data = _extract_cell_faces(ds)
    topo = _build_face_topology(cell_face_data)

    _write_polymesh_files(
        Path(polymesh_dir),
        points=points,
        faces=topo["faces"],
        owner=topo["owner"],
        neighbour=topo["neighbour"],
        n_internal=topo["nInternalFaces"],
        n_cells=topo["nCells"],
    )

    return {
        "nPoints": n_points,
        "nCells": topo["nCells"],
        "nFaces": len(topo["faces"]),
        "nInternalFaces": topo["nInternalFaces"],
    }


# ---------------------------------------------------------------------------
# Round-trip verification helper
# ---------------------------------------------------------------------------


def round_trip_check(
    polymesh_dir: Union[str, Path],
    scratch_dir: Union[str, Path],
) -> Dict[str, object]:
    """
    Verify that read_polymesh_to_vtu -> write_vtu_to_polymesh is topologically
    stable: nPoints, nCells, nFaces, and nInternalFaces must all round-trip
    exactly.  Point coordinates must round-trip to within 1e-12.

    Used by scripts/openfoam_polymesh_convert.py --check.
    """
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    src = read_polymesh(polymesh_dir)
    vtu = scratch / "roundtrip.vtu"
    r_stats = read_polymesh_to_vtu(polymesh_dir, vtu)

    out_dir = scratch / "polymesh_out"
    w_stats = write_vtu_to_polymesh(vtu, out_dir)

    dst = read_polymesh(out_dir)

    checks = {
        "nPoints_match": len(src["points"]) == len(dst["points"]),
        "nCells_match": src["nCells"] == dst["nCells"],
        "nFaces_match": len(src["faces"]) == len(dst["faces"]),
        "nInternalFaces_match": src["nInternalFaces"] == dst["nInternalFaces"],
        "point_max_abs_error": float(
            np.max(np.abs(np.sort(src["points"].ravel()) - np.sort(dst["points"].ravel())))
            if len(src["points"]) == len(dst["points"]) else np.inf
        ),
    }
    return {"source": r_stats, "roundtrip": w_stats, "checks": checks}


__all__ = [
    "read_polymesh",
    "polymesh_to_vtk_unstructured",
    "read_polymesh_to_vtu",
    "write_vtu_to_polymesh",
    "round_trip_check",
]
