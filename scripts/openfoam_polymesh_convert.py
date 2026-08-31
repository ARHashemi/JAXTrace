#!/usr/bin/env python3
"""
CLI for the OpenFOAM polyMesh <-> VTU bridge.

Enables the SCONE / AMLG-Patch benchmark workflow:

    # 1) Ingest one of Kim et al.'s bundled meshes for MALMO
    scripts/openfoam_polymesh_convert.py \\
        --from-polymesh /path/to/SCONE/.../FinalFuelPinTet298 \\
        --to-vtu ./bench_meshes/FinalFuelPinTet298.vtu

    # 2) Export a JAXTrace VTU for SCONE
    scripts/openfoam_polymesh_convert.py \\
        --from-vtu ./bench_meshes/my_mesh.vtu \\
        --to-polymesh /path/to/SCONE/.../my_mesh_polyMesh

    # 3) Round-trip check on any bundled polyMesh
    scripts/openfoam_polymesh_convert.py \\
        --check /path/to/SCONE/.../FinalFuelPinTet298 \\
        --scratch /tmp/polymesh_roundtrip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script from the JAXTrace repo without pip install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenFOAM polyMesh <-> VTU bridge")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-polymesh", metavar="DIR", help="Read polyMesh from DIR")
    g.add_argument("--from-vtu", metavar="FILE", help="Read VTU from FILE")
    g.add_argument("--check", metavar="DIR", help="Round-trip check on polyMesh DIR")

    ap.add_argument("--to-vtu", metavar="FILE", help="Write VTU to FILE")
    ap.add_argument("--to-polymesh", metavar="DIR", help="Write polyMesh to DIR")
    ap.add_argument("--scratch", metavar="DIR", default="/tmp/polymesh_bridge",
                    help="Scratch dir for --check (default: /tmp/polymesh_bridge)")
    args = ap.parse_args()

    # Route
    from jaxtrace.io.openfoam_polymesh import (
        read_polymesh_to_vtu,
        write_vtu_to_polymesh,
        round_trip_check,
    )

    if args.from_polymesh:
        if not args.to_vtu:
            print("--from-polymesh requires --to-vtu", file=sys.stderr)
            return 2
        stats = read_polymesh_to_vtu(args.from_polymesh, args.to_vtu)
        print(json.dumps({"action": "polymesh_to_vtu", **stats}, indent=2))
        return 0

    if args.from_vtu:
        if not args.to_polymesh:
            print("--from-vtu requires --to-polymesh", file=sys.stderr)
            return 2
        stats = write_vtu_to_polymesh(args.from_vtu, args.to_polymesh)
        print(json.dumps({"action": "vtu_to_polymesh", **stats}, indent=2))
        return 0

    if args.check:
        result = round_trip_check(args.check, args.scratch)
        print(json.dumps({"action": "round_trip", **result}, indent=2, default=str))
        ok = all(v is True for k, v in result["checks"].items() if k.endswith("_match"))
        pt_err = result["checks"].get("point_max_abs_error", float("inf"))
        return 0 if ok and pt_err < 1e-9 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
