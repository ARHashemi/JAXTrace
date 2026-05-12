"""Recovery helper for VTKHDF files left half-written by a SIGKILL.

If a JAXTrace run is killed by SLURM (job time limit, OOM) before
``run_tracking.py``'s signal handler can drain the writer queue, the
``.vtkhdf`` file may have a stale "file is open" flag (HDF5 calls this
the "consistency flag"). Most readers will refuse to open such a file.

``h5clear --status`` clears the flag in place; if the file is otherwise
intact, ParaView opens it normally afterwards.

Usage
-----
As a CLI::

    python -m jaxtrace.io.vtkhdf_repair path/to/particles.vtkhdf

Programmatically::

    from jaxtrace.io.vtkhdf_repair import repair_vtkhdf
    repair_vtkhdf("path/to/particles.vtkhdf")

The function calls ``h5clear`` if available (shipped with the
HDF5 tools package; ``apt install hdf5-tools`` or similar); otherwise it
falls back to opening the file with h5py and writing a no-op flush,
which works for milder corruption.

The function never deletes data — at worst it leaves the file unchanged
and prints a clear error.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def repair_vtkhdf(path: str | Path) -> bool:
    """Attempt to clear the HDF5 consistency flag on a VTKHDF file.

    Returns True on success, False if the file is unrecoverable.
    """
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return False

    # First try h5clear (the canonical tool).
    h5clear = shutil.which("h5clear")
    if h5clear is not None:
        print(f"Running h5clear --status on {path}")
        try:
            result = subprocess.run(
                [h5clear, "--status", str(path)],
                check=False, capture_output=True, text=True,
            )
            if result.returncode == 0:
                print("h5clear succeeded; file should be readable now.")
                if result.stdout.strip():
                    print(result.stdout)
                return True
            print(f"h5clear exit {result.returncode}: {result.stderr}",
                  file=sys.stderr)
        except FileNotFoundError:
            pass  # fall through

    # Fallback: open with h5py in read-write mode and re-flush. This works
    # when the consistency flag is set but the on-disk metadata is intact.
    try:
        import h5py
        with h5py.File(str(path), "r+") as f:
            f.flush()
        print("Re-flushed file via h5py; consistency flag cleared.")
        return True
    except Exception as e:
        print(f"Could not repair {path}: {e}", file=sys.stderr)
        print("  Try installing the HDF5 tools (apt/dnf install hdf5-tools)",
              file=sys.stderr)
        print("  and re-running this command.", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    ok = True
    for p in argv:
        ok = ok and repair_vtkhdf(p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
