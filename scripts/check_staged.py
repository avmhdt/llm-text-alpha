"""Reject files that must never enter git: anything under data/, .env, and files over 1 MB.

Used by pre-commit on staged files and by CI on every tracked file.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 1_000_000


def problems(paths: list[str]) -> list[str]:
    """One message per offending path."""
    found: list[str] = []
    for raw in paths:
        posix = PurePosixPath(Path(raw).as_posix())
        if posix.parts and posix.parts[0] == "data":
            found.append(f"{raw}: paths under data/ are never committed")
        if posix.name == ".env":
            found.append(f"{raw}: .env holds secrets")
        path = Path(raw)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            found.append(f"{raw}: {path.stat().st_size} bytes is over 1 MB")
    return found


def main(argv: list[str]) -> int:
    """Print problems and return 1 if there are any."""
    found = problems(argv)
    for message in found:
        print(message, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
