# SPDX-License-Identifier: BSD-3-Clause
"""Locate and invoke an MPI launcher (mpirun / mpiexec)."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Sequence


def find_mpiexec() -> Optional[str]:
    """Return path to an MPI launcher, or None if missing."""
    for key in ("MPIEXEC", "ASV_MPIEXEC", "MPIRUN"):
        val = os.environ.get(key)
        if val and os.path.isfile(val) and os.access(val, os.X_OK):
            return val
    for name in ("mpiexec", "mpirun"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build_mpi_command(
    np: int,
    python: str,
    script_args: Sequence[str],
    *,
    mpiexec: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build ``mpiexec -n <np> [extra] <python> <script_args...>``."""
    launcher = mpiexec or find_mpiexec()
    if not launcher:
        raise FileNotFoundError(
            "No MPI launcher found (set MPIEXEC / ASV_MPIEXEC or install mpirun)"
        )
    if np < 1:
        raise ValueError(f"mpi_np must be >= 1, got {np}")
    cmd = [launcher, "-n", str(int(np))]
    if extra_args:
        cmd.extend(list(extra_args))
    cmd.append(python)
    cmd.extend(list(script_args))
    return cmd
