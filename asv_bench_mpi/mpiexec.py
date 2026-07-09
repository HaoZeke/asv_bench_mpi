# SPDX-License-Identifier: BSD-3-Clause
"""Locate and invoke an MPI launcher (mpirun / mpiexec)."""

from __future__ import annotations

import os
import shutil
from typing import List, Optional, Sequence


def _is_executable(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def path_looks_nix(path: str) -> bool:
    """Nix-store tools break when mixed with system ``libmpiP`` LD_PRELOAD."""
    real = os.path.realpath(path)
    return "/nix/store/" in real or real.startswith("/nix/")


def which_all(name: str) -> List[str]:
    """All executable PATH hits for *name* (realpath-deduped, PATH order)."""
    out: List[str] = []
    seen = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        cand = os.path.join(directory, name)
        if not _is_executable(cand):
            continue
        real = os.path.realpath(cand)
        if real in seen:
            continue
        seen.add(real)
        out.append(cand)
    return out


def prefer_non_nix(candidates: Sequence[str]) -> Optional[str]:
    """Return first non-Nix path, else first candidate, else None."""
    if not candidates:
        return None
    for c in candidates:
        if not path_looks_nix(c):
            return c
    return candidates[0]


def find_mpiexec() -> Optional[str]:
    """Return path to an MPI launcher, or None if missing.

    Preference: ``MPIEXEC`` / ``ASV_MPIEXEC`` / ``MPIRUN`` env, then PATH
    candidates. When several PATH hits exist, prefer non-Nix binaries so a
    system-built ``libmpiP.so`` preloaded via the native launcher does not
    fail resolving libs against a Nix OpenMPI (common on mixed hosts).
    """
    for key in ("MPIEXEC", "ASV_MPIEXEC", "MPIRUN"):
        val = os.environ.get(key)
        if val and _is_executable(val):
            return val

    candidates: List[str] = []
    for name in ("mpiexec", "mpirun"):
        candidates.extend(which_all(name))

    chosen = prefer_non_nix(candidates)
    if chosen:
        return chosen
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
