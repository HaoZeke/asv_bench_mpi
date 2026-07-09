# SPDX-License-Identifier: BSD-3-Clause
"""Compile sample MPI C apps for mpiP end-to-end tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_mpicc() -> str | None:
    for key in ("MPICC", "ASV_MPICC"):
        val = os.environ.get(key)
        if val and os.path.isfile(val):
            return val
    return shutil.which("mpicc")


def compile_pingpong(out_path: str | Path | None = None) -> Path:
    mpicc = find_mpicc()
    if not mpicc:
        raise FileNotFoundError("mpicc not found")
    root = Path(__file__).resolve().parents[1]
    src = root / "examples" / "mpip" / "pingpong.c"
    if out_path is None:
        out_path = Path(tempfile.mkdtemp(prefix="asv_mpip_app_")) / "pingpong"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([mpicc, "-O2", "-g", "-o", str(out_path), str(src)])
    return out_path
