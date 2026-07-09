# SPDX-License-Identifier: BSD-3-Clause
"""Compile sample C/Fortran payloads into shared libraries for tests/examples."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PAYLOAD_DIR = Path(__file__).resolve().parent / "payload"


def payload_dir() -> Path:
    return _PAYLOAD_DIR


def compile_c_kernel(out_path: str | Path | None = None) -> Path:
    """Build libasv_kernel_c.so from payload/kernel_c.c. Return path to .so."""
    src = _PAYLOAD_DIR / "kernel_c.c"
    if not src.is_file():
        raise FileNotFoundError(src)
    if out_path is None:
        out_path = Path(tempfile.mkdtemp(prefix="asv_native_")) / "libasv_kernel_c.so"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    cmd = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-o",
        str(out_path),
        str(src),
    ]
    subprocess.check_call(cmd)
    return out_path


def compile_fortran_kernel(out_path: str | Path | None = None) -> Path | None:
    """Build Fortran shared lib if gfortran is available; else return None."""
    fc = shutil.which(os.environ.get("FC", "gfortran"))
    if not fc:
        return None
    src = _PAYLOAD_DIR / "kernel_f.f90"
    if not src.is_file():
        return None
    if out_path is None:
        out_path = Path(tempfile.mkdtemp(prefix="asv_native_")) / "libasv_kernel_f.so"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [fc, "-shared", "-fPIC", "-O2", "-o", str(out_path), str(src)]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        return None
    return out_path


if __name__ == "__main__":
    p = compile_c_kernel()
    print(p)
    fp = compile_fortran_kernel()
    if fp:
        print(fp)
