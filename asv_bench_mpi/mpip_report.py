# SPDX-License-Identifier: BSD-3-Clause
"""Minimal parser for LLNL mpiP-style text reports (Phase B scaffold)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Union

PathLike = Union[str, Path]


def parse_mpip_report(path: PathLike) -> Dict[str, float]:
    """Extract a few numeric fields from an mpiP report file.

    Real mpiP reports vary by version. This parser is intentionally loose:
    it looks for lines like ``MPI Time: 1.23`` or ``AppTime = 4.56`` and
    returns whatever floats it finds under stable keys when possible.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}
    # Common-ish patterns (mpiP has changed formats over releases)
    patterns = {
        "mpi_time": re.compile(
            r"(?:@?\s*)MPI\s*Time\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", re.I
        ),
        "app_time": re.compile(
            r"(?:@?\s*)App(?:lication)?\s*Time\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)",
            re.I,
        ),
        "mpi_percent": re.compile(
            r"(?:@?\s*)MPI\s*%?\s*[:=]\s*([0-9.]+)", re.I
        ),
    }
    for key, rx in patterns.items():
        m = rx.search(text)
        if m:
            out[key] = float(m.group(1))
    # Fallback: first "name: number" pairs
    if not out:
        for m in re.finditer(
            r"^([A-Za-z][A-Za-z0-9_ %]+?)\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)\s*$",
            text,
            re.M,
        ):
            k = re.sub(r"\s+", "_", m.group(1).strip().lower())
            out[k] = float(m.group(2))
    return out
