# SPDX-License-Identifier: BSD-3-Clause
"""Parse LLNL mpiP text reports into numeric metrics for ASV track types."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Union

PathLike = Union[str, Path]

# Real mpiP reports often look like:
#   @ Command : ./a.out
#   ...
#   Aggregate Time (top twenty, descending, milliseconds):
#   ...
# Or older concise tables. We accept multiple dialects.


def find_mpip_reports(directory: PathLike) -> List[Path]:
    """Find mpiP report files under *directory* (non-recursive).

    Matches ``*.mpiP`` and classic ``exe.nprocs.pid.mpiP`` names.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    found = set(d.glob("*.mpiP"))
    for p in d.iterdir():
        if p.is_file() and ".mpiP" in p.name:
            found.add(p)
    return sorted(found, key=lambda p: p.stat().st_mtime)


def parse_mpip_report(path: PathLike) -> Dict[str, float]:
    """Extract numeric fields from an mpiP report file.

    Returns a dict that always includes whatever stable keys we can map:
    ``mpi_time``, ``app_time``, ``mpi_percent`` when present, plus any
    additional ``name: number`` lines under snake_case keys.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: Dict[str, float] = {}

    patterns = {
        "mpi_time": [
            re.compile(r"MPI\s*Time\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", re.I),
            re.compile(r"Total\s+MPI\s+Time\s*[:=]?\s*([0-9.]+)", re.I),
        ],
        "app_time": [
            re.compile(
                r"App(?:lication)?\s*Time\s*[:=]\s*([0-9.]+(?:[eE][+-]?\d+)?)", re.I
            ),
            re.compile(r"Wall\s*clock\s*[:=]\s*([0-9.]+)", re.I),
        ],
        "mpi_percent": [
            re.compile(r"MPI\s*%\s*[:=]\s*([0-9.]+)", re.I),
            re.compile(r"MPI\s+Time\s*%\s*[:=]?\s*([0-9.]+)", re.I),
            re.compile(r"^\s*MPI\s+([0-9.]+)\s*%", re.I | re.M),
        ],
    }
    for key, rxs in patterns.items():
        for rx in rxs:
            m = rx.search(text)
            if m:
                out[key] = float(m.group(1))
                break

    # Aggregate section: first numeric column after "Call" lines often listed
    # Sum of top-level "time" columns if present
    agg = re.search(
        r"Aggregate Time.*?\n(.*?)(?:\n@|\n\s*\n|\Z)",
        text,
        re.I | re.S,
    )
    if agg:
        nums = [float(x) for x in re.findall(r"\b([0-9]+\.[0-9]+)\b", agg.group(1))]
        if nums and "aggregate_top_ms" not in out:
            out["aggregate_top_ms"] = nums[0]

    if not out:
        for m in re.finditer(
            r"^@?---?\s*([A-Za-z][A-Za-z0-9_ %]+?)\s*[:=]\s*"
            r"([0-9.]+(?:[eE][+-]?\d+)?)\s*$",
            text,
            re.M,
        ):
            k = re.sub(r"\s+", "_", m.group(1).strip().lower())
            k = k.replace("%", "percent")
            out[k] = float(m.group(2))

    return out
