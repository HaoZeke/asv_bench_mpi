# SPDX-License-Identifier: BSD-3-Clause
"""Importable probe benchmarks for integration scripts (has source lines)."""

from __future__ import annotations

_SO = None
_N = 5000


def set_so(path: str, n: int = 5000) -> None:
    global _SO, _N
    _SO = path
    _N = n


def native_time_sum():
    return (_SO, "asv_kernel_sum", _N)


def native_track_sum():
    return (_SO, "asv_kernel_sum", 100)
