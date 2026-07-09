# SPDX-License-Identifier: BSD-3-Clause
"""Importable launch descriptors for mpiP live tests."""

from __future__ import annotations

_APP: str | None = None
_WORK: str | None = None
_LIB: str | None = None


def configure(app: str, work: str, lib: str) -> None:
    global _APP, _WORK, _LIB
    _APP, _WORK, _LIB = app, work, lib


def mpip_track_wall():
    return [_APP]


# attributes set after configure in tests
mpip_track_wall.mpi_np = 2
mpip_track_wall.mpip_metric = "wall_seconds"
mpip_track_wall.unit = "seconds"
