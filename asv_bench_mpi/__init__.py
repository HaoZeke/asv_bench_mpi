# SPDX-License-Identifier: BSD-3-Clause
"""asv_bench_mpi — native C/Fortran launch (+ optional mpiP / mpi4py).

Primary path: free-threaded CPython extension ``asv_bench_mpi._native`` and
benchmark types ``native_time_*`` / ``native_track_*``.

Transitional: ``mpi_time_*`` / ``mpi_track_*`` (mpiexec + mpi4py).

See DESIGN.md for Phase A (native) vs Phase B (mpiP).
"""

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "0.2.0"

__all__ = ["__version__"]
