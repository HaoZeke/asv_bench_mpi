# SPDX-License-Identifier: BSD-3-Clause
"""asv_bench_mpi — MPI / mpi4py benchmark types for airspeed velocity.

Discovered automatically by ``asv_runner`` because the distribution name
starts with ``asv_bench``. See ``asv_bench_mpi.benchmarks``.
"""

try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
