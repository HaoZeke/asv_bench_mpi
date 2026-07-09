# TRANSITIONAL: pure-Python mpiexec+mpi4py path. Prefer native_time_* / native_track_* (C extension, GIL released) and Phase B mpiP for real MPI apps.
# SPDX-License-Identifier: BSD-3-Clause
"""MPI track benchmarks — arbitrary metric under mpiexec + mpi4py."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from asv_runner.benchmarks._base import Benchmark, _get_first_attr
from asv_runner.benchmarks._exceptions import NotRequired

from asv_bench_mpi.mpiexec import build_mpi_command, find_mpiexec

if os.environ.get("ASV_BENCH_MPI_DISABLE", "").strip().lower() in ("1", "true", "yes"):
    raise NotRequired("asv_bench_mpi disabled via ASV_BENCH_MPI_DISABLE")


def _func_import_path(func):
    mod = getattr(func, "__module__", None)
    if not mod or mod == "__main__":
        raise RuntimeError(
            f"MPI benchmarks require an importable module, got {func!r} in {mod!r}"
        )
    q = getattr(func, "__qualname__", func.__name__)
    return mod, q


class MpiTrackBenchmark(Benchmark):
    """Run a callable under MPI and record a numeric result.

    The function should return a float on every rank. Reduction across ranks
    is controlled by ``mpi_reduce``: ``rank0`` (default), ``max``, ``min``,
    ``mean``.

    Other attributes: ``mpi_np``, ``mpiexec``, ``mpiexec_args``, ``timeout``,
    ``unit`` (default ``unit``), ``type`` (default ``track``).
    """

    name_regex = re.compile(r"^(MpiTrack[A-Z_].+)|(mpi_track_.+)$")

    def __init__(self, name, func, attr_sources):
        Benchmark.__init__(self, name, func, attr_sources)
        self.type = _get_first_attr(attr_sources, "type", "track")
        self.unit = _get_first_attr(attr_sources, "unit", "unit")
        self._attr_sources = attr_sources

    def run(self, *param):
        if find_mpiexec() is None and _get_first_attr(self._attr_sources, "mpiexec", None) is None:
            raise RuntimeError(
                "mpi_track_* requires mpiexec/mpirun (set MPIEXEC or ASV_MPIEXEC)"
            )
        try:
            import mpi4py  # noqa: F401
        except ImportError as err:
            raise RuntimeError(
                "mpi_track_* requires mpi4py in the benchmark environment"
            ) from err

        np = int(_get_first_attr(self._attr_sources, "mpi_np", 2))
        launcher = _get_first_attr(self._attr_sources, "mpiexec", None)
        extra = list(_get_first_attr(self._attr_sources, "mpiexec_args", None) or [])
        timeout = float(_get_first_attr(self._attr_sources, "timeout", 120) or 120)
        reduce_how = str(_get_first_attr(self._attr_sources, "mpi_reduce", "rank0"))
        mod_name, qual = _func_import_path(self.func)

        with tempfile.TemporaryDirectory(prefix="asv_mpi_") as td:
            result_path = str(Path(td) / "result.txt")
            env = dict(os.environ)
            env["ASV_MPI_MODE"] = "track"
            env["ASV_MPI_MOD"] = mod_name
            env["ASV_MPI_FUNC"] = qual
            env["ASV_MPI_PARAMS"] = json.dumps(list(param))
            env["ASV_MPI_RESULT"] = result_path
            env["ASV_MPI_REDUCE"] = reduce_how
            env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
            env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")

            code = "from asv_bench_mpi.runner_payload import main; raise SystemExit(main())"
            cmd = build_mpi_command(
                np,
                sys.executable,
                ["-c", code],
                mpiexec=launcher,
                extra_args=extra,
            )
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"mpiexec failed (rc={proc.returncode}):\n"
                    f"cmd={cmd!r}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                )
            with open(result_path, encoding="utf-8") as fh:
                return float(fh.read().strip())


export_as_benchmark = [MpiTrackBenchmark]
