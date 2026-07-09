# SPDX-License-Identifier: BSD-3-Clause
"""Track metrics from a real mpiP profile of a native MPI binary.

The benchmark function should return a launch description:

    (argv_list, metric_key)
    # or just argv_list  → metric_key defaults to ``mpi_percent`` or first key

Attributes (class or function):
- ``mpi_np`` (default 2)
- ``mpip_lib`` / env ``ASV_MPIP_LIB``
- ``mpip_flags`` (default ``-f``)
- ``mpip_metric`` (default ``mpi_percent``)
- ``timeout``
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from asv_runner.benchmarks._base import Benchmark, _get_first_attr
from asv_runner.benchmarks._exceptions import NotRequired

from asv_bench_mpi.mpip_run import find_libmpip, run_with_mpip


class MpipTrackBenchmark(Benchmark):
    """Profile a compiled MPI app with mpiP; record one report metric."""

    name_regex = re.compile(r"^(MpipTrack[A-Z_].+)|(mpip_track_.+)$")

    def __init__(self, name, func, attr_sources):
        if os.environ.get("ASV_BENCH_MPI_DISABLE", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            raise NotRequired("disabled")
        Benchmark.__init__(self, name, func, attr_sources)
        self.type = _get_first_attr(attr_sources, "type", "track")
        self.unit = _get_first_attr(attr_sources, "unit", "unit")
        self._attr_sources = attr_sources

    def run(self, *param):
        ret = self.func(*param) if param else self.func()
        metric_key = _get_first_attr(self._attr_sources, "mpip_metric", "mpi_percent")
        # Forms: "path" | ["path", ...] | (["path", ...], "metric_key")
        if isinstance(ret, str):
            argv = [ret]
        elif isinstance(ret, (list, tuple)) and ret and isinstance(ret[0], str):
            argv = list(ret)
        elif (
            isinstance(ret, (list, tuple))
            and len(ret) >= 1
            and isinstance(ret[0], (list, tuple))
        ):
            argv = list(ret[0])
            if len(ret) >= 2 and isinstance(ret[1], str):
                metric_key = ret[1]
        else:
            raise TypeError(
                "mpip_track_* must return argv list or (argv, metric_key); "
                f"got {type(ret)!r}"
            )

        np = int(_get_first_attr(self._attr_sources, "mpi_np", 2))
        lib = _get_first_attr(self._attr_sources, "mpip_lib", None)
        flags = str(_get_first_attr(self._attr_sources, "mpip_flags", "-f"))
        timeout = float(_get_first_attr(self._attr_sources, "timeout", 120) or 120)
        extra = list(_get_first_attr(self._attr_sources, "mpiexec_args", None) or [])
        work = _get_first_attr(self._attr_sources, "mpip_workdir", None)
        if work is None:
            work = Path.cwd() / ".asv_mpip_work"
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)

        metrics, report, wall = run_with_mpip(
            argv,
            mpi_np=np,
            libmpip=lib,
            mpip_flags=flags,
            workdir=work,
            timeout=timeout,
            mpiexec_args=extra,
        )
        if metric_key not in metrics:
            # prefer sensible fallbacks
            for alt in ("mpi_percent", "mpi_time", "app_time", "wall_seconds"):
                if alt in metrics:
                    metric_key = alt
                    break
            else:
                if metrics:
                    metric_key = next(iter(metrics))
                else:
                    raise RuntimeError(
                        f"mpiP report {report} produced no metrics; keys={metrics}"
                    )
        return float(metrics[metric_key])


export_as_benchmark = [MpipTrackBenchmark]
