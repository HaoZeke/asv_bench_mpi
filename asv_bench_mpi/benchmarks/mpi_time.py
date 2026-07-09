# TRANSITIONAL: pure-Python mpiexec+mpi4py path. Prefer native_time_* / native_track_* (C extension, GIL released) and Phase B mpiP for real MPI apps.
# SPDX-License-Identifier: BSD-3-Clause
"""MPI wall-time benchmarks (mpiexec + mpi4py)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from asv_runner.benchmarks._base import Benchmark, _get_first_attr
from asv_runner.benchmarks._exceptions import NotRequired

from asv_bench_mpi.mpiexec import build_mpi_command, find_mpiexec

# Soft requirement: package still installs; benchmarks skip when MPI stack missing
# at *run* time. Import-time NotRequired would hide the plugin entirely when
# only asv_runner is present (host discovery) — so we only raise NotRequired if
# explicitly disabled.
if os.environ.get("ASV_BENCH_MPI_DISABLE", "").strip().lower() in ("1", "true", "yes"):
    raise NotRequired("asv_bench_mpi disabled via ASV_BENCH_MPI_DISABLE")


def _func_import_path(func):
    """Return (module_name, qualname) for a benchmark callable."""
    mod = getattr(func, "__module__", None)
    if not mod or mod == "__main__":
        raise RuntimeError(
            f"MPI benchmarks require an importable module, got {func!r} in {mod!r}"
        )
    # Prefer class.method for bound methods
    q = getattr(func, "__qualname__", func.__name__)
    return mod, q


class MpiTimeBenchmark(Benchmark):
    """Time a callable under ``mpiexec -n <mpi_np>`` with mpi4py.

    Attributes (function or class):
    - ``mpi_np``: number of ranks (default 2)
    - ``mpiexec``: launcher path (default: find mpiexec/mpirun)
    - ``mpiexec_args``: extra launcher args (list)
    - ``number``: repetitions inside one sample (default 1) — outer repeats
      still controlled by asv timing infrastructure via multiple ``run`` calls
    - ``timeout``: seconds for the mpiexec job (default 120)
    """

    name_regex = re.compile(r"^(MpiTime[A-Z_].+)|(mpi_time_.+)$")

    def __init__(self, name, func, attr_sources):
        Benchmark.__init__(self, name, func, attr_sources)
        self.type = "time"
        self.unit = "seconds"
        self._attr_sources = attr_sources

    def _mpi_settings(self):
        np = int(_get_first_attr(self._attr_sources, "mpi_np", 2))
        launcher = _get_first_attr(self._attr_sources, "mpiexec", None)
        extra = _get_first_attr(self._attr_sources, "mpiexec_args", None) or []
        timeout = float(_get_first_attr(self._attr_sources, "timeout", 120) or 120)
        number = int(_get_first_attr(self._attr_sources, "number", 1) or 1)
        return np, launcher, list(extra), timeout, number

    def run(self, *param):
        if find_mpiexec() is None and _get_first_attr(self._attr_sources, "mpiexec", None) is None:
            raise RuntimeError(
                "mpi_time_* requires mpiexec/mpirun (set MPIEXEC or ASV_MPIEXEC)"
            )
        try:
            import mpi4py  # noqa: F401
        except ImportError as err:
            raise RuntimeError(
                "mpi_time_* requires mpi4py in the *benchmark* environment "
                "(add mpi4py to conf matrix req)"
            ) from err

        np, launcher, extra, timeout, number = self._mpi_settings()
        mod_name, qual = _func_import_path(self.func)

        samples = []
        for _ in range(max(number, 1)):
            with tempfile.TemporaryDirectory(prefix="asv_mpi_") as td:
                result_path = str(Path(td) / "result.txt")
                env = dict(os.environ)
                env["ASV_MPI_MODE"] = "time"
                env["ASV_MPI_MOD"] = mod_name
                env["ASV_MPI_FUNC"] = qual
                env["ASV_MPI_PARAMS"] = json.dumps(list(param))
                env["ASV_MPI_RESULT"] = result_path
                env["ASV_MPI_REDUCE"] = "max"
                # OpenMPI oversubscribe for laptop/CI small hosts
                env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
                env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")

                code = (
                    "from asv_bench_mpi.runner_payload import main; raise SystemExit(main())"
                )
                cmd = build_mpi_command(
                    np,
                    sys.executable,
                    ["-c", code],
                    mpiexec=launcher,
                    extra_args=extra,
                )
                t0 = time.perf_counter()
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                wall = time.perf_counter() - t0
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"mpiexec failed (rc={proc.returncode}):\n"
                        f"cmd={cmd!r}\nstdout={proc.stdout}\nstderr={proc.stderr}"
                    )
                if os.path.isfile(result_path):
                    with open(result_path, encoding="utf-8") as fh:
                        samples.append(float(fh.read().strip()))
                else:
                    # fallback: launcher wall clock
                    samples.append(wall)

        # Same result shape as TimeBenchmark (asv_runner timing pipeline).
        return {"samples": samples, "number": 1}


export_as_benchmark = [MpiTimeBenchmark]
