# SPDX-License-Identifier: BSD-3-Clause
"""Live mpiexec smoke — opt-in via ASV_MPI_LIVE=1."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

if os.environ.get("ASV_MPI_LIVE", "").strip() not in ("1", "true", "yes"):
    pytest.skip("set ASV_MPI_LIVE=1 to run live MPI tests", allow_module_level=True)

pytest.importorskip("mpi4py")

from asv_bench_mpi.mpiexec import build_mpi_command, find_mpiexec

if not find_mpiexec():
    pytest.skip("no mpiexec/mpirun on PATH", allow_module_level=True)


def _run_payload(mode, mod, func, params, reduce="rank0", np=2):
    with tempfile.TemporaryDirectory() as td:
        result = str(Path(td) / "r.txt")
        env = dict(os.environ)
        env.update(
            {
                "ASV_MPI_MODE": mode,
                "ASV_MPI_MOD": mod,
                "ASV_MPI_FUNC": func,
                "ASV_MPI_PARAMS": json.dumps(params),
                "ASV_MPI_RESULT": result,
                "ASV_MPI_REDUCE": reduce,
                "OMPI_MCA_rmaps_base_oversubscribe": "1",
                "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
            }
        )
        code = "from asv_bench_mpi.runner_payload import main; raise SystemExit(main())"
        cmd = build_mpi_command(np, sys.executable, ["-c", code])
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr + proc.stdout
        return float(Path(result).read_text().strip())


def test_track_world_size():
    val = _run_payload("track", "asv_bench_mpi.probes", "world_size", [], reduce="rank0", np=2)
    assert val == 2.0


def test_time_barrier_nonneg():
    t = _run_payload("time", "asv_bench_mpi.probes", "barrier", [], np=2)
    assert t >= 0.0
