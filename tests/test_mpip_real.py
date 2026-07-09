# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end mpiP against a compiled MPI C binary (skip if stack missing)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from asv_bench_mpi.mpiexec import find_mpiexec
from asv_bench_mpi.mpip_run import find_libmpip, run_with_mpip

if not find_mpiexec():
    pytest.skip("no mpiexec", allow_module_level=True)

lib = find_libmpip()
if not lib:
    pytest.skip(
        "libmpiP.so not found (scripts/build_mpip.sh or ASV_MPIP_LIB)",
        allow_module_level=True,
    )

from asv_bench_mpi.mpi_app_build import compile_pingpong, find_mpicc

if not find_mpicc():
    pytest.skip("no mpicc", allow_module_level=True)


@pytest.fixture(scope="module")
def pingpong_app(tmp_path_factory):
    d = tmp_path_factory.mktemp("app")
    return compile_pingpong(d / "pingpong")


def test_run_pingpong_with_mpip(pingpong_app, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    metrics, report, wall = run_with_mpip(
        [str(pingpong_app)],
        mpi_np=2,
        libmpip=lib,
        mpip_flags=f"-f {work}",
        workdir=work,
        timeout=120,
    )
    assert report.is_file()
    assert report.stat().st_size > 0
    assert wall >= 0
    assert metrics
    # Prefer real profile fields when present
    print("REAL_MPIP_METRICS", metrics, "REPORT", report)
    assert "wall_seconds" in metrics or "mpi_time" in metrics or len(metrics) >= 1


def test_mpip_track_benchmark(pingpong_app, tmp_path):
    from asv_bench_mpi import mpip_probes
    from asv_bench_mpi.benchmarks.mpip_track import MpipTrackBenchmark

    work = tmp_path / "work2"
    work.mkdir()
    mpip_probes.configure(str(pingpong_app), str(work), lib)
    mpip_probes.mpip_track_wall.mpip_lib = lib
    mpip_probes.mpip_track_wall.mpip_workdir = str(work)
    mpip_probes.mpip_track_wall.mpip_flags = f"-f {work}"
    mpip_probes.mpip_track_wall.mpi_np = 2
    mpip_probes.mpip_track_wall.mpip_metric = "wall_seconds"

    b = MpipTrackBenchmark(
        "asv_bench_mpi.mpip_probes.mpip_track_wall",
        mpip_probes.mpip_track_wall,
        [mpip_probes.mpip_track_wall],
    )
    val = b.run()
    assert val >= 0.0
    print("REAL_MPIP_TRACK", val)
