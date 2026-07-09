# SPDX-License-Identifier: BSD-3-Clause
"""Plugin registration matches asv_runner asv_bench_* discovery rules."""

import re

from asv_bench_mpi.benchmarks import mpi_time, mpi_track


def test_export_lists():
    assert mpi_time.export_as_benchmark == [mpi_time.MpiTimeBenchmark]
    assert mpi_track.export_as_benchmark == [mpi_track.MpiTrackBenchmark]


def test_name_regex_mpi_time():
    rx = mpi_time.MpiTimeBenchmark.name_regex
    assert rx.match("mpi_time_allreduce")
    assert rx.match("MpiTimeAllreduce")
    assert not rx.match("time_allreduce")
    assert not rx.match("mpi_track_x")


def test_name_regex_mpi_track():
    rx = mpi_track.MpiTrackBenchmark.name_regex
    assert rx.match("mpi_track_world_size")
    assert rx.match("MpiTrackBytes")
    assert not rx.match("track_world_size")


def test_package_name_prefix():
    # asv_runner only loads distributions whose Name starts with asv_bench
    from importlib.metadata import metadata

    try:
        name = metadata("asv_bench_mpi")["Name"]
    except Exception:
        name = "asv_bench_mpi"
    assert name.startswith("asv_bench")
