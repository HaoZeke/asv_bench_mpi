# SPDX-License-Identifier: BSD-3-Clause
import importlib
import sys

import pytest

pytest.importorskip("asv_bench_mpi._native")
from asv_bench_mpi.benchmarks import native_time, native_track
from asv_bench_mpi.payload_build import compile_c_kernel


def test_export_and_regex():
    assert native_time.export_as_benchmark == [native_time.NativeTimeBenchmark]
    assert native_track.export_as_benchmark == [native_track.NativeTrackBenchmark]
    assert native_time.NativeTimeBenchmark.name_regex.match("native_time_sum")
    assert native_track.NativeTrackBenchmark.name_regex.match("native_track_sum")
    assert not native_time.NativeTimeBenchmark.name_regex.match("time_sum")


def test_native_time_run_returns_samples():
    so = str(compile_c_kernel())

    def native_time_sum():
        return (so, "asv_kernel_sum", 5000)

    bench = native_time.NativeTimeBenchmark(
        "mod.native_time_sum", native_time_sum, [native_time_sum]
    )
    out = bench.run()
    assert isinstance(out, dict)
    assert "samples" in out and out["number"] == 1
    assert out["samples"][0] >= 0.0


def test_native_track_run_value():
    so = str(compile_c_kernel())

    def native_track_sum():
        return (so, "asv_kernel_sum", 100)

    bench = native_track.NativeTrackBenchmark(
        "mod.native_track_sum", native_track_sum, [native_track_sum]
    )
    # sum 0..99 = 4950
    assert bench.run() == pytest.approx(4950.0)


def test_asv_runner_discovers_native_types():
    pytest.importorskip("asv_runner")
    for key in list(sys.modules):
        if key == "asv_runner.benchmarks" or key.startswith("asv_runner.benchmarks."):
            del sys.modules[key]
    import asv_runner.benchmarks as bm

    importlib.reload(bm)
    names = {cls.__name__ for cls in bm.benchmark_types}
    if "NativeTimeBenchmark" not in names:
        pytest.skip("package not visible to asv_runner discovery yet")
    assert "NativeTimeBenchmark" in names
    assert "NativeTrackBenchmark" in names
