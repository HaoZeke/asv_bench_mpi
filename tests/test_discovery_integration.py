# SPDX-License-Identifier: BSD-3-Clause
"""Ensure asv_runner picks up our types when the dist is installed."""

import importlib
import sys

import pytest


def test_types_registered_in_asv_runner():
    pytest.importorskip("asv_runner")
    # asv_runner.benchmarks builds benchmark_types at import time; reload so a
    # newly installed asv_bench_* package is scanned in the same process.
    for key in list(sys.modules):
        if key == "asv_runner.benchmarks" or key.startswith("asv_runner.benchmarks."):
            del sys.modules[key]
    import asv_runner.benchmarks as bm

    importlib.reload(bm)
    names = {cls.__name__ for cls in bm.benchmark_types}
    if "MpiTimeBenchmark" not in names and "MpiTrackBenchmark" not in names:
        pytest.skip("asv_bench_mpi not visible to asv_runner (not installed)")
    assert "MpiTimeBenchmark" in names
    assert "MpiTrackBenchmark" in names
