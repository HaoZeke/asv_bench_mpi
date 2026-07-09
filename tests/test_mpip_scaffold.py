# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import pytest

from asv_bench_mpi.mpip_report import parse_mpip_report


def test_parse_sample_report():
    sample = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "mpip"
        / "sample_report.mpiP"
    )
    d = parse_mpip_report(sample)
    assert d["mpi_time"] == pytest.approx(0.0042)
    assert d["app_time"] == pytest.approx(0.0100)
    assert d["mpi_percent"] == pytest.approx(42.0)


def test_mpip_docs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "mpip.md").is_file()
    assert (root / "examples" / "mpip" / "pingpong.c").is_file()
