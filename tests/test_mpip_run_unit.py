# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

from asv_bench_mpi.mpip_report import find_mpip_reports, parse_mpip_report
from asv_bench_mpi.mpip_run import find_libmpip


def test_find_reports_classic_name(tmp_path):
    p = tmp_path / "pingpong.2.12345.mpiP"
    p.write_text("@--- MPI Time: 1.0\n@--- AppTime: 2.0\n@--- MPI %: 50\n")
    found = find_mpip_reports(tmp_path)
    assert p in found
    d = parse_mpip_report(p)
    assert d["mpi_time"] == 1.0


def test_find_libmpip_type():
    lib = find_libmpip()
    assert lib is None or Path(lib).is_file()
