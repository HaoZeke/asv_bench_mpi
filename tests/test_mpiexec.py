# SPDX-License-Identifier: BSD-3-Clause
from asv_bench_mpi.mpiexec import build_mpi_command, find_mpiexec


def test_build_mpi_command_shape():
    cmd = build_mpi_command(4, "/usr/bin/python", ["-c", "pass"], mpiexec="/usr/bin/mpiexec")
    assert cmd[:4] == ["/usr/bin/mpiexec", "-n", "4", "/usr/bin/python"]
    assert cmd[4:] == ["-c", "pass"]


def test_build_rejects_bad_np():
    try:
        build_mpi_command(0, "python", ["-c", "pass"], mpiexec="/usr/bin/mpiexec")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_find_mpiexec_type():
    path = find_mpiexec()
    assert path is None or isinstance(path, str)
