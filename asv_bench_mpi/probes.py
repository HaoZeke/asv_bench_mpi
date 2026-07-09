# SPDX-License-Identifier: BSD-3-Clause
"""Tiny callables for live MPI tests (must be importable in mpiexec children)."""


def world_size():
    from mpi4py import MPI

    return float(MPI.COMM_WORLD.Get_size())


def barrier():
    from mpi4py import MPI

    MPI.COMM_WORLD.Barrier()
