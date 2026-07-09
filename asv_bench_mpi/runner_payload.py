# SPDX-License-Identifier: BSD-3-Clause
"""Child-process entry used under mpiexec (not imported by host discovery).

The host writes a small driver that sets env vars and execs this module's
``main`` logic via ``python -c`` or a temp script. Protocol:

- ``ASV_MPI_MODE``: ``time`` | ``track``
- ``ASV_MPI_MOD``: dotted module path containing the benchmark function
- ``ASV_MPI_FUNC``: attribute name of the function (or Class.method)
- ``ASV_MPI_PARAMS``: JSON list of parameters
- ``ASV_MPI_RESULT``: path to write a single float (rank 0 only for track;
  all ranks participate in the call)

For ``time`` mode, each rank runs the function; rank 0 writes the max
``MPI.Wtime()`` delta across ranks (via Allreduce MAX) as the sample.

For ``track`` mode, the function must return a number on every rank; rank 0
writes the value from rank 0 by default (``ASV_MPI_REDUCE=max|min|mean|rank0``).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time


def _resolve(mod_name: str, func_path: str):
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in func_path.split("."):
        obj = getattr(obj, part)
    # class method: instantiate if class bound
    if isinstance(obj, type):
        raise TypeError(f"resolved {func_path!r} to a class; use Class.method")
    # unbound method on class path "Class.method"
    parts = func_path.split(".")
    if len(parts) >= 2:
        parent = mod
        for part in parts[:-1]:
            parent = getattr(parent, part)
        if isinstance(parent, type):
            instance = parent()
            obj = getattr(instance, parts[-1])
    return obj


def main() -> int:
    mode = os.environ["ASV_MPI_MODE"]
    mod_name = os.environ["ASV_MPI_MOD"]
    func_path = os.environ["ASV_MPI_FUNC"]
    params = json.loads(os.environ.get("ASV_MPI_PARAMS", "[]"))
    result_path = os.environ["ASV_MPI_RESULT"]
    reduce_how = os.environ.get("ASV_MPI_REDUCE", "max" if mode == "time" else "rank0")

    try:
        from mpi4py import MPI
    except ImportError as err:
        if int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))) == 0:
            sys.stderr.write(f"mpi4py required in the benchmark env: {err}\n")
        return 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    func = _resolve(mod_name, func_path)

    if mode == "time":
        comm.Barrier()
        t0 = MPI.Wtime()
        func(*params)
        comm.Barrier()
        elapsed = MPI.Wtime() - t0
        max_elapsed = comm.allreduce(elapsed, op=MPI.MAX)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as fh:
                fh.write(repr(float(max_elapsed)))
        return 0

    if mode == "track":
        value = func(*params)
        value = float(value)
        if reduce_how == "rank0":
            out = value if rank == 0 else None
            out = comm.bcast(out, root=0)
        elif reduce_how == "max":
            out = comm.allreduce(value, op=MPI.MAX)
        elif reduce_how == "min":
            out = comm.allreduce(value, op=MPI.MIN)
        elif reduce_how == "mean":
            s = comm.allreduce(value, op=MPI.SUM)
            out = s / max(comm.Get_size(), 1)
        else:
            raise ValueError(f"unknown ASV_MPI_REDUCE={reduce_how!r}")
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as fh:
                fh.write(repr(float(out)))
        return 0

    raise ValueError(f"unknown ASV_MPI_MODE={mode!r}")


if __name__ == "__main__":
    # allow wall timing without MPI for unit tests of payload import
    if os.environ.get("ASV_MPI_MODE"):
        raise SystemExit(main())
    print("asv_bench_mpi.runner_payload: set ASV_MPI_* env to run", file=sys.stderr)
    raise SystemExit(1)
