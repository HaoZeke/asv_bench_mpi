# asv_bench_mpi

Native **C/Fortran** launch (and future **mpiP**) plugin for
[airspeed velocity](https://asv.readthedocs.io/).

## Design (short)

| Phase | What | Critical path |
|-------|------|----------------|
| **A (primary)** | CPython extension `asv_bench_mpi._native` | Compiled `.so` / binary; **GIL released** |
| **B (scaffold)** | LLNL [mpiP](https://github.com/LLNL/mpiP) | MPI binary + PMPI; Python only parses report |
| Transitional | `mpi_time_*` / `mpi_track_*` (mpiexec + mpi4py) | Python on every rank — **not recommended** |

See [DESIGN.md](DESIGN.md) and [docs/mpip.md](docs/mpip.md).

## Install

```bash
pip install -e ".[test]"   # builds asv_bench_mpi._native
```

Requires a C compiler. Free-threaded CPython: the extension declares
`Py_MOD_GIL_NOT_USED` when built against headers that define `Py_mod_gil`.

## Primary benchmark types

| Prefix | Measures |
|--------|----------|
| `native_time_*` / `NativeTime*` | Wall seconds of `double (*)(long)` in a `.so` |
| `native_track_*` / `NativeTrack*` | Return value of that symbol |

```python
from asv_bench_mpi.payload_build import compile_c_kernel

SO = str(compile_c_kernel())

class Kernel:
    params = [10_000, 1_000_000]
    param_names = ["n"]

    def native_time_sum(self, n):
        return (SO, "asv_kernel_sum", n)

    def native_track_sum(self, n):
        return (SO, "asv_kernel_sum", n)
```

Or set attributes `so_path` / `symbol` / `n` on the function or class.

## Extension API

```python
from asv_bench_mpi import _native
from asv_bench_mpi.payload_build import compile_c_kernel

so = compile_c_kernel()
k = _native.NativeKernel(str(so), "asv_kernel_sum")
print(k.call(1000), k.time(1000))
print(_native.extension_flags())
print(_native.run_executable(["/bin/true"]))
```

## Real mpiP (native MPI apps)

```bash
./scripts/build_mpip.sh
export ASV_MPIP_LIB=$HOME/local/mpip/lib/libmpiP.so
pytest -q tests/test_mpip_real.py
```

Benchmark type: ``mpip_track_*`` / ``MpipTrack*`` — returns a metric from a
**real** mpiP report after ``mpiexec`` + ``LD_PRELOAD=libmpiP.so`` on a
compiled MPI binary (see ``docs/mpip.md``).

## Transitional mpi4py path

`mpi_time_*` / `mpi_track_*` still exist for pure-Python MPI microbenchmarks.
They are **not** the way to profile production C/Fortran MPI codes — use
Phase A binaries + Phase B mpiP.

## Matrix example

```json
{
  "matrix": {
    "req": {
      "asv_bench_mpi": [""]
    }
  }
}
```

For Phase B later: add `openmpi` / `mpich`, build mpiP, and link the app as
in `docs/mpip.md`.
