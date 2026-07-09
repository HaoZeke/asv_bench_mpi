# Design: native launch first, mpiP later

## Why the pure-Python mpi4py / mpiexec design is insufficient

`asv_bench_mpi` originally shipped `mpi_time_*` / `mpi_track_*` types that
shell out to `mpiexec` and import user code under **mpi4py**. That path:

1. Puts **Python on the critical path** of every rank (import, GIL, interpreter).
2. Cannot profile **real C/C++/Fortran MPI applications** the way HPC labs do.
3. Cannot drop the GIL around Fortran kernels when the kernel is not exposed
   through Python.
4. Fights free-threaded CPython: a pure-Python plugin does not declare
   `Py_MOD_GIL_NOT_USED` and adds no free-thread-safe native surface.

LLNL **mpiP** is a PMPI interposition library: it links (or preloads) against
the **application binary**, not against the Python interpreter. The right
ASV integration is therefore:

- **Phase A (this tree):** a **CPython extension** that can load/run compiled
  C and Fortran with the hot path under `Py_BEGIN_ALLOW_THREADS`, using
  **heap types** (`PyType_FromSpec`) and **module free-threading**
  (`Py_mod_gil` / `Py_MOD_GIL_NOT_USED`).
- **Phase B (scaffolded):** launch those same binaries under MPI with
  **mpiP** linked/preloaded; parse mpiP text reports into `track_*` metrics.
  Python only orchestrates; the profiled code never re-enters the interpreter.

## Phase A public extension surface (`asv_bench_mpi._native`)

| API | Role |
|-----|------|
| `NativeKernel(so_path, symbol)` | Heap type holding a `dlopen` handle + symbol |
| `NativeKernel.call(n=0) -> float` | Call `double (*)(long)` **without the GIL** |
| `NativeKernel.time(n=0) -> float` | Wall seconds for one call (GIL released) |
| `run_executable(argv, env=None) -> float` | Spawn binary, wait, wall seconds (GIL released around wait) |
| `extension_flags() -> dict` | Reports free-thread / GIL-not-used build markers |

Sample payloads under `asv_bench_mpi/payload/` are compiled at test/install
time into shared libraries for regression tests.

## Phase B (mpiP)

See `docs/mpip.md`. Not required for Phase A acceptance beyond scaffold
presence and a dry-run parse test of a **checked-in sample report**.

## Transitional pure-Python types

`mpi_time_*` / `mpi_track_*` remain importable for compatibility but are
documented as **transitional** and **non-primary**. Prefer
`native_time_*` / `native_track_*` for compiled work.


## Real mpiP path (implemented)

1. `scripts/build_mpip.sh` — build LLNL mpiP → `libmpiP.so`
2. `mpi_app_build.compile_pingpong()` — `mpicc` sample C app
3. `mpip_run.run_with_mpip(argv)` — `mpiexec -n N -x LD_PRELOAD=libmpiP.so app`
4. `parse_mpip_report` + `MpipTrackBenchmark` (`mpip_track_*`)

Python never enters the MPI ranks' hot path; only orchestrates launch and
parses the text report after finalize.
