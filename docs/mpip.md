# Phase B scaffold: LLNL mpiP

[mpiP](https://github.com/LLNL/mpiP) is a lightweight **PMPI** profiling
library for C/C++/Fortran MPI applications. It does **not** run inside the
Python interpreter. ASV should only:

1. Build or locate an MPI-enabled binary (user code / sample).
2. Link with `-lmpiP` **or** `LD_PRELOAD=libmpiP.so` at launch.
3. Run under `mpiexec` (environment may also use `asv_env_*` backends).
4. Parse the generated `*.mpiP` text report into `track_*` metrics.

## Link / preload (example)

```bash
# after building mpiP and MPI
mpicc -o examples/mpip/pingpong examples/mpip/pingpong.c -lmpi
# profile:
mpiexec -n 2 env MPIP="-f" \
  LD_PRELOAD=$PREFIX/lib/libmpiP.so \
  ./examples/mpip/pingpong
# produces something like: pingpong.<nprocs>.<pid>.mpiP
```

Or link at build time:

```bash
mpicc -o pingpong pingpong.c -L$MPIP_LIB -lmpiP -lbfd -liberty -lm -lunwind -lmpi
```

Exact extra libs depend on the mpiP build (`--with-binutils-dir`, libunwind, etc.).

## Mapping report fields → ASV

A typical mpiP report includes aggregate MPI time, call counts, and top sites.
Phase B Python helper (planned) will expose:

| Report field (illustrative) | ASV type |
|-----------------------------|----------|
| AppTime / MPI time fraction | `native_track_*` or dedicated `mpip_track_*` |
| MPI_Send count | track |
| Top collective time | track |

**Critical path:** the binary under profile is pure C/Fortran + PMPI. Python
only reads the report after the job exits (`run_executable` + parse).

## Sample report fixture

See `examples/mpip/sample_report.mpiP` (checked-in synthetic text) and
`asv_bench_mpi.mpip_report.parse_mpip_report()` for a dry-run parser used in
unit tests without an MPI stack.
