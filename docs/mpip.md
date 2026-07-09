# Real mpiP integration (LLNL)

[mpiP](https://github.com/LLNL/mpiP) is a **PMPI** light-weight profiler for
**C/C++/Fortran MPI binaries**. Python is not on the profiled critical path.

## Build mpiP

```bash
# need: mpicc, libunwind, binutils (libbfd)
./scripts/build_mpip.sh
export ASV_MPIP_LIB=$HOME/local/mpip/lib/libmpiP.so
```

## Profile an application

```python
from asv_bench_mpi.mpi_app_build import compile_pingpong
from asv_bench_mpi.mpip_run import run_with_mpip

app = compile_pingpong()
metrics, report, wall = run_with_mpip([str(app)], mpi_np=2)
print(metrics, report)
```

Or as an ASV benchmark:

```python
class PingPong:
    mpi_np = 2
    mpip_metric = "mpi_percent"  # or wall_seconds, mpi_time, ...
    unit = "percent"

    def mpip_track_app(self):
        from asv_bench_mpi.mpi_app_build import compile_pingpong
        return [str(compile_pingpong())]
```

Launch uses `mpiexec -n N -x LD_PRELOAD -x MPIP ./app` with
`LD_PRELOAD=libmpiP.so`. The extension `run_executable` / process wait can
also be used without MPI for serial native codes (Phase A).

## Report → ASV

`parse_mpip_report()` maps common fields (`mpi_time`, `app_time`,
`mpi_percent`, …). `MpipTrackBenchmark` records one field as a track metric.
