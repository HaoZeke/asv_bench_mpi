# asv_bench_mpi

MPI / **mpi4py** benchmark plugin for [airspeed velocity](https://asv.readthedocs.io/)
(`asv` + `asv_runner`).

Same extension contract as [`asv_bench_memray`](https://github.com/HaoZeke/asv_bench_memray):

- package name starts with `asv_bench`
- types live under `asv_bench_mpi/benchmarks/`
- each module exports `export_as_benchmark = [...]`

## What you get

| Prefix | Type | Measures |
|--------|------|----------|
| `mpi_time_*` / `MpiTime*` | `time` (seconds) | Wall time under `mpiexec -n <mpi_np>` (max `MPI.Wtime()` across ranks) |
| `mpi_track_*` / `MpiTrack*` | `track` | Numeric metric from ranks (`mpi_reduce`: `rank0` / `max` / `min` / `mean`) |

## Install

Host (discover plugin) **and** benchmark env (run under MPI):

```bash
pip install "asv_bench_mpi[mpi]"   # pulls mpi4py; you still need an MPI library + mpiexec
# or matrix:
```

```json
{
  "matrix": {
    "req": {
      "asv_bench_mpi": [""],
      "mpi4py": [""]
    }
  }
}
```

For conda/rattler/pixi backends, prefer conda-forge:

```json
{
  "environment_type": "rattler",
  "matrix": {
    "req": {
      "mpi4py": [""],
      "openmpi": [""],
      "asv_bench_mpi": [""]
    }
  }
}
```

Launcher resolution: `ASV_MPIEXEC` / `MPIEXEC` / `MPIRUN` env, else `mpiexec` / `mpirun` on `PATH`.

## Example benchmarks

```python
# benchmarks/mpi_comm.py
class Allreduce:
    """Scaling-style microbench: Allreduce on a small buffer."""

    mpi_np = 4
    timeout = 60
    params = [1_000, 100_000]
    param_names = ["n"]

    def mpi_time_allreduce(self, n):
        from mpi4py import MPI
        import numpy as np

        comm = MPI.COMM_WORLD
        x = np.ones(n, dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, x, op=MPI.SUM)

    def mpi_track_rank_size(self, n):
        from mpi4py import MPI

        return float(MPI.COMM_WORLD.Get_size())

Allreduce.mpi_track_rank_size.unit = "ranks"
Allreduce.mpi_track_rank_size.mpi_reduce = "rank0"
```

```bash
asv run --bench mpi_time_allreduce
```

## Attributes

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `mpi_np` | `2` | Ranks passed as `mpiexec -n` |
| `mpiexec` | auto | Launcher path |
| `mpiexec_args` | `[]` | Extra launcher flags |
| `mpi_reduce` | `rank0` (track) / `max` (time) | Cross-rank reduction for track metrics |
| `timeout` | `120` | Seconds for the mpiexec job |
| `number` | `1` | Outer repeats of full mpiexec for `mpi_time_*` |

## Design notes

- Each sample launches a **fresh MPI job** (like `timeraw_*` uses a fresh process). Setup that must run once per rank goes inside the benchmark body.
- The callable must live in an **importable module** (normal asv benchmark tree).
- OpenMPI oversubscribe is enabled by default via MCA env for small CI hosts; override if needed.
- This is **not** LLNL `mpip` (the profiler). That could be a future optional path.

## Development

```bash
pip install -e ".[test]"
pytest -q
# with MPI:
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
```
