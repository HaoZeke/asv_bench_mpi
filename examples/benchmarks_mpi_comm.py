# Example ASV suite fragment (copy into your benchmarks/ package).
# Requires asv_bench_mpi + mpi4py in the environment matrix.


class Allreduce:
    mpi_np = 2
    timeout = 60
    params = [1024, 65536]
    param_names = ["n"]

    def mpi_time_allreduce(self, n):
        from mpi4py import MPI
        import numpy as np

        x = np.ones(int(n), dtype=np.float64)
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, x, op=MPI.SUM)

    def mpi_track_world_size(self, n):
        from mpi4py import MPI

        return float(MPI.COMM_WORLD.Get_size())


Allreduce.mpi_track_world_size.unit = "ranks"
Allreduce.mpi_track_world_size.mpi_reduce = "rank0"
