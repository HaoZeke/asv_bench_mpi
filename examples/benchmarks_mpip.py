# Real mpiP benchmark fragment for an ASV suite.

class PingPong:
    mpi_np = 2
    mpip_metric = "wall_seconds"
    unit = "seconds"
    timeout = 120

    def mpip_track_pingpong(self):
        from asv_bench_mpi.mpi_app_build import compile_pingpong

        app = compile_pingpong()
        return [str(app)]
