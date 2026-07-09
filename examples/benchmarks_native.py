# Example ASV suite fragment using native_time_* / native_track_*
# Build the sample .so first:
#   python -c "from asv_bench_mpi.payload_build import compile_c_kernel; print(compile_c_kernel('/tmp/libasv_kernel_c.so'))"

from asv_bench_mpi.payload_build import compile_c_kernel

_SO = None


def _so():
    global _SO
    if _SO is None:
        _SO = str(compile_c_kernel())
    return _SO


class Kernel:
    params = [10_000, 1_000_000]
    param_names = ["n"]

    def native_time_sum(self, n):
        return (_so(), "asv_kernel_sum", n)

    def native_track_sum(self, n):
        return (_so(), "asv_kernel_sum", n)


Kernel.native_track_sum.unit = "checksum"
