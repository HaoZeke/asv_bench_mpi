# SPDX-License-Identifier: BSD-3-Clause
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "asv_bench_mpi._native",
            sources=["asv_bench_mpi/_native.c"],
            extra_compile_args=["-fvisibility=hidden", "-O2"],
        )
    ],
)
