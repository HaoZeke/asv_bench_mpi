# SPDX-License-Identifier: BSD-3-Clause
"""Time compiled C/Fortran kernels via asv_bench_mpi._native (GIL released)."""

from __future__ import annotations

import re

from asv_runner.benchmarks._base import Benchmark, _get_first_attr
from asv_runner.benchmarks._exceptions import NotRequired

try:
    from asv_bench_mpi import _native
except ImportError as err:  # pragma: no cover
    raise NotRequired(f"asv_bench_mpi._native not built: {err}") from err


class NativeTimeBenchmark(Benchmark):
    """Wall-time a ``double (*)(long)`` symbol in a shared library.

    Attributes (function or class):

    - ``so_path`` / ``native_so``: path to ``.so`` / ``.dylib`` / ``.dll``
    - ``symbol`` / ``native_symbol``: exported C symbol (default
      ``asv_kernel_sum``)
    - ``n`` / ``native_arg``: ``long`` argument (default ``0``)
    - ``number``: outer sample count (default ``1``)

    The Python function body is **not** timed; only the native call is.
    """

    name_regex = re.compile(r"^(NativeTime[A-Z_].+)|(native_time_.+)$")

    def __init__(self, name, func, attr_sources):
        Benchmark.__init__(self, name, func, attr_sources)
        self.type = "time"
        self.unit = "seconds"
        self._attr_sources = attr_sources

    def _resolve_so_symbol_n(self, *param):
        so = _get_first_attr(self._attr_sources, "so_path", None)
        if so is None:
            so = _get_first_attr(self._attr_sources, "native_so", None)
        sym = _get_first_attr(self._attr_sources, "symbol", None)
        if sym is None:
            sym = _get_first_attr(self._attr_sources, "native_symbol", "asv_kernel_sum")
        n = _get_first_attr(self._attr_sources, "n", None)
        if n is None:
            n = _get_first_attr(self._attr_sources, "native_arg", 0)
        # Allow parameterized benchmarks: first param overrides n if set
        if param:
            n = param[0]
        if not so:
            # Optional: function returns (so_path, symbol, n)
            ret = self.func(*param) if param else self.func()
            if isinstance(ret, (tuple, list)) and len(ret) >= 1:
                so = ret[0]
                if len(ret) >= 2 and ret[1]:
                    sym = ret[1]
                if len(ret) >= 3 and ret[2] is not None:
                    n = ret[2]
        if not so:
            raise RuntimeError(
                "native_time_* requires so_path/native_so attribute "
                "or a return value (so_path[, symbol[, n]])"
            )
        return str(so), str(sym), int(n)

    def run(self, *param):
        number = int(_get_first_attr(self._attr_sources, "number", 1) or 1)
        so, sym, n = self._resolve_so_symbol_n(*param)
        kernel = _native.NativeKernel(so, sym)
        samples = []
        for _ in range(max(number, 1)):
            samples.append(float(kernel.time(n)))
        return {"samples": samples, "number": 1}


export_as_benchmark = [NativeTimeBenchmark]
