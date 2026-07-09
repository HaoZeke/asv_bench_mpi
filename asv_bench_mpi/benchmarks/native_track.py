# SPDX-License-Identifier: BSD-3-Clause
"""Track numeric result of a compiled kernel via asv_bench_mpi._native."""

from __future__ import annotations

import re

from asv_runner.benchmarks._base import Benchmark, _get_first_attr
from asv_runner.benchmarks._exceptions import NotRequired

try:
    from asv_bench_mpi import _native
except ImportError as err:  # pragma: no cover
    raise NotRequired(f"asv_bench_mpi._native not built: {err}") from err


class NativeTrackBenchmark(Benchmark):
    """Call ``double (*)(long)`` and record the return value (GIL released).

    Attributes: ``so_path`` / ``native_so``, ``symbol`` / ``native_symbol``,
    ``n`` / ``native_arg``, ``unit`` (default ``unit``).
    """

    name_regex = re.compile(r"^(NativeTrack[A-Z_].+)|(native_track_.+)$")

    def __init__(self, name, func, attr_sources):
        Benchmark.__init__(self, name, func, attr_sources)
        self.type = _get_first_attr(attr_sources, "type", "track")
        self.unit = _get_first_attr(attr_sources, "unit", "unit")
        self._attr_sources = attr_sources

    def run(self, *param):
        so = _get_first_attr(self._attr_sources, "so_path", None)
        if so is None:
            so = _get_first_attr(self._attr_sources, "native_so", None)
        sym = _get_first_attr(self._attr_sources, "symbol", None)
        if sym is None:
            sym = _get_first_attr(self._attr_sources, "native_symbol", "asv_kernel_sum")
        n = _get_first_attr(self._attr_sources, "n", None)
        if n is None:
            n = _get_first_attr(self._attr_sources, "native_arg", 0)
        if param:
            n = param[0]
        if not so:
            ret = self.func(*param) if param else self.func()
            if isinstance(ret, (tuple, list)) and len(ret) >= 1:
                so = ret[0]
                if len(ret) >= 2 and ret[1]:
                    sym = ret[1]
                if len(ret) >= 3 and ret[2] is not None:
                    n = ret[2]
        if not so:
            raise RuntimeError("native_track_* requires so_path or return tuple")
        kernel = _native.NativeKernel(str(so), str(sym))
        return float(kernel.call(int(n)))


export_as_benchmark = [NativeTrackBenchmark]
