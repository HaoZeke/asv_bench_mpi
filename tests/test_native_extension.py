# SPDX-License-Identifier: BSD-3-Clause
"""Drive the shipped _native extension (not a reimplementation)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("asv_bench_mpi._native")
from asv_bench_mpi import _native
from asv_bench_mpi.payload_build import compile_c_kernel


def test_extension_flags_mark_free_thread_and_gil():
    flags = _native.extension_flags()
    assert flags["heap_type"] is True
    assert flags["py_begin_allow_threads"] is True
    # gil_not_used True when built against headers with Py_mod_gil
    assert "gil_not_used" in flags
    assert flags["api"] == "native-launch+exec"
    assert "fork" in flags["system_calls"]
    assert "execve" in flags["system_calls"]


def test_source_has_pytype_spec_and_gil_slots():
    root = Path(__file__).resolve().parents[1]
    src = (root / "asv_bench_mpi" / "_native.c").read_text()
    assert "PyType_Spec" in src
    assert "PyType_FromSpec" in src
    assert "Py_BEGIN_ALLOW_THREADS" in src
    assert "Py_mod_gil" in src
    assert "Py_MOD_GIL_NOT_USED" in src


def test_native_kernel_call_and_time():
    so = compile_c_kernel()
    k = _native.NativeKernel(str(so), "asv_kernel_sum")
    # sum 0..9999 = 9999*10000/2 = 49995000
    val = k.call(10_000)
    assert val == pytest.approx(49995000.0)
    t = k.time(10_000)
    assert math.isfinite(t) and t >= 0.0


def test_run_executable_true():
    # /bin/true or true on PATH — system launch in C (fork/exec/wait)
    import shutil

    true = shutil.which("true") or "/bin/true"
    r = _native.run_executable([true])
    assert isinstance(r, dict)
    assert r["ok"] == 1
    assert r["returncode"] == 0
    assert math.isfinite(r["wall_seconds"]) and r["wall_seconds"] >= 0.0


def test_run_executable_env_and_false():
    import shutil

    false = shutil.which("false") or "/bin/false"
    r = _native.run_executable([false])
    assert r["ok"] == 0
    assert r["returncode"] != 0

    # env merge: empty command that checks env would need a shell; use env binary
    envbin = shutil.which("env")
    if envbin:
        r2 = _native.run_executable([envbin, "true"], env={"ASV_NATIVE_TEST": "1"})
        assert r2["ok"] == 1


def test_source_documents_system_calls():
    root = Path(__file__).resolve().parents[1]
    src = (root / "asv_bench_mpi" / "_native.c").read_text()
    for s in ("fork(", "execve(", "execvp(", "waitpid(", "chdir("):
        assert s in src, s
