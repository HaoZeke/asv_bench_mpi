# SPDX-License-Identifier: BSD-3-Clause
"""Launch an MPI binary under LLNL mpiP via the C extension.

The process spawn is **not** Python subprocess: ``asv_bench_mpi._native.run_executable``
performs fork/execve/waitpid with the GIL released. Python only builds argv/env
and parses the mpiP text report after the native job exits.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from asv_bench_mpi import _native
from asv_bench_mpi.mpiexec import find_mpiexec
from asv_bench_mpi.mpip_report import find_mpip_reports, parse_mpip_report

PathLike = Union[str, Path]


def find_libmpip() -> Optional[str]:
    """Locate libmpiP.so via env or common prefixes."""
    for key in ("ASV_MPIP_LIB", "MPIP_LIB", "MPIP_SO"):
        val = os.environ.get(key)
        if val and os.path.isfile(val):
            return val
        if val and os.path.isdir(val):
            cand = os.path.join(val, "libmpiP.so")
            if os.path.isfile(cand):
                return cand
    prefixes = []
    if os.environ.get("CONDA_PREFIX"):
        prefixes.append(os.environ["CONDA_PREFIX"])
    prefixes.extend(
        [
            str(Path.home() / "local" / "mpip"),
            str(Path.home() / "opt" / "mpip"),
            "/usr/local",
            "/usr",
        ]
    )
    for pref in prefixes:
        for sub in ("lib", "lib64"):
            cand = os.path.join(pref, sub, "libmpiP.so")
            if os.path.isfile(cand):
                return cand
    return None


def run_with_mpip(
    argv: Sequence[str],
    *,
    mpi_np: int = 2,
    mpiexec: Optional[str] = None,
    mpiexec_args: Optional[Sequence[str]] = None,
    libmpip: Optional[str] = None,
    mpip_flags: str = "-f .",
    workdir: Optional[PathLike] = None,
    timeout: float = 120.0,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, float], Path, float]:
    """Run *argv* under mpiexec with mpiP preloaded via C fork/exec.

    Parameters
    ----------
    timeout
        Soft limit checked only around the native call (extension does not
        implement alarm yet); used for documentation / future hard kill.
    """
    del timeout  # reserved; extension wait is unbounded for now
    lib = libmpip or find_libmpip()
    if not lib:
        raise FileNotFoundError(
            "libmpiP.so not found; set ASV_MPIP_LIB or run scripts/build_mpip.sh"
        )
    if not argv:
        raise ValueError("argv must be non-empty")
    exe = str(argv[0])
    if not os.path.isfile(exe) and shutil.which(exe) is None:
        raise FileNotFoundError(f"MPI executable not found: {exe}")

    wd = Path(workdir or os.getcwd())
    wd.mkdir(parents=True, exist_ok=True)
    before = set(find_mpip_reports(wd))

    launcher = mpiexec or find_mpiexec()
    if not launcher:
        raise FileNotFoundError("mpiexec/mpirun not found")

    # Full command line executed by _native (single fork of mpiexec).
    cmd: List[str] = [launcher, "-n", str(int(mpi_np))]
    if mpiexec_args:
        cmd.extend(list(mpiexec_args))
    for var in ("LD_PRELOAD", "MPIP", "OMPI_MCA_rmaps_base_oversubscribe",
                "PRTE_MCA_rmaps_default_mapping_policy",
                "OMPI_ALLOW_RUN_AS_ROOT", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"):
        cmd.extend(["-x", var])
    cmd.extend(str(a) for a in argv)

    child_env = {
        "LD_PRELOAD": lib,
        "MPIP": mpip_flags,
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    if env:
        child_env.update({str(k): str(v) for k, v in env.items()})
        # if caller set LD_PRELOAD, prepend libmpiP
        if "LD_PRELOAD" in env and lib not in env["LD_PRELOAD"]:
            child_env["LD_PRELOAD"] = env["LD_PRELOAD"] + ":" + lib

    # ---- THE ACTUAL SYSTEM LAUNCH IS IN C (_native.run_executable) ----
    t_mark = time.time()
    result = _native.run_executable(cmd, env=child_env, cwd=str(wd))
    wall = float(result["wall_seconds"])
    if not result.get("ok"):
        raise RuntimeError(
            f"native run_executable failed: returncode={result.get('returncode')} "
            f"argv0={result.get('argv0')} wall={wall}"
        )

    after = set(find_mpip_reports(wd))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if not new:
        all_r = find_mpip_reports(wd)
        new = [p for p in all_r if p.stat().st_mtime >= t_mark - 1.0]
        new.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not new:
        raise FileNotFoundError(f"mpiP ran but no report found in {wd}")
    report = new[0]
    metrics = parse_mpip_report(report)
    metrics.setdefault("wall_seconds", wall)
    metrics["_native_returncode"] = float(result["returncode"])
    return metrics, report, wall
