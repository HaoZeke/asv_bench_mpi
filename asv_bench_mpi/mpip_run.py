# SPDX-License-Identifier: BSD-3-Clause
"""Launch an MPI binary under LLNL mpiP and collect the report.

Python is only used to set ``LD_PRELOAD`` / link-time is separate: the profiled
process is pure C/Fortran + PMPI. Critical path never re-enters the interpreter.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from asv_bench_mpi.mpiexec import build_mpi_command, find_mpiexec
from asv_bench_mpi.mpip_report import parse_mpip_report, find_mpip_reports

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
    which = shutil.which("libmpiP.so")
    if which:
        return which
    # conda-style / local installs
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
    # ldconfig
    try:
        out = subprocess.check_output(["ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "libmpiP.so" in line and "=>" in line:
                path = line.split("=>")[-1].strip()
                if os.path.isfile(path):
                    return path
    except (OSError, subprocess.CalledProcessError):
        pass
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
    """Run *argv* under mpiexec with mpiP preloaded.

    Parameters
    ----------
    argv
        Executable and arguments of the **native** MPI application (not Python).
    mpip_flags
        Value of ``MPIP`` env (default ``-f`` = put report in cwd).

    Returns
    -------
    metrics, report_path, wall_seconds
    """
    lib = libmpip or find_libmpip()
    if not lib:
        raise FileNotFoundError(
            "libmpiP.so not found; set ASV_MPIP_LIB or install mpiP "
            "(see docs/mpip.md / scripts/build_mpip.sh)"
        )
    if not argv:
        raise ValueError("argv must be non-empty")
    exe = str(argv[0])
    if not os.path.isfile(exe) and shutil.which(exe) is None:
        raise FileNotFoundError(f"MPI executable not found: {exe}")

    wd = Path(workdir or os.getcwd())
    wd.mkdir(parents=True, exist_ok=True)

    # Clear old reports in workdir matching common patterns
    before = set(find_mpip_reports(wd))

    child_env = dict(os.environ)
    if env:
        child_env.update({str(k): str(v) for k, v in env.items()})
    # Preload only the app ranks: put LD_PRELOAD in env for mpiexec children.
    # Some launchers need -x LD_PRELOAD; we set both.
    preload = lib
    if child_env.get("LD_PRELOAD"):
        preload = child_env["LD_PRELOAD"] + ":" + lib
    child_env["LD_PRELOAD"] = preload
    child_env["MPIP"] = mpip_flags
    # OpenMPI oversubscribe for small hosts
    child_env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    child_env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")

    launcher = mpiexec or find_mpiexec()
    if not launcher:
        raise FileNotFoundError("mpiexec/mpirun not found")

    # mpiexec -n N -x LD_PRELOAD -x MPIP ./app
    cmd: List[str] = [launcher, "-n", str(int(mpi_np))]
    if mpiexec_args:
        cmd.extend(list(mpiexec_args))
    # Portable-ish export of env to ranks (Open MPI / PRRTE)
    for var in ("LD_PRELOAD", "MPIP"):
        cmd.extend(["-x", var])
    cmd.extend([str(a) for a in argv])

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(wd),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"mpiexec+mpiP failed rc={proc.returncode}\n"
            f"cmd={cmd!r}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    after = set(find_mpip_reports(wd))
    new = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if not new:
        # fallback: any report newer than t0
        all_r = find_mpip_reports(wd)
        new = [p for p in all_r if p.stat().st_mtime >= t0 - 1.0]
        new.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not new:
        raise FileNotFoundError(
            f"mpiP ran but no report found in {wd}; stderr was:\n{proc.stderr}"
        )
    report = new[0]
    metrics = parse_mpip_report(report)
    metrics.setdefault("wall_seconds", wall)
    return metrics, report, wall
