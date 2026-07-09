#!/usr/bin/env bash
# Full real path: build mpiP, build pingpong, LD_PRELOAD, parse report, pytest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v mpicc >/dev/null; then
  echo "mpicc required" >&2
  exit 1
fi
if ! command -v mpiexec >/dev/null; then
  echo "mpiexec required" >&2
  exit 1
fi

export ASV_MPIP_PREFIX="${ASV_MPIP_PREFIX:-$HOME/local/mpip}"
export ASV_MPIP_SRC="${ASV_MPIP_SRC:-$HOME/tmp/mpiP-src}"
bash "$ROOT/scripts/build_mpip.sh"
export ASV_MPIP_LIB="${ASV_MPIP_LIB:-$ASV_MPIP_PREFIX/lib/libmpiP.so}"
if [[ ! -f "$ASV_MPIP_LIB" ]]; then
  # search build tree
  ASV_MPIP_LIB="$(find "$ASV_MPIP_SRC" "$ASV_MPIP_PREFIX" -name 'libmpiP.so' 2>/dev/null | head -1 || true)"
fi
if [[ -z "${ASV_MPIP_LIB}" || ! -f "$ASV_MPIP_LIB" ]]; then
  echo "libmpiP.so still missing" >&2
  exit 1
fi
export ASV_MPIP_LIB
echo "Using ASV_MPIP_LIB=$ASV_MPIP_LIB"

python -m pip install -e ".[test]" -q
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=":oversubscribe"
python -m pytest -q tests/test_mpip_real.py tests/test_mpip_scaffold.py tests/test_mpip_run_unit.py -v --tb=short
echo "REAL_MPIP_E2E_OK"
