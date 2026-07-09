#!/usr/bin/env bash
# Build LLNL mpiP against the active MPI (mpicc on PATH).
set -euo pipefail
PREFIX="${ASV_MPIP_PREFIX:-$HOME/local/mpip}"
SRC="${ASV_MPIP_SRC:-$HOME/tmp/mpiP-src}"

mkdir -p "$(dirname "$SRC")" "$PREFIX"
if [[ ! -d "$SRC/.git" ]]; then
  rm -rf "$SRC"
  git clone --depth 1 https://github.com/LLNL/mpiP.git "$SRC"
fi
cd "$SRC"

# Generate configure if needed (autotools projects sometimes ship it)
if [[ ! -x ./configure ]]; then
  if [[ -f ./configure.ac ]]; then
    autoreconf -if || true
  fi
fi

export CC="${CC:-mpicc}"
export CXX="${CXX:-mpicxx}"

# Prefer shared library
CFG=(--prefix="$PREFIX")
# binutils for BFD if present
if [[ -d /usr/include ]]; then
  CFG+=(--with-binutils-dir=/usr)
fi

if [[ ! -f Makefile ]]; then
  ./configure "${CFG[@]}" CC="$CC" CXX="$CXX" || ./configure --prefix="$PREFIX" CC="$CC"
fi
make -j"$(nproc 2>/dev/null || echo 2)"
make install || true

LIB=""
for cand in \
  "$PREFIX/lib/libmpiP.so" \
  "$PREFIX/lib64/libmpiP.so" \
  "$SRC/libmpiP.so" \
  "$SRC/.libs/libmpiP.so"
do
  if [[ -f "$cand" ]]; then LIB="$cand"; break; fi
done
if [[ -z "$LIB" ]]; then
  LIB="$(find "$SRC" "$PREFIX" -name 'libmpiP.so' 2>/dev/null | head -1 || true)"
fi
if [[ -z "$LIB" || ! -f "$LIB" ]]; then
  echo "libmpiP.so not found after build; tree:" >&2
  find "$SRC" -name '*mpiP*' 2>/dev/null | head -40 >&2
  exit 1
fi
mkdir -p "$PREFIX/lib"
if [[ "$LIB" != "$PREFIX/lib/libmpiP.so" ]]; then
  cp -a "$LIB" "$PREFIX/lib/libmpiP.so" 2>/dev/null || cp -a "$LIB" "$PREFIX/lib/"
  LIB="$PREFIX/lib/libmpiP.so"
fi
echo "Installed libmpiP to $LIB"
echo "export ASV_MPIP_LIB=$LIB"
