#!/usr/bin/env bash
# Build McRave (Protoss/Zerg/Terran, MIT; BWEM + BWEB + Horizon included) as a BWAPI 4.2 (OpenBW)
# AI module for use as a T1 opponent (spec 021).
# Output: <URANIUM_OSS_BOTS_DIR>/mcrave/{module.so, VERSION, bwapi-data/}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/mcrave"
COMMIT=7d1719a22d8b896f957abae50e2ea5efff974fe2
OUT="$OSS_OUT_ROOT/mcrave"

oss_git_fetch "$SRC" https://github.com/Cmccrave/McRave.git "$COMMIT"
# Patch 0009: Linux entry points, and MSVC-only constructs GCC rejects (rvalues bound to
# non-const references, void ternaries, lookups into incomplete types, Windows headers).
oss_patch "$SRC" "$ROOT/third_party/patches/0009-mcrave-linux-bwapi42.patch"

S="$SRC/Source"
mapfile -t srcs < <(find "$S" -name '*.cpp' ! -name Dll.cpp | sort)
mapfile -t objs < <(oss_compile "$ROOT/build/obj/mcrave" -std=c++20 -O2 -fPIC -fpermissive -w -DNOMINMAX \
  -include "$S/McRave/Main/linux/compat.h" -include climits -include cfloat -include cmath \
  -I"$BWAPI_INC" -I"$S/BWEB" -I"$S/BWEM" -I"$S/Horizon" -I"$S/McRave" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish mcrave "${COMMIT:0:7}"
