#!/usr/bin/env bash
# Build Iron (Terran, MIT/X11, AIIDE 2017 version written for BWAPI 4.1.2; includes BWEM) as a
# BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Source: the AIIDE 2019 bot archive (Iron returned with its 2017 version), pinned by sha256.
# Output: <URANIUM_OSS_BOTS_DIR>/iron/{module.so, VERSION, bwapi-data/}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/iron"
OUT="$OSS_OUT_ROOT/iron"

oss_zip_fetch "$SRC" https://davechurchill.ca/starcraft/aiide/results/2019/bots/Iron.zip d9a3eefe04c627550d57af611c17828971fba1667da402ccf708523b6b892f33
# Patch 0010: Linux entry points, include-path case (Windows file names are case-insensitive),
# MSVC shims (__pragma, strncpy_s) and template lookups GCC resolves at definition time.
oss_patch "$SRC" "$ROOT/third_party/patches/0010-iron-linux-bwapi42.patch"

S="$SRC/Iron/src/Iron"
# EasyBMP only serves BWEM's map printer, which Iron compiles out (BWEM_USE_MAP_PRINTER 0);
# behavior/wall.cpp is a stale file outside the Visual Studio project (duplicate symbols).
mapfile -t srcs < <(find "$S" -name '*.cpp' ! -name dllmain.cpp ! -path '*/behavior/wall.cpp' ! -path '*/EasyBMP_1.06/*' | sort)
mapfile -t objs < <(oss_compile "$ROOT/build/obj/iron" -std=c++17 -O2 -fPIC -fpermissive -w -DNOMINMAX \
  -include "$S/linux/compat.h" -include climits -include cfloat -include cmath -include stdexcept -include algorithm \
  -I"$BWAPI_INC" -I"$S" -I"$S/bwem/src" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish iron aiide2017
