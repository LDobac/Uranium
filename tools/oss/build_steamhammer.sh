#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
VER="$(pin steamhammer_version)"
SRC="$ROOT/third_party/steamhammer"
CFG="$ROOT/third_party/steamhammer_dll"
OUT="$OSS_OUT_ROOT/steamhammer"
URL="$(pin steamhammer_url)/$VER"
SRC_SHA="$(pin steamhammer_source_sha256)"
CFG_SHA="$(pin steamhammer_dll_sha256)"
PATCH="$(pin steamhammer_patch)"

oss_zip_fetch "$SRC" "$URL/Steamhammer_source.zip" "$SRC_SHA"
oss_zip_fetch "$CFG" "$URL/Steamhammer_dll.zip" "$CFG_SHA"
oss_patch "$SRC" "$PATCH"

S="$SRC/Steamhammer_source/Steamhammer/Source"
srcs=()
for f in "$S"/*.cpp "$S"/linux/*.cpp "$SRC"/Steamhammer_source/RC/Source/*.cpp; do
  case "$(basename "$f")" in Dll.cpp|"OpsBoss with immobile.cpp") continue ;; esac
  srcs+=("$f")
done
mapfile -t objs < <(oss_compile "$ROOT/build/obj/steamhammer" -std=c++20 -O2 -fPIC -fpermissive -w \
  -include "$S/linux/compat.h" -include climits -include cfloat -include cmath -I"$BWAPI_INC" -I"$S" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
cp "$CFG/Steamhammer_dll/Steamhammer_$VER.json" "$OUT/bwapi-data/AI/"
oss_finish steamhammer "$VER"
