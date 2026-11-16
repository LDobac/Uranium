#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/iron"
OUT="$OSS_OUT_ROOT/iron"
URL="$(pin iron_url)"
SHA="$(pin iron_sha256)"
PATCH="$(pin iron_patch)"
VER="$(pin iron_version)"

oss_zip_fetch "$SRC" "$URL" "$SHA"
oss_patch "$SRC" "$PATCH"

S="$SRC/Iron/src/Iron"
mapfile -t srcs < <(find "$S" -name '*.cpp' ! -name dllmain.cpp ! -path '*/behavior/wall.cpp' ! -path '*/EasyBMP_1.06/*' | sort)
mapfile -t objs < <(oss_compile "$ROOT/build/obj/iron" -std=c++17 -O2 -fPIC -fpermissive -w -DNOMINMAX \
  -include "$S/linux/compat.h" -include climits -include cfloat -include cmath -include stdexcept -include algorithm \
  -I"$BWAPI_INC" -I"$S" -I"$S/bwem/src" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish iron "$VER"
