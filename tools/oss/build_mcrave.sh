#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/mcrave"
COMMIT="$(pin mcrave_commit)"
URL="$(pin mcrave_url)"
PATCH="$(pin mcrave_patch)"
OUT="$OSS_OUT_ROOT/mcrave"

git_fetch_pinned "$SRC" "$URL" "$COMMIT"
oss_patch "$SRC" "$PATCH"

S="$SRC/Source"
mapfile -t srcs < <(find "$S" -name '*.cpp' ! -name Dll.cpp | sort)
mapfile -t objs < <(oss_compile "$ROOT/build/obj/mcrave" -std=c++20 -O2 -fPIC -fpermissive -w -DNOMINMAX \
  -include "$S/McRave/Main/linux/compat.h" -include climits -include cfloat -include cmath \
  -I"$BWAPI_INC" -I"$S/BWEB" -I"$S/BWEM" -I"$S/Horizon" -I"$S/McRave" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish mcrave "${COMMIT:0:7}"
