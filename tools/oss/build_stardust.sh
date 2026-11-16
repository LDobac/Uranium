#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/stardust"
COMMIT="$(pin stardust_commit)"
URL="$(pin stardust_url)"
PATCH="$(pin stardust_patch)"
OUT="$OSS_OUT_ROOT/stardust"
OBJ="$ROOT/build/obj/stardust"

git_fetch_pinned "$SRC" "$URL" "$COMMIT"
oss_patch "$SRC" "$PATCH"

Z="$SRC/3rdparty/zstd/lib"
zflags=(-O2 -fPIC -DNDEBUG -DZSTD_DISABLE_ASM -I"$Z" -I"$Z/common")
oss_flags_stamp "$OBJ/zstd" "${zflags[@]}"
zobjs=()
for f in "$Z"/common/*.c "$Z"/compress/*.c "$Z"/decompress/*.c; do
  o="$OBJ/zstd/$(basename "$f" .c).o"
  [[ -f "$o" && "$o" -nt "$f" ]] || gcc "${zflags[@]}" -c "$f" -o "$o"
  zobjs+=("$o")
done

inc=(-I"$SRC/3rdparty" -I"$SRC/3rdparty/BWEM/include" -I"$SRC/3rdparty/FAP" -I"$SRC/3rdparty/bitsery/include"
     -I"$SRC/3rdparty/zstdstream" -I"$Z" -I"$SRC/3rdparty/nlohmann")
for d in "" Builder General Instrumentation Map Map/PathFinding Players Producer Strategist Units Util Workers; do inc+=(-I"$SRC/src/$d"); done
mapfile -t srcs < <(find "$SRC/src" "$SRC/3rdparty/BWEM/src" "$SRC/3rdparty/zstdstream" -name '*.cpp' | sort)
mapfile -t objs < <(oss_compile "$OBJ" -std=c++20 -O2 -fPIC -w -include cstdint -include climits -include memory \
  -I"$BWAPI_INC" "${inc[@]}" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}" "${zobjs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish stardust "${COMMIT:0:7}"
