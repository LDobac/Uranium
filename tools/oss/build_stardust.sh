#!/usr/bin/env bash
# Build Stardust (Protoss; MIT with a no-tournament-forks condition) as an AI module for our
# OpenBW BWAPI 4.2 fork, for use as a T1 opponent (spec 021).
# Output: <URANIUM_OSS_BOTS_DIR>/stardust/{module.so, VERSION, bwapi-data/}
#
# Stardust's own Linux build targets the author's extended OpenBW fork (IS_OPENBW). Those
# extensions (mining-path simulation, CherryVis instrumentation) are only used under
# IS_OPENBW / INSTRUMENTATION_ENABLED, so we build without them against our headers, as the
# tournament (Windows) build does.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/stardust"
COMMIT=22d93d7a55d0a0494384474a456fd7ee26baee97
OUT="$OSS_OUT_ROOT/stardust"
OBJ="$ROOT/build/obj/stardust"

oss_git_fetch "$SRC" https://github.com/bmnielsen/Stardust.git "$COMMIT"
oss_patch "$SRC" "$ROOT/third_party/patches/0008-stardust-linux-bwapi42.patch"

# Vendored zstd (C) for zstdstream, which the mining-path serializer links against.
Z="$SRC/3rdparty/zstd/lib"
mkdir -p "$OBJ/zstd"
zobjs=()
for f in "$Z"/common/*.c "$Z"/compress/*.c "$Z"/decompress/*.c; do
  o="$OBJ/zstd/$(basename "$f" .c).o"
  [[ -f "$o" && "$o" -nt "$f" ]] || gcc -O2 -fPIC -DZSTD_DISABLE_ASM -I"$Z" -I"$Z/common" -c "$f" -o "$o"
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
