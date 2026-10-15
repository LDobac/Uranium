#!/usr/bin/env bash
# Build Steamhammer 5.3.6 as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Output: <URANIUM_OSS_BOTS_DIR>/steamhammer/{module.so, VERSION, bwapi-data/AI/Steamhammer_5.3.6.json}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
VER=5.3.6
SRC="$ROOT/third_party/steamhammer"
CFG="$ROOT/third_party/steamhammer_dll"
OUT="$OSS_OUT_ROOT/steamhammer"
URL=https://satirist.org/downloads/starcraft/steamhammer/$VER

# Source archive (MIT) and the release package that carries the bot's JSON configuration.
oss_zip_fetch "$SRC" "$URL/Steamhammer_source.zip" d7ccc9f54cb8a9b1d11befc136272646767a7fd494dd937924a82e03581ad522
oss_zip_fetch "$CFG" "$URL/Steamhammer_dll.zip" afde6ff7a697abae5364c3e6c623aa579524f05e3015f662e4ef7bce379cc83c
oss_patch "$SRC" "$ROOT/third_party/patches/0006-steamhammer-linux-bwapi42.patch"

S="$SRC/Steamhammer_source/Steamhammer/Source"
srcs=()
for f in "$S"/*.cpp "$S"/linux/*.cpp "$SRC"/Steamhammer_source/RC/Source/*.cpp; do
  # Dll.cpp: Windows entry points (replaced by linux/UraniumModule.cpp); the other file is an
  # unused variant that is not part of the Visual Studio project.
  case "$(basename "$f")" in Dll.cpp|"OpsBoss with immobile.cpp") continue ;; esac
  srcs+=("$f")
done
mapfile -t objs < <(oss_compile "$ROOT/build/obj/steamhammer" -std=c++20 -O2 -fPIC -fpermissive -w \
  -include "$S/linux/compat.h" -include climits -include cfloat -include cmath -I"$BWAPI_INC" -I"$S" -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
cp "$CFG/Steamhammer_dll/Steamhammer_$VER.json" "$OUT/bwapi-data/AI/"
oss_finish steamhammer "$VER"
