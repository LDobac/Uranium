#!/usr/bin/env bash
# Build UAlbertaBot as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Output: <URANIUM_OSS_BOTS_DIR>/ualbertabot/{module.so, VERSION, bwapi-data/AI/UAlbertaBot_Config.txt}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/ualbertabot"
COMMIT=558899d8793456f4a6ec4196efbb5235552e24db
OUT="$OSS_OUT_ROOT/ualbertabot"

oss_git_fetch "$SRC" https://github.com/davechurchill/ualbertabot.git "$COMMIT"
oss_patch "$SRC" "$ROOT/third_party/patches/0005-ualbertabot-linux-bwapi42.patch"

srcs=()
for f in "$SRC"/UAlbertaBot/Source/*.cpp "$SRC"/UAlbertaBot/Source/linux/*.cpp "$SRC"/BOSS/source/*.cpp "$SRC"/SparCraft/source/*.cpp; do
  case "$(basename "$f")" in main.cpp|BOSS_main.cpp|StarCraftGUI.cpp|TutorialCode.cpp|UnitInfoManager.cpp) continue ;; esac
  srcs+=("$f")
done
mapfile -t objs < <(oss_compile "$ROOT/build/obj/ualbertabot" -std=c++17 -O2 -fPIC -DURANIUM_BOSS_BWAPI \
  -include "$SRC/UAlbertaBot/Source/linux/compat.h" -I"$BWAPI_INC" -I"$SRC/UAlbertaBot/Source" -w -- "${srcs[@]}")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI"
cp "$SRC/UAlbertaBot/bin/UAlbertaBot_Config.txt" "$OUT/bwapi-data/AI/"
oss_finish ualbertabot "${COMMIT:0:7}"
