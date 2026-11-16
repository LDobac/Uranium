#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/ualbertabot"
COMMIT="$(pin ualbertabot_commit)"
URL="$(pin ualbertabot_url)"
PATCH="$(pin ualbertabot_patch)"
OUT="$OSS_OUT_ROOT/ualbertabot"

git_fetch_pinned "$SRC" "$URL" "$COMMIT"
oss_patch "$SRC" "$PATCH"

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
