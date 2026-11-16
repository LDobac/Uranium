#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/zzzkbot"
COMMIT="$(pin zzzkbot_commit)"
URL="$(pin zzzkbot_url)"
PATCH="$(pin zzzkbot_patch)"
OUT="$OSS_OUT_ROOT/zzzkbot"

git_fetch_pinned "$SRC" "$URL" "$COMMIT"
oss_patch "$SRC" "$PATCH"

S="$SRC/ZZZKBot/Source"
mapfile -t objs < <(oss_compile "$ROOT/build/obj/zzzkbot" -std=c++17 -O2 -fPIC -fpermissive -w \
  -include "$S/linux/compat.h" -I"$BWAPI_INC" -I"$S" -- "$S/ZZZKBotAIModule.cpp" "$S/linux/UraniumModule.cpp")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish zzzkbot "${COMMIT:0:7}"
