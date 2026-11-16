#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/oss/common.sh"
SRC="$ROOT/third_party/zzzkbot"
COMMIT=7183e37b6b416ea53c1040c83e639a3a3c395eed
OUT="$OSS_OUT_ROOT/zzzkbot"

oss_git_fetch "$SRC" https://github.com/chriscoxe/zzzkbot.git "$COMMIT"
oss_patch "$SRC" "$ROOT/third_party/patches/0007-zzzkbot-linux-bwapi42.patch"

S="$SRC/ZZZKBot/Source"
mapfile -t objs < <(oss_compile "$ROOT/build/obj/zzzkbot" -std=c++17 -O2 -fPIC -fpermissive -w \
  -include "$S/linux/compat.h" -I"$BWAPI_INC" -I"$S" -- "$S/ZZZKBotAIModule.cpp" "$S/linux/UraniumModule.cpp")
oss_link "$OUT/module.so" "${objs[@]}"
mkdir -p "$OUT/bwapi-data/AI" "$OUT/bwapi-data/read" "$OUT/bwapi-data/write"
oss_finish zzzkbot "${COMMIT:0:7}"
