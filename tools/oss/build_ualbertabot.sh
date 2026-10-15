#!/usr/bin/env bash
# Build UAlbertaBot as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Output: $OUT/ualbertabot/{module.so, bwapi-data/AI/UAlbertaBot_Config.txt}
# OUT defaults to $URANIUM_OSS_BOTS_DIR (from .env) or <repo>/build/oss_bots.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/env.sh"; uranium_load_env "$ROOT"
SRC="$ROOT/third_party/ualbertabot"
COMMIT=558899d8793456f4a6ec4196efbb5235552e24db
PATCH="$ROOT/third_party/patches/0005-ualbertabot-linux-bwapi42.patch"
BWAPI_INC="$ROOT/third_party/bwapi/bwapi/include"
BWAPI_LIB="$ROOT/third_party/bwapi/build/lib"
OUT="${URANIUM_OSS_BOTS_DIR:-$ROOT/build/oss_bots}/ualbertabot"
OBJ="$ROOT/build/obj/ualbertabot"
[[ -f "$BWAPI_LIB/libBWAPILIB.so" ]] || { echo "run tools/openbw/build.sh first" >&2; exit 1; }

[[ -d "$SRC/.git" ]] || git clone -q https://github.com/davechurchill/ualbertabot.git "$SRC"
if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ]]; then git -C "$SRC" fetch -q origin "$COMMIT"; git -C "$SRC" checkout -q "$COMMIT"; fi
if git -C "$SRC" apply --reverse --check "$PATCH" 2>/dev/null; then echo "[uab] patch already applied"; else git -C "$SRC" apply "$PATCH"; echo "[uab] patch applied"; fi

mkdir -p "$OBJ" "$OUT/bwapi-data/AI"
CXXFLAGS=(-std=c++17 -O2 -fPIC -DURANIUM_BOSS_BWAPI -include "$SRC/UAlbertaBot/Source/linux/compat.h" -I"$BWAPI_INC" -I"$SRC/UAlbertaBot/Source" -w)
objs=()
for f in "$SRC"/UAlbertaBot/Source/*.cpp "$SRC"/UAlbertaBot/Source/linux/*.cpp "$SRC"/BOSS/source/*.cpp "$SRC"/SparCraft/source/*.cpp; do
  case "$(basename "$f")" in main.cpp|BOSS_main.cpp|StarCraftGUI.cpp|TutorialCode.cpp|UnitInfoManager.cpp) continue ;; esac
  o="$OBJ/$(echo "${f#$SRC/}" | tr / _).o"
  if [[ ! -f "$o" || "$f" -nt "$o" ]]; then echo "$f"; fi
  objs+=("$o")
done | xargs -r -P "$(nproc)" -I{} sh -c 'f="{}"; o="'"$OBJ"'/$(echo "${f#'"$SRC"'/}" | tr / _).o"; g++ '"${CXXFLAGS[*]}"' -c "$f" -o "$o"'
for f in "$SRC"/UAlbertaBot/Source/*.cpp "$SRC"/UAlbertaBot/Source/linux/*.cpp "$SRC"/BOSS/source/*.cpp "$SRC"/SparCraft/source/*.cpp; do
  case "$(basename "$f")" in main.cpp|BOSS_main.cpp|StarCraftGUI.cpp|TutorialCode.cpp|UnitInfoManager.cpp) continue ;; esac
  objs+=("$OBJ/$(echo "${f#$SRC/}" | tr / _).o")
done
g++ -shared -o "$OUT/module.so" "${objs[@]}" -L"$BWAPI_LIB" -lBWAPILIB -Wl,-rpath,"$BWAPI_LIB" -pthread
cp "$SRC/UAlbertaBot/bin/UAlbertaBot_Config.txt" "$OUT/bwapi-data/AI/"
echo "[uab] built $OUT/module.so"
