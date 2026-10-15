#!/usr/bin/env bash
# Build OpenBW + BWAPI (OpenBW fork) headless for Uranium (spec 011, M1(b)).
# Clones pinned commits into third_party/, applies third_party/patches/*.patch, builds Release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TP="$ROOT/third_party"
OPENBW_COMMIT=4b046d5f65302b10cb0a745f0fecd37ec85b20a8
BWAPI_COMMIT=48124ba8ed1b4d52b3dfd52acbaf34afb9a37fe2
JOBS="${JOBS:-$(nproc)}"

fetch() { # dir url commit
  local dir="$1" url="$2" commit="$3"
  if [[ ! -d "$dir/.git" ]]; then
    git clone -q "$url" "$dir"
  fi
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$dir" fetch -q origin "$commit"
    git -C "$dir" checkout -q "$commit"
  fi
}

apply_patch() { # dir patch
  local dir="$1" patch="$2"
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "[build] already applied: $(basename "$patch")"
  else
    git -C "$dir" apply "$patch"
    echo "[build] applied: $(basename "$patch")"
  fi
}

fetch "$TP/openbw" https://github.com/OpenBW/openbw.git "$OPENBW_COMMIT"
fetch "$TP/bwapi" https://github.com/OpenBW/bwapi.git "$BWAPI_COMMIT"
for p in "$TP"/patches/0*-openbw-*.patch; do [[ -e "$p" ]] && apply_patch "$TP/openbw" "$p"; done
for p in "$TP"/patches/0*-bwapi-*.patch; do [[ -e "$p" ]] && apply_patch "$TP/bwapi" "$p"; done

cmake -S "$TP/bwapi" -B "$TP/bwapi/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENBW_DIR="$TP/openbw" \
  -DOPENBW_ENABLE_UI=0 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS="-include cstdint"
cmake --build "$TP/bwapi/build" -j "$JOBS"

echo "[build] done:"
find "$TP/bwapi/build" -maxdepth 3 -type f \( -name 'BWAPILauncher' -o -name '*.so' \) -printf '  %p\n'
