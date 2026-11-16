#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/tools/build_common.sh"
TP="$ROOT/third_party"
OPENBW_URL="$(pin openbw_url)"
OPENBW_COMMIT="$(pin openbw_commit)"
BWAPI_URL="$(pin bwapi_url)"
BWAPI_COMMIT="$(pin bwapi_commit)"
JOBS="${JOBS:-$(nproc)}"

git_fetch_pinned "$TP/openbw" "$OPENBW_URL" "$OPENBW_COMMIT"
git_fetch_pinned "$TP/bwapi" "$BWAPI_URL" "$BWAPI_COMMIT"
for p in "$TP"/patches/0*-openbw-*.patch; do [[ -e "$p" ]] && git_apply_patch "$TP/openbw" "$p" "[build]"; done
for p in "$TP"/patches/0*-bwapi-*.patch; do [[ -e "$p" ]] && git_apply_patch "$TP/bwapi" "$p" "[build]"; done

cmake -S "$TP/bwapi" -B "$TP/bwapi/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPENBW_DIR="$TP/openbw" \
  -DOPENBW_ENABLE_UI=0 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS="-include cstdint"
cmake --build "$TP/bwapi/build" -j "$JOBS"

echo "[build] done:"
find "$TP/bwapi/build" -maxdepth 3 -type f \( -name 'BWAPILauncher' -o -name '*.so' \) -printf '  %p\n'
