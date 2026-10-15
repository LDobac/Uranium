#!/usr/bin/env bash
# Build liburanium_bridge.so (spec 011): Rust staticlib + thin C++ shim. Requires tools/openbw/build.sh first.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -f "$ROOT/third_party/bwapi/build/lib/libBWAPILIB.so" ]] || { echo "run tools/openbw/build.sh first" >&2; exit 1; }
cargo build --release -p uranium-bridge-openbw --manifest-path "$ROOT/Cargo.toml"
cmake -S "$ROOT/bridge/openbw" -B "$ROOT/bridge/openbw/build" -DCMAKE_BUILD_TYPE=Release \
  -DRUST_STATICLIB="$ROOT/target/release/liburanium_bridge_openbw.a"
cmake --build "$ROOT/bridge/openbw/build" -j "$(nproc)"
echo "[bridge] $ROOT/bridge/openbw/build/liburanium_bridge.so"
