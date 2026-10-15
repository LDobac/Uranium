#!/usr/bin/env bash
# M2 harness: scripted T0 Terran vs itself on OpenBW, N games, P in parallel (spec 011-AC1..AC4/AC7).
# Usage: tools/openbw/selfmatch.sh <fighting_spirit|circuit_breaker> <games> [parallel] [seed_base] [extra agent args...]
# Env: OUT_DIR (default runs/selfmatch-<map>-<ts>), PORT (default 7600), SAME_SEED=1 (all games use seed_base)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAP_KEY="$1"; GAMES="$2"; PAR="${3:-4}"; SEED_BASE="${4:-1000}"; shift $(( $# < 4 ? $# : 4 ))
case "$MAP_KEY" in
  fighting_spirit) MAP="$HOME/sc_data/maps/fighting_spirit_1.3.scx" ;;
  circuit_breaker) MAP="$HOME/sc_data/maps/circuit_breaker_1.0.scx" ;;
  *) echo "unknown map $MAP_KEY" >&2; exit 1 ;;
esac
OUT_DIR="${OUT_DIR:-$ROOT/runs/selfmatch-$MAP_KEY-$(date +%Y%m%d-%H%M%S)}"
PORT="${PORT:-7600}"
SOCK_DIR="/run/user/$(id -u)/uranium"  # short path: unix socket names are limited to 108 bytes
BWAPI="$ROOT/third_party/bwapi/build"
BRIDGE="$ROOT/bridge/openbw/build/liburanium_bridge.so"
AGENT="$ROOT/target/release/uranium-agent"
for f in "$MAP" "$BRIDGE" "$AGENT" "$BWAPI/bin/BWAPILauncher"; do [[ -e "$f" ]] || { echo "missing $f" >&2; exit 1; }; done
mkdir -p "$OUT_DIR" "$SOCK_DIR"

"$AGENT" serve --listen "127.0.0.1:$PORT" --out "$OUT_DIR/results.jsonl" "$@" > "$OUT_DIR/agent.log" 2>&1 &
AGENT_PID=$!
trap 'kill $AGENT_PID 2>/dev/null || true' EXIT
sleep 0.5

play() { # game index
  local i="$1" dir="$OUT_DIR/g$1" sock="$SOCK_DIR/$PORT-$1.sock" seed
  if [[ "${SAME_SEED:-0}" == 1 ]]; then seed="$SEED_BASE"; else seed=$((SEED_BASE + i)); fi
  mkdir -p "$dir/p0/bwapi-data" "$dir/p1/bwapi-data"; rm -f "$sock"
  for p in p0 p1; do
    (cd "$dir/$p" && env URANIUM_DATA_DIR="$HOME/sc_data/openbw" OPENBW_ENABLE_UI=0 OPENBW_LAN_MODE=LOCAL OPENBW_LOCAL_PATH="$sock" \
      URANIUM_AGENT_ADDR="127.0.0.1:$PORT" URANIUM_GAME_ID="$i" URANIUM_SEED="$seed" \
      BWAPI_CONFIG_AI__AI="$BRIDGE" BWAPI_CONFIG_AUTO_MENU__AUTO_MENU=LAN BWAPI_CONFIG_AUTO_MENU__MAP="$MAP" \
      BWAPI_CONFIG_AUTO_MENU__RACE=Terran BWAPI_CONFIG_AUTO_MENU__GAME_TYPE=MELEE BWAPI_CONFIG_AUTO_MENU__WAIT_FOR_MIN_PLAYERS=2 \
      BWAPI_CONFIG_AUTO_MENU__SAVE_REPLAY= BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART=OFF \
      timeout 900 "$BWAPI/bin/BWAPILauncher" > launcher.log 2>&1; echo $? > exit_code) &
    sleep 0.2
  done
  wait
  rm -f "$sock"
  echo "game $i seed $seed exits $(cat "$dir/p0/exit_code") $(cat "$dir/p1/exit_code")"
}
export -f play
export OUT_DIR SOCK_DIR PORT SEED_BASE MAP BRIDGE BWAPI ROOT
seq 1 "$GAMES" | xargs -P "$PAR" -I{} bash -c 'play {}' | tee "$OUT_DIR/games.txt"
sleep 1
kill $AGENT_PID 2>/dev/null || true
"$AGENT" summarize "$OUT_DIR/results.jsonl" --out "$OUT_DIR/summary.md" --title "Selfmatch $MAP_KEY ($GAMES games)"
cat "$OUT_DIR/summary.md"
