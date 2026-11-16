source "$ROOT/tools/env.sh"; uranium_load_env "$ROOT"
source "$ROOT/tools/build_common.sh"
BWAPI_INC="$ROOT/third_party/bwapi/bwapi/include"
BWAPI_LIB="$ROOT/third_party/bwapi/build/lib"
OSS_OUT_ROOT="${URANIUM_OSS_BOTS_DIR:-$ROOT/build/oss_bots}"
JOBS="${JOBS:-$(nproc)}"
[[ -f "$BWAPI_LIB/libBWAPILIB.so" ]] || { echo "run tools/openbw/build.sh first" >&2; exit 1; }

oss_zip_fetch() {
  local dir="$1" url="$2" sha="$3" zip="$ROOT/third_party/$(basename "$1").zip"
  if [[ ! -d "$dir/.git" ]]; then
    [[ -f "$zip" ]] || curl -fsSL -o "$zip" "$url"
    echo "$sha  $zip" | sha256sum -c --quiet - || { echo "checksum mismatch: $zip" >&2; exit 1; }
    mkdir -p "$dir"
    python3 -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$zip" "$dir"
    git -C "$dir" init -q
    git -C "$dir" add -A
    git -C "$dir" -c user.name=uranium -c user.email=uranium@localhost commit -qm "pristine $(basename "$zip") $sha"
  fi
}

oss_patch() {
  git_apply_patch "$1" "$ROOT/third_party/patches/$2" "[oss] patch"
}

oss_stale() {
  local o="$1" d="${1%.o}.d" dep
  [[ -f "$o" && -f "$d" ]] || return 0
  while IFS= read -r dep; do
    [[ -n "$dep" && ( ! -e "$dep" || "$dep" -nt "$o" ) ]] && return 0
  done < <(sed -e 's/\\$//' -e 's/^[^:]*://' "$d" | tr ' ' '\n')
  return 1
}

oss_flags_stamp() {
  local obj="$1"; shift
  local stamp="$obj/.flags" want
  want="$(printf '%s\n' "$@")"
  mkdir -p "$obj"
  if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$want" ]]; then
    rm -f "$obj"/*.o "$obj"/*.d
    printf '%s\n' "$want" > "$stamp"
  fi
}

oss_compile() {
  local obj="$1"; shift
  local flags=(-DNDEBUG)
  while [[ "$1" != "--" ]]; do flags+=("$1"); shift; done
  shift
  oss_flags_stamp "$obj" "${flags[@]}"
  local f o todo=()
  for f in "$@"; do
    o="$obj/$(echo "${f#$ROOT/third_party/}" | tr '/ ' '__').o"
    oss_stale "$o" && todo+=("$f"$'\t'"$o")
    echo "$o"
  done
  if ((${#todo[@]})); then
    printf '%s\n' "${todo[@]}" | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r f o <<< "$1"; shift; rm -f "$o"; g++ "$@" -MMD -c "$f" -o "$o" || { echo "FAILED: $f" >&2; exit 255; }' _ {} "${flags[@]}" >&2
  fi
}

oss_link() {
  local out="$1"; shift
  local objs=() extra=()
  while (($#)) && [[ "$1" != "--" ]]; do objs+=("$1"); shift; done
  (($#)) && { shift; extra=("$@"); }
  local o
  for o in "${objs[@]}"; do [[ -f "$o" ]] || { echo "compile failed: $o missing" >&2; exit 1; }; done
  mkdir -p "$(dirname "$out")"
  g++ -shared -o "$out" "${objs[@]}" "${extra[@]}" -L"$BWAPI_LIB" -lBWAPILIB -Wl,-rpath,"$BWAPI_LIB" -pthread
}

oss_finish() {
  echo "$1@$2" > "$OSS_OUT_ROOT/$1/VERSION"
  echo "[oss] built $OSS_OUT_ROOT/$1/module.so ($1@$2)"
}
