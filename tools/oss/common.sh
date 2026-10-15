# Shared helpers for tools/oss/build_<bot>.sh (spec 021 T1). Source after setting ROOT.
# Loads .env, defines the BWAPI 4.2 (OpenBW fork) paths and the output layout:
#   <URANIUM_OSS_BOTS_DIR>/<bot>/{module.so, VERSION, bwapi-data/...}
source "$ROOT/tools/env.sh"; uranium_load_env "$ROOT"
BWAPI_INC="$ROOT/third_party/bwapi/bwapi/include"
BWAPI_LIB="$ROOT/third_party/bwapi/build/lib"
OSS_OUT_ROOT="${URANIUM_OSS_BOTS_DIR:-$ROOT/build/oss_bots}"
JOBS="${JOBS:-$(nproc)}"
[[ -f "$BWAPI_LIB/libBWAPILIB.so" ]] || { echo "run tools/openbw/build.sh first" >&2; exit 1; }

# oss_git_fetch <dir> <url> <commit>: clone once, then pin to the commit (021-R2).
oss_git_fetch() {
  local dir="$1" url="$2" commit="$3"
  [[ -d "$dir/.git" ]] || git clone -q "$url" "$dir"
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$dir" fetch -q origin "$commit"
    git -C "$dir" checkout -q "$commit"
  fi
}

# oss_zip_fetch <dir> <url> <sha256>: download a source archive once, verify, unpack into dir.
# The unpacked tree is turned into a local git repo so patches apply like for git sources.
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

# oss_patch <dir> <patch>: apply once (idempotent).
oss_patch() {
  local dir="$1" patch="$2"
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "[oss] patch already applied: $(basename "$patch")"
  else
    git -C "$dir" apply "$patch"
    echo "[oss] patch applied: $(basename "$patch")"
  fi
}

# oss_compile <obj dir> <cxx flags...> -- <sources...>: parallel incremental g++ -c.
# Prints the object list (one per line) on stdout.
oss_compile() {
  local obj="$1"; shift
  local flags=()
  while [[ "$1" != "--" ]]; do flags+=("$1"); shift; done
  shift
  mkdir -p "$obj"
  local f o todo=()
  for f in "$@"; do
    o="$obj/$(echo "${f#$ROOT/third_party/}" | tr '/ ' '__').o"
    [[ -f "$o" && "$o" -nt "$f" ]] || todo+=("$f"$'\t'"$o")
    echo "$o"
  done
  if ((${#todo[@]})); then
    printf '%s\n' "${todo[@]}" | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r f o <<< "$1"; shift; g++ "$@" -c "$f" -o "$o" || { echo "FAILED: $f" >&2; exit 255; }' _ {} "${flags[@]}" >&2
  fi
}

# oss_link <module.so> <objs...> [-- extra link args]: fails if any object is missing
# (oss_compile runs inside a process substitution, so its exit status is lost).
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

# oss_finish <bot> <version>: write the opponent ID file (021-R3).
oss_finish() {
  echo "$1@$2" > "$OSS_OUT_ROOT/$1/VERSION"
  echo "[oss] built $OSS_OUT_ROOT/$1/module.so ($1@$2)"
}
