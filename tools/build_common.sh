pin() {
  local v
  v="$(awk -v k="$1" '$1 == k { sub(/\r$/, "", $2); print $2; exit }' "$ROOT/tools/pins.txt")"
  [[ -n "$v" ]] || { echo "no pin $1 in tools/pins.txt" >&2; return 1; }
  printf '%s\n' "$v"
}

git_fetch_pinned() {
  local dir="$1" url="$2" commit="$3"
  [[ -d "$dir/.git" ]] || git clone -q "$url" "$dir"
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$dir" fetch -q origin "$commit"
    git -C "$dir" checkout -q "$commit"
  fi
}

git_apply_patch() {
  local dir="$1" patch="$2" tag="$3"
  if git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
    echo "$tag already applied: $(basename "$patch")"
  else
    git -C "$dir" apply "$patch"
    echo "$tag applied: $(basename "$patch")"
  fi
}
