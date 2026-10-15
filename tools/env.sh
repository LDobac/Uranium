# Source this file, then call `uranium_load_env <repo root>` (spec 001-R9).
# Loads <repo root>/.env (or $URANIUM_ENV_FILE) and exports each KEY=VALUE.
# Variables already set in the process environment win over .env values.
# Supported syntax: `KEY=value`, optional `export ` prefix, one level of matching quotes,
# whole-line `#` comments. No `~`/`$VAR` expansion (same rule as the Rust loader).
uranium_load_env() {
  local file="${URANIUM_ENV_FILE:-$1/.env}" line key val
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    line="${line#export }"
    if [[ ! "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      echo "env.sh: skipping malformed line in $file" >&2
      continue
    fi
    key="${BASH_REMATCH[1]}"; val="${BASH_REMATCH[2]}"
    if [[ "$val" =~ ^\"(.*)\"$ || "$val" =~ ^\'(.*)\'$ ]]; then val="${BASH_REMATCH[1]}"; fi
    [[ -n "${!key+x}" ]] && continue
    export "$key=$val"
  done < "$file"
}
