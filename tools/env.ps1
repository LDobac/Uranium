# Dot-source this file, then call `Import-UraniumEnv <repo root>` (spec 001-R9).
# Loads <repo root>\.env (or $env:URANIUM_ENV_FILE) into the process environment.
# Variables already set in the process environment win over .env values.
# Supported syntax: `KEY=value`, optional `export ` prefix, one level of matching quotes,
# whole-line `#` comments. No `~`/`$VAR` expansion (same rule as tools/env.sh and the Rust loader).
function Import-UraniumEnv([string]$Root) {
    $file = if ($env:URANIUM_ENV_FILE) { $env:URANIUM_ENV_FILE } else { Join-Path $Root '.env' }
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { return }
    foreach ($line in [System.IO.File]::ReadAllLines($file)) {
        if ($line -match '^\s*(#|$)') { continue }
        if ($line.StartsWith('export ')) { $line = $line.Substring(7) }
        if ($line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
            Write-Warning "env.ps1: skipping malformed line in $file"
            continue
        }
        $key = $Matches[1]; $val = $Matches[2]
        if ($val -match '^"(.*)"$' -or $val -match "^'(.*)'$") { $val = $Matches[1] }
        if (Test-Path -LiteralPath "Env:$key") { continue }
        Set-Item -LiteralPath "Env:$key" -Value $val
    }
}
