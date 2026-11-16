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
