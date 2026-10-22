# Build the SC:R bridge DLL (spec 012) and install it in the samase mod folder.
# bridge/scr is its own Cargo workspace (x86_64-pc-windows-msvc). Mod folder: URANIUM_SCR_MOD_DIR,
# default <repo>\runs\scr\mod (git-ignored). See docs/scr.md.
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\env.ps1"
Import-UraniumEnv $Root

function Invoke-Native([string]$Exe, [string[]]$Argv) {
    # PowerShell 5.1 turns redirected native stderr into errors; judge by exit code only.
    $ErrorActionPreference = 'Continue'
    & $Exe @Argv 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Argv -join ' ') failed ($LASTEXITCODE)" }
}

Invoke-Native cargo @('test', '--manifest-path', "$Root\bridge\scr\Cargo.toml")
Invoke-Native cargo @('build', '--release', '--manifest-path', "$Root\bridge\scr\Cargo.toml")
$Mod = if ($env:URANIUM_SCR_MOD_DIR) { $env:URANIUM_SCR_MOD_DIR } else { "$Root\runs\scr\mod" }
New-Item -ItemType Directory -Force "$Mod\samase" | Out-Null
Copy-Item "$Root\bridge\scr\target\release\uranium_bridge_scr.dll" "$Mod\samase\uranium_bridge_scr.dll" -Force
# samase loads the plugins listed in <mod>\samase\plugins, one path per line.
[System.IO.File]::WriteAllText("$Mod\samase\plugins", "samase/uranium_bridge_scr.dll`n")
Write-Host "[scr] $Mod\samase\uranium_bridge_scr.dll"
