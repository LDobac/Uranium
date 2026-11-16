$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\env.ps1"
. "$Root\tools\build_common.ps1"
Import-UraniumEnv $Root

Invoke-Native cargo @('test', '--manifest-path', "$Root\bridge\scr\Cargo.toml")
Invoke-Native cargo @('build', '--release', '--manifest-path', "$Root\bridge\scr\Cargo.toml")
$Mod = if ($env:URANIUM_SCR_MOD_DIR) { $env:URANIUM_SCR_MOD_DIR } else { "$Root\runs\scr\mod" }
New-Item -ItemType Directory -Force "$Mod\samase" | Out-Null
Copy-Item "$Root\bridge\scr\target\release\uranium_bridge_scr.dll" "$Mod\samase\uranium_bridge_scr.dll" -Force
[System.IO.File]::WriteAllText("$Mod\samase\plugins", "samase/uranium_bridge_scr.dll`n")
Write-Host "[scr] $Mod\samase\uranium_bridge_scr.dll"
