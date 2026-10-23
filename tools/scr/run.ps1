# Start SC:R with the Uranium bridge through samase (spec 012). Start the agent first
# (`python -m uranium scr --checkpoint ...`), then this script, then start a game in SC:R by hand.
# Settings come from .env (001-R9): URANIUM_SCR_SAMASE, URANIUM_SCR_MOD_DIR, URANIUM_SCR_AGENT_ADDR,
# URANIUM_SCR_LOG_DIR. See docs/scr.md.
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\env.ps1"
Import-UraniumEnv $Root

# samase 0.10.28-64, the loader spec 012 was checked with (012 Scope).
$SamaseSha256 = '8cc8d84d2c8c533ee048760b13cf49ed369502a821840f6e35ebe537afec668c'
if (-not $env:URANIUM_SCR_SAMASE) { throw 'URANIUM_SCR_SAMASE is not set (path of samase64.exe, docs/scr.md)' }
$hash = (Get-FileHash -LiteralPath $env:URANIUM_SCR_SAMASE -Algorithm SHA256).Hash.ToLower()
if ($hash -ne $SamaseSha256) { throw "samase64.exe has SHA-256 $hash, expected $SamaseSha256 (012)" }

$Mod = if ($env:URANIUM_SCR_MOD_DIR) { $env:URANIUM_SCR_MOD_DIR } else { "$Root\runs\scr\mod" }
if (-not (Test-Path "$Mod\samase\uranium_bridge_scr.dll")) { throw "no bridge DLL in $Mod\samase: run tools\scr\build.ps1" }
if (-not $env:URANIUM_SCR_LOG_DIR) { $env:URANIUM_SCR_LOG_DIR = "$Root\runs\scr\logs" }
New-Item -ItemType Directory -Force $env:URANIUM_SCR_LOG_DIR | Out-Null
# samase writes samase.log into its working folder.
$Work = Split-Path -Parent $Mod
Write-Host "[scr] agent $(if ($env:URANIUM_SCR_AGENT_ADDR) { $env:URANIUM_SCR_AGENT_ADDR } else { '127.0.0.1:7700' }), bridge log $env:URANIUM_SCR_LOG_DIR"
Start-Process -FilePath $env:URANIUM_SCR_SAMASE -ArgumentList "`"$Mod`"" -WorkingDirectory $Work
