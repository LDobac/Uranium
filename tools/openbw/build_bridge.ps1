$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$Generator = if ($env:URANIUM_CMAKE_GENERATOR) { $env:URANIUM_CMAKE_GENERATOR } else { 'Visual Studio 18 2026' }

function Invoke-Native([string]$Exe, [string[]]$Argv) {
    $ErrorActionPreference = 'Continue'
    & $Exe @Argv 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Argv -join ' ') failed ($LASTEXITCODE)" }
}

if (-not (Test-Path "$Root\third_party\bwapi\build\lib\Release\BWAPILIB.lib")) { throw 'run tools\openbw\build.ps1 first' }
Invoke-Native cargo @('build', '--release', '-p', 'uranium-bridge-openbw', '--manifest-path', "$Root\Cargo.toml")
$Staticlib = "$Root\target\release\uranium_bridge_openbw.lib" -replace '\\', '/'
Invoke-Native cmake @('-S', "$Root\bridge\openbw", '-B', "$Root\bridge\openbw\build", '-G', $Generator, '-A', 'x64',
    "-DRUST_STATICLIB=$Staticlib")
Invoke-Native cmake @('--build', "$Root\bridge\openbw\build", '--config', 'Release', '--parallel')
Write-Host "[bridge] $Root\bridge\openbw\build\Release\uranium_bridge.dll"
