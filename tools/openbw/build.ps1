$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\build_common.ps1"
$TP = Join-Path $Root 'third_party'
$Generator = if ($env:URANIUM_CMAKE_GENERATOR) { $env:URANIUM_CMAKE_GENERATOR } else { 'Visual Studio 18 2026' }

Get-PinnedGit "$TP\openbw" (Get-Pin openbw_url) (Get-Pin openbw_commit)
Get-PinnedGit "$TP\bwapi" (Get-Pin bwapi_url) (Get-Pin bwapi_commit)
Get-ChildItem "$TP\patches\0*-openbw-*.patch" | Sort-Object Name | ForEach-Object { Add-GitPatch "$TP\openbw" $_.FullName '[build]' }
Get-ChildItem "$TP\patches\0*-bwapi-*.patch" | Sort-Object Name | ForEach-Object { Add-GitPatch "$TP\bwapi" $_.FullName '[build]' }

$OpenbwDir = "$TP\openbw" -replace '\\', '/'
Invoke-Native cmake @('-S', "$TP\bwapi", '-B', "$TP\bwapi\build", '-G', $Generator, '-A', 'x64',
    "-DOPENBW_DIR=$OpenbwDir", '-DOPENBW_ENABLE_UI=0', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
    '-DCMAKE_CXX_FLAGS=/FIcstdint /EHsc')
Invoke-Native cmake @('--build', "$TP\bwapi\build", '--config', 'Release', '--parallel')

Write-Host '[build] done:'
Get-ChildItem "$TP\bwapi\build" -Recurse -Include BWAPILauncher.exe, *.dll, BWAPILIB.lib |
    ForEach-Object { Write-Host "  $($_.FullName)" }
