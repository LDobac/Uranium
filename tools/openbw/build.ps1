# Build OpenBW + BWAPI (OpenBW fork) headless with MSVC for Uranium (spec 011, 001-R8).
# Windows counterpart of build.sh: same pinned commits and patches, Release build.
# Usage: powershell -ExecutionPolicy Bypass -File tools\openbw\build.ps1
$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
$TP = Join-Path $Root 'third_party'
$OpenbwCommit = '4b046d5f65302b10cb0a745f0fecd37ec85b20a8'
$BwapiCommit = '48124ba8ed1b4d52b3dfd52acbaf34afb9a37fe2'
# Override with e.g. "Visual Studio 17 2022" on hosts with an older Visual Studio.
$Generator = if ($env:URANIUM_CMAKE_GENERATOR) { $env:URANIUM_CMAKE_GENERATOR } else { 'Visual Studio 18 2026' }

function Invoke-Native([string]$Exe, [string[]]$Argv) {
    # PowerShell 5.1 turns redirected native stderr into errors; judge by exit code only.
    $ErrorActionPreference = 'Continue'
    & $Exe @Argv 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Argv -join ' ') failed ($LASTEXITCODE)" }
}

function Get-Pinned([string]$Dir, [string]$Url, [string]$Commit) {
    if (-not (Test-Path (Join-Path $Dir '.git'))) {
        # Patches use LF line endings: keep checkouts byte-identical to Linux.
        Invoke-Native git @('-c', 'core.autocrlf=false', 'clone', '-q', $Url, $Dir)
        Invoke-Native git @('-C', $Dir, 'config', 'core.autocrlf', 'false')
    }
    if ((git -C $Dir rev-parse HEAD) -ne $Commit) {
        Invoke-Native git @('-C', $Dir, 'fetch', '-q', 'origin', $Commit)
        Invoke-Native git @('-C', $Dir, 'checkout', '-q', $Commit)
    }
}

function Add-Patch([string]$Dir, [string]$Patch) {
    # PowerShell 5.1 turns native stderr into a terminating error under 'Stop'.
    $ErrorActionPreference = 'Continue'
    git -C $Dir apply --reverse --check $Patch 2>$null
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[build] already applied: $(Split-Path $Patch -Leaf)"
    } else {
        Invoke-Native git @('-C', $Dir, 'apply', $Patch)
        Write-Host "[build] applied: $(Split-Path $Patch -Leaf)"
    }
}

Get-Pinned "$TP\openbw" 'https://github.com/OpenBW/openbw.git' $OpenbwCommit
Get-Pinned "$TP\bwapi" 'https://github.com/OpenBW/bwapi.git' $BwapiCommit
Get-ChildItem "$TP\patches\0*-openbw-*.patch" | Sort-Object Name | ForEach-Object { Add-Patch "$TP\openbw" $_.FullName }
Get-ChildItem "$TP\patches\0*-bwapi-*.patch" | Sort-Object Name | ForEach-Object { Add-Patch "$TP\bwapi" $_.FullName }

$OpenbwDir = "$TP\openbw" -replace '\\', '/'
Invoke-Native cmake @('-S', "$TP\bwapi", '-B', "$TP\bwapi\build", '-G', $Generator, '-A', 'x64',
    "-DOPENBW_DIR=$OpenbwDir", '-DOPENBW_ENABLE_UI=0', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
    '-DCMAKE_CXX_FLAGS=/FIcstdint /EHsc')
Invoke-Native cmake @('--build', "$TP\bwapi\build", '--config', 'Release', '--parallel')

Write-Host '[build] done:'
Get-ChildItem "$TP\bwapi\build" -Recurse -Include BWAPILauncher.exe, *.dll, BWAPILIB.lib |
    ForEach-Object { Write-Host "  $($_.FullName)" }
