# Shared helpers for tools\oss\build_<bot>.ps1 (spec 021 T1, Windows counterpart of common.sh).
# Dot-source after setting $Root. Output layout matches Linux except for the module name:
#   <URANIUM_OSS_BOTS_DIR>\<bot>\{module.dll, VERSION, bwapi-data\...}
# Each bot is built by a generated CMake project (MSVC, /MP) under build\obj\<bot>.
$ErrorActionPreference = 'Stop'
. "$Root\tools\env.ps1"
Import-UraniumEnv $Root
$BwapiInc = "$Root\third_party\bwapi\bwapi\include"
$BwapiLib = "$Root\third_party\bwapi\build\lib\Release\BWAPILIB.lib"
$OssOutRoot = if ($env:URANIUM_OSS_BOTS_DIR) { $env:URANIUM_OSS_BOTS_DIR } else { "$Root\build\oss_bots" }
$Generator = if ($env:URANIUM_CMAKE_GENERATOR) { $env:URANIUM_CMAKE_GENERATOR } else { 'Visual Studio 18 2026' }
if (-not (Test-Path $BwapiLib)) { throw 'run tools\openbw\build.ps1 first' }

function Invoke-Native([string]$Exe, [string[]]$Argv) {
    # PowerShell 5.1 turns redirected native stderr into errors; judge by exit code only.
    $ErrorActionPreference = 'Continue'
    & $Exe @Argv 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Argv -join ' ') failed ($LASTEXITCODE)" }
}

# Clone once, then pin to the commit (021-R2). LF checkouts so the patches apply.
function Get-OssGit([string]$Dir, [string]$Url, [string]$Commit) {
    if (-not (Test-Path "$Dir\.git")) {
        Invoke-Native git @('-c', 'core.autocrlf=false', 'clone', '-q', $Url, $Dir)
        Invoke-Native git @('-C', $Dir, 'config', 'core.autocrlf', 'false')
    }
    if ((git -C $Dir rev-parse HEAD) -ne $Commit) {
        Invoke-Native git @('-C', $Dir, 'fetch', '-q', 'origin', $Commit)
        Invoke-Native git @('-C', $Dir, 'checkout', '-q', $Commit)
    }
}

# Download a source archive once, verify, unpack into a local git repo (same as common.sh).
function Get-OssZip([string]$Dir, [string]$Url, [string]$Sha256) {
    if (Test-Path "$Dir\.git") { return }
    $zip = "$Root\third_party\$(Split-Path $Dir -Leaf).zip"
    if (-not (Test-Path $zip)) { Invoke-WebRequest -UseBasicParsing $Url -OutFile $zip }
    if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $Sha256) { throw "checksum mismatch: $zip" }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $Dir)
    Invoke-Native git @('-C', $Dir, 'init', '-q')
    Invoke-Native git @('-C', $Dir, 'config', 'core.autocrlf', 'false')
    Invoke-Native git @('-C', $Dir, 'add', '-A')
    Invoke-Native git @('-C', $Dir, '-c', 'user.name=uranium', '-c', 'user.email=uranium@localhost', 'commit', '-qm', "pristine $(Split-Path $zip -Leaf) $Sha256")
}

function Add-OssPatch([string]$Dir, [string]$Patch) {
    $ErrorActionPreference = 'Continue'
    git -C $Dir apply --reverse --check $Patch 2>$null
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[oss] patch already applied: $(Split-Path $Patch -Leaf)"
    } else {
        Invoke-Native git @('-C', $Dir, 'apply', $Patch)
        Write-Host "[oss] patch applied: $(Split-Path $Patch -Leaf)"
    }
}

# Build <OssOutRoot>\<Bot>\module.dll from the given sources.
#  -UraniumModule: sources use the patch's linux\UraniumModule.cpp entry points, which are
#   declared with GCC visibility attributes; they are exported through a .def file instead.
function Build-OssModule {
    param(
        [string]$Bot, [string[]]$Sources, [string[]]$Includes = @(), [string[]]$Defines = @(),
        [string[]]$ForceIncludes = @(), [string]$Std = 'c++17', [switch]$UraniumModule, [string]$Prelude = '',
        [string[]]$CSources = @(), [string[]]$CIncludes = @(), [string[]]$CDefines = @()
    )
    $obj = "$Root\build\obj\$Bot"
    New-Item -ItemType Directory -Force $obj | Out-Null
    $q = { param($s) '"' + ($s -replace '\\', '/') + '"' }
    $fi = @($ForceIncludes)
    # -Prelude: extra declarations (MSVC gaps such as ssize_t) force-included before every C++ file.
    if ($Prelude) {
        Set-Content -Encoding ascii "$obj\prelude.h" "#pragma once`r`n$Prelude"
        $fi = @("$obj\prelude.h") + $fi
    }
    $lines = @('cmake_minimum_required(VERSION 3.16)', "project(oss_$Bot C CXX)",
        "add_library(module SHARED $(($Sources | ForEach-Object { & $q $_ }) -join ' '))")
    if ($UraniumModule) {
        Set-Content -Encoding ascii "$obj\uranium_msvc.h" "#pragma once`r`n// GCC attributes in the patches' entry points; exports come from module.def.`r`n#define __attribute__(x)"
        Set-Content -Encoding ascii "$obj\module.def" "EXPORTS`r`n  gameInit`r`n  newAIModule"
        $fi = @("$obj\uranium_msvc.h") + $fi
        $lines += "target_sources(module PRIVATE $(& $q "$obj\module.def"))"
    }
    # /Zi + /DEBUG: module.pdb next to module.dll for crash analysis (cdb); code is still /O2.
    $opts = @('/MP', '/O2', '/Zi', '/w', '/EHsc', '/bigobj', '/permissive', "/std:$Std") + ($fi | ForEach-Object { "/FI$($_ -replace '\\', '/')" })
    $lines += "target_compile_options(module PRIVATE `$<`$<COMPILE_LANGUAGE:CXX>:$($opts -join ';')>)"
    $lines += 'target_link_options(module PRIVATE /DEBUG /OPT:REF /OPT:ICF)'
    $lines += "target_include_directories(module PRIVATE $((@($BwapiInc) + $Includes | ForEach-Object { & $q $_ }) -join ' '))"
    # _HAS_AUTO_PTR_ETC: MSVC drops std::random_shuffle & co. in C++17; libstdc++ keeps them.
    $lines += "target_compile_definitions(module PRIVATE _CRT_SECURE_NO_WARNINGS _HAS_AUTO_PTR_ETC=1 $($Defines -join ' '))"
    if ($CSources.Count) {
        $lines += "add_library(cdeps STATIC $(($CSources | ForEach-Object { & $q $_ }) -join ' '))"
        $lines += "target_include_directories(cdeps PRIVATE $(($CIncludes | ForEach-Object { & $q $_ }) -join ' '))"
        $lines += "target_compile_definitions(cdeps PRIVATE $($CDefines -join ' '))"
        $lines += 'target_compile_options(cdeps PRIVATE /MP /O2 /w)'
        $lines += 'target_link_libraries(module PRIVATE cdeps)'
    }
    $lines += "target_link_libraries(module PRIVATE $(& $q $BwapiLib))"
    Set-Content -Encoding utf8 "$obj\CMakeLists.txt" ($lines -join "`n")
    Invoke-Native cmake @('-S', $obj, '-B', "$obj\build", '-G', $Generator, '-A', 'x64') | Out-Null
    Invoke-Native cmake @('--build', "$obj\build", '--config', 'Release', '--parallel') |
        Where-Object { $_ -match 'error|module\.vcxproj ->' }
    $out = "$OssOutRoot\$Bot"
    New-Item -ItemType Directory -Force $out | Out-Null
    Copy-Item "$obj\build\Release\module.dll", "$obj\build\Release\module.pdb" $out -Force
}

# Create bwapi-data dirs under the bot output.
function New-OssData([string]$Bot, [string[]]$Dirs = @('AI', 'read', 'write')) {
    foreach ($d in $Dirs) { New-Item -ItemType Directory -Force "$OssOutRoot\$Bot\bwapi-data\$d" | Out-Null }
}

# Write the opponent ID file (021-R3).
function Complete-Oss([string]$Bot, [string]$Version) {
    [IO.File]::WriteAllText("$OssOutRoot\$Bot\VERSION", "$Bot@$Version`n")
    Write-Host "[oss] built $OssOutRoot\$Bot\module.dll ($Bot@$Version)"
}
