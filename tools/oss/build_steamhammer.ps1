# Build Steamhammer as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Windows counterpart of build_steamhammer.sh.
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Ver = '5.3.6'
$Src = "$Root\third_party\steamhammer"
$Cfg = "$Root\third_party\steamhammer_dll"
$Url = "https://satirist.org/downloads/starcraft/steamhammer/$Ver"

Get-OssZip $Src "$Url/Steamhammer_source.zip" 'd7ccc9f54cb8a9b1d11befc136272646767a7fd494dd937924a82e03581ad522'
Get-OssZip $Cfg "$Url/Steamhammer_dll.zip" 'afde6ff7a697abae5364c3e6c623aa579524f05e3015f662e4ef7bce379cc83c'
Add-OssPatch $Src "$Root\third_party\patches\0006-steamhammer-linux-bwapi42.patch"

$S = "$Src\Steamhammer_source\Steamhammer\Source"
$srcs = Get-ChildItem "$S\*.cpp", "$S\linux\*.cpp", "$Src\Steamhammer_source\RC\Source\*.cpp" |
    Where-Object { 'Dll.cpp', 'OpsBoss with immobile.cpp' -notcontains $_.Name } | ForEach-Object { $_.FullName }
Build-OssModule -Bot steamhammer -Sources $srcs -Includes $S -Defines NOMINMAX -Std 'c++20' -ForceIncludes climits, cfloat, cmath -UraniumModule
New-OssData steamhammer
Copy-Item "$Cfg\Steamhammer_dll\Steamhammer_$Ver.json" "$OssOutRoot\steamhammer\bwapi-data\AI\" -Force
Complete-Oss steamhammer $Ver
