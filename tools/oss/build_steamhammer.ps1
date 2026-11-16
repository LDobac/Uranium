$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Ver = Get-Pin steamhammer_version
$Src = "$Root\third_party\steamhammer"
$Cfg = "$Root\third_party\steamhammer_dll"
$Url = "$(Get-Pin steamhammer_url)/$Ver"

Get-OssZip $Src "$Url/Steamhammer_source.zip" (Get-Pin steamhammer_source_sha256)
Get-OssZip $Cfg "$Url/Steamhammer_dll.zip" (Get-Pin steamhammer_dll_sha256)
Add-OssPatch $Src (Get-Pin steamhammer_patch)

$S = "$Src\Steamhammer_source\Steamhammer\Source"
$srcs = Get-ChildItem "$S\*.cpp", "$S\linux\*.cpp", "$Src\Steamhammer_source\RC\Source\*.cpp" |
    Where-Object { 'Dll.cpp', 'OpsBoss with immobile.cpp' -notcontains $_.Name } | ForEach-Object { $_.FullName }
Build-OssModule -Bot steamhammer -Sources $srcs -Includes $S -Defines NOMINMAX -Std 'c++20' -ForceIncludes climits, cfloat, cmath -UraniumModule
New-OssData steamhammer
Copy-Item "$Cfg\Steamhammer_dll\Steamhammer_$Ver.json" "$OssOutRoot\steamhammer\bwapi-data\AI\" -Force
Complete-Oss steamhammer $Ver
