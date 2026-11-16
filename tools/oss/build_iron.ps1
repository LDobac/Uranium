$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\iron"

Get-OssZip $Src (Get-Pin iron_url) (Get-Pin iron_sha256)
Add-OssPatch $Src (Get-Pin iron_patch)

$S = "$Src\Iron\src\Iron"
$srcs = Get-ChildItem $S -Recurse -Filter *.cpp |
    Where-Object { $_.Name -ne 'dllmain.cpp' -and $_.FullName -notlike '*\behavior\wall.cpp' -and $_.FullName -notlike '*\EasyBMP_1.06\*' } |
    ForEach-Object { $_.FullName } | Sort-Object
Build-OssModule -Bot iron -Sources $srcs -Includes $S, "$S\bwem\src" -Defines NOMINMAX `
    -ForceIncludes climits, cfloat, cmath, stdexcept, algorithm -UraniumModule
New-OssData iron
Complete-Oss iron (Get-Pin iron_version)
