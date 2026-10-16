# Build Iron (AIIDE 2017 version) as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent
# (spec 021). Windows counterpart of build_iron.sh.
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\iron"

Get-OssZip $Src 'https://davechurchill.ca/starcraft/aiide/results/2019/bots/Iron.zip' 'd9a3eefe04c627550d57af611c17828971fba1667da402ccf708523b6b892f33'
Add-OssPatch $Src "$Root\third_party\patches\0010-iron-linux-bwapi42.patch"

$S = "$Src\Iron\src\Iron"
# behavior\wall.cpp is not part of the original VS project.
$srcs = Get-ChildItem $S -Recurse -Filter *.cpp |
    Where-Object { $_.Name -ne 'dllmain.cpp' -and $_.FullName -notlike '*\behavior\wall.cpp' -and $_.FullName -notlike '*\EasyBMP_1.06\*' } |
    ForEach-Object { $_.FullName } | Sort-Object
Build-OssModule -Bot iron -Sources $srcs -Includes $S, "$S\bwem\src" -Defines NOMINMAX `
    -ForceIncludes climits, cfloat, cmath, stdexcept, algorithm -UraniumModule
New-OssData iron
Complete-Oss iron aiide2017
