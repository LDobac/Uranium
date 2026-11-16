$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\mcrave"
$Commit = '7d1719a22d8b896f957abae50e2ea5efff974fe2'

Get-OssGit $Src 'https://github.com/Cmccrave/McRave.git' $Commit
Add-OssPatch $Src "$Root\third_party\patches\0009-mcrave-linux-bwapi42.patch"

$S = "$Src\Source"
$srcs = Get-ChildItem $S -Recurse -Filter *.cpp | Where-Object { $_.Name -ne 'Dll.cpp' } | ForEach-Object { $_.FullName } | Sort-Object
Build-OssModule -Bot mcrave -Sources $srcs -Includes "$S\BWEB", "$S\BWEM", "$S\Horizon", "$S\McRave" -Defines NOMINMAX -Std 'c++20' `
    -ForceIncludes climits, cfloat, cmath, stdexcept -UraniumModule
New-OssData mcrave
Complete-Oss mcrave ($Commit.Substring(0, 7))
