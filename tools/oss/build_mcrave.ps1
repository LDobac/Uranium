$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\mcrave"
$Commit = Get-Pin mcrave_commit

Get-PinnedGit $Src (Get-Pin mcrave_url) $Commit
Add-OssPatch $Src (Get-Pin mcrave_patch)

$S = "$Src\Source"
$srcs = Get-ChildItem $S -Recurse -Filter *.cpp | Where-Object { $_.Name -ne 'Dll.cpp' } | ForEach-Object { $_.FullName } | Sort-Object
Build-OssModule -Bot mcrave -Sources $srcs -Includes "$S\BWEB", "$S\BWEM", "$S\Horizon", "$S\McRave" -Defines NOMINMAX -Std 'c++20' `
    -ForceIncludes climits, cfloat, cmath, stdexcept -UraniumModule
New-OssData mcrave
Complete-Oss mcrave ($Commit.Substring(0, 7))
