$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\ualbertabot"
$Commit = Get-Pin ualbertabot_commit

Get-PinnedGit $Src (Get-Pin ualbertabot_url) $Commit
Add-OssPatch $Src (Get-Pin ualbertabot_patch)

$skip = 'main.cpp', 'BOSS_main.cpp', 'StarCraftGUI.cpp', 'TutorialCode.cpp', 'UnitInfoManager.cpp'
$srcs = Get-ChildItem "$Src\UAlbertaBot\Source\*.cpp", "$Src\UAlbertaBot\Source\linux\*.cpp", "$Src\BOSS\source\*.cpp", "$Src\SparCraft\source\*.cpp" |
    Where-Object { $skip -notcontains $_.Name } | ForEach-Object { $_.FullName }
Build-OssModule -Bot ualbertabot -Sources $srcs -Includes "$Src\UAlbertaBot\Source" -Defines URANIUM_BOSS_BWAPI, NOMINMAX -UraniumModule
New-OssData ualbertabot @('AI')
Copy-Item "$Src\UAlbertaBot\bin\UAlbertaBot_Config.txt" "$OssOutRoot\ualbertabot\bwapi-data\AI\" -Force
Complete-Oss ualbertabot ($Commit.Substring(0, 7))
