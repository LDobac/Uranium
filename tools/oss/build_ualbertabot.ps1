# Build UAlbertaBot as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Windows counterpart of build_ualbertabot.sh.
# Output: <URANIUM_OSS_BOTS_DIR>\ualbertabot\{module.dll, VERSION, bwapi-data\AI\UAlbertaBot_Config.txt}
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\ualbertabot"
$Commit = '558899d8793456f4a6ec4196efbb5235552e24db'

Get-OssGit $Src 'https://github.com/davechurchill/ualbertabot.git' $Commit
Add-OssPatch $Src "$Root\third_party\patches\0005-ualbertabot-linux-bwapi42.patch"

$skip = 'main.cpp', 'BOSS_main.cpp', 'StarCraftGUI.cpp', 'TutorialCode.cpp', 'UnitInfoManager.cpp'
$srcs = Get-ChildItem "$Src\UAlbertaBot\Source\*.cpp", "$Src\UAlbertaBot\Source\linux\*.cpp", "$Src\BOSS\source\*.cpp", "$Src\SparCraft\source\*.cpp" |
    Where-Object { $skip -notcontains $_.Name } | ForEach-Object { $_.FullName }
# linux\UraniumModule.cpp is the module entry on Windows too: it runs SparCraft/BOSS init.
Build-OssModule -Bot ualbertabot -Sources $srcs -Includes "$Src\UAlbertaBot\Source" -Defines URANIUM_BOSS_BWAPI, NOMINMAX -UraniumModule
New-OssData ualbertabot @('AI')
Copy-Item "$Src\UAlbertaBot\bin\UAlbertaBot_Config.txt" "$OssOutRoot\ualbertabot\bwapi-data\AI\" -Force
Complete-Oss ualbertabot ($Commit.Substring(0, 7))
