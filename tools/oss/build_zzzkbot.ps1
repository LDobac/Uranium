# Build ZZZKBot as a BWAPI 4.2 (OpenBW) AI module for use as a T1 opponent (spec 021).
# Windows counterpart of build_zzzkbot.sh.
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\zzzkbot"
$Commit = '7183e37b6b416ea53c1040c83e639a3a3c395eed'

Get-OssGit $Src 'https://github.com/chriscoxe/zzzkbot.git' $Commit
Add-OssPatch $Src "$Root\third_party\patches\0007-zzzkbot-linux-bwapi42.patch"

$S = "$Src\ZZZKBot\Source"
Build-OssModule -Bot zzzkbot -Sources "$S\ZZZKBotAIModule.cpp", "$S\linux\UraniumModule.cpp" -Includes $S -Defines NOMINMAX -UraniumModule
New-OssData zzzkbot
Complete-Oss zzzkbot ($Commit.Substring(0, 7))
