$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\zzzkbot"
$Commit = Get-Pin zzzkbot_commit

Get-PinnedGit $Src (Get-Pin zzzkbot_url) $Commit
Add-OssPatch $Src (Get-Pin zzzkbot_patch)

$S = "$Src\ZZZKBot\Source"
Build-OssModule -Bot zzzkbot -Sources "$S\ZZZKBotAIModule.cpp", "$S\linux\UraniumModule.cpp" -Includes $S -Defines NOMINMAX -UraniumModule
New-OssData zzzkbot
Complete-Oss zzzkbot ($Commit.Substring(0, 7))
