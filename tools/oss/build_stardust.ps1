$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\oss\common.ps1"
$Src = "$Root\third_party\stardust"
$Commit = Get-Pin stardust_commit

Get-PinnedGit $Src (Get-Pin stardust_url) $Commit
Add-OssPatch $Src (Get-Pin stardust_patch)

$Z = "$Src\3rdparty\zstd\lib"
$zsrcs = Get-ChildItem "$Z\common\*.c", "$Z\compress\*.c", "$Z\decompress\*.c" | ForEach-Object { $_.FullName }
$inc = @()
foreach ($d in '', 'Builder', 'General', 'Instrumentation', 'Map', 'Map\PathFinding', 'Players', 'Producer', 'Strategist', 'Units', 'Util', 'Workers') {
    $inc += "$Src\src\$d".TrimEnd('\')
}
$inc += "$Src\3rdparty", "$Src\3rdparty\BWEM\include", "$Src\3rdparty\FAP", "$Src\3rdparty\bitsery\include",
    "$Src\3rdparty\zstdstream", $Z, "$Src\3rdparty\nlohmann"
$srcs = Get-ChildItem "$Src\src", "$Src\3rdparty\BWEM\src", "$Src\3rdparty\zstdstream" -Recurse -Filter *.cpp |
    ForEach-Object { $_.FullName } | Sort-Object
Build-OssModule -Bot stardust -Sources $srcs -Includes $inc -Defines NOMINMAX -Std 'c++20' -ForceIncludes cstdint, climits, memory `
    -Prelude "#include <BaseTsd.h>`r`ntypedef SSIZE_T ssize_t;" `
    -CSources $zsrcs -CIncludes $Z, "$Z\common" -CDefines ZSTD_DISABLE_ASM
New-OssData stardust
Complete-Oss stardust ($Commit.Substring(0, 7))
