function Invoke-Native([string]$Exe, [string[]]$Argv) {
    $ErrorActionPreference = 'Continue'
    & $Exe @Argv 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Argv -join ' ') failed ($LASTEXITCODE)" }
}

function Get-Pin([string]$Key) {
    foreach ($line in Get-Content "$Root\tools\pins.txt") {
        $k, $v = $line.Trim() -split '\s+', 2
        if ($k -eq $Key -and $v) { return $v }
    }
    throw "no pin $Key in tools\pins.txt"
}

function Get-PinnedGit([string]$Dir, [string]$Url, [string]$Commit) {
    if (-not (Test-Path "$Dir\.git")) {
        Invoke-Native git @('-c', 'core.autocrlf=false', 'clone', '-q', $Url, $Dir)
        Invoke-Native git @('-C', $Dir, 'config', 'core.autocrlf', 'false')
    }
    if ((git -C $Dir rev-parse HEAD) -ne $Commit) {
        Invoke-Native git @('-C', $Dir, 'fetch', '-q', 'origin', $Commit)
        Invoke-Native git @('-C', $Dir, 'checkout', '-q', $Commit)
    }
}

function Add-GitPatch([string]$Dir, [string]$Patch, [string]$Tag) {
    $ErrorActionPreference = 'Continue'
    git -C $Dir apply --reverse --check $Patch 2>$null
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$Tag already applied: $(Split-Path $Patch -Leaf)"
    } else {
        Invoke-Native git @('-C', $Dir, 'apply', $Patch)
        Write-Host "$Tag applied: $(Split-Path $Patch -Leaf)"
    }
}
