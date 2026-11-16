param(
    [Parameter(Mandatory = $true)][string]$Job,
    [Parameter(Mandatory = $true)][string]$LocalHostd,
    [int]$Priority = 10,
    [string]$Backend = 'cuda'
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path "$PSScriptRoot\..\..").Path
. "$Root\tools\env.ps1"
Import-UraniumEnv $Root

$Python = if ($env:URANIUM_PYTHON) { $env:URANIUM_PYTHON } else { "$Root\.venv\Scripts\python.exe" }
$Hostd = "$Root\target\release\uranium-hostd.exe"
if (-not (Test-Path $Hostd)) { throw "no $Hostd`: cargo build --release -p uranium-hostd" }
Set-Location $Root
& $Python -m uranium eval-worker --job $Job --priority $Priority --backend $Backend --local-hostd $LocalHostd --hostd $Hostd --log "$Root\runs\eval-worker.log"
Write-Host "[eval] worker exited ($LASTEXITCODE)"
