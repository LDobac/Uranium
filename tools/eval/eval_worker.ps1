# Evaluation worker of the Windows CUDA host (spec 020-R15): this PC's hostd and the eval worker
# together. The worker starts hostd in its own console window, out of reach of Ctrl+C here.
# Ctrl+C in this window: the worker stops handing out games, marks this PC's hostd as leaving (no
# new game goes there), lets the lease go (a Linux ROCm worker takes the next games), finishes the
# games in flight and then stops hostd. Ctrl+C again stops the games in flight now (they are
# played again later). Run it again to rejoin.
#   tools\eval\eval_worker.ps1 -Job eval/<job> -LocalHostd <this PC's address in the job's hosts file>
# Settings come from .env (001-R9): URANIUM_SHARED_DIR, URANIUM_AGENT_LISTEN / _PUBLIC_ADDR,
# URANIUM_HOSTD_LISTEN, URANIUM_PYTHON (else .venv). See docs/commands.md.
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
