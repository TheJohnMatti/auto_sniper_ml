<#
.SYNOPSIS
    Run src.scraper.fetch_descriptions in spaced batches until the queue is drained.

.DESCRIPTION
    fetch_descriptions is already resumable (it appends to data/raw/descriptions.csv
    and skips listings it has). This just loops it with jittered sleeps so a full
    backfill can run unattended overnight without hammering Facebook.

.EXAMPLE
    # Fresh listings only (recommended first pass), ~8 batches:
    pwsh scripts/backfill_descriptions.ps1 -Since 20260828

    # Everything, larger batches, also score when done:
    pwsh scripts/backfill_descriptions.ps1 -BatchSize 200 -RunSentiment

.NOTES
    Requires the machine awake and online for the duration. Stop with Ctrl+C at
    any time; re-run to continue.
#>
param(
    [int]    $BatchSize   = 150,
    [string] $Since       = "",          # YYYYMMDD; "" = all listings
    [int]    $Concurrency = 3,            # pages in flight per batch; keep low (2-3)
    [int]    $MaxRuns     = 30,
    [int]    $MinSleepMin = 12,
    [int]    $MaxSleepMin = 22,
    [switch] $RunSentiment
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$env:PYTHONPATH = $repo
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$py = $null
if (Get-Command poetry -ErrorAction SilentlyContinue) {
    try { $envPath = (& poetry env info -p 2>$null); if ($envPath) { $py = Join-Path $envPath "Scripts\python.exe" } } catch {}
}
if (-not $py -or -not (Test-Path $py)) {
    $vdir = "$env:LOCALAPPDATA\pypoetry\Cache\virtualenvs"
    $py = Get-ChildItem $vdir -Filter "auto-sniper-ml-*" -Directory -ErrorAction SilentlyContinue |
          ForEach-Object { Join-Path $_.FullName "Scripts\python.exe" } |
          Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $py -or -not (Test-Path $py)) { throw "Could not locate the project venv python. Set `$py in this script." }

$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("backfill_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "[backfill] python : $py"
Write-Host "[backfill] log    : $log"

$moduleArgs = @("-u", "-m", "src.scraper.fetch_descriptions", "$BatchSize", "--concurrency=$Concurrency")
if ($Since) { $moduleArgs += "--since=$Since" }

$lastPending = [int]::MaxValue
for ($run = 1; $run -le $MaxRuns; $run++) {
    $stamp = Get-Date -Format "HH:mm:ss"
    Write-Host "`n[backfill] === run $run/$MaxRuns  ($stamp) ==="
    "`n=== run $run  $(Get-Date -Format o) ===" | Add-Content $log

    $out = & $py @moduleArgs 2>&1
    $out | Tee-Object -FilePath $log -Append | Out-Host

    $joined  = ($out -join "`n")
    $pending = if ($joined -match "(\d+)\s+listings still pending") { [int]$Matches[1] } else { $null }

    if ($joined -match "Nothing to do" -or $pending -eq 0) {
        Write-Host "[backfill] queue drained. Done."
        break
    }
    if ($null -ne $pending -and $pending -ge $lastPending) {
        Write-Host "[backfill] no progress this run ($pending pending) - stopping so it can be looked at."
        break
    }
    if ($null -ne $pending) {
        $lastPending = $pending
        Write-Host "[backfill] $pending still pending."
    }

    if ($run -lt $MaxRuns) {
        $sleepSec = Get-Random -Minimum ($MinSleepMin * 60) -Maximum ($MaxSleepMin * 60)
        Write-Host ("[backfill] sleeping {0:n1} min..." -f ($sleepSec / 60))
        Start-Sleep -Seconds $sleepSec
    }
}

if ($RunSentiment) {
    Write-Host "`n[backfill] scoring descriptions..."
    & $py -X utf8 -m src.ml.sentiment  2>&1 | Tee-Object -FilePath $log -Append | Out-Host
    & $py -X utf8 -m src.ml.valuation  2>&1 | Tee-Object -FilePath $log -Append | Out-Host
}

Write-Host "`n[backfill] finished. Full log: $log"
