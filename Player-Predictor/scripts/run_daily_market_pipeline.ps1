$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logsDir = Join-Path $repoRoot "logs\scheduled"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir ("daily_market_pipeline_" + $stamp + ".log")

$pythonCmd = Get-Command python -ErrorAction Stop
$pythonExe = $pythonCmd.Source

Write-Host ("Running daily market pipeline with log: " + $logPath)

& $pythonExe "scripts\run_daily_market_pipeline.py" @args 2>&1 | Tee-Object -FilePath $logPath

if ($LASTEXITCODE -ne 0) {
    throw "Daily market pipeline failed with exit code $LASTEXITCODE"
}
