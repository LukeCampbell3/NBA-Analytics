param(
    [string]$TaskName = "PlayerPredictorDailyMarketPipeline",
    [string]$Time = "02:00",
    [switch]$UseLatest
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runnerPath = Join-Path $PSScriptRoot "run_daily_market_pipeline.ps1"
if (-not (Test-Path $runnerPath)) {
    throw "Runner script not found: $runnerPath"
}

$args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", ('"' + $runnerPath + '"')
)
if ($UseLatest) {
    $args += "--latest"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($args -join " ")
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Runs the Player Predictor daily market pipeline at 2am." `
    -Force | Out-Null

Write-Host "Scheduled task installed."
Write-Host ("Task name: " + $TaskName)
Write-Host ("Time:      " + $Time)
Write-Host ("Runner:    " + $runnerPath)
