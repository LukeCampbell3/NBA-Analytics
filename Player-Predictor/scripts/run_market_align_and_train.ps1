Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host $Label
    Write-Host ("=" * 80)

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label (exit code $LASTEXITCODE)"
    }
}

Push-Location $repoRoot
try {
    $seasonFiles = Get-ChildItem -Path (Join-Path $repoRoot "Data-Proc") -Directory |
        ForEach-Object { Get-ChildItem -Path $_.FullName -Filter "*_processed_processed.csv" -File } |
        ForEach-Object {
            if ($_.BaseName -match '^(\d{4})_processed_processed$') {
                [int]$matches[1]
            }
        } |
        Sort-Object -Unique

    if (-not $seasonFiles) {
        throw "No processed season CSVs found under Data-Proc."
    }

    foreach ($season in $seasonFiles) {
        Invoke-Step "Align Historical Market Lines - Season $season" {
            python scripts\align_historical_market_lines.py --season $season --skip-market-anchor
        }
    }

    if ($seasonFiles -contains 2026) {
        Invoke-Step "Align Historical Market Lines - Season 2026 (Final Refresh)" {
            python scripts\align_historical_market_lines.py --season 2026 --skip-market-anchor
        }
    }

    Invoke-Step "Train Improved LSTM In Docker" {
        docker run --rm --gpus all -v "${repoRoot}:/workspace" -w /workspace qdi-rust-catboost python train.py --mode improved_lstm
    }

    Write-Host ""
    Write-Host ("=" * 80)
    Write-Host "MARKET ALIGNMENT AND TRAINING COMPLETE"
    Write-Host ("=" * 80)
}
finally {
    Pop-Location
}
