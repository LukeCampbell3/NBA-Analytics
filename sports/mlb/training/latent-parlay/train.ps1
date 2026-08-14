param(
    [string]$BeforeDate = "2026-08-06",
    [string]$Image = "nba-analytics/mlb-latent-parlay:cuda12.8"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$dockerfile = Join-Path $PSScriptRoot "Dockerfile"

docker build --file $dockerfile --tag $Image $root
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

docker run --rm --gpus all `
    --volume "${root}:/workspace" `
    $Image `
    --processed-root /workspace/Player-Predictor/Data-Proc-MLB `
    --before-date $BeforeDate `
    --output-json /workspace/sports/mlb/data/predictions/calibration/latent_parlay_model_2026.json `
    --report-json /workspace/sports/mlb/data/predictions/backtests/latent_parlay_model_2026_report.json
if ($LASTEXITCODE -ne 0) { throw "GPU training failed" }
