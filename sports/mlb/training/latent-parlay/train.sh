#!/usr/bin/env bash
set -euo pipefail

before_date="${1:-2026-08-06}"
image="${MLB_LATENT_IMAGE:-nba-analytics/mlb-latent-parlay:cuda12.8}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

docker build --file "$root/sports/mlb/training/latent-parlay/Dockerfile" --tag "$image" "$root"
docker run --rm --gpus all \
  --volume "$root:/workspace" \
  "$image" \
  --processed-root /workspace/Player-Predictor/Data-Proc-MLB \
  --before-date "$before_date" \
  --output-json /workspace/sports/mlb/data/predictions/calibration/latent_parlay_model_2026.json \
  --report-json /workspace/sports/mlb/data/predictions/backtests/latent_parlay_model_2026_report.json
