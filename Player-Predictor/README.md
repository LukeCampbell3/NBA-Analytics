# Player Predictor

NBA player prop prediction pipeline.

## Core Commands

- Build market plays:
  - `python scripts/run_market_pipeline.py --season 2026 --market-wide-path model/analysis/daily_runs/20260326/current_market_snapshot_20260326.parquet --slate-csv-out model/analysis/upcoming_market_slate.csv --selector-csv-out model/analysis/upcoming_market_play_selector.csv --final-csv-out model/analysis/final_market_plays.csv --final-json-out model/analysis/final_market_plays.json`
- Run daily pipeline:
  - `python scripts/run_daily_market_pipeline.py --season 2026`
- Export web payload:
  - `python scripts/export_daily_predictions_web.py --manifest model/analysis/daily_runs/20260326/daily_market_pipeline_manifest_20260326.json --out-json ../web/data/daily_predictions.json --out-dist ../dist/data/daily_predictions.json`

## Notes

- If historical backtest CSV is unavailable, the selector now falls back to heuristic edge calibration.
- Upcoming slate prediction generation supports Covers-style abbreviated player names (for example `J_Brunson`).
