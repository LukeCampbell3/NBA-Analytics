# MLB Pipeline

MLB-native data collection, feature engineering, validation, and training stack.
The trainer enforces pregame-safe features only and compares ML models against rolling baselines.

## Layout

- `schema/mlb_native_player_schema_v1.json`: machine-readable MLB contract.
- `MLB_MODEL_DATA_REQUIREMENTS.md`: human-readable requirements.
- `data/raw/`: collected raw game logs.
- `data/processed/`: per-player processed files + aggregate role files.
- `scripts/collect_mlb_data.py`: collect raw logs from MLB Stats API.
- `scripts/build_mlb_features.py`: build processed training features.
- `scripts/validate_mlb_processed_contract.py`: validate processed contract.
- `scripts/train_mlb_models.py`: train hitter and pitcher models.
- `scripts/run_mlb_pipeline.py`: orchestrate full pipeline.
- `scripts/build_mlb_daily_prediction_pool.py`: generate daily pregame prediction pool.
- `scripts/score_mlb_prediction_pool.py`: score daily pool vs actual outcomes.
- `scripts/run_mlb_daily_prediction_pipeline.py`: daily orchestrator for collect/build/validate/infer/score.

## Quick Start

```bash
python sports/mlb/scripts/collect_mlb_data.py --start-date 2026-03-20 --end-date 2026-04-08
python sports/mlb/scripts/build_mlb_features.py --season 2026
python sports/mlb/scripts/validate_mlb_processed_contract.py
python sports/mlb/scripts/train_mlb_models.py --season 2026 --min-rows 200
```

Full orchestrator:

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-03-20 --end-date 2026-04-08 --min-train-rows 200
```

Short-range smoke test (small date windows):

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-04-01 --end-date 2026-04-03 --min-processed-rows 1 --min-train-rows 200
```

Daily prediction pool:

```bash
python sports/mlb/scripts/build_mlb_daily_prediction_pool.py --run-date 2026-04-10 --season 2026
```

One-command best picks (build pool + reduce to final shortlist):

```bash
python sports/mlb/scripts/select_mlb_best_predictions.py --run-date 2026-04-10 --season 2026 --top-n 50 --min-abs-edge 0.35 --min-history-rows 10 --max-per-player 1
```

Score a pool against real outcomes:

```bash
python sports/mlb/scripts/score_mlb_prediction_pool.py --pool-csv sports/mlb/data/predictions/daily_runs/20260410/daily_prediction_pool_20260410.csv
```

Daily orchestrator (NBA-style run folder + manifest):

```bash
python sports/mlb/scripts/run_mlb_daily_prediction_pipeline.py --run-date 2026-04-10 --score-all-unscored
```
