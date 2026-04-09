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
