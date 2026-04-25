# MLB Workspace Scaffold

This folder is reserved for MLB pages, prediction pipelines, and model artifacts, and now includes a small `web/` entry point so baseball has its own frontend route.

Suggested next folders:

- `web/` placeholder MLB landing page
- `dist/`
- `pipeline/`
- `predictions/`
- `tests/`

## High-Precision Selection

The repo now includes an MLB tightening script at `sports/mlb/scripts/select_high_precision_predictions.py`.
It takes a large daily pool and produces a smaller board optimized for hit probability instead of raw volume.

Example:

```bash
python sports/mlb/scripts/select_high_precision_predictions.py ^
  --pool-csv sports/mlb/data/predictions/daily_runs/20260410/daily_prediction_pool_20260410.csv
```

By default the selector:

- removes baseline-only rows
- keeps only supported count targets
- estimates directional hit probability from the model mean and line
- filters out weak edge / stale history / high-push plays
- limits concentration by player, game, and team
