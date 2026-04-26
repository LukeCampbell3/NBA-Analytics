# MLB Workspace

This folder now includes a published MLB landing page, bounty-style prediction board, and prediction method page under the shared multi-sport site.

Suggested next folders:

- `web/` active MLB frontend pages
- `pipeline/`
- `predictions/`
- `tests/`

## Frontend Pages

- `/mlb/` MLB home page
- `/mlb/predictions/` MLB prediction bounty board
- `/mlb/prediction-about/` MLB prediction method page

These pages are published through the repo-root `dist/` bundle.

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

## Web Payload Export

The predictions pages read from `sports/mlb/web/data/daily_predictions.json`.
To rebuild that payload from the latest high-precision selector output:

```bash
python sports/mlb/scripts/export_web_prediction_payload.py
```

For the shared published site, the preferred one-shot command is:

```bash
python sports/site/pipeline/run_daily_predictions.py
```

That command checks local time and runs at `2:00 AM` by default. When it runs, it selects the latest MLB high-precision board, updates the MLB web payload, refreshes NBA, and rebuilds the shared `dist/` bundle.
