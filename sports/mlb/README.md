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

The repo now includes:

- `sports/mlb/scripts/generate_daily_prediction_pool.py`: builds a raw MLB daily pool from `Player-Predictor/Data-Proc-MLB`
- `sports/mlb/scripts/select_high_precision_predictions.py`: tightens the raw pool into a smaller board optimized for hit probability instead of raw volume

Example:

```bash
python sports/mlb/scripts/generate_daily_prediction_pool.py --run-date 2026-04-05

python sports/mlb/scripts/select_high_precision_predictions.py ^
  --pool-csv sports/mlb/data/predictions/daily_runs/20260410/daily_prediction_pool_20260410.csv
```

By default the selector:

- removes baseline-only rows
- keeps only supported count targets
- estimates directional hit probability from the model mean and line
- calibrates those hit rates with empirical target/direction/line win buckets from processed MLB history
- filters out weak edge / stale history / high-push plays
- limits concentration by player, game, team, and exact market bucket so one prop shape cannot dominate the board

The production action profile is stricter than the research defaults. It requires a real market from at least five books, at least 35 prior target observations, a projection of at least `0.10`, a valid selected-side price, and current matchup identity. Core plays require at least `0.35` absolute edge; the separately validated R/TB OVER profile uses a bounded `0.15` to `0.35` edge corridor. Scheduled-slate projections regress recent form toward longer-run player rates and aggregate rate statistics over prior games instead of treating one-game `wOBA`, `ISO`, or barrel rate as a forecast. Published calibrated probability is capped at 80% until prospective calibration supports a higher ceiling.

## Batter/Pitcher Matchup Network

`sports/mlb/decision_engine/matchup_network.py` links each hitter to the listed probable starter. The network combines target-specific batter skill, starter vulnerability, pitcher-profile uncertainty, heavily shrunk prior batter-versus-starter results, and the batter's prior performance against statistically similar starter profiles. Similar-starter evidence is reliability weighted, limited to the previous 60 games, and shrunk by effective support. Exact BvP rows are excluded from that neighborhood so they cannot be counted twice. It is a residual adjustment on top of the existing projection, so general hitter form is not counted twice. Pitcher uncertainty suppresses profile assumptions; it only increases the relative value of direct matchup history, and missing pitcher data alone produces no edge.

The adjustment is capped separately for H, TB, R, HR, and RBI and receives no independent confidence or parlay bonus. The v1 and v2 chronological ablations are stored in `sports/validation/mlb_matchup_network_ablation.json` and `sports/validation/mlb_matchup_network_v2_ablation.json`.

## Leakage-Aware Backtesting

Run the expanding-window MLB evaluation with:

```bash
python sports/mlb/scripts/backtest_prediction_method.py
```

The evaluator rebuilds priors using only dates before each slate, re-grades the selected direction, and separates real-market evidence from synthetic-line research. Reports are written under `sports/mlb/data/predictions/backtests/`. The current evidence verdict is shadow-only; synthetic replay rates must not be presented as executable ROI.

To tune the compact-board thresholds on chronological training and validation windows, then score the winner once on an untouched holdout:

```bash
python sports/mlb/scripts/optimize_walk_forward_policy.py --refresh-candidates
```

The deployed `walk_forward_balanced_v1` profile adds real-market placeability support, valid selected-side prices, a 12% push ceiling, tighter probability floors, and two-card market-bucket concentration. It does not fall back to synthetic or one-book cards solely to fill the board. Price-confirmed promotion still requires prospective evidence; flat `-110` replay profit is reported only as a comparison proxy.

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

That command checks local time and runs at `2:00 AM` by default. It generates a fresh MLB research pool from `Data-Proc-MLB`, applies the production action policy, updates the MLB web payload, refreshes NBA, and rebuilds the shared `dist/` bundle. A current slate that does not pass publication gates is withheld rather than replaced with an older or synthetic slate.
