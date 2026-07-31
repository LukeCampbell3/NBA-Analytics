# NFL Predictor

The NFL workspace contains a reproducible player-yardage model for passing,
rushing, and receiving yards. It adapts the NBA predictor's lagged sequence,
gradient-boosting, regularized regression, and stacked-model approach to weekly
football data.

## Train and backtest

```bash
python sports/nfl/scripts/train_nfl_predictor.py
```

The default run downloads and locally caches nflverse weekly player statistics
for 2018–2024, uses 2022–2023 as chronological stacking folds, and evaluates an
untouched 2024 holdout. Generated outputs:

- `data/evaluation/backtest_report.json`: committed model evidence
- `data/evaluation/backtest_rows.csv`: row-level holdout audit
- `model/nfl_yardage_stack.joblib`: local deployable model artifact (gitignored)
- `web/data/daily_predictions.json`: static model-report payload

All rolling player and opponent inputs are shifted by at least one game. The
published accuracy is projection accuracy, not sportsbook win rate; archived
market lines are not available in the training source.

## Build the static site

```bash
python sports/site/pipeline/build_static_site.py
```

The NFL report is then available under `/nfl/predictions/` and its methodology
under `/nfl/prediction-about/`.
