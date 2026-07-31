# NFL Predictor

This workspace contains a leakage-aware player-yardage model for passing,
rushing, and receiving yards. Pregame rolling features are shared across a
compact tabular candidate set, but each stat selects its own architecture on
expanding chronological folds. The untouched 2024 season is used once for the
reported projection test.

## Train and test projections

```bash
python sports/nfl/scripts/train_nfl_predictor.py
```

The default run uses nflverse regular-season weekly statistics from 2018-2024,
selects architectures on 2020-2023, and evaluates 2024. Generated outputs:

- `data/evaluation/backtest_report.json`: model-selection and holdout evidence
- `data/evaluation/backtest_rows.csv`: row-level 2024 projection audit
- `model/nfl_yardage_stack.joblib`: local deployable model artifact (gitignored)
- `web/data/daily_predictions.json`: static research payload

All player and opponent inputs are shifted by at least one game. Eligibility is
based on prior opportunity volume rather than a hard position whitelist. The
report includes position-by-stat row counts and MAE so impossible or unintended
target populations are visible. In the current holdout, every passing-yard row
is a QB row.

## Acquire historical sportsbook lines

The Odds API offers paid historical event-level player props from May 2023
onward. Estimate quota before making any paid calls:

```bash
python sports/nfl/scripts/fetch_historical_nfl_props.py --season 2024
```

After setting `THE_ODDS_API_KEY`, add `--execute` only after reviewing the quota
estimate. The collector queries each game shortly before kickoff, retains the
actual snapshot timestamp, and writes book-specific lines and prices. Raw
historical lines are gitignored.

## Grade true betting hit rate

```bash
python sports/nfl/scripts/backtest_nfl_markets.py \
  --lines sports/nfl/data/raw/historical_player_props.csv
```

The evaluator rejects synthetic/result-derived rows and odds captured at or
after kickoff. It grades win/loss/push, hit rate with a Wilson 95% interval,
American-price ROI, and stat/position breakdowns. Only players with a genuinely
posted line enter the betting denominator, which removes non-marketed backups
from sportsbook accuracy without using the outcome to filter them.

Training can attach the same archive directly:

```bash
python sports/nfl/scripts/train_nfl_predictor.py \
  --market-lines sports/nfl/data/raw/historical_player_props.csv
```

Static promotion remains blocked unless both the projection gate and authentic
historical-market gate pass. A projection holdout alone is never labeled as a
verified betting backtest.

## Build the static site

```bash
python sports/site/pipeline/build_static_site.py
```

The NFL report is available under `/nfl/predictions/` and its methodology under
`/nfl/prediction-about/`. Until market validation passes, the payload is marked
`research_only` and should not be merged as a published betting board.
