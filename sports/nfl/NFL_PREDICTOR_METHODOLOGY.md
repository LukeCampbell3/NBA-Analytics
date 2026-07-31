# NFL Predictor and Market-Selection Methodology

## Status

The NFL system has two distinct layers:

1. A leakage-aware player-yardage model that predicts passing, rushing, and
   receiving yards.
2. A selective market model that estimates whether a posted passing-yard line
   will finish over or under, then publishes no more than 12 qualified picks per
   week.

The locked historical replay supports **research effectiveness**, not live
deployment. The passing board finished 127-83 (60.48%) with +13.00% flat-stake
ROI on 210 later-season decisions. Static deployment remains blocked because
the free historical line archive does not include capture timestamps.

## Data and eligibility

- Player statistics come from nflverse weekly data, with 2025 supplementation
  from play-by-play aggregation where needed.
- Every player, opponent, volume, and latent feature is shifted so the current
  game's outcome is unavailable at prediction time.
- Eligibility is based on prior opportunity: passing attempts, rushing carries,
  or receiving targets. It is not restricted to a hard-coded position list.
- Market evaluation includes only players who had a genuinely posted line.
- The free market archive contains Bovada player props for 2021 and 2022 with
  both side prices. Result columns supplied by the archive are discarded.
- One earliest available posted observation is retained per player, statistic,
  week, and book. Line movement and closing lines are not used.

## Yardage projection architecture

Pregame features summarize each player's strictly prior history:

- Three- and five-game yardage and opportunity averages
- Five-game volatility and expanding career averages
- Lagged opponent yardage allowances
- Season progress and games played
- Position indicators and target-specific efficiency/context features

A predictive latent encoder reads the previous eight player games and learns a
16-dimensional state by predicting the next statistical state. Ridge, histogram
boosting, Extra Trees, XGBoost, CatBoost, and fixed regularized blends are
compared in expanding season folds.

The latent representation is retained for point projections because its hybrid
improved weighted holdout MAE with a favorable paired week-bootstrap interval.
No current-game result is used to fit its fold-local representation.

## Why the betting layer is simpler

Lower yardage MAE did not translate automatically into profitable over/under
decisions. The market layer therefore optimizes the actual decision: the
probability that actual yards exceed the posted line.

For each target, the candidates are:

- Regularized logistic classification on raw lagged features
- Regularized logistic classification on raw plus frozen-latent features
- CatBoost classification on raw features
- CatBoost classification on raw plus frozen-latent features

Architecture selection uses expanding 2021 folds and chooses the lowest Brier
score, then log loss, then the smaller feature set. The regularized raw-feature
classifier won for passing. Adding latent features reduced calibration quality
for this smaller line-labeled problem, so they are not forced into production.

## Probability and target gates

A candidate pick requires:

- Estimated selected-side probability of at least 0.56
- At least 0.025 probability advantage over the no-vig book probability
- Valid prices for both sides

Each statistic is evaluated independently. A target needs at least 150 graded
decisions, eight weeks, at least 58% hit rate, a Wilson 95% lower bound above
50%, and positive ROI in both the development and later-season gates. Passing
qualified. Rushing and receiving remain diagnostic-only.

## MLB-style weekly pruning

Eligible passing picks are ranked by estimated side probability and then by
probability advantage. Weekly caps of 6, 8, 10, and 12 were compared using only
the 2021 walk-forward pool.

Low-volume caps are rejected. An eligible cap needs at least 60 development
decisions, eight weeks, positive ROI, and a Wilson lower bound above 50%. The
highest Wilson lower bound selected a maximum of 12 picks per week. The 2022
season was not read during cap selection.

| Evaluation | Picks | Record | Hit rate | ROI |
|---|---:|---:|---:|---:|
| 2021 expanding development | 95 | 59-36 | 62.11% | +15.72% |
| 2022 locked final replay | 210 | 127-83 | 60.48% | +13.00% |

## Production-style replay

The replay does not trust stored outcomes. It independently recomputes every
side, result, price, and unit return from actual yards and the posted line. It
fails closed on:

- Missing columns or schema drift
- Duplicate player props
- Unvalidated targets
- Invalid prices or probability ranges
- Threshold, probability-arithmetic, price-side, or architecture mismatches
- Nondeterministic ranking
- Selection that changes after outcomes are mutated
- Stored grading that differs from recomputed grading
- More than 12 picks in any week

The locked replay passed contract, operational, effectiveness, and stability
gates:

- Wilson hit-rate interval: 53.73%-66.84%
- Week-clustered hit-rate interval: 54.07%-66.35%
- Week-clustered ROI interval: +0.99%-24.02%
- Exact one-sided p-value versus 50%: 0.001457
- First half: 60.78%; second half: 60.19%
- Maximum drawdown: 7.23 units

On the identical cohort, the original point-projection side finished 52.86%
with -1.11% ROI. The selector was significantly better in a paired comparison
(one-sided p=0.0361).

Always-under finished 57.14% with +6.62% ROI. The selector improved that to
60.48% and +13.00%, but the incremental paired result is not statistically
significant (p=0.1239). This unresolved under-bias comparison is a required
caveat, not a hidden failure.

## Deployment boundary

The statistical evidence demonstrates historical effectiveness under the locked
protocol. It does not prove that every archived line was captured before
kickoff, because the free source provides no timestamps. The frontend must
therefore label the board `research_only_source_blocked`.

Promotion requires either:

- A timestamp-authenticated historical opening-line replay, or
- A prospective shadow run that records each line before kickoff and passes the
  same operational, effectiveness, stability, and ROI gates.

## Reproduction

```bash
python sports/nfl/scripts/train_nfl_market_selector.py \
  --development-market-rows sports/nfl/tmp/2021/market_rows_edge0.csv \
  --final-market-rows sports/nfl/tmp/2022/market_rows_edge0.csv

python sports/nfl/scripts/run_nfl_production_replay.py
python sports/nfl/scripts/validate_nfl_production_pipeline.py
python sports/nfl/scripts/export_nfl_validation_web.py
```

Primary evidence files:

- `data/evaluation/market_selector_report.json`
- `data/evaluation/production_replay_report.json`
- `data/evaluation/production_pipeline_validation_report.json`
- `data/evaluation/production_replay_picks.csv`
- `data/evaluation/production_replay_weekly.csv`
