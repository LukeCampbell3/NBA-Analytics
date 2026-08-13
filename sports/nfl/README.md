# NFL Predictor

This workspace contains a leakage-aware player-yardage model for passing,
rushing, and receiving yards. A predictive 16-dimensional player-state encoder
learns from each player's previous eight games and augments the regularized
tabular models. The untouched 2025 season is used once for the reported
replacement test.

## Daily live-shadow board

The production-shaped NFL path is independent of NBA and MLB. It captures the
complete two-sided NFL player-prop slate from The Odds API, joins offered players
to lagged nflverse form and opponent features, scores the frozen market selector
and a separate NFL loss-aware meta-policy, and publishes only executable
passing-yard candidates with at least two books
and one common US sportsbook:

```bash
python sports/nfl/scripts/backtest_nfl_daily_policy.py
python sports/nfl/scripts/train_nfl_pick_meta_selector.py
python sports/nfl/scripts/bootstrap_nfl_artifacts.py
python sports/nfl/scripts/refresh_nfl_yardage_artifact.py
python sports/nfl/scripts/run_nfl_daily_predictions.py --run-date YYYY-MM-DD
```

Set `SPORTSGAMEODDS_API_KEY` or `THE_ODDS_API_KEY` for the live run. The default
provider order is SportsGameOdds followed by The Odds API and can be changed
with `NFL_ODDS_PROVIDER_PRIORITY`. SportsGameOdds filtering is performed
client-side so lower subscription tiers are not rejected for requesting locked
bookmaker filters. Only same-book, same-line, currently available over/under
pairs with provider timestamps enter the complete slate; consensus-only lines
remain diagnostic and cannot become picks. Current snapshots are immutable JSON
ledgers under `data/production/snapshots/`. The model artifact is refit only
after a newly completed regular-season week appears; policy thresholds and
market scope remain frozen. The current board is shadow-only until prospective
certificate evidence exists, and every candidate is explicitly marked
unauthorized for staking.

The runtime artifacts are NFL-specific and independently typed:

- `model/nfl_yardage_latent_hybrid.joblib`: yardage projections
- `model/nfl_market_selector.joblib`: line-side scoring
- `model/nfl_pick_meta_selector.joblib`: loss-aware survivor policy

The meta-policy was selected on settled 2025 weeks 1-12 and frozen for weeks
13-18, where it retained 36 passing-yard candidates at 26-10 (72.22%) and
34.86% ROI. It uses a 0.58 side-probability floor, 0.10 no-vig advantage,
prices from -130 to +130, and a six-pick weekly cap. The 2025 source consists
of explicit SportsGameOdds provider consensus closes, so this is research
evidence rather than proof that a named sportsbook offered the same execution.
The live gate still requires current, named-book, two-sided prices.

Confidence calibration is independently replayed on the surviving policy pool.
Grouped development tests compare identity, shrinkage, Platt, beta, offset, and
isotonic transforms by Brier score and log loss. The identity transform remains
active because it is best on both development and locked recent evidence: over
the full 2025 profile, average confidence is 65.87% versus a 64.65% realized hit
rate. Live output retains raw and calibrated fields separately and abstains when
confidence falls outside the observed 58.56%-80.68% support range.

The legacy locked 2022 singles replay is 127-83 (60.48%) with +13.00% ROI across
210 selections. The deterministic distinct-game two-leg parlay failed its
locked replay at 2-16 and -61.89% ROI, so the pipeline may construct that ticket
for observation but always withholds it from recommendation. Rushing and
receiving props remain captured in the complete slate but are not selected
because they failed the target-level holdout gate.

## Train and test projections

```bash
python sports/nfl/scripts/train_nfl_predictor.py
```

The default run uses nflverse weekly statistics from 2018-2024, aggregates the
official 2025 play-by-play release into the same contract, selects architectures
on 2021-2024, and evaluates 2025. Generated outputs:

- `data/evaluation/backtest_report.json`: model-selection and holdout evidence
- `data/evaluation/backtest_rows.csv`: row-level 2025 projection and challenger audit
- `model/nfl_yardage_latent_hybrid.joblib`: deployable hybrid artifact (gitignored)
- `web/data/daily_predictions.json`: static research payload

All player and opponent inputs are shifted by at least one game. Eligibility is
based on prior opportunity volume rather than a hard position whitelist. The
report includes position-by-stat row counts and MAE so impossible or unintended
target populations are visible. In the current holdout, every passing-yard row
is a QB row.

## Latent replacement evidence

The latent challenger is not used alone. It is blended 50/50 with the previous
per-stat champion, preserving the stable raw-feature model while adding a
future-predictive player-state representation. On the untouched 2025 holdout:

- previous system weighted MAE: 25.1256 yards
- latent hybrid weighted MAE: 24.8836 yards
- relative improvement: 0.96%
- paired week-bootstrap MAE delta 95% interval: -0.393 to -0.099 yards
- passing improvement: 2.52%
- rushing improvement: 0.34%
- receiving improvement: 0.29%

The interval is entirely favorable and every stat improved, so the predeclared
replacement gate passed. Reproduce the comparison with:

```bash
python sports/nfl/scripts/benchmark_latent_challenger.py
```

The encoder is refit inside every chronological fold. No validation or holdout
season is used to train its latent space before that season is scored.

## Acquire historical sportsbook lines

The free starting point is XSportsbook's intentionally downloadable Bovada
archive: 12,598 regular-season player-prop rows for 2021 and 12,913 for 2022,
including two-sided prices. Download and normalize both seasons with:

```bash
python sports/nfl/scripts/fetch_xsportsbook_bovada_props.py
```

The importer discards the source result fields and retains only authentic
posted lines/prices. The archive is large enough for a development-season
threshold (2021) and an untouched final market test (2022). It does not contain
capture timestamps or an explicit opening-line guarantee, so it is eligible for
the report's performance gate but not the stricter static-deployment provenance
gate.

After producing leakage-free prediction rows for each season, freeze the edge
on 2021 and test 2022 once:

```bash
python sports/nfl/scripts/validate_nfl_market_holdouts.py \
  --lines sports/nfl/data/raw/xsportsbook_bovada_player_props.csv \
  --development-predictions sports/nfl/tmp/2021/backtest_rows.csv \
  --final-predictions sports/nfl/tmp/2022/backtest_rows.csv
```

The Odds API is the primary source because it offers paid, timestamped
event-level player-prop snapshots from May 2023 onward. Its current 20,000-credit
plan is enough for two regular seasons of the three target markets at one region
(maximum event-odds estimate: 16,320 credits). Estimate quota before making any
paid calls:

```bash
python sports/nfl/scripts/fetch_historical_nfl_props.py --season 2024
```

After setting `THE_ODDS_API_KEY`, add `--execute` only after reviewing the quota
estimate. The collector queries each game shortly before kickoff, retains the
actual snapshot timestamp, and writes book-specific lines and prices. Raw
historical lines are gitignored.

SportsGameOdds is integrated as a bulk closing-line source and cross-check. It
returns explicit per-book open/close fields in one paginated event feed:

```bash
python sports/nfl/scripts/fetch_sportsgameodds_historical_props.py \
  --season 2025 --weeks 1
```

After setting `SPORTSGAMEODDS_API_KEY`, add `--execute`. Historical access is a
Pro-or-higher feature, so first verify one week has all three target markets and
two-sided prices. The adapter never substitutes current/live values for missing
closing fields. See [MARKET_BACKTEST_DATA.md](MARKET_BACKTEST_DATA.md) for the
source comparison and acquisition protocol.

The market evaluator retains only the earliest available valid pregame
observation for each player/stat/book. Later movement and closing lines are not
used by the selective pipeline. The free archive contains one unstamped posted
observation rather than a time series, so no line-movement data is required or
manufactured.

## Train the selective betting layer

The point-projection model and betting selector solve different objectives. The
selector trains directly on whether a player cleared the posted line, compares
regularized raw-feature and latent-feature classifiers in expanding 2021 folds,
and tests the selected architecture once on 2022. Run it after generating
edge-zero market rows for both seasons:

```bash
python sports/nfl/scripts/train_nfl_market_selector.py \
  --development-market-rows sports/nfl/tmp/2021/market_rows_edge0.csv \
  --final-market-rows sports/nfl/tmp/2022/market_rows_edge0.csv
```

The run also writes auditable week-by-week exports:

- `data/evaluation/market_selector_pool_2021.csv`: expanding-fold development
  picks from weeks 11-18; weeks 1-10 are reserved for calibration
- `data/evaluation/market_selector_pool_2022.csv`: every eligible later-season
  test pick from weeks 1-18
- `data/evaluation/market_selector_validated_pool_2021.csv` and
  `market_selector_validated_pool_2022.csv`: passing-only pools after applying
  the target-level validation gate
- `data/evaluation/market_selector_weekly_validation.csv`: overall and
  per-target wins, losses, hit rate, ROI, and target validation status for every
  week

Each detailed row retains the projected side probability, no-vig probability,
posted line and price, actual yards, settled result, `pass`/`fail` validation,
selected architecture, and final target-level gate. Candidate picks from failed
targets remain visible for diagnosis but are not deployment-eligible.

An MLB-style board cap is selected using only the 2021 walk-forward pool. Caps
of 6, 8, 10, and 12 picks per week are compared; candidates need at least 60
development decisions, eight weeks, a Wilson lower bound above 50%, and positive
ROI. The highest Wilson lower bound selects top 12 per week. This reduces the
2022 validated board from 315 to 210 passing picks and improves it to 127-83
(60.48%) with +13.00% ROI. The final season is not used to choose the cap.

The pruned exports are:

- `data/evaluation/market_selector_pruned_pool_2021.csv`
- `data/evaluation/market_selector_pruned_pool_2022.csv`
- `data/evaluation/market_selector_pruned_weekly_validation.csv`

## Production-style effectiveness replay

The locked policy can be replayed without retraining or trusting stored result
labels:

```bash
python sports/nfl/scripts/run_nfl_production_replay.py
```

The replay fails closed on schema drift, duplicate props, unvalidated targets,
invalid prices, threshold violations, probability arithmetic mismatches, wrong
architectures, grading mismatches, nondeterministic ranking, outcome-dependent
selection, or more than 12 weekly picks. It independently regrades every play
from actual yards and the posted line, resamples complete weeks for clustered
confidence intervals, and compares against always-under and point-projection
baselines.

The locked 2022 replay passes its operational, statistical, and stability gates:

- 127-83 (60.48%), +13.00% ROI, +27.30 units
- exact one-sided hit-rate p-value versus 50%: 0.001457
- week-clustered 95% hit-rate interval: 54.07%-66.35%
- week-clustered 95% ROI interval: +0.99%-24.02%
- first-half and second-half hit rates: 60.78% and 60.19%
- top-8, top-10, and top-12 sensitivity boards all remain profitable
- maximum weekly-sequence drawdown: 7.23 units

The selector is statistically superior to the original point-projection side on
the matched cohort (paired one-sided p=0.0361). It is directionally better than
always-under (60.48% versus 57.14%), but that incremental paired comparison is
not yet significant (p=0.1239). The system therefore proves market-relative
historical effectiveness and price-confirmed profitability, while prospective
data is still required to prove that the learned side selection beats the
archive's broad under bias.

Run the complete selector-training and replay pipeline together with:

```bash
python sports/nfl/scripts/validate_nfl_production_pipeline.py
```

This regenerates the model artifact and pools in an isolated work directory,
runs the locked replay, and checks the expected 210 decisions and 127 wins. Its
research pipeline passes; deployment remains correctly blocked because the free
archive has no capture timestamps.

At the fixed 0.56 side-probability floor, only passing yards passes the
target-level holdout gate: 188-127 (59.68%), +11.67% flat-stake ROI over 315
2022 decisions. The regularized raw-feature classifier beat the latent and
CatBoost candidates on development Brier score. Rushing and receiving are
suppressed because they did not generalize. This is a partial statistical
validation, not permission to publish the static betting board: the free source
lacks timestamps proving that its observations were available before kickoff.

## Grade true betting hit rate

```bash
python sports/nfl/scripts/backtest_nfl_markets.py \
  --lines sports/nfl/data/raw/historical_player_props.csv
```

The evaluator rejects synthetic/result-derived rows and odds captured at or
after kickoff. When multiple timestamped observations exist, it keeps the first
valid pregame observation and discards later movement. It grades win/loss/push,
hit rate with a Wilson 95% interval,
American-price ROI, and stat/position breakdowns. Only players with a genuinely
posted line enter the betting denominator, which removes non-marketed backups
from sportsbook accuracy without using the outcome to filter them. The
performance gate answers whether the model beat the available lines. Promotion
additionally requires verified pregame timing, real prices on every wager, and
coverage across at least eight season-weeks.

Training can attach the same archive directly:

```bash
python sports/nfl/scripts/train_nfl_predictor.py \
  --market-lines sports/nfl/data/raw/historical_player_props.csv
```

Static promotion remains blocked unless both the projection gate and authentic
historical-market gate pass. A projection holdout alone is never labeled as a
verified betting backtest.

## Build the static site

Build the separate full-PPR fantasy draft model before packaging the site:

```bash
python sports/nfl/scripts/refresh_nfl_depth_chart.py --season 2026
python sports/nfl/scripts/build_fantasy_draft_rankings.py \
  --season 2026 --simulations 2000 --players 200
```

The command selects regularized point-model architectures on 2024, reports
separate fitted/seen and chronological unseen-week results, calibrates confidence
with out-of-sample conformal residuals, and writes
`web/data/fantasy_draft_rankings.json` only when every validation gate passes.
The current depth chart then supplies starter/rotation probabilities. A second
publication gate verifies that player scenarios exactly share each current
offense's finite pass-attempt, target, carry, passing-TD, receiving-TD, and
rushing-TD budgets. Players changing teams retain their history as a skill
prior, but receive their opportunity and schedule from the new team.
The draft board is available under `/nfl/fantasy/`; it includes per-game and
season-total stat projections, P10–P90 fantasy-point ranges, position ranks,
tiers, value over replacement, and a short player assessment.

```bash
python sports/site/pipeline/build_static_site.py
```

The NFL report is available under `/nfl/predictions/` and its methodology under
`/nfl/prediction-about/`. Until market validation passes, the payload is marked
`research_only` and should not be merged as a published betting board.
