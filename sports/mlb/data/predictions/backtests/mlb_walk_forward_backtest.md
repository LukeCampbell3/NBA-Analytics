# MLB Prediction Method Backtest

Generated: 2026-07-31T20:14:59.769593+00:00

## Method

Every evaluation date uses only rows from earlier dates to construct historical and recent-form priors. Candidate direction is re-graded from the selected side. Flat-stake units use recorded odds when available and -110 only as a research proxy otherwise.

## Results

| Policy | Window | W-L-P | Hit rate | 95% interval | Model estimate | Net units* |
|---|---:|---:|---:|---:|---:|---:|
| production_action_board | long window | 49-5-1 | 90.7% | 80.1%-96.0% | 79.5% | +27.46 |
| production_action_board | short window | 0-0-0 | n/a | n/a | n/a | n/a |
| published_real_market | long window | 242-90-5 | 72.9% | 67.9%-77.4% | 72.1% | +151.94 |
| published_real_market | short window | 2-2-0 | 50.0% | 15.0%-85.0% | 71.5% | +1.21 |
| directional_model_replay | long window | 5240-648-4 | 89.0% | 88.2%-89.8% | 75.5% | +4111.07 |
| directional_model_replay | short window | 264-28-0 | 90.4% | 86.5%-93.3% | 76.0% | +211.36 |
| guardrailed_short_board | long window | 748-45-0 | 94.3% | 92.5%-95.7% | 78.7% | +631.39 |
| guardrailed_short_board | short window | 40-2-0 | 95.2% | 84.2%-98.7% | 78.7% | +34.36 |

*Recorded prices are used when present; unpriced research rows use a -110 proxy and are not executable ROI.*

## Observed Boards

- Archived boards: 16-9 (64.0%) on 25 graded picks.
- June 17 after deduplication: 19-16 (54.3%) on 35 graded picks.
- June 19 raw top-edge partial audit: 5-10 on 15 completed-game rows; this is a raw-pool diagnostic, not a finalized-board result.
- Combined observed direction: 35-25 (58.3%); 95% interval 45.7%-69.9%.

## Calibration Audit

The raw 90-100% estimate bucket realized 89.5% on 48886 outcomes, versus an average estimate of 97.8%.

**Verdict: SHADOW_ONLY_NOT_VALIDATED**

## Interpretation

- The production action policy contains 54 graded historical picks across 8 dates; it remains shadow-only until the priced sample is materially larger.
- The placeable real-market sample contains 332 graded picks across 11 dates; it is too small to support a long-term claim.
- The long-window directional replay hit 89.0% on 5888 graded picks; this measures stored-line directional accuracy, not realizable betting profit.
- The tighter six-play board hit 94.3% on 793 graded picks with a 95% interval of 92.5%-95.7%.

## Limits

- The historical universe ends on 2026-07-29; real sportsbook rows cover 2026-04-27 through 2026-07-29 across 11 dates.
- Synthetic-line rows test model ranking and grading logic but cannot establish executable ROI or closing-line value.
- The replay covers every configured count target: ER, H, HR, K, R, RBI, TB.
- Lineup confirmation, roster validation, duplicate suppression, and stale-data withholding reduce publishing risk but do not create predictive edge.
- No backtest can guarantee short-term or long-term wins; promotion requires prospective, timestamped shadow results.
