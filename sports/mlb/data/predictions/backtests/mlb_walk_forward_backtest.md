# MLB Prediction Method Backtest

Generated: 2026-06-20T12:05:58.289015+00:00

## Method

Every evaluation date uses only rows from earlier dates to construct historical and recent-form priors. Candidate direction is re-graded from the selected side. Flat-stake units use recorded odds when available and -110 only as a research proxy otherwise.

## Results

| Policy | Window | W-L-P | Hit rate | 95% interval | Model estimate | Net units* |
|---|---:|---:|---:|---:|---:|---:|
| production_action_board | long window | 10-4-1 | 71.4% | 45.4%-88.3% | 79.5% | +3.77 |
| production_action_board | short window | 6-2-0 | 75.0% | 40.9%-92.9% | 79.1% | +3.60 |
| published_real_market | long window | 112-35-1 | 76.2% | 68.7%-82.4% | 72.6% | +64.99 |
| published_real_market | short window | 25-12-0 | 67.6% | 51.5%-80.4% | 70.5% | +11.48 |
| directional_model_replay | long window | 3751-439-1 | 89.5% | 88.6%-90.4% | 75.4% | +2967.35 |
| directional_model_replay | short window | 289-33-0 | 89.8% | 86.0%-92.6% | 76.2% | +230.89 |
| guardrailed_short_board | long window | 540-33-0 | 94.2% | 92.0%-95.9% | 78.7% | +456.27 |
| guardrailed_short_board | short window | 37-5-0 | 88.1% | 75.0%-94.8% | 78.9% | +28.53 |

*Recorded prices are used when present; unpriced research rows use a -110 proxy and are not executable ROI.*

## Observed Boards

- Archived boards: 16-9 (64.0%) on 25 graded picks.
- June 17 after deduplication: 19-16 (54.3%) on 35 graded picks.
- June 19 raw top-edge partial audit: 5-10 on 15 completed-game rows; this is a raw-pool diagnostic, not a finalized-board result.
- Combined observed direction: 35-25 (58.3%); 95% interval 45.7%-69.9%.

## Calibration Audit

The raw 90-100% estimate bucket realized 82.3% on 13402 outcomes, versus an average estimate of 97.5%.

**Verdict: SHADOW_ONLY_NOT_VALIDATED**

## Interpretation

- The production action policy contains 14 graded historical picks across 4 dates; it remains shadow-only until the priced sample is materially larger.
- The placeable real-market sample contains 147 graded picks across 6 dates; it is too small to support a long-term claim.
- The long-window directional replay hit 89.5% on 4190 graded picks; this measures stored-line directional accuracy, not realizable betting profit.
- The tighter six-play board hit 94.2% on 573 graded picks with a 95% interval of 92.0%-95.9%.

## Limits

- The historical universe ends on 2026-06-19; real sportsbook rows cover 2026-04-27 through 2026-06-19 across 6 dates.
- Synthetic-line rows test model ranking and grading logic but cannot establish executable ROI or closing-line value.
- The replay covers H, K, R, and TB; the current published board also includes HR and ER, which lack this backtest universe.
- Lineup confirmation, roster validation, duplicate suppression, and stale-data withholding reduce publishing risk but do not create predictive edge.
- No backtest can guarantee short-term or long-term wins; promotion requires prospective, timestamped shadow results.
