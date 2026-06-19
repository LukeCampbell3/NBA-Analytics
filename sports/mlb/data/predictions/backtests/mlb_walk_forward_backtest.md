# MLB Prediction Method Backtest

Generated: 2026-06-19T19:00:54.758011+00:00

## Method

Every evaluation date uses only rows from earlier dates to construct historical and recent-form priors. Candidate direction is re-graded from the selected side. Flat-stake units use recorded odds when available and -110 only as a research proxy otherwise.

## Results

| Policy | Window | W-L-P | Hit rate | 95% interval | Model estimate | Net units* |
|---|---:|---:|---:|---:|---:|---:|
| published_real_market | long window | 46-9-0 | 83.6% | 71.7%-91.1% | 79.4% | +43.65 |
| published_real_market | short window | 46-9-0 | 83.6% | 71.7%-91.1% | 79.4% | +43.65 |
| directional_model_replay | long window | 1644-216-3 | 88.4% | 86.9%-89.8% | 77.9% | +1284.96 |
| directional_model_replay | short window | 285-30-3 | 90.5% | 86.7%-93.2% | 81.5% | +235.50 |
| guardrailed_short_board | long window | 249-19-0 | 92.9% | 89.2%-95.4% | 86.0% | +205.38 |
| guardrailed_short_board | short window | 42-0-0 | 100.0% | 91.6%-100.0% | 89.3% | +36.20 |

*Recorded prices are used when present; unpriced research rows use a -110 proxy and are not executable ROI.*

## Observed Boards

- Archived boards: 16-9 (64.0%) on 25 graded picks.
- June 17 after deduplication: 19-16 (54.3%) on 35 graded picks.
- Combined observed direction: 35-25 (58.3%); 95% interval 45.7%-69.9%.

## Calibration Audit

The raw 90-100% estimate bucket realized 77.6% on 7191 outcomes, versus an average estimate of 97.9%.

**Verdict: SHADOW_ONLY_NOT_VALIDATED**

## Interpretation

- The placeable real-market sample contains 55 graded picks across 3 dates; it is too small to support a long-term claim.
- The long-window directional replay hit 88.4% on 1860 graded picks; this measures stored-line directional accuracy, not realizable betting profit.
- The tighter six-play board hit 92.9% on 268 graded picks with a 95% interval of 89.2%-95.4%.

## Limits

- The historical universe ends on 2026-04-29 and real sportsbook coverage exists only on 2026-04-27 through 2026-04-29.
- Synthetic-line rows test model ranking and grading logic but cannot establish executable ROI or closing-line value.
- The replay covers H, K, R, and TB; the current published board also includes HR and ER, which lack this backtest universe.
- Lineup confirmation, roster validation, duplicate suppression, and stale-data withholding reduce publishing risk but do not create predictive edge.
- No backtest can guarantee short-term or long-term wins; promotion requires prospective, timestamped shadow results.
