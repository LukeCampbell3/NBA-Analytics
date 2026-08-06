# MLB Policy Thesis Retrospective Audit

Generated: 2026-08-06T21:51:59.029680+00:00
Evidence: `RETROSPECTIVE_FULL_CANDIDATE_RECONSTRUCTION`

## Verdict

**CURRENT_PROFILE_REJECTED_BROADER_THEORY_UNPROVEN**

- Current-profile singles did not produce a positive holdout calendar return.
- The multiplicity-adjusted bounded return LCB did not clear the 1% deployment margin.
- Captured date and action-day volume is far below certificate requirements.
- The parlay return LCB did not clear the deployment margin.
- The current profile did not outperform the broader playable-OVER baseline on calendar return.

## Chronological Split

- Development audit dates: 2026-04-27, 2026-05-01, 2026-06-19, 2026-06-20
- Retrospective holdout dates: 2026-06-21, 2026-06-26, 2026-06-27
- The holdout was not locked before earlier policy development and cannot certify production.
- Development current-profile result: 0-2-0 across 2 picks.

## Holdout Results

| Policy | Picks/tickets | W-L-P | Hit rate | Calendar return | Return LCB | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Current-profile singles | 6 | 2-4-0 | 33.3% | -40.4% | -100.0% | 100.0% |
| Playable-over baseline | 9 | 5-4-0 | 55.6% | 27.9% | -100.0% | 100.0% |
| Two-leg parlay | 3 | 2-1-0 | 66.7% | 241.4% | -100.0% | 100.0% |

## Interpretation

- Returns use the latest captured pregame price at the exact line and book.
- Every positive-edge modeled OVER candidate with an exact historical quote was retained before policy filtering.
- A positive point estimate is descriptive; the bounded held-out LCB is the decision criterion.
- Pitcher workload fields and RBI lines were unavailable in the captured historical scope.
- This audit does not create or activate a policy certificate.
