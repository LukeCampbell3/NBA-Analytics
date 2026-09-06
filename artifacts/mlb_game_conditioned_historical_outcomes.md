# MLB Game-Conditioned Historical Outcome Collection

Evidence: `HISTORICAL_REALIZED_OUTCOME_LABEL_NOT_PREGAME_FEATURE_EVIDENCE`

Season: **2026**

Collected hitter-games: **47,297** across **679** hitters, **2,333** games, and **178** dates.

| Target | Threshold | Rows | Clears | Observed clear rate |
|---|---:|---:|---:|---:|
| H | O 0.5 | 47,297 | 26,194 | 55.38% |
| TB | O 1.5 | 47,297 | 14,761 | 31.21% |
| HR | O 0.5 | 47,297 | 4,912 | 10.39% |

The ledger contains only realized H/TB/HR/PA/AB outcomes plus identity/provenance. Projection, market, matchup, rolling-form, and model fields are intentionally excluded so historical settlement can be joined after pregame feature construction.

Historical labels are valid retrospective outcomes, but they do not by themselves prove that reconstructed historical features were available exactly as represented before first pitch.

## Skipped rows

- `NON_HITTER`: 17,603
- `ZERO_PA`: 1,610
