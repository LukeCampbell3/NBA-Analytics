# Unified MLB historical evidence recovery

Recovery checkpoint: `c871e1652064184ac2cdd36a8f576d8eb237229b`  
Checkpoint CI: `33353124131`  
Evidence state: **LOCKED_HISTORICAL_VALIDATION**

## Result

The exact corpus expanded from one to eight observations without changing the
frozen policy or evidence thresholds. All eight are from August 30, so the
independent-slate count remains one.

| Capability | Exact observations | Wins | Losses |
|---|---:|---:|---:|
| Batter Hits | 6 | 2 | 4 |
| Batter Total Bases | 2 | 1 | 1 |
| Total | 8 | 3 | 5 |

The aggregate realized hit rate is 37.5% and realized flat-stake ROI is
−37.4%. These are descriptive values for one slate, not stable estimates. They
do not authorize threshold changes or production promotion.

## Admitted workflow evidence

Seven additional observations came from retained `mlb-frontend-20260830`
GitHub Actions artifacts `9734123174`, `9735656814`, and `9737132236`.
Each admitted observation preserves the complete candidate payload, workflow
and artifact identity, workflow head commit, artifact ZIP SHA-256, canonical
candidate SHA-256, pregame timestamps, final/usable probability, exact line
and FanDuel price, confirmed lineup state, and an independently hashed MLB
StatsAPI settlement.

Repeated Cal Raleigh, Austin Riley, and Pete Crow-Armstrong artifacts were
deduplicated by semantic candidate identity. The earliest exact pregame object
remains authoritative.

## Diagnostic-only sources

The August 28 and August 29 V16 artifacts contain 13 pregame candidates, but
they do not preserve `final_hit_probability`. The 146-row path-evidence
reservoir and 242,425-row historical universe also lack the complete frozen
decision intersection. No current or closing quote was substituted.

## Remaining gate

The frozen requirements remain literal and capability-specific:

* independent slates: 1 of 20;
* Batter Hits: 6 of 50;
* Batter Total Bases: 2 of 50.

Fifty mixed selections cannot certify either capability. The regression suite
now enforces this directly.
