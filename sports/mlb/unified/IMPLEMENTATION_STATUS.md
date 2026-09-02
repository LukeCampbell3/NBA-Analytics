# Unified MLB Engine — Implementation Status

Branch: `unified-mlb-engine`  
Production fallback: `static-deployment` legacy MLB pipeline  
Migration authority: **none** until M9 gates pass

## Current pipeline map (M0)

`approved MLB history / schedule / probable starters` →
`generate_daily_prediction_pool.py` → player/pitcher estimators + matchup network →
`select_high_precision_predictions.py` → legacy singles exporter.

In parallel:

* `game_simulation_model.py` + `pitching_enriched_win_model.py` → moneyline/game-total/F5 candidates → same-game selector.
* `pitcher_strikeout_model.py` → pitcher alternate-line frontier → pitcher parlay selector.
* selected player props → legacy/frozen/high-hit cross-game parlay selectors.
* independent JSON artifacts → `predictions.js` rendering.
* aggregate player/game settlement is handled by `settle_published_predictions.py`; calibration ledgers have separate schemas.
* `.github/workflows/mlb-predictions.yml` orchestrates generation and publication.

This fragmentation permits inconsistent probability/EV definitions and independently rendered products. The unified branch adds one candidate/decision/ticket contract while retaining legacy generation as rollback.

## Existing model inventory

| Component | Inputs | Output | Evidence/authority | Limitation |
|---|---|---|---|---|
| Player projection pool | strictly prior processed player history, schedule, live prop lines | H/TB/R/RBI/HR and pitcher targets | legacy production input; calibrators vary | candidates do not share one final probability contract |
| Matchup network | batter/pitcher profiles and prior history | diagnostic matchup adjustment | production feature, bounded adjustment | does not create a coherent game world |
| Pitcher K model | starter history/workload/opponent context | strikeout distribution | pitcher shadow/parlay use | aggregate game K only; no inning path |
| Pitching-enriched win model | team/starter/bullpen history | team runs/win inputs | same-game shadow | no player accounting identity |
| Game simulation | team expected runs and starter/bullpen state | ML, total and F5 probabilities; shared trials | same-game shadow | aggregate trials; player outcomes absent |
| Historical/live calibration | settled buckets and live-board evidence | adjusted hit probability | mixed legacy authority | support and probability fields differ by product |
| V4 singles | balanced/market blend | shadow ranking | `PROSPECTIVE_SHADOW`, uncertified | H over 0.5 research scope |
| Legacy/V2/high-hit parlays | selected prop rows | two-leg tickets | frozen/shadow states vary | separate value semantics |

## Market capability matrix

| Market | Outcomes | Pregame features | Live quote | Historical quote | Exact identity | Model | Settlement | Status / blocker |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| H | yes | yes | yes | partial | yes | yes | yes | SUPPORTED |
| TB | yes | yes | yes | partial | yes | yes | yes | SUPPORTED |
| R | yes | yes | yes | partial | yes | yes | yes | SUPPORTED |
| RBI | yes | yes | yes | partial | yes | yes | yes | SUPPORTED |
| HR | yes | yes | yes | partial | yes | yes | yes | SUPPORTED |
| Pitcher K | yes | yes | yes | partial | yes | yes | yes | SUPPORTED |
| Pitcher outs | yes | yes | provider-dependent | sparse | yes | partial | aggregate hook | SHADOW_ONLY / quote and calibration support |
| Moneyline | yes | yes | yes | partial | game | yes | yes | SUPPORTED_SHADOW |
| Game total | yes | yes | yes | partial | game | yes | yes | SUPPORTED_SHADOW |
| F5 total | yes | yes | yes | partial | game/period | yes | yes | SUPPORTED_SHADOW |
| Team runs/total | yes | yes | provider-dependent | insufficient | team/game | structural distribution exists | aggregate hook | DISCOVERY / exact two-sided quote support |
| Team hits | yes | partial | unknown daily | unavailable | team/game | no coherent player-sum model | aggregate hook | MODEL_REQUIRED: team hits must equal summed player hits |
| Runs by inning | derivable from play-by-play | partial | discovery only | unavailable | inning | none | hook only | EVENT_MODEL_REQUIRED |
| Team runs by inning | derivable | partial | discovery only | unavailable | team/inning | none | hook only | EVENT_MODEL_REQUIRED |
| Ks by inning | derivable | partial | discovery only | unavailable | pitcher/inning | none | hook only | EVENT_MODEL_REQUIRED |
| Pitches by inning | derivable | partial | discovery only | unavailable | pitcher/inning | none | hook only | EVENT_MODEL_REQUIRED |
| PA pitch count | derivable | partial | discovery only | unavailable | unresolved PA ordinal | none | hook only | EVENT_IDENTITY_UNAVAILABLE |

## Invariants

1. Missing probability, price, identity, role, support, or evidence fails closed.
2. `usable_probability` alone determines displayed edge and conservative EV.
3. No negative-conservative-EV candidate enters a selected ticket.
4. Same-game joints require common-world masks; otherwise `SHADOW_ONLY` with no executable EV claim.
5. Two-, three-, and four-leg classes select independently and may independently abstain.
6. Parlay legs never become singles unless the singles gate independently admits them.
7. New mechanisms start `DEVELOPMENT` or `PROSPECTIVE_SHADOW`.
8. Legacy publication remains the rollback until M9 explicitly migrates authority.

## Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 audit | COMPLETE | this document |
| M1 contracts | COMPLETE | conservative decision-contract tests |
| M2 supported adapters | COMPLETE_SHADOW | legacy player and aggregate-product adapters; unsupported uncertainty fails closed |
| M3 universal parlays | COMPLETE_SHADOW | bounded, independently tested 2/3/4 constructors; SGP common-mask requirement |
| M4 unified artifact/UI | COMPLETE_SHADOW | one JSON source and grouped ticket component; legacy remains fallback |
| M5 player/team dependencies | FOUNDATION_COMPLETE | smoothed shares, coherent event allocation, conditional metrics; no production authority |
| M6 market conditioning | FOUNDATION_COMPLETE | declared identification level, clipped weights, ESS gate; diagnostic-only when underidentified |
| M7 event/exotic foundation | COMPLETE_FAIL_CLOSED | event model and exact identity are mandatory; no fabricated probabilities |
| M8 validation | INTERFACE_COMPLETE | point-in-time labels, metrics, bankroll paths and development benchmark; representative replay pending |
| M9 migration | BLOCKED | locked/prospective evidence and representative production-vs-shadow comparisons unavailable |

## Current functional shadow result

The checked-in development run normalized 19 current artifacts, admitted one
independent single, rejected 18 candidates with explicit reasons, and produced
no 2/3/4 ticket because only one candidate survived the safe gate. This is a
functional contract result, not a statistical certification.

## Explicit evidence limitations

* No unified mechanism is `CERTIFIED`.
* Existing archived slates do not contain complete point-in-time quotes and
  calibration state for every requested market; those rows remain
  `UNAVAILABLE` rather than reconstructed from current prices.
* Team-hit and pitch/inning/PA capabilities remain blocked as stated in the
  matrix. Interfaces, identity checks, and deterministic settlement hooks are
  present, but no model probability is fabricated.
* The legacy daily publisher is still authoritative and is the rollback path.
