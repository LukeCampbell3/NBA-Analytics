# MLB Sequential PA Model Validation

Run date: `2026-09-05`  
Model: `sequential_pa_contact_model_v1`  
Authority: `NEGATIVE_AUTHORITY_UNTIL_INDEPENDENT_ADVANCED_MODEL_CALIBRATION`

## Data

- Baseball Savant / Statcast: **UNKNOWN**; coverage `None` / `None` active-slate entities.
- FanGraphs-compatible pitching: **UNKNOWN**; fields `none reported`.
- Effective/as-of cutoff: `None`; fetched `None`.
- Profiles: `None` batters, `None` pitchers, `None` direct BvP pairs.
- Incremental cache: `{}`.

Raw season-scale pitch data are not committed. Pybaseball caches upstream responses; production keeps bounded dated feature partitions with source/fetch/effective timestamps and MLBAM IDs.

## Architecture

`EXPECTED PA -> batter × pitcher -> K | BB | HBP | HR | NON_HR_CONTACT | OTHER -> contact quality -> average-context expected result -> defense/park residual -> PA result -> state update -> next PA -> full-night H/TB distribution`

PA and AB are separate. HR is exclusive and not double-counted. Hits O0.5 is exactly `1-P(H=0)` and TB O1.5 is directly simulated `P(TB>=2)`. Defense is a zero-centered residual against Statcast average-context expected outcomes; Sprint Speed is not double-counted.

## Historical validation

Evidence: `ROLLING_ORIGIN_HIGH_FIDELITY_DIAGNOSTIC_NOT_CERTIFICATION`; observations `240`.

| Target | Verdict | Legacy Brier | Seq raw Brier | Δ Brier | Legacy log loss | Seq raw log loss | Δ log loss | Seq usable Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H | REGRESSES_PREDICTIVE_PROBABILITY | 0.2611 | 0.2881 | 0.0270 | 0.7203 | 0.7744 | 0.0540 | 0.2700 |
| TB | REGRESSES_PREDICTIVE_PROBABILITY | 0.1482 | 0.1982 | 0.0500 | 0.4449 | 0.5873 | 0.1425 | 0.1776 |

Zero-hit diagnostic: predicted `0.3431`, observed `0.5417`, n=`120`.

No ROI claim is made without exact preserved decision-time sportsbook prices.

## Daily production test

- Evaluated `None`, modeled `None`, blocked `None`.
- Freshness: `None`; statuses `{}`.
- Blocked reasons: `{}`.

## Static publication

Source/dist/protected sequential artifacts all exist: **False**; byte-identical: **False**.

## GitHub Actions

Parent `MLB Sequential PA Validation` run `34006852299`; report run `34006920463`.

## Limitations

- Specific fielder OAA/location assignment remains non-authoritative when reliable fielder/location data are unavailable.
- Direct BvP is strongly shrunk because samples are usually small.
- Rolling-origin replay lacks complete historical pitch-level xFIP/SIERA/OAA snapshots and is diagnostic rather than certification.
- No historical ROI claim is made without exact preserved decision-time prices.
- The sequential model remains negative-authority until independent calibration evidence earns promotion.

## Promotion decision

The advanced model remains **negative-authority**. The historical proxy regressed versus legacy, so it cannot raise public H/TB confidence or force picks. It may only veto/down-rank where fresh advanced evidence and the existing integrity/lineup/quote gates all pass.
