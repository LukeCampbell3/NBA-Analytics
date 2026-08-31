# Unified MLB production promotion report

Final state: **UNIFIED_MLB_SHADOW_ONLY**

## 1–5. Repository, baseline and frozen policy

Repository: `LukeCampbell3/NBA-Analytics`. Production baseline:
`static-deployment@c42bf2c1579d140b72efc5597fb9d074834ddfb4`. Frozen policy and
initial implementation: `7c56729a9914eb9f903edffe9ca58b1a0a749ad4`.
Canonical policy hash:
`5f8b247e7781717ffb39a01f581dd36c7466f9da061aa7519c5ab6777b73b67b`.

The policy manifest records the decision and ticket thresholds, capability
registry, source-code hashes, market-conditioning configuration, uncertainty
semantics, and disabled staking. Operational hardening does not alter these
thresholds.

## 6–9. Data inventory, fidelity, temporal audit and replay

The repository contains 242,425 settled historical pool candidates spanning
156 slates from March 1 through August 5, 2026. Those rows cannot reproduce the
frozen unified selector because they lack its final/usable probability,
uncertainty, lineup/role status, and exact calibration state. They are
`RECONSTRUCTED_WEAK` and diagnostic only.

Immutable Git history preserves one deduplicated `EXACT` pregame unified
candidate, but not its settlement. The locked corpus consequently has 0
eligible settled candidates across 0 slates. No future outcome or present-day
quote was used to fill missing state.

Locked result: `HISTORICAL_VALIDATION_FAIL`.

## 10–20. Statistical and economic results

Calibration, selector discrimination, accepted/rejected controls, singles,
2-leg, 3-leg, 4-leg, concentration, bankroll, baselines and ablations are all
`UNAVAILABLE` for certification because the exact eligible sample is empty.
Metrics are stored as `null`, not zero.

The predeclared gate required at least 20 independent slates, 50 selected
singles per capability and 30 tickets per parlay class, plus calibration,
economic, discrimination, dependency and concentration requirements. Thresholds
were not changed after the inventory result.

## 21–22. Capability states

| Capability | State | Reason |
|---|---|---|
| H, TB, R, RBI, HR, pitcher K | VALIDATION_ONLY | frozen settled replay unavailable |
| Pitcher outs, ML, game total, F5, team total | VALIDATION_ONLY | insufficient unified calibration/evidence |
| Cross-game 2/3/4-leg | VALIDATION_ONLY | no eligible ticket corpus |
| Same-game parlays | SHADOW | no certified common-world ticket corpus and real combined-price evidence |
| Team hits | BLOCKED | coherent trained player/team hit model unavailable |
| Inning markets | BLOCKED | event-state model/history/quotes unavailable |
| PA pitch count | BLOCKED | exact event identity unavailable |

No capability is `CERTIFIED` or `PRODUCTION_ACTIVE`.

## 23. Production build results

The production build was exercised in isolated output directories. The first
smoke test found that the existing explicit allowlist removed
`unified-contract.js`; this was repaired and regression-tested. The rebuilt
page, primary JavaScript, unified contract, unified JSON, and engine manifest
all returned HTTP 200 from a local static server. Web, dist and compatibility
JSON mirrors were byte-identical.

Artifacts now use temporary-file validation, fsync and atomic replacement.
Generation IDs are append-only and idempotent. Older generation timestamps are
rejected. All mutating MLB workflows use one non-canceling concurrency group.

## 24–28. Dark deploy, canary, links, frontend and deployed site

Dark deployment: **NOT AUTHORIZED** because M15 failed. No merge to
`static-deployment` was performed. The live canary, live distribution check,
and unified deployed-site smoke test were therefore not run or mislabeled as
successful.

Frontend failure simulations passed for HTTP 404, malformed JSON, empty JSON,
stale JSON, schema mismatch and timeout. Every path terminates in an explicit
visible state while legacy remains active. Exact FanDuel links require both
market and selection IDs; missing exact links are nonfatal and are not rendered
as exact-market navigation.

The public production site remains on the unchanged legacy baseline.

## 29. CI

Prior frozen implementation CI: GitHub Actions run `33347254031`, successful.
Local promotion validation: **783 MLB tests passed**, frontend syntax and all
six failure-state simulations passed, workflow YAML parsed, and the isolated
static build produced byte-identical public/compatibility unified artifacts.
Promotion branch hardening commit: `f8183b405af3f691d6d4a15063aaf5ccc3045505`.
Remote CI is recorded after this report-triggering commit completes; it does not
override the failed historical gate.

## 30–31. Authority and rollback

Active engine: `legacy`. Unified authority: false. The immediate rollback
reference is the untouched production baseline
`c42bf2c1579d140b72efc5597fb9d074834ddfb4` plus the existing legacy artifact
path. A production rollback tag was intentionally not created because the
sequence never reached the authorized pre-merge M21 boundary.

## 32. Known limitations

The repository must prospectively preserve the unified policy hash, source
snapshots, final/usable probability, uncertainty, identity/role/lineup state,
exact quote and timestamp, candidates and rejections, tickets, and later
settlement. Accumulation is based on independent slates/effective sample size,
not elapsed days.

## 33. Final production decision

**UNIFIED_MLB_SHADOW_ONLY**

The implementation is operationally hardened but did not earn statistical
promotion. Proceeding to dark merge or activation would violate the declared
state machine and the user's no-fabrication/no-certification-from-tests rules.
Production remains fully operational on legacy with no interruption.
