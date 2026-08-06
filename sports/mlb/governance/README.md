# MLB Policy Governance

This directory governs MLB betting decisions as versioned policies. It does not certify individual player props or model probabilities.

## Evidence flow

```text
complete market snapshot
  -> immutable model-feature snapshot
  -> frozen bounded policy family
  -> locked validation
  -> prospective shadow
  -> policy certificate
  -> candidate authorization
  -> continued monitoring
```

The current policies are `POLICY_DEVELOPMENT`. They have no active prospective certificate, cannot authorize a candidate, and cannot enable staking.

## Evidence labels

- `FULL_SLATE_SNAPSHOT`: all provider observations available at one acquisition time, including rejected rows, are retained with content hashes.
- `SELECTED_ONLY_HISTORY`: historical rows were filtered by an earlier selector.
- `FULL_POLICY_REPLAY_UNAVAILABLE`: a materially different selector cannot be certified from selected-only history.

## Daily evidence contract

Locked and prospective return rows must identify `slate_id`, immutable `snapshot_id`, exact `policy_version`
and `policy_digest`, evidence partition, complete-slate capture label, and pre-event decision-freeze time. There
must be exactly one row per daily slate. CSV booleans are parsed strictly; missing, ambiguous, duplicate, or
cross-policy evidence is rejected before inference.

## Statistical methods

- Bounded finite policy families use Learn-then-Test or an explicitly multiplicity-controlled held-out test.
- Prospective daily returns use a named confidence-sequence construction with its assumptions recorded in the certificate.
- The initial `HOEFFDING_UNION_BOUND_V1` monitor is deliberately conservative. It allocates error over all sample sizes and requires a passed dependence stress check.
- Original Conformal Risk Control is reserved for bounded monotone losses. It is not used for oscillating ROI policy selection.
- Off-policy evaluation is disabled unless logging propensities and overlap are explicitly available.

Primary references:

- [Learn then Test](https://arxiv.org/abs/2110.01052)
- [Time-uniform confidence sequences](https://doi.org/10.1214/20-AOS1991)
- [Selective classification and risk-coverage](https://jmlr.csail.mit.edu/papers/v11/el-yaniv10a.html)
- [Off-policy confidence sequences](https://proceedings.mlr.press/v139/karampatziakis21a.html)
- [Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
- [Non-monotonic CRC over finite grids](https://arxiv.org/abs/2604.01502)
- [Stability-based non-monotonic CRC](https://arxiv.org/abs/2602.20151)

## Blocking state

Production authorization remains blocked until all of the following exist for an exact policy version:

1. Frozen deterministic rules and implementation digest.
2. Non-overlapping locked-validation and prospective periods.
3. Successful full-slate replay for locked validation.
4. Prospective executable prices and daily settlements.
5. Return LCB above the predeclared deployment margin.
6. Minimum action days, selections, eligible days, and coverage.
7. Frozen support and shift gates passing.
8. For parlays, a validated downside/CVaR method in addition to bounded payout and losing-slate gates.

Showing a row in the shadow candidate pool is not candidate authorization.
