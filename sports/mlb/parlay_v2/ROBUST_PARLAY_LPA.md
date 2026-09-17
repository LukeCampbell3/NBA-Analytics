# Robust MLB parlay valuation + LPA* search (research/shadow)

This layer deliberately separates **probability estimation** from **search**.

```text
PCA edge/support evidence
        -> marginal probability distributions
        -> joint/dependence evidence
        -> robust pair valuation
        -> LPA* combination search
```

LPA* never estimates whether a leg or pair will win. It only selects among pair valuations that already exist.

## Required output per pair

Every pair reports:

- `p_joint`: mean joint win probability.
- `p_joint_lcb`: lower quantile of the joint probability distribution.
- `d_fair = 1 / p_joint`.
- `d_book`: executable combined decimal price.
- `ev_mean = p_joint * d_book - 1`.
- `ev_lcb = p_joint_lcb * d_book - 1`.
- `p_ev_positive`: posterior/bootstrap probability EV is positive.
- `p_loss = 1 - p_joint`.
- `cvar`: lower-tail unit-stake return CVaR.
- mean and conservative Kelly fractions.
- `dependency_burden`: relative departure of joint probability from the marginal-independence reference.
- leg and joint evidence support.
- expected-log-growth of the parlay and the same total exposure split across the two singles.
- `growth_advantage = G_parlay - G_separate_singles`.

## Probability uncertainty

Upstream pipelines should provide bootstrap/posterior samples for each marginal probability. If no samples are supplied, the module can construct a beta approximation from the point estimate and effective support, but such a fallback is **not eligible for `ROBUST_PAIR_CANDIDATE` by default**.

This prevents a high point estimate from masquerading as a high-confidence estimate.

## Dependence

Supported modes are:

1. `INDEPENDENT` — cross-game only; joint probability samples are products of independently re-sampled marginal probability samples.
2. `JOINT_SAMPLES` — an upstream joint model provides direct samples of `P(A & B)`.
3. `COMMON_WORLD` — both legs are settled inside the same simulated/historical worlds. This is the preferred same-game path.
4. `UNSUPPORTED` — no defensible dependency evidence; the pair is rejected.

Same-game pairs can never use `INDEPENDENT` mode.

## Research classifications

`ROBUST_PAIR_CANDIDATE` means the proposal passed the robust risk gates. It is **not** policy certification and does not authorize staking. By default it requires:

- both legs carry the PCA/A* edge evidence flag;
- minimum leg support;
- explicit pipeline probability samples;
- supported dependence treatment;
- positive 10th-percentile EV (`ev_lcb > 0`);
- `P(EV > 0) >= 0.80`;
- mean joint probability at least 0.50;
- loss probability at most 0.50;
- dependency burden within policy limit;
- positive expected-log-growth advantage versus staking the same total exposure on the two singles separately.

`PROBABLE_PAIR` has positive mean EV but misses at least one robust risk gate. It is shadow only.

`LOTTERY_TAIL` has low joint probability and is never treated as a robust pair candidate.

`UNSUPPORTED` has missing/insufficient pricing, dependence, support, or non-positive mean EV.

All thresholds are explicit `RobustParlayPolicy` configuration. They are research defaults, not proven production constants.

## LPA* state graph

For each frozen slate, the search graph is:

```text
START -> LEG::<canonical first leg> -> PAIR::<pair_id> -> GOAL
```

Only classifications permitted by the requested search mode connect to `GOAL`. The pair-to-goal edge cost is a positive monotone transform of the pair's robust utility:

```text
G_parlay - G_separate_singles
  - uncertainty penalty
  - dependency penalty
  - loss-concentration penalty
```

The heuristic is `h=0`, so it is admissible. Search correctness therefore does not depend on a learned model.

When an existing pair's price/probability/dependence evidence changes, only the affected pair-to-goal edge is updated and LPA* repairs the shortest path. A newly appearing pair changes topology and triggers a rebuild.

## Exact-event bridge

`robust_parlay_bridge.py` joins existing `PairCandidate` objects to PCA/A* probability evidence using the full event key:

```text
(player_id, game_id, target, side, line)
```

The bridge does not create missing evidence. Cross-game pairs may use the structural independence mode; same-game pairs with no joint/common-world evidence remain `UNSUPPORTED`. Missing PCA probability samples cannot silently become robust pair candidates.

## CLI

```bash
python -m sports.mlb.parlay_v2.run_robust_parlay_shadow \
  --input-json path/to/robust_pair_inputs.json \
  --output-json path/to/robust_pair_output.json
```

The CLI is research/shadow only, always emits `production_authorized: false`, and reports `robust_search` separately from the broader `shadow_search`.

## Production boundary

This module does **not** change `run_parlay_v2.py`, the current certification state machine, or staking authorization. The word `ROBUST_PAIR_CANDIDATE` intentionally does not mean policy certification. Promotion requires a separate locked chronological validation using exact pregame probability samples, exact executable prices, settlement identity, and common-world/dependence evidence where required. Only `sports/mlb/research/parlay_certification_v2/` can certify the policy under the existing repository governance boundary.
