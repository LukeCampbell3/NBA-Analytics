# NBA Conditional Chain V1.1

This package implements the report's NBA research architecture without changing the existing predictor or publishing policy.

## Frozen contracts

- Selector: `ROBUST_STATE_INTERSECTION_Q25_V1`
- Allocation representation: `NBA_ALLOCATION_PATH_V1_1_FROZEN`
- Chain policy: `NBA_CONDITIONAL_CHAIN_V0_SHADOW`
- Survival policy: `NBA_RECENT_REGIME_SURVIVAL_V1_SHADOW`
- Joint outcome set: `NBA_BINARY_OUTCOME_SET_V1_SHADOW`
- Representation unit: one event/team/market
- Statistical unit: one game event for path confirmation and one final slate decision for parlay evaluation

The selector corrects UNDER probability semantics, uses the prior 20 date-safe outcomes against the current exact line, applies a Jeffreys 10% lower credible bound, and keeps the exact four-leg publication floor from the project report.

The path builder uses player-points allocation at `T-240`, `T-120`, `T-60`, `T-30`, and `T-5` minutes. A quote must be observed no later than its checkpoint and be at most 20 minutes old. Every player coordinate requires at least two independent pricing engines. Pregame team identity is mandatory and cannot be repaired from outcomes.

## Confirmation rule

T3 compares identical `StandardScaler + Ridge(alpha=1)` models. The endpoint model receives closing state only; the path model receives closing state plus trajectory features. Both are evaluated with chronological expanding windows.

Path information passes only at the frozen 20/30/50-event checkpoints when:

```text
one-sided paired-bootstrap LCB(mean endpoint MAE - path MAE) > 0.005
and
one-sided sign-flip p < 0.0167
```

The per-checkpoint alpha is `0.05 / 3` so the frozen 20/30/50-event looks retain a familywise 5% error budget.

Passing T3 does not authorize a parlay. It only opens chain-policy development. The chain resolver remains shadow-only until a separately frozen policy passes prospective validation.

The downstream extension model is trained only on earlier final decisions whose preceding prefix survived. It estimates each extension conditionally, but its evidence remains clustered and reported at one final slate decision, not at the generated-combination level.

## Parlay hit-rate finding

The recovered 53,957-row, 130-date ledger reproduces the report's final research block exactly:

- frozen four-leg control: 13/27 (48.15%);
- frozen control legs: 89/108 (82.41%);
- independence benchmark: `0.8241 ** 4 = 46.13%`;
- every one of the 27 slates contains at least four winners in the frozen top-10 reservoir.

The four-leg result is therefore close to what its marginal leg rate implies. The candidate reservoir contains enough winners, but the static features tested so far do not identify the compatible chain reliably. A date-safe role/allocation reranker improved its calibration block and then regressed to 10/27 on the reported block, so it is rejected.

A same-version rank-reliability policy, frozen only on the 34 earlier published slates, selected ranks 1 and 4 as a two-leg research core. It hit 21/27 (77.78%) on the later block. It remains shadow-only: applying those ranks to the different April predictor produced only 6/17, proving that rank reliability cannot be transferred across model or selector versions. Each exact version must earn its own core policy and prospective certificate.

The replacement survival policy does not use rank or model-version identity as a feature. It adjusts the frozen robust score with a date-safe 30-day market/side posterior, shrunk toward the recent overall hit rate with prior strength 20. It then evaluates the full combination set using an independence reference and a Frechet lower reference. Two legs are the primary board, three legs are a separately measured extension, and four legs are rejected.

Across the 41-slate expanding replay plus the 17-slate predictor-transfer block:

- survival pair: 40/58 (68.97%) versus 36/58 (62.07%) for the ordinary top-two control;
- April transfer pair: 12/17 (70.59%) versus 9/17 for top two and 6/17 for the rejected fixed-rank core;
- three-leg extension: 33/58 (56.90%) versus 29/58 (50.00%), but no transfer-block improvement;
- selective pair at the frozen 0.42 lower-reference floor: 23/31 (74.19%) at 53.45% slate coverage, versus 19/31 (61.29%) for the control on the same action slates.

The selective paired result has `p=0.0625`, so it is promising but not confirmation-grade. These windows and synthetic thresholds have been repeatedly inspected. The policy remains shadow-only and must reproduce prospectively on executable prices before it can replace publication logic.

A hierarchical pair-lift tree, forced game diversification, and a generic residual Gaussian copula were also tested and rejected. The pair tree regressed the earlier holdout, diversification did not transfer, and the strongest pooled relation correlation across 802 prior games was only 0.051. Their results are retained in `artifacts/survival_replay/rejected_joint_models.json`.

The historical Q25 ledger is not executable market evidence. Only 13.0% of the reported control lines are integers or half-points, every row uses an assumed `-110`, and the rows have no book, quote timestamp, or raw-source hash. It is labeled `SYNTHETIC_THRESHOLD_HISTORY` and cannot authorize publication.

Publication now requires all of the following:

- the exact candidate policy version matches an active prospective certificate;
- the path representation has independently passed its incremental-value test;
- every candidate carries a fresh book quote, executable decimal price, timestamp, parser version, and raw-source hash;
- candidate lineup, player, identity, support, feature-cutoff, model-version, and exposure gates pass;
- minimum prospective action-slate, selection, and coverage evidence is satisfied;
- staking remains disabled independently of candidate publication.

This prevents a high synthetic hit rate or a high model score from being presented as a validated betting parlay.

## Exact binary outcome-path layer

The original binary-state theory is preserved, but its claim is made testable. For a reservoir of `M` bets, let the joint settlement world be:

```text
y in {0, 1}^M
```

where `1` is a settled full-payout win and `0` is any result that prevents the full parlay payout. Pushes are therefore mapped to `0` for this specific proof target. At checkpoint `t`, `C_t` is the set of joint worlds still retained by the frozen model and path evidence. Define:

```text
G_t = intersection over y in C_t of {i : y_i = 1}
```

The exact finite-state theorem is:

```text
an n-leg perfect parlay exists inside C_t if and only if |G_t| >= n
```

This is necessary and sufficient, not a greedy approximation. The implementation enumerates all `2^M` worlds for the frozen maximum `M=10`, updates their probabilities with checkpoint-level joint log evidence, constructs a chronological label-powerset APS outcome set, and exhaustively evaluates every requested 2-, 3-, or 4-leg subset. A candidate receives a logical certificate only when no retained world is a counterexample. An empty outcome set forces abstention and can never create a vacuous certificate.

The distinction between existence and identification is essential. If a slate has at least `n` winners, a winning subset exists after settlement. It is identifiable before settlement only if observed information removes every conflicting world. If the same observable endpoint and path can arise under both a parlay-win world and a parlay-loss world, no classifier can perfectly distinguish them; the unresolved conditional entropy is irreducible from those inputs.

The chronological replay used 20 prior slates for calibration and then evaluated 58 slates across the historical and cross-version blocks:

- realized joint-world coverage: 54/58 (93.10%) against a 90% marginal target;
- ex-post winning 2-, 3-, and 4-leg subsets existed in 58/58 reservoirs, with at least five realized winners in every top-10;
- logical 2-, 3-, and 4-leg certificates: 0/58 for every length;
- mean retained set: 430.7 of 1,024 worlds;
- best exhaustive pair frontier: 39/58 (67.24%), exactly tied with ordinary top two;
- best exhaustive triple frontier: 32/58 (55.17%) versus 31/58 for ordinary top three;
- best exhaustive four-leg frontier: 24/58 (41.38%), exactly tied with ordinary top four;
- mean pair proof gap: 229.8 counterexample worlds carrying 33.05% of retained-set mass.

The one-win triple difference is repeatedly inspected research evidence, not validation. More importantly, the exhaustive search does not improve pairs or four-leg chains. That falsifies the idea that another static reordering of current marginal information is enough.

The synthetic mechanism audit then applies shared-state evidence directly to joint worlds. At the replay's final chronological APS threshold, coherent pair evidence reduces the retained set from 13/16 worlds to 3/16 and makes the same two coordinates wins in every retained world. Exact reversal restores the original 13/16 set and removes the certificate. The software also exhaustively verifies the existence theorem across all 255 nonempty outcome sets for three candidates and all three leg counts: 765/765 checks pass. This proves the mechanism and implementation, not NBA accuracy.

The resulting path is:

```text
frozen candidate reservoir
  -> exact joint binary worlds
  -> shared-state checkpoint evidence
  -> chronological conformal outcome set
  -> exhaustive counterexample search
  -> logical shadow certificate or abstention
  -> separate prospective action-conditional risk test
```

Marginal outcome-set coverage is not a promise that a selected parlay wins 90% of the time. Promotion still requires real timestamped paths to show incremental predictive value, followed by a fresh prospective test of the failure rate specifically on action slates. This separation follows the coverage/set construction used in [classification with adaptive coverage](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html), recent [multi-label confidence-set enumeration](https://proceedings.mlr.press/v337/ledaguenel26a.html), and [conformal structured prediction](https://arxiv.org/abs/2410.06296).

## Commands

```powershell
python -m sports.nba.conditional_chain.cli repository-audit `
  --output sports/nba/conditional_chain/artifacts/repository_path_audit.json

python -m sports.nba.conditional_chain.cli build-dataset `
  --quotes path/to/repeated_quotes.csv `
  --outcomes path/to/independent_outcomes.csv `
  --output-dir out/nba_allocation_path_v1_1

python -m sports.nba.conditional_chain.cli confirm `
  --settled-features out/nba_allocation_path_v1_1/allocation_path_settled_player_features.csv `
  --output-dir out/nba_allocation_path_v1_1/confirmation

python -m sports.nba.conditional_chain.cli synthetic-audit `
  --output sports/nba/conditional_chain/artifacts/synthetic_null_power_audit.json

python -m sports.nba.conditional_chain.cli backtest-selector `
  --candidate-pool sports/validation/validation_recent_pool_selector_20260406_20260430_rows.csv `
  --data-proc-dir sports/nba/predictions/Player-Predictor/Data-Proc `
  --output-dir sports/nba/conditional_chain/artifacts/selector_replay

python -m sports.nba.conditional_chain.cli backtest-master `
  --master-ledger C:/Users/jcthi/Code/Predictor/backtest/data/backtest_master_dataset.csv `
  --holdout-start 2026-02-11 `
  --output-dir sports/nba/conditional_chain/artifacts/full_replay

python -m sports.nba.conditional_chain.cli backtest-survival `
  --research-reservoir sports/nba/conditional_chain/artifacts/full_replay/frozen_reservoir_replay.csv `
  --transfer-candidate-pool sports/validation/validation_recent_pool_selector_20260406_20260430_rows.csv `
  --data-proc-dir sports/nba/predictions/Player-Predictor/Data-Proc `
  --warmup-slates 20 `
  --output-dir sports/nba/conditional_chain/artifacts/survival_replay

python -m sports.nba.conditional_chain.cli backtest-outcome-set `
  --research-reservoir sports/nba/conditional_chain/artifacts/full_replay/frozen_reservoir_replay.csv `
  --transfer-reservoir sports/nba/conditional_chain/artifacts/survival_replay/transfer_reservoir_replay.csv `
  --output-dir sports/nba/conditional_chain/artifacts/outcome_set_replay

python -m sports.nba.conditional_chain.cli binary-path-audit `
  --aps-threshold 0.9007125852032632 `
  --output sports/nba/conditional_chain/artifacts/outcome_set_replay/binary_path_sensitivity.json
```

The documented `gs://nba-scraped-data/odds-api/player-props-history` archive is a supported upstream shape once exported to CSV or Parquet, but it returns HTTP 403 without project credentials. The committed repository sequence is audited honestly and is not promoted to confirmation evidence when fixed-checkpoint, identity, or quote-provenance gates fail.
