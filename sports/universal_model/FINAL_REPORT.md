# FINAL REPORT — DRM-Guided Universal Multi-Sport Transformer/MoE

All numbers in this report are read directly from the committed JSON
artifacts under `manifests/` and `reports/` (linked inline). Nothing here
is estimated or extrapolated beyond what those files record.

## STATUS

Implemented, trained, and evaluated end-to-end on the two sports this
repository has genuine settled-outcome training data for (MLB, NFL).
**Not** validated as a 5-sport universal system — three of the five
target sports (NBA, golf, F1) have real adapters but no usable historical
training data in this repository (see SPORTS EXCLUDED). Final label:
**`DENSE_SHARED_MODEL_SUFFICIENT`** (see FINAL RESEARCH DECISION).

## REPOSITORIES INSPECTED

- **`LukeCampbell3/NBA-Analytics`** (this repo): fully inspected —
  `sports/`, `sports/shared/`, `sports/validation/`, `sports/{mlb,nba,nfl,f1,golf}/`,
  existing prediction datasets, feature generation, odds/market ingestion,
  validation protocols, historical partitions, existing model code,
  settlement/betting outputs. See `reports/INVENTORY.md`.
- **`LukeCampbell3/drm-linux-kernel-coder`**: **NOT inspected.** Read
  access was requested via this session's repo-attach tool and explicitly
  denied earlier in this session. The DRM structural controller
  (`drm_controller/`) is therefore implemented from the architectural
  principles as described in the mission specification text itself
  (OBSERVE/DERIVE/COMMIT, provisional/permanent structure, bounded
  mutation vocabulary, function-preserving expert birth, deferred
  consolidation), not from that repository's actual source. This is a
  real limitation, carried forward rather than hidden.

## SPORTS INCLUDED

| sport | rows | events | seasons | targets | usable odds coverage |
|---|---|---|---|---|---|
| MLB | 242,425 | 2,034 | 2026 (Mar 1 – Aug 5) | H, TB, R, HR, RBI, K, ER | 4,390/242,425 rows (1.8%) have a real quoted American price; concentrated 2026-04-28..06-28 |
| NFL | 5,328 | 544 | 2025 (weeks 1–18) | passing, rushing, receiving (fantasy points) | none (no market line in source data) |

## SPORTS EXCLUDED

| sport | reason |
|---|---|
| NBA | 81 raw ESPN team-game files exist, but no compiled settled per-observation outcome ledger. The one candidate file (`simulation_backtest_2025_preseason_rows.csv`) has 0 data rows (header only, verified directly). |
| Golf | Only 17 current-tournament ESPN leaderboard snapshots exist; no historical settled per-golfer outcome ledger. |
| F1 | No `sports/f1/data/raw` directory exists at all; `predictions/data_source.py` fetches live with no persisted archive. |

All three have real, working `SportAdapter` implementations (`adapters/{nba,golf,f1}.py`) that honestly report `sufficient_for_training=False` with the measured reason above, rather than being forced into training with fabricated rows.

## UNIVERSAL FEATURE SCHEMA

`data/schema.py`: `UniversalEvent` (32 fields — identity, temporal, market,
target, settlement, provenance) + `UniversalFeature` (two-level: `namespace`
= "universal" or a sport code, `semantic_family`, typed value + explicit
missing flag). `schema_hash()` = `435e420869d72bd6`.

## FEATURE COUNTS

From `manifests/feature_registry.json` (49 real source columns audited,
21 allowed for training):

| category | count |
|---|---|
| UNIVERSAL (allowed) | 7 |
| SPORT_SPECIFIC (allowed) | 2 |
| MARKET (allowed) | 12 |
| TARGET | 5 |
| IDENTIFIER_ONLY (rejected, not a leak — just identity) | 11 |
| POSTGAME_FORBIDDEN (rejected — leakage risk) | 3 |
| UNUSABLE (rejected — circular with an existing incumbent model's own output) | 9 |
| numeric (of the 21 allowed) | 11 |
| categorical (of the 21 allowed) | 10 |

The 9 UNUSABLE columns (MLB `Prediction`/`Edge`/`Model_Selected`/
`Model_Members`/`Model_Val_MAE`/`Model_Val_RMSE`; NFL `prediction`/
`current_prediction`/`challenger_prediction`) are an existing per-sport
model's own output, excluded to avoid circularity against the section-52
shadow comparison.

## DATASET

`manifests/universal_dataset_manifest.json`: 247,753 rows / 2,578 events,
Parquet shards partitioned by sport/season under
`manifests/dataset/{sport=mlb,sport=nfl}/season=*.parquet` (6.2 MB total).
`schema_hash=435e420869d72bd6`, `feature_registry_hash=9d2d7a76e3bed63c`.

## SPLITS

Chronological, event-date granularity, per `manifests/split_manifest.json`:

| sport | DERIVE | SELECT | TEST |
|---|---|---|---|
| MLB (per-sport) | 2026-03-02..06-19 (169,315 rows / 1,429 events) | 06-20..07-11 (36,056 rows / 298 events) | 07-12..08-06 (37,054 rows / 307 events) |
| NFL (per-sport) | weeks 1–11 (3,478 rows / 356 events) | weeks 12–13 (924 rows / 92 events) | weeks 14–16 (926 rows / 96 events) |

Global (pooled) split: NFL's 2025 season fully predates MLB's 2026 season,
so the single global cutover places 100% of NFL rows in DERIVE — a real
property of the data, documented in the manifest rather than hidden;
NFL's own per-sport split (above) is unaffected.

## LEAKAGE AUDIT

**PASS.** `audit_no_cross_split_event_leakage` found 0 events split across
a DERIVE/SELECT/TEST boundary, for both the per-sport and global splits
(`split_manifest.json["leakage_audit"]["pass"] == true`). Adapter-level
`validate_timestamps`/`validate_provenance` also ran clean on all 247,753
compiled rows (0 violations) before compilation.

## MODEL ARCHITECTURES

All three share one `UniversalTransformerStem` (hidden_dim=96, 4 heads, 4
blocks, dropout 0.1) and one `FeatureTokenizer` (8 typed tokens:
SPORT/ENTITY/ROLE/OPPORTUNITY/TEMPORAL/MARKET/UNCERTAINTY/TARGET) — only
the FFN sublayer in the last 2 of 4 blocks differs:

| | blocks 1-2 | blocks 3-4 |
|---|---|---|
| dense | Dense FFN | Dense FFN |
| Switch Top-1 | Dense FFN | Switch (top_k=1, 8 experts) |
| Top-2 MoE | Dense FFN | Top-2 MoE (top_k=2, 8 experts) |
| DRM-MoE (final) | Dense FFN | Top-2 MoE, 9 experts (1 committed birth survived; see DRM STRUCTURAL CHANGES) |

Reduced scale, disclosed: CPU-only, no GPU. hidden_dim=96 is well below
the spec's suggested 384–768 range — a deliberate, disclosed reduction
(spec section 59), not an oversight.

## PARAMETERS

| model | total params | active params/token | checkpoint size |
|---|---|---|---|
| dense_baseline | 1,241,858 | 1,241,666 | 15.0 MB |
| switch_baseline | 2,282,322 | 1,243,218 | 27.5 MB |
| top2_moe | 2,282,322 | 1,391,634 | 27.5 MB |
| drm_final | 2,430,932 | 1,391,828 | 9.75 MB* |

\* smaller because it was saved without optimizer state (a disclosed
reproducibility gap — see REMAINING LIMITATIONS).

FLOPs/example were not separately profiled (no `fvcore`/`ptflops` in this
environment); active-parameter accounting above is the used proxy, per
spec section 42's own framing ("quality per active parameter").

## TRAINING

Hardware: 4 CPU cores, 15 GB RAM, **no GPU**
(`torch.cuda.is_available() == False`). Precision: FP32 (no AMP available
without CUDA). `torch==2.13.0+cpu`, installed for this build.

| model | steps | batch | wall time | throughput |
|---|---|---|---|---|
| dense_baseline | 3,000 | 64 | 200.6 s | 957 examples/sec |
| switch_baseline | 3,000 | 64 | 305.0 s | 629 examples/sec |
| top2_moe | 3,000 | 64 | 319.8 s | 600 examples/sec |

Temperature sport sampling, α=0.5 (spec section 12, never tuned on
TEST/SELECT): gave NFL ~12.5% effective training-batch share from ~2.2%
of raw rows (`reports/dense_baseline_results.json["sampler_effective_contribution"]`).

## DENSE BASELINE RESULTS

SELECT (final, step 3000): brier=0.1884, log_loss=0.5570, auc=0.7010,
ece=0.0122. TEST (frozen, touched once): brier=0.1863, log_loss=0.5518,
auc=0.7007, ece=0.0332. (`reports/dense_baseline_results.json`, `reports/test_results.json`)

## SWITCH RESULTS

SELECT: brier=0.1911, log_loss=0.5637, auc=0.6924. TEST: brier=0.1891,
log_loss=0.5596, auc=0.6928. (`reports/switch_baseline_results.json`, `reports/test_results.json`)

## TOP-2 MOE RESULTS

SELECT: brier=0.1904, log_loss=0.5620, auc=0.6956. TEST: brier=0.1888,
log_loss=0.5587, auc=0.6952. (`reports/top2_moe_results.json`, `reports/test_results.json`)

**Honest reading at this scale:** the dense baseline beats both sparse
configs on every classification metric, on both SELECT and TEST. The MoE
thesis's basic precondition (active ≪ total params) held (see PARAMETERS),
but that structural property did not translate into a quality win here.

## DRM STRUCTURAL CHANGES

Bounded, 4-of-8 spec-section-26 tiers implemented (`parameter_adaptation`,
`router_repair`, `expert_birth`, `shared_width_expansion`; the other 4 —
expert width/local repair, expert merge/split, additional MoE layer, added
temporal/state capacity — are explicitly deferred, see
`drm_controller/mutations.py` docstring). Thresholds for OBSERVE
(brier>0.15, ece>0.02) are this build's own admission criteria, set below
the achieved top2_moe baseline so the mechanism had a real residual to act
on rather than idling (disclosed in `drm_controller/residuals.py`).

Real run, `reports/drm_mutation_history.json`: 3 cycles, 6 attempts.

| # | tier | status | reason |
|---|---|---|---|
| 1 | parameter_adaptation | **PERMANENT** | J improved 0.5859→0.5830 |
| 2 | expert_birth | REJECTED | J worsened 0.5830→0.5835 |
| 3 | parameter_adaptation | REJECTED | J worsened 0.5830→0.5857 |
| 4 | expert_birth | **PERMANENT** | J improved 0.5830→0.5824 |
| 5 | parameter_adaptation | **PERMANENT** | J improved 0.5824→0.5823 |
| 6 | expert_birth | REJECTED | J worsened 0.5823→0.5839 |

**Proposed: 6. Committed: 3. Rejected: 3.** `router_repair` never fired —
routing entropy never dropped enough to register a collapse residual (a
real finding: this build's routing did not collapse). Every rejection was
rolled back exactly (verified by `tests/test_universal_model.py::test_S`/`test_V`).
Total active-param growth from commits: 194 (two committed expert births
each added one router-gate row: 96+1 weights ×2 layers = 194 — top-k
stayed fixed at 2, so per-token active FFN compute did not grow, only
router capacity did, which is the correct MoE semantics).

**Bug found and fixed while running this for real** (3 separate stale
"how many experts do I have" counters existed across `nn.Linear` metadata,
`ExpertBank`/`Router` attributes, and the FFN layer's own attribute — see
git history for the three fix commits) — each was caught by an actual
rollback failing, not by inspection, and each is now covered by a
regression test.

## FINAL DRM-MOE ARCHITECTURE

`manifests/drm_final_config.json`: hidden_dim=96, 4 heads, blocks
`[dense, dense, top2_moe, top2_moe]`, **9 experts** (grew from 8 via one
committed birth), top_k=2. Total params 2,430,932, active params/token
1,391,828.

## EXPERT SPECIALIZATION

Not run as a full semantic-family breakdown: this dataset's real feature
set (21 allowed columns, mostly market/identity/recency) does not yet
carry the rich per-sport semantic families (opportunity, interaction,
environment, etc.) the spec's ideal expert-specialization analysis
assumes — there is not enough real, distinct semantic signal in this
repository's current features to meaningfully attribute expert
specialization to semantic family vs. noise. What **was** measured: routing
entropy stayed healthy throughout (no collapse residual ever fired — see
DRM STRUCTURAL CHANGES), so the specific anti-pattern the spec warns about
(one expert per sport) was not observed, but this is a weaker claim than a
positive semantic-specialization finding.

## MARKET-ONLY COMPARISON

TEST has **zero** priced rows (real market-price capture in this dataset
spans only 2026-04-28..06-28, entirely inside DERIVE/SELECT — see
`reports/market_only_baseline.json`). Computed on SELECT's 2,709-row real-priced
subset (7.5% of SELECT) instead, disclosed as such: single-sided implied
probability (not vig-adjusted — no simultaneous two-sided price exists in
this data) gets brier=0.1884, log_loss=0.5531, auc=0.7021 — closely
matching, and on AUC fractionally beating, this build's own dense/DRM
results. The market is already hard to beat on the subset where a real
price exists.

## WITHIN-SPORT RESULTS

MLB per-sport TEST (dense): brier=0.1863. NFL per-sport TEST: no
classification metric (no market line to build `y_over` from; regression
only). Both from `reports/test_results.json`.

## MACRO CROSS-SPORT RESULTS

Classification macro-average reduces to MLB alone (NFL contributes 0 rows
to the classification head by construction, not by exclusion). Regression
macro (unweighted mean of per-sport MAE, dense baseline, SELECT): MLB
0.6476, NFL 0.5317 → macro MAE 0.5897, vs. micro (pooled) MAE 0.6447 —
macro is *better* than micro here because NFL (fewer, easier-to-predict
observations under this simple regression head) pulls the unweighted
average down; reported both ways per spec section 44 rather than picking
the flattering one.

## LEAVE-ONE-SPORT-OUT RESULTS

2-sport scope (mlb, nfl only — the only two with usable data; disclosed
limitation, not the full 5-sport version). `reports/leave_one_sport_out_results.json`
+ `reports/negative_transfer_audit.json`:

- MLB solo-only: brier=0.18847, MAE=0.6630. Pooled model on MLB:
  brier=0.18841, MAE=0.6476. **No negative transfer**; pooled model is
  marginally better on both metrics.
- NFL solo-only: MAE=0.5889. Pooled model on NFL: MAE=0.5317 (**9.7%
  better**).

## SMALL-DATA TRANSFER RESULTS

NFL solo-only, DERIVE fraction ∈ {1.0, 0.5, 0.25, 0.1}
(`reports/small_data_transfer_results.json`): MAE 0.5828 → 0.6520 → 0.7158
→ 0.6611. Full data is clearly best; the 25%→10% ordering is noisy (a
single run on 870–3,478 rows, no repeated seeds — too small to smooth).
Combined with the negative-transfer result above (pooled MAE 0.5317 beats
even NFL's own 100%-data solo MAE of 0.5828), this is genuine, if
narrowly-scoped, support for the core small-data transfer hypothesis.

## NEGATIVE TRANSFER AUDIT

See LEAVE-ONE-SPORT-OUT RESULTS above — `negative_transfer: false` for
MLB (the only sport with a classification metric to check); NFL's
regression also improved under pooling. No negative transfer detected in
either sport measured.

## CALIBRATION RESULTS

`reports/calibration_report.json`: TEST ECE (drm_final) = 0.0319. Priced
subset: 0 rows (see MARKET-ONLY COMPARISON — TEST has no real prices).
Full reliability curve (10 bins) is in the JSON; well-behaved, no bin with
>15 percentage points of predicted-vs-empirical gap at n≥200.

## COMPUTE EFFICIENCY

`reports/compute_benchmark.json`. Single-example inference latency: dense
1.71 ms, switch 2.61 ms, top2_moe 2.65 ms, drm_final 3.37 ms. **The sparse
configs are NOT faster than dense here** — disclosed and expected: this
implementation evaluates all experts and masks the unselected ones (no
custom sparse-dispatch kernel at this scale, see `model/router.py`
docstring), so wall-clock cannot show the theoretical sparse advantage;
only the analytic active-parameter accounting (PARAMETERS, above) does.
Dataset compile: 247,753 rows in 19.3 s (12,852 rows/sec).

## ABLATIONS

Per `reports/ablation_report.json` (A/B/C on TEST, input-masking on
drm_final; F is a real retrain — see module docstrings for why D/E/G/H/I
are references to already-reported runs, not separate ablation executions):

| ablation | brier | Δ vs. baseline (0.1869) |
|---|---|---|
| baseline (drm_final) | 0.1869 | — |
| A: sport identity removed | 0.1869 | 0 (**disclosed no-op** — MLB is sport_id=0 already, and NFL contributes 0 classification rows, so this ablation could not change anything for this metric) |
| B: role features removed | 0.1886 | +0.0017 (small) |
| C: market prior removed | 0.2010 | **+0.0141 (largest)** — market/line is this model's single biggest real signal source |
| D: MoE→dense | — | see TOP-2 MOE RESULTS vs. DENSE BASELINE RESULTS: dense wins by 0.0025 brier |
| E: Top-1 vs Top-2 | — | see SWITCH vs. TOP-2 MOE RESULTS: Top-2 wins by 0.0003 brier (negligible) |
| F: router balance loss removed | 0.1902 (SELECT) | ~0 vs. top2_moe's 0.1904 SELECT — negligible |
| G: DRM disabled | — | = TOP-2 MOE RESULTS (pre-DRM) |
| H: DRM expert-birth only | not run separately (the real run's escalation order interleaves parameter_adaptation; isolating birth-only was not executed — disclosed gap) |
| I: full bounded DRM | — | = FINAL DRM-MOE ARCHITECTURE / DRM STRUCTURAL CHANGES |

## TEST RESULTS

Touched exactly once, in one script execution (`validation/run_full_validation.py`), `reports/test_results.json`:

| model | brier | log_loss | auc | ece |
|---|---|---|---|---|
| dense_baseline | **0.1863** | **0.5518** | **0.7007** | 0.0332 |
| switch_baseline | 0.1891 | 0.5596 | 0.6928 | 0.0482 |
| top2_moe | 0.1888 | 0.5587 | 0.6952 | 0.0502 |
| drm_final | 0.1869 | 0.5533 | 0.6985 | 0.0319 |

## EXISTING REGRESSION SUITE

`reports/regression_run.json`. Confirmed via `git diff origin/static-deployment...HEAD --stat`
that this entire build touched only `.gitignore` and `sports/universal_model/` —
zero files under any existing sport's directory. 592 existing tests
passed; 6 failures + 3 collection errors, **all pre-existing and unrelated**:
a frontend-copy drift in `sports/mlb/tests` from before this branch
started, a missing `h5py` module in this environment
(`sports/nba/tests`), and a missing trained `.joblib` artifact
(`sports/nfl/tests`). Zero impact from this build.

## DAILY UNIVERSAL INFERENCE

`python -m sports.universal_model.inference.predict --sport mlb --date 2026-07-15`
verified for real (see git history) — real predictions from `drm_final.pt`.
`tests/test_universal_model.py::test_X` confirms the identical checkpoint
path executes for both `mlb` and `nfl` sport args (structural
same-checkpoint proof, per spec section 39).

## SHADOW INTEGRATION STATUS

`inference/shadow.py` joins the existing MLB `daily_prediction_pool` CSV
against universal-model output on `(player_id, target, line)`. Real run,
2026-04-05: 2 rows joined (1 direction agreement, 1 disagreement); 
2026-04-10: 0 rows joined. **Disclosed limitation:** the live daily-pool
pipeline and the historical backtest ledger this adapter reads from are
two independently-run pipelines with low real key-overlap on any given
date — the mechanism works end-to-end but daily coverage is small in this
dataset. Never touches `PolicyStatus`/`G_C`/`G_L`/`G_V` (verified: this
package imports nothing from `sports.mlb.research.parlay_certification_v2`).

## FINAL RESEARCH DECISION

**`DENSE_SHARED_MODEL_SUFFICIENT`**

The dense shared Transformer stem beat both the Switch Top-1 and Top-2 MoE
architectures on every TEST classification metric (brier, log_loss, AUC),
at matched attention depth/hidden size/training budget. DRM development
produced a real, auditable, bounded improvement over the undeveloped Top-2
MoE (brier 0.1888→0.1869 on TEST) — genuine incremental value from the DRM
process itself — but did not close the remaining gap to the dense
baseline. Sparse specialization's structural precondition held (active ≪
total params in both Switch and Top-2 MoE) without translating into a
quality advantage at this scale.

This is not a rejection of the MoE/DRM hypothesis in general — it is a
finding scoped honestly to this build's real constraints: ~248K training
rows across 2 sports, hidden_dim=96, 3,000 training steps, CPU-only. The
mission's own framework anticipates exactly this outcome as a legitimate,
pre-authorized result rather than a failure.

Separately and positively: the core small-data multi-sport transfer
hypothesis found real support on the one sport small enough to test it
(NFL) — pooled training beat NFL's own solo-trained model by 9.7% MAE, and
neither sport showed negative transfer from pooling. This is evidence for
the *representation-sharing* half of the mission's core hypothesis even
though the *sparse-specialization* half did not show a benefit here.

## REMAINING LIMITATIONS

1. **DRM reference repository unavailable** — controller built from spec
   text only (see REPOSITORIES INSPECTED).
2. **2 of 5 sports usable for training** — NBA/golf/F1 excluded for real,
   measured data-insufficiency reasons, not by choice.
3. **Reduced scale, CPU-only** — hidden_dim=96 vs. spec's suggested
   384–768; 3,000 training steps; no GPU. Directionally informative, not a
   claim of full-scale results.
4. **Sparse routing has no real dispatch kernel** — wall-clock inference
   latency cannot show the MoE compute advantage at this scale (see
   COMPUTE EFFICIENCY).
5. **`drm_final.pt` checkpoint saved without optimizer state** — fully
   sufficient for inference/evaluation (all TEST/ablation/calibration
   results above used it directly), but not sufficient to resume training
   from exactly where DRM development left off.
6. **Market-only comparison uses SELECT, not TEST** — real price coverage
   in this dataset does not reach the TEST date window at all.
7. **Ablation A (sport identity removed) was a no-op** for the reported
   metric, a vocab-assignment artifact, not a null finding about sport
   identity's importance.
8. **Ablation H (DRM expert-birth-only) was not run as a separate,
   isolated experiment** — the real DRM run's escalation order interleaves
   `parameter_adaptation`, so H's clean isolation is a gap, not evidence
   either way.
9. **No deep expert-specialization-by-semantic-family analysis** — this
   dataset's real feature set does not yet carry enough distinct semantic
   families to support one meaningfully (see EXPERT SPECIALIZATION).
10. **Shadow-integration daily coverage is small** — real but sparse
    join overlap between two independently-run real pipelines.

## NEXT STEP

If more real historical data becomes available for NBA, golf, or F1 (a
compiled, settled, per-observation, dated outcome ledger — not just live
snapshots), the architecture requires zero changes to onboard them (see
`reports/NEW_SPORT_ONBOARDING.md` and the real `test_I` fixture) — only a
new/completed adapter. Independent of new data, the highest-value next
experiment given today's evidence is a controlled scale-up (larger
hidden_dim, more training steps, GPU) specifically on the dense-vs-Top-2-MoE
question, since the result here (dense wins) could plausibly flip at
larger scale/data, which this build's own resource constraints cannot
rule out either way.
