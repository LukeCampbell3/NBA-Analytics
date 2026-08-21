# MIGRATION — PARLAY_CERTIFICATION_V2

## 1. Audit: old certification/risk/authorization entry points

Searched the whole repo for `production_authorized`, `PRODUCTION_AUTHORIZED`,
certificate/risk-gate logic, and parlay authorization strings.

| Location | What it was | Live in CI/production? | Disposition |
|---|---|---|---|
| `joint_position_builder_v2/risk_gate.py` (`SelectiveRiskCertificate`, `build_selective_risk_certificate`, `gate_and_rank_day`) | Single-endpoint Clopper-Pearson empirical-risk bound; final JOINT POSITION gating | No — manual/local research script only, never wired into any workflow | **Archived** → `joint_position_builder_v2/legacy/risk_gate_v1_ARCHIVED.py` |
| `joint_position_builder_v2/pairs.py` (`build_pair_certificate`) | Pair-specific logical certificate; already required `retained_count > 0` but had no *separate* positive-probability-mass check | No | **Superseded** for new decisions by `world_certificate.build_nonvacuous_world_certificate`; left in place unchanged as a predictive/world-layer helper (still correct for its original narrower claim, still exercised by its own theorem tests) |
| `joint_position_builder_v2/manifest.py` (`PRODUCTION_AUTHORIZED`, `STATUS`) | Ad hoc research status flags | No | Left in place (historical record), annotated `CERTIFICATION_AUTHORITY_SUPERSEDED_BY = "PARLAY_CERTIFICATION_V2"` |
| `joint_position_builder_v2/ablation.py`, `run_development.py` | 2×2 diagnostic ablation built on `risk_gate.py` | No | Left runnable (reproducibility of `reports/`), banner added marking both ARCHIVED/DIAGNOSTIC-ONLY, import path updated to the archived module |
| `sports/parlay_analysis.py`, `sports/mlb/scripts/select_daily_parlay.py` (CONTROL) | ML probability-threshold gates (`min_leg_probability`, `HIT_SURVIVAL_MIN_PROBABILITY`/`_CONSENSUS`), FanDuel betslip building | **Yes** — called from `sports/site/pipeline/run_daily_predictions.py`, which the `mlb-predictions.yml` GitHub Action runs daily | **Out of scope**, audited and explicitly excluded — see §3 below |
| `sports/mlb/conditional_chain/outcome_worlds.py` (`certify_perfect_parlay`, `guaranteed_winner_indices`, `PerfectParlayCertificate`) | Earlier (pre-`joint_position_builder_v2`) logical parlay certificate | Shadow-only job in `mlb-predictions.yml` (`mlb-path-evidence-shadow`), additive, never authorizes anything live | **Audited, left in place** — already guards `outcome_set.world_count > 0` at both `guaranteed_winner_indices` and the `logical_proof` computation, so it does NOT have the vacuous-empty-set bug section 5 targets; `production_authorized` is already hardcoded `False` unconditionally. It predates the simultaneous coverage/loss/return G-process framework and is a sibling system (path-conditioned evidence, not pair-parlay certification) — replacing it is a separate, differently-scoped task, not implied by this one. |

Call-graph check (confirms no other importer of the archived module):
```
grep -rln "risk_gate" sports/          # only ablation.py + its own test, before migration
grep -rn  "PRODUCTION_AUTHORIZED"      # only joint_position_builder_v2/manifest.py (research flag, never read by any pipeline)
```

## 2. New authoritative path

```
sports/mlb/research/parlay_certification_v2/
  eligibility.py        OPERATIONAL ELIGIBILITY (external, immutable, no model/pair fields possible)
  settlement.py          SETTLEMENT (bounded R, explicit status enum, no universal R=D*W-1)
  world_certificate.py   NONVACUOUS_WORLD_CERTIFICATE (nonempty AND positive-mass AND zero counterexamples)
  policy.py               DECISION POLICY (acts/abstains; 1 action/eligible day; price-bound rejection)
  evidence_store.py        Atomic, append-only, idempotent slate-day evidence, one file per policy_version
  anytime_monitor.py       PARLAY_REFERENCE_ANYTIME_CERT_V1 (G_C/G_L/G_V, union-bound Hoeffding radii)
  state_machine.py         Reversible PolicyStatus state machine
  theory.py                 Stationary-oracle documentation only + terminal research labels
  manifest.py               Frozen config, versions, PRODUCTION_AUTHORIZED=False, MIGRATION_SUMMARY
```

Predictive/world-layer code (`joint_position_builder_v2.pairs.CandidatePair`,
`outcome_worlds.build_world_distribution`, etc.) is **reused unchanged** as an
input to `policy.CandidateWager` — it proposes wagers; it never sets `action`
or any authorization flag. Only `policy.select_action_for_day` (driven by
`world_certificate`) sets `A_t`, and only `anytime_monitor` +
`state_machine` may ever advance `PolicyStatus` toward
`SUPPORTED_CURRENT`.

## 3. Why CONTROL (`select_daily_parlay.py`) is out of scope

- It shares no code, vocabulary, or theory with the world-certificate /
  G-process framework this mission upgrades (`grep` for
  `outcome_worlds|build_world_distribution|certify_perfect_parlay` in both
  CONTROL files returns nothing).
- Its authorization outputs are already hard-labeled non-live in the
  payload itself: `"authorization": "shadow_only"`,
  `"shadow_authorization": "diagnostic_only"` — no downstream consumer can
  mistake them for a real, live-money authorization.
- It is a large (~1,350-line), independently-evolved, FanDuel-integrated
  production system (batting-order lookups, latent models, betslip deep
  links). Rewriting its authorization logic as an incidental side effect of
  this migration — without the user reviewing that specific change —
  carries real operational risk to a currently-working live pipeline, for
  a system the mission's own frozen theory (§1–§10) does not describe.
- This research thread has maintained "CONTROL untouched" as an explicit
  boundary since it began (`joint_position_builder_v2/REPORT.md`); this
  migration preserves that boundary rather than silently crossing it.

If a future task explicitly asks to bring CONTROL onto the V2 theory, the
right shape is an adapter: keep `select_daily_parlay.py`'s ML layer as a
PREDICTIVE/WORLD-MODEL input (it already estimates leg probabilities) and
route its output through `policy.py`/`eligibility.py`/`anytime_monitor.py`
instead of its own `min_leg_probability`/consensus gates — but that is a
separate, explicitly-scoped task, not implied by this one.

## 4. Next prospective step

`manifest.STATUS = "DEVELOPMENT"`. To move to
`FROZEN_PROSPECTIVE_INCONCLUSIVE`:
1. Review and explicitly re-affirm (or revise, once, before any prospective
   data is collected) `C_MIN_COVERAGE`/`R_MAX_LOSS_RISK`/`DELTA_MIN_RETURN`/
   `R_MAX_ACCEPTED`/`ALPHA_TOTAL` in `manifest.py`.
2. Wire `policy.select_action_for_day` to a real daily candidate feed
   (reusing `joint_position_builder_v2`'s multi-target universe / pair
   builder as the predictive/world-model input) and a real
   `eligibility.EligibilityInputs` source.
3. Begin appending real `FinalEvidenceRecord`s via `EvidenceStore` as MLB
   slate days settle. Do not touch `TEST_STAMPS`; this is necessarily new,
   forward-only data by construction (an anytime monitor cannot be run
   retroactively on frozen historical evidence without violating its own
   sequential/predictable-mean assumptions).
4. Evaluate `anytime_monitor.evaluate_simultaneous_certificate` at each new
   eligible day; drive `state_machine.next_status` from its
   `fully_supported` field.
