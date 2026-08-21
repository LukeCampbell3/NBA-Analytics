# PARLAY_POLICY_V2 — architecture

## Two-stream evidence architecture

This system keeps two structurally separate, never-conflated streams.

### A. PREDICTIVE CALIBRATION STREAM (`parlay_v2/calibration/`)

Purpose: tell today's candidate adapter whether a candidate has enough
*prior, settled* history to be evaluated at all.

- `schema.CalibrationObservation` — one settled predictive event. Exact
  event identity (`player, game, target, side, line, book`) is baked into
  `observation_id`/`row_hash`, so a 0.5-line and a 1.5-line event can
  never collapse into one record (mission section 6/14, same invariant
  `candidate_adapter.py` already enforced for live rows).
- `store.CalibrationStore` — forward-only, append-only, idempotent by
  `observation_id`. `observations_as_of(cutoff)` is the **only** read path
  and returns rows with `calibration_admitted_at` **strictly before**
  `cutoff` — this one method is what makes same-day information
  unavailable to same-day decisions (see "Same-day exclusion" below).
  Refuses to admit an observation before its own settlement is final, and
  refuses to mix `calibration_version`s.
- `snapshot.CalibrationSnapshot` — immutable, content-hashed (`sha256` over
  sorted `observation_id`s + `as_of` + version), reproducible from the
  ledger alone. `assert_snapshot_precedes_decision` enforces
  `calibration_as_of < decision_frozen_at` with a **strict** comparison.
- `support.CandidateSupport` — multidimensional
  (`market_support, line_support, state_support, joint_support,
  recent_support, calibration_error, shift_status`). `joint_support` and
  `shift_status` are honestly `UNESTABLISHED` (no validated research
  exists for either in this repo yet) — `in_support` is therefore always
  `False` today. This is a documented limitation, not a bug: mission
  section 5 explicitly forbids inventing a threshold to make a candidate
  actionable.
- `replay.replay_calibration_as_of` — pure re-derivation from the ledger;
  same ledger state + same `as_of`/buckets ⇒ identical snapshot hash and
  support classification.

The calibration stream may influence which candidates the predictive
layer treats as evaluable. It can **never** certify the policy, move
`PolicyStatus`, or touch `production_authorized`.

### B. POLICY EVIDENCE STREAM (`parlay_certification_v2/evidence_store.py`, unchanged)

Purpose: record exactly what the frozen policy *did* on each
operationally eligible slate, and drive `G_C/G_L/G_V` → `PolicyStatus`.

One `FinalEvidenceRecord` per eligible slate — never per-candidate, never
per-leg, never more than one even if a two-leg parlay settles across two
different game windows (settlement is atomic at the *slate* level, not
per-leg). `parlay_certification_v2/replay.py` (new) replays a frozen
policy's evidence rows into identical cumulative `G` values, anytime
bounds, and `PolicyStatus` transitions, including first-support and
demotion horizons.

## Same-day exclusion / next-day admission

```
day t candidate adapter runs
  -> reads calibration_as_of = timestamp captured BEFORE any candidate work
  -> CalibrationStore.observations_as_of(calibration_as_of)
       returns only rows admitted strictly before that timestamp
  -> day t's own outcomes cannot exist in the ledger yet (they aren't
     settled), so this is automatically enforced, not just by convention
  -> day t decision is frozen (decision_frozen_at > calibration_as_of, asserted)
  -> day t settles
  -> a SEPARATE admission step (not run by run_parlay_v2.py) writes day t's
     settled observations into the ledger with calibration_admitted_at >= settled_at
  -> day t+1's candidate adapter's own calibration_as_of cutoff now includes them
```

`run_parlay_v2.py` never writes to the calibration ledger — it only
reads from it (via `observations_as_of`) and only ever writes to the
*policy* evidence stream, and only after settlement, via the existing
`EvidenceStore` (unchanged). No code path in this package can make day
t's own outcome visible to day t's own decision.

## Prospective freeze boundary

`parlay_certification_v2/prospective_boundary.py` (built previous turn,
reused unchanged): a one-way per-`policy_version` marker file.
`freeze_prospective.py` is the only sanctioned way to activate it —
`python -m sports.mlb.parlay_v2.freeze_prospective --policy <id> --confirm`
verifies a clean relevant git tree, computes `code_hash`/`config_hash`,
and refuses to run if `freeze_ready=False`. **As of this document, no
boundary has been activated for any real policy version** (see
`test_no_boundary_file_exists_for_the_real_policy_version_yet`) —
`freeze_ready` is `False` because `joint_support`/`shift_status` remain
`UNESTABLISHED`. 2026-08-21 (today) therefore remains `DEVELOPMENT_SHADOW`
for every policy version that exists, by construction (no boundary ⇒
`prospective_boundary.is_prospective(...)` is always `False`).

## Program-level alpha budget

`parlay_v2/program_alpha.ProgramAlphaLedger` (built previous turn):
`sum(alpha_policy_k) <= alpha_program`, append-only, idempotent per
`policy_version`, never resets on a failed/demoted policy.
`freeze_prospective.py`'s `config_hash` includes `alpha_program`,
`alpha_total`, `alpha_c/l/v` so a frozen policy's alpha allocation is part
of its immutable, hashed configuration.

## Branch / repository topology

See the final response for the full branch audit. Summary: `main` (the
GitHub default branch) has **no shared git history** with the branch this
entire research program has been developed on
(`feature/mlb-path-conditioned-evidence-v1`) or any of its siblings
(`MLB-dev`, `NFL_dev`, `frontend_dev`, `server-deployment`,
`static-deployment`, `feature/nba-conditional-chain-v1`) — `main`'s recent
commits are authored by a different automated agent ("Codex
<codex@openai.com>") and its tree is a fraction of the size (1,416 vs.
13,243 files; no `sports/mlb/research/`, `conditional_chain/`,
`parlay_v2/`, `governance/` at all). This directly contradicts the
assumption that a normal merge into `main` is possible or safe. No merge
or branch deletion was performed this session — see the final response's
branch-audit section for the full classification and the explicit
decision to escalate rather than guess.
