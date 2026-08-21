# MLB Path-Conditioned Evidence V1 (shadow-only)

This package is a **scoped port** of `sports/nba/conditional_chain`'s newest
layer — `path_world_evidence.py` / `path_conditioned_backtest.py` /
`path_conditioned_cli.py` — to MLB, plus the sport-agnostic joint
binary-outcome-world machinery those modules sit on (`outcome_worlds.py`,
`proof_trajectory.py`, both copied unchanged: they only touch generic
`candidate_id` / `player` / probability columns, nothing NBA-specific).

## What did *not* get ported

Everything upstream of "already-scored candidate pool" stayed NBA-only:
`frozen_selector.py`, `chain_resolver.py`, `authorization.py`, `freeze.py`,
`confirmation.py`, `core_policy.py`, `dataset.py`, `allocation_path.py`,
`survival_builder.py`, `survival_backtest.py`, `outcome_set_backtest.py`,
`conditional_extension.py`, `synthetic_audit.py`, `binary_path_audit.py`,
`research_replay.py`, `snapshot_ledger.py`. This was a deliberate scope
reduction, not an oversight: instead of re-deriving a `survival_probability`
from raw history the way NBA's `survival_builder.score_recent_regime_candidates`
does, this package takes as **input** an already-scored MLB reservoir row
per candidate leg:

- `survival_probability` — the day-of model prior. In practice this is fed
  from the existing production pipeline's own prior estimate (e.g.
  `estimated_hit_probability` in the published board /
  `sports/mlb/web/data/history/*.json`, or `pick_survival_model.py`'s own
  `survival_probability` output — same field name, independently arrived at).
- `robust_score` — a stable historical base-rate feature (e.g.
  `historical_bucket_win_rate` from the published board), used only as a
  path-model input feature, exactly as NBA's `robust_score` is.

`path_conditioned_backtest.chronological_path_conditioned_replay` only adds
checkpoint path evidence on top of that already-scored pool. It never
re-ranks or re-derives the prior itself, and it never touches the live
production pool's actual selection or publication.

## Checkpoints

NBA's path uses five fixed clock offsets (`T-240/-120/-60/-30/-5` minutes)
backed by a dedicated five-phase snapshot collector. MLB prop odds in this
repository are instead recovered from opportunistically fetched, timestamped
snapshots (`fetched_at_utc` rows under
`sports/mlb/data/raw/market_odds/mlb/**/normalized/*.csv`, plus git-history
recovery via `sports/mlb/scripts/recover_historical_market_snapshots.py`).
There is no dedicated intraday MLB collector today.

`protocol.AllocationPathProtocol.checkpoints_minutes` uses the same
open → intraday → injury/lineup-sensitive → prelock → close phase design as
NBA's own `Player-Predictor/configs/market_snapshot_collection_schedule_v9_6.json`,
adapted to MLB's longer pregame window: `T-1440 / T-360 / T-90 / T-15 / T-2`
minutes before first pitch. Each checkpoint is filled by the nearest
available snapshot within `max_checkpoint_age_minutes` (45m, wider than
NBA's 20m tolerance, since MLB collection is irregular); a leg missing a
usable snapshot at any checkpoint is excluded, never interpolated.

**`share` means something different here than in NBA.** NBA's `share_m*` is
a player's fraction of their team's total projected allocation (only
meaningful because NBA's path is a single `player_points` market). MLB
candidates can carry different prop markets per player per game, so this
package tracks, per `(event, player, market)`: the no-vig implied
probability of the taken side from `over_price`/`under_price` at each
checkpoint (`share_m*`) and the prop line at that checkpoint (`line_m*`).
`path_world_evidence.merge_candidates_with_paths` therefore joins on
`(event_id or event_date, player, market)` — NBA's version only joins on
`(event, player)`, which would silently collide two different props for the
same player in the same game.

## Inputs

Two adapters build the CLI's `--reservoir` / `--path-features` CSVs from
data already produced by the existing MLB pipeline, rather than requiring a
new collector:

- `build_reservoir_from_history.py` reads
  `sports/mlb/web/data/history/*.json` (published board history) and joins
  each settled play against `Player-Predictor/Data-Proc-MLB/<Player>/`
  processed game logs to compute `leg_result` (OVER hits if actual > line,
  UNDER hits if actual < line, push at 0.5 — the same rule
  `sports/mlb/predictions/scripts/settle_mlb_production_shadow.py` uses).
- `build_path_features_from_market_snapshots.py` reads the normalized
  odds-snapshot history and buckets it into the five checkpoints above.

Both adapters skip rows they cannot ground in real data rather than
fabricating values, and both report how many rows they dropped and why.

## Status discipline (same invariants as NBA)

- `path_mode` / `publication_mode` are `shadow_only_until_*` — identical
  posture to NBA's protocol.
- `chronological_path_conditioned_replay(...).report["production_authorized"]`
  is unconditionally `False`.
- `MECHANISM_STATUS` starts at
  `REAL_MLB_PATH_MECHANISM_IMPLEMENTED_INCREMENTAL_VALUE_UNPROVEN` and can
  only become `REAL_MLB_PATH_INCREMENTAL_VALUE_SUPPORTED` once real-path
  evidence beats endpoint-only, shuffled-path, and inverted-path controls on
  paired bootstrap and sign-flip tests over **real, settled MLB history** —
  never on synthetic fixtures.
- `certify_perfect_parlay(...).production_authorized` is unconditionally
  `False`.

This package makes **no claim of real MLB predictive value**. It exists to
measure, on real settled history as it accumulates, whether MLB's pregame
line-movement path carries incremental evidence beyond the pool's existing
day-of score — nothing here changes what the production pipeline selects or
publishes today.
