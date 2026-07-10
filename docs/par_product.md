# PAR/PAR-F Player Metrics Product

Canonical identities:

- `par_model_version`: `par_pvg_v0_5`
- `parf_model_version`: `parf_v0_6`
- `points_per_win`: `30.4`
- `PAR = BoxVisiblePAR + ConfirmedHiddenRolePAR + ShrunkProxyPAR - OverlapLeakage`
- `PAR_1000 = (PAR / Minutes) * 1000`
- `WAR_equivalent = PAR / 30.4`
- `PVGScore = 50 + 45 * tanh(PAR_1000 / 210)`

The authoritative model config is `sports/nba/analytics/par/config.py`. Tests,
CLI tools, exports, API routes, and the static page consume that configuration.

## Build Commands

```bash
python -m nba_cv_normalizer.cli.main build-par --season 2025-26 --out out/par/2025-26
python -m nba_cv_normalizer.cli.main build-par-f --season-from 2025-26 --season-to 2026-27 --out out/parf/2026-27
python -m nba_cv_normalizer.cli.main build-player-metrics --season 2025-26 --forecast-season 2026-27 --out out/player_metrics --copy-to-web
python -m nba_cv_normalizer.cli.main prove-par-product --metrics-dir out/player_metrics
python -m nba_cv_normalizer.cli.main validate-par-f --metrics-dir out/player_metrics --out out/player_metrics
```

## Artifacts

The build emits:

- `players.json`
- `player_par_components.json`
- `player_par_atoms.jsonl`
- `player_par_forecasts.json`
- `player_par_forecast_atoms.json`
- `par_leaderboard.json`
- `par_validation.json`
- `par_build_manifest.json`
- `par_product_proof.json`
- `par_model.json`
- `replacement_baselines.json`

The frontend reads static JSON from `sports/nba/web/data` when `--copy-to-web`
is used. It does not calculate PAR on page load.

## Routes

Static routes:

- `/nba/par.html`
- `/nba/par.html?player={player_id}`
- `/nba/par/player/{player_id}` when served by `sports/nba/pipeline/serve_web.py`

API routes:

- `GET /api/par/players`
- `GET /api/par/players/{player_id}`
- `GET /api/par/players/{player_id}/atoms`
- `GET /api/par/players/{player_id}/forecast`
- `GET /api/par/leaderboard`
- `GET /api/par/model`

## Source Governance

The frozen source document is `docs/par_frozen_model.md`.

The evidence-readiness contract is
`sports/nba/analytics/par/evidence_contract.json`.

The current production-ready source is `box_score_direct_v0_5`, which supports
only direct visible atoms. Tracking, proxy, hidden-role, and residual sources are
declared but not production-ready, so they contribute zero PAR and appear as
limited evidence in the UI until upstream evidence readiness is supplied.
