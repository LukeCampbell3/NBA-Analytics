# NBA Analytics Workspace

NBA code is isolated under `sports/nba`, and the shared site now mounts it beneath `/nba/` inside the repo-root `dist/` bundle.

## Layout

- `web/`: NBA frontend source
- `pipeline/prepare_web_data.py`: Build `web/data/cards.json` + `web/data/valuations.json`
- `pipeline/prepare_web_college_data.py`: Build `web/data/college_cards.json` + `web/data/college_valuations.json`
- `pipeline/serve_web.py`: Serve NBA web app locally
- `predictions/Player-Predictor/`: NBA daily market prediction/model stack
- `tests/test_conditional_framework.py`: NBA predictor gate/regression tests

For the combined static site, use `python sports/site/pipeline/build_static_site.py`.

## Common Commands

Build NBA web data:

```bash
python sports/nba/pipeline/prepare_web_data.py
python sports/nba/pipeline/prepare_web_college_data.py
```

Serve the unified site locally:

```bash
python sports/site/pipeline/serve_web.py
```

Run conditional framework tests:

```bash
pytest sports/nba/tests/test_conditional_framework.py
```

Build PAR/PAR-F player metrics:

```bash
python -m nba_cv_normalizer.cli.main build-player-metrics --season 2025-26 --forecast-season 2026-27 --out out/player_metrics --copy-to-web
python -m nba_cv_normalizer.cli.main prove-par-product --metrics-dir out/player_metrics
```

The PAR product docs are in `docs/par_product.md`, and the static leaderboard is
served from `/nba/par.html`.

Run daily market pipeline:

```bash
python sports/nba/predictions/Player-Predictor/scripts/run_daily_market_pipeline.py
```

Rebuild and validate the selected-board confidence calibrator:

```bash
python sports/nba/predictions/Player-Predictor/scripts/train_selected_board_calibrator.py
```

The trainer uses chronological rolling development followed by five locked slate
dates. Production accepts the calibrator only when its frozen policy improves
both Brier score and log loss on that locked period. Raw and calibrated
probabilities remain separate in the frontend payload, and scores outside the
historical target/direction support are marked unsupported.

For the shared published site, the preferred entrypoint is:

```bash
python sports/site/pipeline/run_daily_predictions.py
```

That command checks local time and runs once `2:00 AM` local time has passed whenever the current-day payloads are stale or missing. When it runs, it refreshes NBA, refreshes MLB, and rebuilds the unified `dist/` bundle in one pass. Use `--force-run` for a manual refresh outside the scheduled window.

## Data Paths

- NBA frontend payloads: `sports/nba/web/data/*.json`
- Published NBA static payloads: `dist/nba/data/*.json`
- NBA prediction artifacts: `sports/nba/predictions/Player-Predictor/model/analysis/...`
