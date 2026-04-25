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

Run daily market pipeline:

```bash
python sports/nba/predictions/Player-Predictor/scripts/run_daily_market_pipeline.py
```

## Data Paths

- NBA frontend payloads: `sports/nba/web/data/*.json`
- Published NBA static payloads: `dist/nba/data/*.json`
- NBA prediction artifacts: `sports/nba/predictions/Player-Predictor/model/analysis/...`
