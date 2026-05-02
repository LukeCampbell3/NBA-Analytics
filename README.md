# Multi-Sport Analytics Workspace

This repository is organized by sport so NBA, MLB, and NFL can evolve independently without path collisions, while still shipping through one shared landing page.

## Repository Structure

```text
sports/
  site/
    web/                   # Shared landing page source
    pipeline/              # Multi-sport build + local serving scripts
  nba/
    web/                   # NBA web app source
    pipeline/              # NBA web/data build + local serving scripts
    predictions/
      Player-Predictor/    # NBA market prediction engine + model artifacts
    tests/
  mlb/                     # MLB scaffold
  nfl/                     # NFL scaffold
```

## Multi-Sport Quick Start

1. Run the shared daily prediction refresh:

```bash
python sports/site/pipeline/run_daily_predictions.py
```

This command now checks local time and runs once the local clock has passed `2:00 AM` by default whenever the current-day payloads are stale or missing. At that point it refreshes NBA predictions, generates a fresh MLB raw pool when processed MLB data is available, tightens the MLB board for publication, and rebuilds the unified static site into `dist/`.

For a manual run outside the 2:00 AM window:

```bash
python sports/site/pipeline/run_daily_predictions.py --force-run
```

2. Build the shared site only:

```bash
python sports/site/pipeline/build_static_site.py
```

3. Serve the shared site locally:

```bash
python sports/site/pipeline/serve_web.py
```

This gives you a landing page at `/` and sport workspaces like `/nba/`, `/mlb/`, and `/nfl/`.
The combined static output is written to `dist/` at the repo root.

## Odds API Key

For local market fetches, you can store your Odds API key in a repo-local `config.local.yaml` file instead of setting a shell environment variable each time:

```yaml
odds_api:
  api_key: "paste-your-key-here"
```

The fetchers check in this order:

1. `--api-key`
2. `THE_ODDS_API_KEY` or `ODDS_API_KEY`
3. `config.local.yaml`
4. `.env.local` or `.env`

For MLB, the market fetch now pulls live player prop tables from RotoWire's MLB props page and writes them into the same normalized snapshot contract the prediction pipeline already consumes.

## NBA Quick Start

1. Build NBA web payloads:

```bash
python sports/nba/pipeline/prepare_web_data.py
python sports/nba/pipeline/prepare_web_college_data.py
```

2. Build NBA static site bundle:

```bash
python sports/nba/pipeline/build_static_site.py
```

3. Serve NBA site locally:

```bash
python sports/nba/pipeline/serve_web.py
```

4. Run NBA prediction pipeline only:

```bash
python sports/nba/predictions/Player-Predictor/scripts/run_daily_market_pipeline.py
```

For the shared published site, prefer `python sports/site/pipeline/run_daily_predictions.py` so both NBA and MLB payloads refresh together and `dist/` stays in sync.

See `sports/nba/README.md` for full NBA pipeline details.
