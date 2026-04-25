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

1. Build the shared site:

```bash
python sports/site/pipeline/build_static_site.py
```

2. Serve the shared site locally:

```bash
python sports/site/pipeline/serve_web.py
```

This gives you a landing page at `/` and sport workspaces like `/nba/`, `/mlb/`, and `/nfl/`.
The combined static output is written to `dist/` at the repo root.

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

4. Run NBA prediction pipeline:

```bash
python sports/nba/predictions/Player-Predictor/scripts/run_daily_market_pipeline.py
```

See `sports/nba/README.md` for full NBA pipeline details.
