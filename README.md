# Multi-Sport Analytics Workspace

This repository is now organized by sport so NBA, MLB, and NFL can evolve independently without path collisions.

## Repository Structure

```text
sports/
  nba/
    web/                   # NBA web app source
    dist/                  # Built static bundle
    pipeline/              # NBA web/data build + local serving scripts
    predictions/
      Player-Predictor/    # NBA market prediction engine + model artifacts
    tests/
  mlb/                     # MLB scaffold
  nfl/                     # NFL scaffold
```

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
