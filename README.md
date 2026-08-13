# Multi-Sport Analytics Workspace

This repository is organized by sport so NBA, MLB, NFL, and Formula 1 can evolve independently without path collisions, while still shipping through one shared landing page.

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
  f1/                      # Formula 1 race model and market board
```

## Multi-Sport Quick Start

1. Run the shared daily prediction refresh:

```bash
python sports/site/pipeline/run_daily_predictions.py
```

This command checks local time and runs once the local clock has passed `2:00 AM` by default whenever the current-day payloads are stale or missing. It refreshes prediction data and rebuilds two separate outputs: the public shell in `dist/` and protected release source in `paywall/private-content/app/`.

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

This previews only the public shell at `/`. Sport boards and their prediction payloads are deliberately absent from `dist/`; they are delivered through the authenticated Go gateway from private R2.

## Market Data

NBA and MLB production runs scrape their public RotoWire player-props pages.
The fetchers enforce the requested board date, normalize multi-book lines and
prices into the existing market snapshot contract, and require no API key.

## Automated Daily Publication

`.github/workflows/main.yml` runs at 8:17 AM America/New_York and can also be
started manually with a date and sport selection. GitHub requires scheduled
workflow definitions to exist on the repository's default branch (`main`), so
the workflow starts there and explicitly checks out, builds, validates, and
updates the deployable `static-deployment` branch.

The job rejects stale payloads before committing. `dist/` contains only the
public landing, pricing, login, payment-return, legal, catalog metadata, and
shared presentation assets. Prediction payload sources remain available to the
controlled publication workflow but are never copied into the public static
artifact.

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

5. Rebuild the 2026-27 opening-night projection pool:

```bash
python sports/nba/pipeline/build_opening_night_pool.py
```

The opening-night payload is a research-only projection watchlist built from
the existing preseason simulation cards. It does not authorize picks until
current rosters, availability, and authentic two-sided prop lines are attached.

For the shared published site, prefer `python sports/site/pipeline/run_daily_predictions.py` so both NBA and MLB payloads refresh together and both public/private outputs stay in sync.

See `sports/nba/README.md` for full NBA pipeline details.

## NFL Quick Start

Train the leakage-aware player-yardage stack and reproduce its chronological
holdout report:

```bash
python sports/nfl/scripts/train_nfl_predictor.py
```

The static model report is included in the protected release at
`/app/nfl/predictions/` and is not copied to `dist/`. See `sports/nfl/README.md`
for data, model, and evaluation details.

## Formula 1 Quick Start

Build the next-race model board from credential-free public data and market
feeds:

```bash
python -m pip install -r sports/f1/requirements.txt
python sports/f1/scripts/run_f1_daily_predictions.py
python sports/f1/scripts/validate_f1_publication.py
```

See `sports/f1/README.md` for model features, chronological evaluation, free
Polymarket/Kalshi market sourcing, and shadow-publication controls.
