# Sports Site Hub

This folder contains the shared landing page and build/serve scripts for the multi-sport site.

## Commands

Build the combined site:

```bash
python sports/site/pipeline/build_static_site.py
```

Run the shared daily predictor refresh for the published site:

```bash
python sports/site/pipeline/run_daily_predictions.py
```

That command now checks local time and runs only at `2:00 AM` by default. When the schedule gate passes, it refreshes the NBA board, tightens and exports the latest MLB board, then rebuilds the unified static bundle.

For a manual refresh outside the scheduled window:

```bash
python sports/site/pipeline/run_daily_predictions.py --force-run
```

Serve the built site locally:

```bash
python sports/site/pipeline/serve_web.py
```

## How It Works

- `web/`: landing page source for `/`
- `pipeline/build_static_site.py`: copies the landing page and mounts each `sports/*/web/` site under its own route
- `pipeline/run_daily_predictions.py`: shared daily predictor entrypoint for NBA + MLB + dist rebuild
- `pipeline/serve_web.py`: serves the built site from the repo-root `dist/`

Each sport can publish its own `site.json` metadata file so the landing page can describe it without hardcoding every card.

## Output Directory

The unified deployable static bundle now defaults to:

```text
dist/
```

That folder contains the landing page plus every published sport page and asset, so it can be deployed directly as a static site.
