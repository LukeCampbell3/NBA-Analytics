# Sports Site Hub

This folder contains the shared landing page and build/serve scripts for the multi-sport site.

## Commands

Build the public site and compatibility release source:

```bash
python sports/site/pipeline/build_static_site.py
```

Run the shared daily predictor refresh for the published site:

```bash
python sports/site/pipeline/run_daily_predictions.py
```

That command checks local time and runs once `2:00 AM` local time has passed whenever the current-day payloads are stale or missing. It refreshes the prediction boards and rebuilds both outputs, including the public sport routes and prediction payloads.

For a manual refresh outside the scheduled window:

```bash
python sports/site/pipeline/run_daily_predictions.py --force-run
```

Serve the built site locally:

```bash
python sports/site/pipeline/serve_web.py
```

## How It Works

- `web/`: explicitly public landing, pricing, login, payment-return, legal, and presentation assets
- `pipeline/build_static_site.py`: creates the public prediction site and a compatibility release source
- `pipeline/run_daily_predictions.py`: shared daily predictor entrypoint plus public/private rebuild
- `pipeline/serve_web.py`: serves the built site from the repo-root `dist/`

Each sport can publish its own `site.json` metadata file so the landing page can describe it without hardcoding every card.

## Output Directory

The outputs are:

```text
dist/                           # public static artifact, including sport routes
paywall/private-content/app/   # compatibility release source
```

Only `dist/` is deployed to the public static component. Sport prediction pages,
scripts, and data are available below `dist/<sport>/` and can be previewed with
the local static server. The compatibility output remains available for legacy
release tooling.
