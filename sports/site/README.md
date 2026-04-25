# Sports Site Hub

This folder contains the shared landing page and build/serve scripts for the multi-sport site.

## Commands

Build the combined site:

```bash
python sports/site/pipeline/build_static_site.py
```

Serve the built site locally:

```bash
python sports/site/pipeline/serve_web.py
```

## How It Works

- `web/`: landing page source for `/`
- `pipeline/build_static_site.py`: copies the landing page and mounts each `sports/*/web/` site under its own route
- `pipeline/serve_web.py`: serves the built site from `dist/`

Each sport can publish its own `site.json` metadata file so the landing page can describe it without hardcoding every card.
