# NBA Analytics Engine

Mechanism-first NBA player valuation and breakout analysis with a local web app.

This project builds player cards, computes value and breakout signals, and serves an interactive site for filtering and deep-dive scenario analysis.

## What The Project Does

1. Creates standardized player cards from raw stat data.
2. Computes player value (including EPM/LEBRON-informed signals where available).
3. Runs breakout/fit analysis with constraints and confidence governance.
4. Prepares a web-ready dataset and serves a local frontend.

## Project Layout

- `src/create_cards.py`: Generate player card JSON files from CSV/Parquet.
- `src/backfill_usage_rates.py`: Recompute/backfill usage rate on existing cards.
- `src/value_players.py`: Value engine (wins-added style valuation, aging, surplus).
- `src/analyze_players.py`: Breakout, portability, sanity checks, and report outputs.
- `prepare_web_data.py`: Consolidate cards + valuations into `web/data`.
- `serve_web.py`: Local multi-page server (`/`, `/about`, etc.).
- `web/`: Frontend application.

## Requirements

- Python 3.10+ recommended
- Pip

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (only if you use Parquet input):

```bash
pip install pyarrow
```

## Quick Start (Local Build + Run)

### 1. Create a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build/refresh web data

If you already have final cards in `data/processed/player_cards/*_final.json`:

```bash
python prepare_web_data.py
```

### 3. Serve the site locally

```bash
python serve_web.py
```

Then open:

- `http://localhost:8000/` (landing page)
- `http://localhost:8000/about` (methodology page)

## Full Pipeline (From Raw Stats)

### A. Generate cards from raw data

```bash
python src/create_cards.py --input data/raw/practical_player_card_data.csv --output data/processed/player_cards
```

### B. (Optional) Backfill accurate usage rates

```bash
python src/backfill_usage_rates.py --raw data/raw/practical_player_card_data.csv --cards data/processed/player_cards
```

### C. Run valuation outputs

```bash
python src/value_players.py --cards data/processed/player_cards --output data/valuations
```

### D. Run analysis outputs

```bash
python src/analyze_players.py --cards data/processed/player_cards --output data/breakout
```

### E. Build frontend data bundle and run site

```bash
python prepare_web_data.py
python serve_web.py
```

## Expected Input Data

Your raw CSV should include (at minimum):

- Player identifiers: `player_name` or `name`, `team`, `season`, `position`, `age`
- Box stats: `points_per_game`, `assists_per_game`, `rebounds_per_game`
- Supporting stats: `steals_per_game`, `blocks_per_game`, `turnovers_per_game`
- Volume fields: `field_goal_attempts_per_game`, `three_point_attempts_per_game`
- Time/sample: `minutes_per_game`, `games_played`
- Impact/context fields when available: `usage_rate`, `plus_minus`

## Output Artifacts

- `data/processed/player_cards/*_final.json`: canonical card objects
- `data/valuations/`: valuation reports and summaries
- `data/breakout/`: breakout and fit analysis reports
- `web/data/cards.json`: combined card payload for frontend
- `web/data/valuations.json`: combined valuation payload for frontend

## Notes

- The web app reads from `web/data/cards.json` and `web/data/valuations.json`.
- `serve_web.py` supports clean routes for HTML pages (example: `/about` -> `about.html`).
- If frontend changes do not appear, do a hard refresh (`Ctrl+F5`).
