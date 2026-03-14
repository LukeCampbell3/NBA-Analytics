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

If you also want the college card page payload:

```bash
python prepare_web_college_data.py
```

### 3. Serve the site locally

```bash
python serve_web.py
```

Then open:

- `http://localhost:8000/` (landing page)
- `http://localhost:8000/college` (college cards page)
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

## College Data Backend (Robots-Compliant)

Build raw and canonical college player-season data (Sports-Reference CBB) with robots.txt enforcement:

```bash
python src/build_college_player_data.py --season-start 2021 --season-end 2025
```

If season-level player pages are unavailable, auto mode falls back to player profile crawling:

```bash
python src/build_college_player_data.py --season-start 2024 --season-end 2025 --collection-mode auto
```

For quick smoke tests:

```bash
python src/build_college_player_data.py --season-start 2024 --season-end 2024 --collection-mode player_pages --max-player-pages 250
```

Outputs:

- `data/raw/college/players_per_game_raw/*.csv`
- `data/raw/college/players_advanced_raw/*.csv`
- `data/processed/college/players_season.csv`
- `data/processed/college/build_summary.json`

Compliance behavior:

- checks `robots.txt` before each request
- enforces crawl-delay where available
- throttles requests with conservative fallback delay
- skips disallowed URLs and records them in summary output

## College Player Valuation (NBA-Style + Verification)

Run the same valuation framework used for NBA cards against college player-season rows:

```bash
python src/value_college_players.py --input data/processed/college/players_season.csv --output data/valuations/college
```

The valuation run also writes a compliance/logic verification report:

- `data/valuations/college/college_valuation_summary.json`
- `data/valuations/college/college_valuation_verification.json`

Verification checks include:

- required core/provenance columns (`player_key`, `team_key`, `source_url`, etc.)
- robots enforcement status from `data/processed/college/build_summary.json`
- valuation invariants (finite values, surplus consistency, NPV consistency, trade-band ordering)

Use a non-failing verification mode if you want output even when checks fail:

```bash
python src/value_college_players.py --non-strict-verify
```

## College Pillar Parity Audit

Validate whether college rows can compute the same core pillars used for NBA valuation + breakout:

```bash
python src/validate_college_metric_parity.py --input data/processed/college/players_season.csv
```

Output:

- `data/processed/college/metric_parity_report.json`

The audit checks three groups:

- valuation pillars (`wins_added`, market curve, surplus/NPV, trade bands, aging)
- breakout pillars (opportunity, signal strength, confidence, defense portability, impact sanity)
- isolation pillars (trust/uncertainty + player-signal + provenance availability)

## Notes

- The web app reads from `web/data/cards.json` and `web/data/valuations.json`.
- `serve_web.py` supports clean routes for HTML pages (example: `/about` -> `about.html`).
- If frontend changes do not appear, do a hard refresh (`Ctrl+F5`).
