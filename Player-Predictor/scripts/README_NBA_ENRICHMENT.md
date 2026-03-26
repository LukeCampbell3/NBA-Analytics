# NBA Enrichment Scraper

Use [`fetch_nba_enrichment.py`](/c:/Users/jcthi/Code/Player-Predictor/scripts/fetch_nba_enrichment.py) to pull richer official NBA data for training.

It fetches:
- player game logs
- box score advanced splits
- player tracking splits
- scoring splits
- misc splits
- four-factor splits
- hustle splits
- matchup splits
- live play-by-play actions

Example:

```powershell
python scripts/fetch_nba_enrichment.py --seasons 2025
```

Smoke test:

```powershell
python scripts/fetch_nba_enrichment.py --seasons 2025 --max-games 2
```

Outputs land in:

```text
data copy/raw/nba_enrichment/season=2025/
```

Key files:
- `player_game_logs.parquet`
- `boxscore_advanced_player.parquet`
- `boxscore_playertrack_player.parquet`
- `boxscore_matchups_matchups.parquet`
- `playbyplay_actions.parquet`
- `playbyplay_player_summary.parquet`
- `manifest.json`
