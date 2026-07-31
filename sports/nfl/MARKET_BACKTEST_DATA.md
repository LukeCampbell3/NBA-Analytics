# NFL market-backtest data decision

The promotion test needs authentic, book-specific player passing, rushing, and
receiving yard lines with real prices. A line must either carry a timestamp
before kickoff or be an explicit provider closing line. Scraped result pages,
consensus numbers without a book, and lines reconstructed from actual results
are not eligible.

## Recommended acquisition order

1. **The Odds API historical event snapshots** are the primary source. Player
   props are available from May 3, 2023 at five-minute intervals. A three-market,
   one-region event call costs at most 30 credits, so one 272-game regular season
   costs at most 8,160 event-odds credits plus inexpensive event discovery. The
   repository collector requests the last snapshot 30 minutes before kickoff and
   retains both requested and returned timestamps.
2. **SportsGameOdds Pro history** is the bulk/cross-check source. Its event feed
   exposes per-book `openOdds`, `closeOdds`, `openOverUnder`, and
   `closeOverUnder`, plus a normalized player directory. The adapter accepts only
   the explicit closing fields and ignores current/live values. Historical depth
   can vary and must be probed for the requested NFL seasons before purchase.
3. **SportsDataIO Vault** is an enterprise fallback. It has a dedicated NFL
   `BettingPropsArchive` endpoint, but access requires a separate archive key and
   a sales-enabled agreement.

OpticOdds is not suitable for this backfill because its detailed historical odds
endpoint retains only a rolling two months. Scraping sportsbook pages is not the
primary plan: it cannot reliably prove the line existed before kickoff, schemas
change without notice, and access terms vary by book.

## Acquisition commands

Dry-run the auditable snapshot source:

```bash
python sports/nfl/scripts/fetch_historical_nfl_props.py --season 2025
```

With `THE_ODDS_API_KEY` set, add `--execute`. For the bulk closing-line source:

```bash
python sports/nfl/scripts/fetch_sportsgameodds_historical_props.py --season 2025
```

With a Pro-or-higher `SPORTSGAMEODDS_API_KEY`, add `--execute`. Run a one-week
probe first and inspect the manifest's event, row, two-sided-price, bookmaker,
and target coverage before paying for or fetching a full archive.

## Promotion protocol

Use a model prediction generated without that game, join only players who had a
posted line, and choose over/under strictly from `prediction - line`. Grade with
the independent game result. The gate requires at least 200 decisions, 50 per
yardage target, eight distinct season-weeks, a Wilson 95% hit-rate lower bound
above 50%, positive real-price ROI, real prices on every wager, and pregame proof
on every line. The threshold must be selected on a development season; the final
reported hit rate must come from a later untouched season.

Provider references:

- https://the-odds-api.com/historical-odds-data/
- https://the-odds-api.com/liveapi/guides/v4/
- https://sportsgameodds.com/use-cases/historical-odds-data-api
- https://sportsgameodds.com/docs/endpoints/getEvents
- https://sportsdata.io/help/historical-data-integration-guide
