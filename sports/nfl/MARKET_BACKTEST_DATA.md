# NFL market-backtest data decision

The promotion test needs authentic, book-specific player passing, rushing, and
receiving yard lines with real prices. Only one posted pregame observation is
needed per player/stat/book; line movement and closing values are discarded. A
line must carry a timestamp before kickoff for deployment provenance. Scraped result pages,
consensus numbers without a book, and lines reconstructed from actual results
are not eligible for promotion.

## Recommended acquisition order

1. **XSportsbook's downloadable Bovada archive** is the free research source.
   The publisher explicitly offers 12,598 rows for 2021 and 12,913 for 2022 for
   model training. Both side prices, player, market, opponent, and week are
   present. The importer keeps only passing/rushing/receiving lines and prices
   and discards the supplied results. Because the files have no capture
   timestamp and are not explicitly described as opening lines, they can pass
   the statistical performance gate but cannot pass the deployment provenance
   gate by themselves.
2. **The Odds API historical event snapshots** are the primary timestamped
   source. Player
   props are available from May 3, 2023 at five-minute intervals. A three-market,
   one-region event call costs at most 30 credits, so one 272-game regular season
   costs at most 8,160 event-odds credits plus inexpensive event discovery. The
   repository collector can request a single pregame snapshot and retains both
   requested and returned timestamps. Repeated snapshots are unnecessary.
3. **SportsGameOdds Pro history** remains a cross-check source, but its closing
   fields are not required for this pipeline. Historical depth can vary and must
   be probed for the requested NFL seasons before purchase.
4. **SportsDataIO Vault** is an enterprise fallback. It has a dedicated NFL
   `BettingPropsArchive` endpoint, but access requires a separate archive key and
   a sales-enabled agreement.

OpticOdds is not suitable for this backfill because its detailed historical odds
endpoint retains only a rolling two months. Scraping sportsbook pages is not the
primary plan: it cannot reliably prove the line existed before kickoff, schemas
change without notice, and access terms vary by book.

## Acquisition commands

Download the free, intentionally published 2021-2022 archive:

```bash
python sports/nfl/scripts/fetch_xsportsbook_bovada_props.py
```

The collector pins the observed SHA-256 checksums, records a manifest, maps
kickoffs through nflverse, rejects ambiguous duplicate rows, and writes only
the three yardage markets. Raw outputs remain gitignored.

For a leakage-safe market test, train separate 2021 and 2022 holdouts, then let
2021 select the policy and apply it once to 2022:

```bash
python sports/nfl/scripts/validate_nfl_market_holdouts.py \
  --lines sports/nfl/data/raw/xsportsbook_bovada_player_props.csv \
  --development-predictions sports/nfl/tmp/2021/backtest_rows.csv \
  --final-predictions sports/nfl/tmp/2022/backtest_rows.csv
```

The fixed candidate grid and complete development table are retained in the
report, so an unsuccessful final season cannot be hidden by retuning.

The higher-precision selector tests whether the MLB-style calibrated selection
layer transfers to NFL props, including raw and frozen-latent candidates:

```bash
python sports/nfl/scripts/train_nfl_market_selector.py \
  --development-market-rows sports/nfl/tmp/2021/market_rows_edge0.csv \
  --final-market-rows sports/nfl/tmp/2022/market_rows_edge0.csv
```

Its fixed 0.56 probability floor is chosen before the final test. Architectures
are selected only by expanding 2021 Brier score, and each stat has its own final
promotion gate. This prevents a failed receiving market from being hidden by a
successful passing market.

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

Use features generated without that game, join only players who had a posted
line, and estimate each side's probability directly. Grade with the independent
game result. The selective per-target gate requires at least 150 decisions,
eight distinct season-weeks, at least 58% hit rate, a Wilson 95% lower bound
above 50%, and positive real-price ROI. The probability threshold and model
architecture must be selected on a development season; the final reported hit
rate must come from a later untouched season. Static deployment additionally
requires timestamped pregame proof for every line.

Provider references:

- https://xsportsbook.com/nfl-player-prop-betting-detail-2021/
- https://xsportsbook.com/nfl-player-prop-betting-detail-2022/
- https://the-odds-api.com/historical-odds-data/
- https://the-odds-api.com/liveapi/guides/v4/
- https://sportsgameodds.com/use-cases/historical-odds-data-api
- https://sportsgameodds.com/docs/endpoints/getEvents
- https://sportsdata.io/help/historical-data-integration-guide
