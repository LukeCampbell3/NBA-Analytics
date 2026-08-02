#!/usr/bin/env python3
"""Compact SmartStake MLB quote history into exact pregame book offers.

The source dataset contains minute-level side quotes. This importer keeps the
last quote strictly before each game's start for every player/market/line/book,
then pivots OVER and UNDER into the normalized long format consumed by the MLB
daily-pool generator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "sports"
    / "mlb"
    / "data"
    / "raw"
    / "market_odds"
    / "mlb"
    / "smartstake"
    / "history_player_props_long.parquet"
)
DEFAULT_BOOKS = ["bet365", "caesars", "draftkings", "fanduel", "fanatics", "betmgm", "mgm"]
MARKET_KEY_MAP = {
    "player bases": "batter_total_bases",
    "player hits": "batter_hits",
    "player rbis": "batter_rbis",
    "player home runs": "batter_home_runs",
    "player strikeouts": "pitcher_strikeouts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import exact pregame MLB player-prop closing lines.")
    parser.add_argument("--source", action="append", required=True, help="Parquet URL/path; repeat for partitions.")
    parser.add_argument("--start-date", required=True, help="First Eastern event date, inclusive (YYYY-MM-DD).")
    parser.add_argument("--end-date", required=True, help="Last Eastern event date, inclusive (YYYY-MM-DD).")
    parser.add_argument("--book", action="append", dest="books", default=None, help="Book key to retain; repeatable.")
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def decimal_to_american(value: float) -> int:
    decimal = float(value)
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise ValueError("decimal odds must be finite and greater than 1.0")
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def source_expression(sources: list[str]) -> str:
    return "[" + ",".join(sql_string(value) for value in sources) + "]"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def build_compaction_sql(
    *,
    sources: list[str],
    start_date: str,
    end_date: str,
    books: list[str],
) -> str:
    source_list = source_expression(sources)
    book_list = ",".join(sql_string(value.lower()) for value in books)
    market_cases = " ".join(
        f"WHEN market = {sql_string(source)} THEN {sql_string(target)}"
        for source, target in MARKET_KEY_MAP.items()
    )
    return f"""
        WITH source_rows AS (
            SELECT
                game_id,
                start_time,
                CAST(timezone('America/New_York', start_time AT TIME ZONE 'UTC') AS DATE) AS event_date_et,
                lower(trim(player)) AS player,
                lower(trim(market)) AS market,
                line,
                lower(trim(side)) AS side,
                lower(trim(book)) AS book,
                ts,
                odds,
                result,
                won
            FROM read_parquet({source_list}, union_by_name=true)
            WHERE result IS NOT NULL
              AND ts < start_time
              AND odds > 1.0
              AND lower(trim(side)) IN ('over', 'under')
              AND lower(trim(book)) IN ({book_list})
              AND lower(trim(market)) IN ({','.join(sql_string(value) for value in MARKET_KEY_MAP)})
        ), filtered AS (
            SELECT *
            FROM source_rows
            WHERE event_date_et BETWEEN DATE {sql_string(start_date)} AND DATE {sql_string(end_date)}
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY game_id, start_time, player, market, line, side, book
                ORDER BY ts DESC
            ) AS quote_rank
            FROM filtered
        ), closing AS (
            SELECT * FROM ranked WHERE quote_rank = 1
        )
        SELECT
            NULL::VARCHAR AS fetched_at_utc,
            game_id AS event_id,
            start_time AS commence_time_utc,
            event_date_et,
            '' AS home_team,
            '' AS away_team,
            CASE WHEN book IN ('betmgm', 'mgm') THEN 'mgm' ELSE book END AS bookmaker_key,
            CASE
                WHEN book = 'fanduel' THEN 'FanDuel'
                WHEN book = 'draftkings' THEN 'DraftKings'
                WHEN book = 'bet365' THEN 'bet365'
                WHEN book IN ('betmgm', 'mgm') THEN 'BetMGM'
                WHEN book = 'caesars' THEN 'Caesars'
                WHEN book = 'fanatics' THEN 'Fanatics'
                ELSE book
            END AS bookmaker_title,
            CASE {market_cases} END AS market_key,
            player AS player_name_raw,
            regexp_replace(player, '[^a-z0-9]+', '_', 'g') AS player_name_norm,
            line,
            max(CASE WHEN side = 'over' THEN
                CASE WHEN odds >= 2.0 THEN round((odds - 1.0) * 100.0)
                     ELSE round(-100.0 / (odds - 1.0)) END
            END)::INTEGER AS over_price,
            max(CASE WHEN side = 'under' THEN
                CASE WHEN odds >= 2.0 THEN round((odds - 1.0) * 100.0)
                     ELSE round(-100.0 / (odds - 1.0)) END
            END)::INTEGER AS under_price,
            max(CASE WHEN side = 'over' THEN ts END) AS over_quote_time_utc,
            max(CASE WHEN side = 'under' THEN ts END) AS under_quote_time_utc,
            max(result) AS settled_result,
            'smartstake_hf' AS provider,
            'per_offer_closing' AS snapshot_mode
        FROM closing
        GROUP BY ALL
        HAVING over_price IS NOT NULL OR under_price IS NOT NULL
        ORDER BY event_date_et, commence_time_utc, event_id, player_name_norm, market_key, line, bookmaker_key
    """


def main() -> None:
    args = parse_args()
    try:
        import duckdb
    except ImportError as exc:
        raise SystemExit("DuckDB is required: python -m pip install duckdb") from exc

    books = sorted({str(value).strip().lower() for value in (args.books or DEFAULT_BOOKS) if str(value).strip()})
    if not books:
        raise SystemExit("At least one sportsbook key is required.")
    query = build_compaction_sql(
        sources=args.source,
        start_date=args.start_date,
        end_date=args.end_date,
        books=books,
    )
    out_parquet = args.out_parquet.resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    escaped_output = sql_string(str(out_parquet))
    connection.execute(f"COPY ({query}) TO {escaped_output} (FORMAT PARQUET, COMPRESSION ZSTD)")
    stats = connection.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count(DISTINCT event_date_et) AS event_date_count,
            min(event_date_et)::VARCHAR AS first_event_date,
            max(event_date_et)::VARCHAR AS last_event_date,
            count(DISTINCT bookmaker_key) AS bookmaker_count,
            count(DISTINCT event_id) AS event_count
        FROM read_parquet({escaped_output})
        """
    ).fetchone()
    market_counts = dict(
        connection.execute(
            f"SELECT market_key, count(*) FROM read_parquet({escaped_output}) GROUP BY market_key ORDER BY market_key"
        ).fetchall()
    )
    report = {
        "source": "SmartStake/mlb-player-props closing quote strictly before game start",
        "license": "CC-BY-4.0",
        "synthetic_events_used": False,
        "source_partition_count": len(args.source),
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "books": books,
        "row_count": int(stats[0]),
        "event_date_count": int(stats[1]),
        "first_event_date": stats[2],
        "last_event_date": stats[3],
        "bookmaker_count": int(stats[4]),
        "event_count": int(stats[5]),
        "market_counts": {str(key): int(value) for key, value in market_counts.items()},
        "output_parquet": portable_path(out_parquet),
    }
    report_path = (args.report_json or out_parquet.with_suffix(".json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
