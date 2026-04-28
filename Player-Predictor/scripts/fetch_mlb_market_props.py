#!/usr/bin/env python3
"""
Fetch and normalize MLB player prop markets from RotoWire into a stable contract.

The public MLB player-props page embeds the current tables directly in inline
JavaScript. This fetcher parses those tables and publishes the same normalized
artifacts the rest of the MLB pipeline already expects:

- raw page HTML + extracted bundle payloads
- normalized long + wide tables
- rolling append-only history outputs
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io"
ROTOWIRE_URL = "https://www.rotowire.com/betting/mlb/player-props.php"

BOOK_TITLES = {
    "betrivers": "BetRivers",
    "caesars": "Caesars",
    "circasports": "Circa Sports",
    "draftkings": "DraftKings",
    "fanatics": "Fanatics",
    "fanduel": "FanDuel",
    "hardrock": "Hard Rock",
    "mgm": "BetMGM",
    "thescore": "theScore",
}

MARKET_WIDE_COLUMNS = [
    "Market_Date",
    "Player",
    "Market_Player_Raw",
    "Market_Event_ID",
    "Market_Commence_Time_UTC",
    "Market_Home_Team",
    "Market_Away_Team",
    "Market_H",
    "Market_TB",
    "Market_R",
    "Market_HR",
    "Market_RBI",
    "Market_K",
    "Market_ER",
    "Market_ERA",
    "Market_Source_H",
    "Market_Source_TB",
    "Market_Source_R",
    "Market_Source_HR",
    "Market_Source_RBI",
    "Market_Source_K",
    "Market_Source_ER",
    "Market_Source_ERA",
    "Market_H_books",
    "Market_TB_books",
    "Market_R_books",
    "Market_HR_books",
    "Market_RBI_books",
    "Market_K_books",
    "Market_ER_books",
    "Market_ERA_books",
    "Market_H_over_price",
    "Market_TB_over_price",
    "Market_R_over_price",
    "Market_HR_over_price",
    "Market_RBI_over_price",
    "Market_K_over_price",
    "Market_ER_over_price",
    "Market_ERA_over_price",
    "Market_H_under_price",
    "Market_TB_under_price",
    "Market_R_under_price",
    "Market_HR_under_price",
    "Market_RBI_under_price",
    "Market_K_under_price",
    "Market_ER_under_price",
    "Market_ERA_under_price",
    "Market_H_line_std",
    "Market_TB_line_std",
    "Market_R_line_std",
    "Market_HR_line_std",
    "Market_RBI_line_std",
    "Market_K_line_std",
    "Market_ER_line_std",
    "Market_ERA_line_std",
    "Market_Fetched_At_UTC",
]

VALUE_MAP = {
    "batter_hits": "Market_H",
    "batter_total_bases": "Market_TB",
    "batter_runs_scored": "Market_R",
    "batter_home_runs": "Market_HR",
    "batter_rbis": "Market_RBI",
    "pitcher_strikeouts": "Market_K",
    "pitcher_earned_runs": "Market_ER",
    "pitcher_era": "Market_ERA",
}
SOURCE_MAP = {
    "batter_hits": "Market_Source_H",
    "batter_total_bases": "Market_Source_TB",
    "batter_runs_scored": "Market_Source_R",
    "batter_home_runs": "Market_Source_HR",
    "batter_rbis": "Market_Source_RBI",
    "pitcher_strikeouts": "Market_Source_K",
    "pitcher_earned_runs": "Market_Source_ER",
    "pitcher_era": "Market_Source_ERA",
}
BOOKS_MAP = {
    "batter_hits": "Market_H_books",
    "batter_total_bases": "Market_TB_books",
    "batter_runs_scored": "Market_R_books",
    "batter_home_runs": "Market_HR_books",
    "batter_rbis": "Market_RBI_books",
    "pitcher_strikeouts": "Market_K_books",
    "pitcher_earned_runs": "Market_ER_books",
    "pitcher_era": "Market_ERA_books",
}
OVER_MAP = {
    "batter_hits": "Market_H_over_price",
    "batter_total_bases": "Market_TB_over_price",
    "batter_runs_scored": "Market_R_over_price",
    "batter_home_runs": "Market_HR_over_price",
    "batter_rbis": "Market_RBI_over_price",
    "pitcher_strikeouts": "Market_K_over_price",
    "pitcher_earned_runs": "Market_ER_over_price",
    "pitcher_era": "Market_ERA_over_price",
}
UNDER_MAP = {
    "batter_hits": "Market_H_under_price",
    "batter_total_bases": "Market_TB_under_price",
    "batter_runs_scored": "Market_R_under_price",
    "batter_home_runs": "Market_HR_under_price",
    "batter_rbis": "Market_RBI_under_price",
    "pitcher_strikeouts": "Market_K_under_price",
    "pitcher_earned_runs": "Market_ER_under_price",
    "pitcher_era": "Market_ERA_under_price",
}
STD_MAP = {
    "batter_hits": "Market_H_line_std",
    "batter_total_bases": "Market_TB_line_std",
    "batter_runs_scored": "Market_R_line_std",
    "batter_home_runs": "Market_HR_line_std",
    "batter_rbis": "Market_RBI_line_std",
    "pitcher_strikeouts": "Market_K_line_std",
    "pitcher_earned_runs": "Market_ER_line_std",
    "pitcher_era": "Market_ERA_line_std",
}

MONEYLINE_PROP_MAP = {
    "onehit": "batter_hits",
    "onehomerun": "batter_home_runs",
    "onerbi": "batter_rbis",
}
TABLE_PROP_MAP = {
    "bases": "batter_total_bases",
    "runs": "batter_runs_scored",
    "strikeouts": "pitcher_strikeouts",
    "er": "pitcher_earned_runs",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_player_name(value: object) -> str:
    out = str(value or "").strip()
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def safe_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def to_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch normalized MLB player prop lines from RotoWire.")
    parser.add_argument(
        "--provider",
        type=str,
        default="rotowire",
        choices=["rotowire", "odds_api", "snapshot"],
        help="Market data provider. 'odds_api' is kept as a backward-compatible alias for the RotoWire parser.",
    )
    parser.add_argument(
        "--event-date",
        type=str,
        default=None,
        help="Requested action date (YYYY-MM-DD). The RotoWire page only exposes the current board date.",
    )
    parser.add_argument("--input-path", type=Path, default=None, help="Input CSV/parquet for --provider snapshot.")
    parser.add_argument("--page-url", type=str, default=ROTOWIRE_URL, help="RotoWire MLB player props page URL.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP timeout for the page request.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for raw + normalized snapshots.")
    return parser.parse_args()


def extract_day_token(html: str) -> str:
    match = re.search(r'const dayMLB = "([0-9]{4}-[0-9]{2}-[0-9]{2})"', html)
    if not match:
        raise RuntimeError("Unable to locate RotoWire page date (dayMLB) in the MLB props page.")
    return str(match.group(1))


def _extract_data_array_literal(script_text: str) -> str:
    data_idx = script_text.find("data:")
    if data_idx < 0:
        raise ValueError("settings.data array not found in script block")
    start = script_text.find("[", data_idx)
    if start < 0:
        raise ValueError("settings.data opening '[' not found")

    depth = 0
    in_string = False
    string_char = ""
    escaped = False
    for pos in range(start, len(script_text)):
        char = script_text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            string_char = char
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                return script_text[start : pos + 1]

    raise ValueError("Unterminated settings.data array in script block")


def _detect_bundle_kind(script_text: str) -> str | None:
    if "container: 'moneyline-props'" in script_text or "_onehit" in script_text:
        return "moneyline"
    match = re.search(r'const prop = "([a-z]+)"', script_text)
    if not match:
        return None
    prop = str(match.group(1)).strip().lower()
    return prop if prop in TABLE_PROP_MAP else None


def extract_rotowire_page_payload(html: str) -> tuple[str, dict[str, list[dict[str, object]]]]:
    page_date = extract_day_token(html)
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL)

    bundles: dict[str, list[dict[str, object]]] = {}
    for script_text in scripts:
        if "const settings" not in script_text or "data:" not in script_text:
            continue
        kind = _detect_bundle_kind(script_text)
        if kind is None:
            continue
        array_literal = _extract_data_array_literal(script_text)
        rows = json.loads(array_literal)
        if isinstance(rows, list) and rows:
            bundles[kind] = rows

    if not bundles:
        raise RuntimeError("No supported RotoWire prop data bundles were found on the MLB props page.")
    return page_date, bundles


def _derive_matchup(row: dict[str, object]) -> tuple[str | None, str | None]:
    team = str(row.get("team") or "").strip().upper()
    opp = str(row.get("opp") or "").strip().upper()
    if not team:
        return None, None
    if opp.startswith("@"):
        return opp[1:] or None, team
    return team, opp or None


def _book_keys_for_moneyline(row: dict[str, object], prop_key: str) -> list[str]:
    suffix = f"_{prop_key}"
    books: set[str] = set()
    for key in row:
        if key.endswith(suffix):
            books.add(key[: -len(suffix)])
    return sorted(books)


def _book_keys_for_table(row: dict[str, object], prop_key: str) -> list[str]:
    suffixes = (f"_{prop_key}", f"_{prop_key}Over", f"_{prop_key}Under")
    books: set[str] = set()
    for key in row:
        for suffix in suffixes:
            if key.endswith(suffix):
                books.add(key[: -len(suffix)])
                break
    return sorted(books)


def _long_row(
    *,
    fetched_at_utc: str,
    market_date: str,
    market_key: str,
    row: dict[str, object],
    bookmaker_key: str,
    line: float | None,
    over_price: float | None,
    under_price: float | None,
) -> dict[str, object] | None:
    if line is None and over_price is None and under_price is None:
        return None

    player_raw = str(row.get("name") or "").strip()
    player_norm = normalize_player_name(player_raw)
    if not player_raw or not player_norm:
        return None

    home_team, away_team = _derive_matchup(row)
    return {
        "fetched_at_utc": fetched_at_utc,
        "event_id": str(row.get("gameID") or ""),
        "commence_time_utc": None,
        "event_date_et": market_date,
        "home_team": home_team,
        "away_team": away_team,
        "bookmaker_key": bookmaker_key,
        "bookmaker_title": BOOK_TITLES.get(bookmaker_key, bookmaker_key.title()),
        "market_key": market_key,
        "player_name_raw": player_raw,
        "player_name_norm": player_norm,
        "line": line,
        "over_price": over_price,
        "under_price": under_price,
    }


def build_rotowire_frames(
    *,
    market_date: str,
    bundles: dict[str, list[dict[str, object]]],
    fetched_at_utc: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: list[dict[str, object]] = []

    for row in bundles.get("moneyline", []):
        for prop_key, market_key in MONEYLINE_PROP_MAP.items():
            for bookmaker_key in _book_keys_for_moneyline(row, prop_key):
                over_price = to_float(row.get(f"{bookmaker_key}_{prop_key}"))
                candidate = _long_row(
                    fetched_at_utc=fetched_at_utc,
                    market_date=market_date,
                    market_key=market_key,
                    row=row,
                    bookmaker_key=bookmaker_key,
                    line=0.5 if over_price is not None else None,
                    over_price=over_price,
                    under_price=None,
                )
                if candidate is not None:
                    long_rows.append(candidate)

    for prop_key, market_key in TABLE_PROP_MAP.items():
        for row in bundles.get(prop_key, []):
            for bookmaker_key in _book_keys_for_table(row, prop_key):
                line = to_float(row.get(f"{bookmaker_key}_{prop_key}"))
                over_price = to_float(row.get(f"{bookmaker_key}_{prop_key}Over"))
                under_price = to_float(row.get(f"{bookmaker_key}_{prop_key}Under"))
                candidate = _long_row(
                    fetched_at_utc=fetched_at_utc,
                    market_date=market_date,
                    market_key=market_key,
                    row=row,
                    bookmaker_key=bookmaker_key,
                    line=line,
                    over_price=over_price,
                    under_price=under_price,
                )
                if candidate is not None:
                    long_rows.append(candidate)

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return long_df, pd.DataFrame(columns=MARKET_WIDE_COLUMNS)

    for column in ["line", "over_price", "under_price"]:
        long_df[column] = pd.to_numeric(long_df[column], errors="coerce")

    consensus = (
        long_df.groupby(["event_date_et", "player_name_norm", "player_name_raw", "market_key"], dropna=False)
        .agg(
            market_line=("line", "median"),
            market_line_std=("line", "std"),
            over_price_avg=("over_price", "mean"),
            under_price_avg=("under_price", "mean"),
            book_count=("bookmaker_key", "nunique"),
            first_event_id=("event_id", "first"),
            first_commence_time_utc=("commence_time_utc", "first"),
            first_home_team=("home_team", "first"),
            first_away_team=("away_team", "first"),
        )
        .reset_index()
    )
    consensus["market_line_std"] = consensus["market_line_std"].fillna(0.0)

    def _pivot(metric_col: str, rename_map: dict[str, str]) -> pd.DataFrame:
        wide = (
            consensus.pivot_table(
                index=["event_date_et", "player_name_norm", "player_name_raw"],
                columns="market_key",
                values=metric_col,
                aggfunc="first",
            )
            .rename(columns=rename_map)
            .reset_index()
        )
        wide.columns.name = None
        return wide

    metadata = consensus[
        [
            "event_date_et",
            "player_name_norm",
            "player_name_raw",
            "first_event_id",
            "first_commence_time_utc",
            "first_home_team",
            "first_away_team",
        ]
    ].drop_duplicates(subset=["event_date_et", "player_name_norm", "player_name_raw"], keep="last")

    wide_df = metadata.merge(
        _pivot("market_line", VALUE_MAP),
        how="left",
        on=["event_date_et", "player_name_norm", "player_name_raw"],
    )
    for metric_col, rename_map in [
        ("book_count", BOOKS_MAP),
        ("over_price_avg", OVER_MAP),
        ("under_price_avg", UNDER_MAP),
        ("market_line_std", STD_MAP),
    ]:
        wide_df = wide_df.merge(
            _pivot(metric_col, rename_map),
            how="left",
            on=["event_date_et", "player_name_norm", "player_name_raw"],
        )

    wide_df = wide_df.rename(
        columns={
            "event_date_et": "Market_Date",
            "player_name_norm": "Player",
            "player_name_raw": "Market_Player_Raw",
            "first_event_id": "Market_Event_ID",
            "first_commence_time_utc": "Market_Commence_Time_UTC",
            "first_home_team": "Market_Home_Team",
            "first_away_team": "Market_Away_Team",
        }
    )

    for market_key, value_col in VALUE_MAP.items():
        source_col = SOURCE_MAP[market_key]
        if value_col not in wide_df.columns:
            wide_df[value_col] = np.nan
        wide_df[source_col] = np.where(wide_df[value_col].notna(), "real", "missing")

    wide_df["Market_Fetched_At_UTC"] = fetched_at_utc
    for column in MARKET_WIDE_COLUMNS:
        if column not in wide_df.columns:
            wide_df[column] = np.nan

    numeric_columns = [
        column
        for column in MARKET_WIDE_COLUMNS
        if column.startswith("Market_")
        and column
        not in {
            "Market_Date",
            "Market_Player_Raw",
            "Market_Event_ID",
            "Market_Commence_Time_UTC",
            "Market_Home_Team",
            "Market_Away_Team",
            "Market_Source_H",
            "Market_Source_TB",
            "Market_Source_R",
            "Market_Source_HR",
            "Market_Source_RBI",
            "Market_Source_K",
            "Market_Source_ER",
            "Market_Source_ERA",
            "Market_Fetched_At_UTC",
        }
    ]
    for column in numeric_columns:
        wide_df[column] = pd.to_numeric(wide_df[column], errors="coerce")

    wide_df = wide_df[MARKET_WIDE_COLUMNS].drop_duplicates(subset=["Market_Date", "Player"], keep="last").copy()
    return long_df, wide_df


def normalize_wide_snapshot(df: pd.DataFrame, fetched_at_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=MARKET_WIDE_COLUMNS)

    out = df.copy()
    rename_map = {}
    if "event_date_et" in out.columns and "Market_Date" not in out.columns:
        rename_map["event_date_et"] = "Market_Date"
    if "player_name_norm" in out.columns and "Player" not in out.columns:
        rename_map["player_name_norm"] = "Player"
    if "player_name_raw" in out.columns and "Market_Player_Raw" not in out.columns:
        rename_map["player_name_raw"] = "Market_Player_Raw"
    if rename_map:
        out = out.rename(columns=rename_map)

    if "Player" not in out.columns or "Market_Date" not in out.columns:
        raise ValueError("Snapshot provider requires Player and Market_Date columns.")

    out["Player"] = out["Player"].astype(str).map(normalize_player_name)
    out["Market_Date"] = pd.to_datetime(out["Market_Date"], errors="coerce").dt.date.astype(str)
    if "Market_Player_Raw" not in out.columns:
        out["Market_Player_Raw"] = out["Player"]
    if "Market_Fetched_At_UTC" not in out.columns:
        out["Market_Fetched_At_UTC"] = fetched_at_utc

    for column in MARKET_WIDE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    wide_df = out[MARKET_WIDE_COLUMNS].drop_duplicates(subset=["Market_Date", "Player"], keep="last").copy()

    long_rows: list[dict[str, object]] = []
    for market_key, value_col in VALUE_MAP.items():
        short_source = SOURCE_MAP[market_key]
        books_col = BOOKS_MAP[market_key]
        over_col = OVER_MAP[market_key]
        under_col = UNDER_MAP[market_key]
        for _, row in wide_df.iterrows():
            line = to_float(row.get(value_col))
            if line is None:
                continue
            long_rows.append(
                {
                    "fetched_at_utc": row.get("Market_Fetched_At_UTC", fetched_at_utc),
                    "event_id": row.get("Market_Event_ID", np.nan),
                    "commence_time_utc": row.get("Market_Commence_Time_UTC", np.nan),
                    "event_date_et": row.get("Market_Date"),
                    "home_team": row.get("Market_Home_Team", np.nan),
                    "away_team": row.get("Market_Away_Team", np.nan),
                    "bookmaker_key": np.nan,
                    "bookmaker_title": row.get(short_source, np.nan),
                    "market_key": market_key,
                    "player_name_raw": row.get("Market_Player_Raw", row.get("Player")),
                    "player_name_norm": row.get("Player"),
                    "line": line,
                    "over_price": row.get(over_col, np.nan),
                    "under_price": row.get(under_col, np.nan),
                    "book_count": row.get(books_col, np.nan),
                }
            )
    long_df = pd.DataFrame(long_rows)
    return long_df, wide_df


def fetch_from_rotowire(
    args: argparse.Namespace,
    fetched_at_utc: str,
) -> tuple[str, dict[str, list[dict[str, object]]], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    response = requests.get(
        args.page_url,
        timeout=float(args.timeout_seconds),
        headers={"User-Agent": "NBA-Analytics/1.0"},
    )
    response.raise_for_status()
    html = response.text

    page_date, bundles = extract_rotowire_page_payload(html)
    requested_event_date = str(args.event_date).strip() if args.event_date else page_date
    if requested_event_date != page_date:
        raise RuntimeError(
            "RotoWire MLB props page date mismatch: "
            f"requested {requested_event_date}, but the live page currently exposes {page_date}. "
            "Historical date selection is not supported by this source."
        )

    long_df, wide_df = build_rotowire_frames(
        market_date=page_date,
        bundles=bundles,
        fetched_at_utc=fetched_at_utc,
    )
    manifest = {
        "provider": "rotowire",
        "provider_alias_used": args.provider == "odds_api",
        "fetched_at_utc": fetched_at_utc,
        "source_url": args.page_url,
        "page_date": page_date,
        "event_date_requested": requested_event_date,
        "bundle_kinds": sorted(bundles),
        "bundle_rows": {kind: int(len(rows)) for kind, rows in bundles.items()},
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "errors": [],
    }
    return html, bundles, long_df, wide_df, manifest


def fetch_from_snapshot(
    args: argparse.Namespace,
    fetched_at_utc: str,
) -> tuple[str, dict[str, list[dict[str, object]]], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if args.input_path is None or not args.input_path.exists():
        raise RuntimeError("Snapshot provider requires --input-path pointing to an existing CSV/parquet file.")

    snapshot_df = load_table(args.input_path)
    long_df, wide_df = normalize_wide_snapshot(snapshot_df, fetched_at_utc)
    manifest = {
        "provider": "snapshot",
        "fetched_at_utc": fetched_at_utc,
        "input_path": str(args.input_path),
        "long_rows": int(len(long_df)),
        "wide_rows": int(len(wide_df)),
        "errors": [],
    }
    return "", {}, long_df, wide_df, manifest


def write_snapshot(
    outdir: Path,
    stamp: str,
    raw_html: str,
    bundles: dict[str, list[dict[str, object]]],
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    manifest: dict[str, object],
) -> None:
    raw_dir = outdir / "raw" / stamp
    norm_dir = outdir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    if raw_html:
        (raw_dir / "page.html").write_text(raw_html, encoding="utf-8")
    safe_write_json(raw_dir / "bundles.json", bundles)
    safe_write_json(raw_dir / "manifest.json", manifest)

    if not long_df.empty:
        long_df.to_parquet(norm_dir / f"player_props_long_{stamp}.parquet", index=False)
        long_df.to_csv(norm_dir / f"player_props_long_{stamp}.csv", index=False)
        long_df.to_parquet(outdir / "latest_player_props_long.parquet", index=False)
        long_df.to_csv(outdir / "latest_player_props_long.csv", index=False)

    if not wide_df.empty:
        wide_df.to_parquet(norm_dir / f"player_props_wide_{stamp}.parquet", index=False)
        wide_df.to_csv(norm_dir / f"player_props_wide_{stamp}.csv", index=False)
        wide_df.to_parquet(outdir / "latest_player_props_wide.parquet", index=False)
        wide_df.to_csv(outdir / "latest_player_props_wide.csv", index=False)

    safe_write_json(outdir / "latest_manifest.json", manifest)


def append_history(outdir: Path, long_df: pd.DataFrame, wide_df: pd.DataFrame) -> dict[str, int]:
    summary = {"history_long_rows": 0, "history_wide_rows": 0}

    if not long_df.empty:
        history_long_path = outdir / "history_player_props_long.parquet"
        if history_long_path.exists():
            history_long = pd.read_parquet(history_long_path)
            combined_long = pd.concat([history_long, long_df], ignore_index=True)
        else:
            combined_long = long_df.copy()
        long_dedupe_cols = [
            "event_date_et",
            "player_name_norm",
            "market_key",
            "bookmaker_key",
            "line",
            "over_price",
            "under_price",
            "fetched_at_utc",
        ]
        combined_long = combined_long.drop_duplicates(
            subset=[column for column in long_dedupe_cols if column in combined_long.columns],
            keep="last",
        )
        write_table(combined_long, history_long_path)
        write_table(combined_long, outdir / "history_player_props_long.csv")
        summary["history_long_rows"] = int(len(combined_long))

    if not wide_df.empty:
        history_wide_path = outdir / "history_player_props_wide.parquet"
        if history_wide_path.exists():
            history_wide = pd.read_parquet(history_wide_path)
            combined_wide = pd.concat([history_wide, wide_df], ignore_index=True)
        else:
            combined_wide = wide_df.copy()
        wide_dedupe_cols = ["Market_Date", "Player", "Market_Fetched_At_UTC"]
        combined_wide = combined_wide.drop_duplicates(
            subset=[column for column in wide_dedupe_cols if column in combined_wide.columns],
            keep="last",
        )
        write_table(combined_wide, history_wide_path)
        write_table(combined_wide, outdir / "history_player_props_wide.csv")
        summary["history_wide_rows"] = int(len(combined_wide))

    return summary


def main() -> None:
    args = parse_args()
    stamp = utc_compact_timestamp()
    fetched_at_utc = utc_now_iso()

    provider = "rotowire" if args.provider in {"rotowire", "odds_api"} else args.provider
    if provider == "rotowire":
        raw_html, bundles, long_df, wide_df, manifest = fetch_from_rotowire(args, fetched_at_utc)
    elif provider == "snapshot":
        raw_html, bundles, long_df, wide_df, manifest = fetch_from_snapshot(args, fetched_at_utc)
    else:
        raise RuntimeError(f"Unsupported provider: {args.provider}")

    write_snapshot(args.outdir, stamp, raw_html, bundles, long_df, wide_df, manifest)
    history_summary = append_history(args.outdir, long_df, wide_df)
    manifest.update(history_summary)
    safe_write_json(args.outdir / "latest_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("MLB MARKET PROPS FETCH COMPLETE")
    print("=" * 80)
    print(f"Provider:         {manifest.get('provider')}")
    print(f"Page date:        {manifest.get('page_date', 'n/a')}")
    print(f"Long rows:        {len(long_df)}")
    print(f"Wide rows:        {len(wide_df)}")
    print(f"History wide:     {history_summary['history_wide_rows']}")
    print(f"Output:           {args.outdir}")


if __name__ == "__main__":
    main()
