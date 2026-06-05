#!/usr/bin/env python3
"""Build historical NBA pre-lock availability snapshots from official NBA injury-report PDFs."""
from __future__ import annotations

import argparse
import io
import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import pdfplumber
import requests
from PyPDF2 import PdfReader
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PDF_STEM = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
ET = ZoneInfo("America/New_York")
STATUS_PROBABILITY = {
    "Out": 1.0,
    "Doubtful": 0.8,
    "Questionable": 0.45,
    "Probable": 0.15,
    "Available": 0.0,
}
STATUS_CONFIDENCE = {
    "Out": 1.0,
    "Doubtful": 0.85,
    "Questionable": 0.75,
    "Probable": 0.75,
    "Available": 1.0,
}
TEAM_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}
TEAM_TOKEN_MAP = {tuple(team.split()): abbr for team, abbr in TEAM_TO_ABBR.items()}
TEAM_COMPACT_TO_ABBR = {team.replace(" ", ""): abbr for team, abbr in TEAM_TO_ABBR.items()}
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
SHORT_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
MATCHUP_RE = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")


def _report_url(ts_et: datetime) -> str:
    date_part = ts_et.date().strftime("%Y-%m-%d")
    if ts_et >= datetime(2025, 12, 22, 9, 0, tzinfo=ET):
        time_part = ts_et.strftime("%I_%M%p")
    else:
        time_part = ts_et.replace(minute=0).strftime("%I%p")
    return PDF_STEM.format(stamp=f"{date_part}_{time_part}")


def _to_utc_iso(dt_et: datetime) -> str:
    return dt_et.astimezone(timezone.utc).isoformat()


def _clean_lines(text: str) -> list[str]:
    skip = {
        "Injury",
        "Report:",
        "Game",
        "Date",
        "Time",
        "Matchup",
        "Team",
        "Player",
        "Name",
        "Current",
        "Status",
        "Reason",
        "Page",
        "of",
    }
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in skip or line.isdigit():
            continue
        if SHORT_DATE_RE.match(line):
            continue
        lines.append(line)
    return lines


def _fetch_pdf_lines(ts_et: datetime, timeout: int) -> tuple[list[str], str]:
    url = _report_url(ts_et)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "NBA-Analytics-v9.5/1.0"})
    response.raise_for_status()
    reader = PdfReader(io.BytesIO(response.content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _clean_lines(text), url


def _fetch_pdf_frame(ts_et: datetime, timeout: int) -> tuple[pd.DataFrame, str]:
    url = _report_url(ts_et)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "NBA-Analytics-v9.5/1.0"})
    response.raise_for_status()
    rows: list[dict] = []
    current_date = None
    current_time = None
    current_matchup = None
    current_team = None
    current_abbr = None
    last_row_idx = None
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
            line_groups: dict[int, list[dict]] = {}
            for word in words:
                if word["top"] < 120:
                    continue
                key = int(round(word["top"] / 3.0) * 3)
                line_groups.setdefault(key, []).append(word)
            for _, line_words in sorted(line_groups.items()):
                cols = {"date": [], "time": [], "matchup": [], "team": [], "player": [], "status": [], "reason": []}
                for word in sorted(line_words, key=lambda w: w["x0"]):
                    x0 = float(word["x0"])
                    text = str(word["text"]).strip()
                    if x0 < 105:
                        cols["date"].append(text)
                    elif x0 < 190:
                        cols["time"].append(text)
                    elif x0 < 255:
                        cols["matchup"].append(text)
                    elif x0 < 420:
                        cols["team"].append(text)
                    elif x0 < 580:
                        cols["player"].append(text)
                    elif x0 < 660:
                        cols["status"].append(text)
                    else:
                        cols["reason"].append(text)

                date_text = " ".join(cols["date"]).strip()
                time_text = " ".join(cols["time"]).replace("(ET)", "").strip()
                matchup_text = " ".join(cols["matchup"]).strip()
                team_text = " ".join(cols["team"]).strip()
                player_text = " ".join(cols["player"]).strip()
                status_text = " ".join(cols["status"]).strip()
                reason_text = " ".join(cols["reason"]).strip()

                if DATE_RE.match(date_text):
                    current_date = date_text
                if TIME_RE.match(time_text):
                    current_time = time_text
                if MATCHUP_RE.match(matchup_text):
                    current_matchup = matchup_text
                if team_text:
                    compact_team = team_text.replace(" ", "")
                    if compact_team in TEAM_COMPACT_TO_ABBR:
                        current_team = team_text
                        current_abbr = TEAM_COMPACT_TO_ABBR[compact_team]

                status_title = status_text.title()
                if player_text and status_title in STATUS_PROBABILITY and current_date and current_time and current_matchup and current_abbr:
                    rows.append({
                        "snapshot_time": _to_utc_iso(ts_et),
                        "game_start_time": _game_start_iso(current_date, current_time),
                        "date": datetime.strptime(current_date, "%m/%d/%Y").date().isoformat(),
                        "team": current_abbr,
                        "team_name": current_team,
                        "player": _player_to_first_last(player_text.split()),
                        "status": status_title.lower(),
                        "out_probability": STATUS_PROBABILITY[status_title],
                        "availability_confidence": STATUS_CONFIDENCE[status_title],
                        "source": "nba_official_injury_report",
                        "matchup": current_matchup,
                        "reason": reason_text,
                        "source_url": url,
                    })
                    last_row_idx = len(rows) - 1
                elif reason_text and last_row_idx is not None and not player_text and not status_text:
                    rows[last_row_idx]["reason"] = (rows[last_row_idx]["reason"] + " " + reason_text).strip()
    return pd.DataFrame(rows), url


def _match_team(lines: list[str], idx: int) -> tuple[str | None, str | None, int]:
    for tokens, abbr in sorted(TEAM_TOKEN_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if tuple(lines[idx: idx + len(tokens)]) == tokens:
            return " ".join(tokens), abbr, len(tokens)
    return None, None, 0


def _find_status(lines: list[str], idx: int, limit: int = 10) -> int | None:
    for j in range(idx, min(len(lines), idx + limit)):
        if lines[j] in STATUS_PROBABILITY:
            return j
    return None


def _looks_like_next_row(lines: list[str], idx: int) -> bool:
    if idx >= len(lines):
        return True
    if DATE_RE.match(lines[idx]) or MATCHUP_RE.match(lines[idx]) or _match_team(lines, idx)[1]:
        return True
    status_idx = _find_status(lines, idx)
    if status_idx is None:
        return False
    reason_starts = {"Injury/Illness", "Personal", "Reasons", "Rest", "G", "League", "-", "Not"}
    if lines[idx] in reason_starts:
        return False
    return any("," in token for token in lines[idx:min(status_idx, idx + 2)])


def _player_to_first_last(name_tokens: list[str]) -> str:
    text = " ".join(name_tokens)
    if "," not in text:
        return text.replace(" ", "_")
    last, first = text.split(",", 1)
    return f"{first.strip()}_{last.strip()}".replace(" ", "_")


def _game_start_iso(game_date: str, game_time: str) -> str:
    dt = datetime.strptime(f"{game_date} {game_time}", "%m/%d/%Y %H:%M").replace(tzinfo=ET)
    if dt.hour < 12:
        dt = dt.replace(hour=dt.hour + 12)
    return _to_utc_iso(dt)


def _normalize_player_name(value: object) -> str:
    return str(value).strip().replace(" ", "_")


def _load_game_log_team_map(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        return {}
    logs = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    rename = {"PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team"}
    logs = logs.rename(columns={k: v for k, v in rename.items() if k in logs.columns})
    if not {"player", "team"}.issubset(logs.columns):
        return {}
    logs["player"] = logs["player"].map(_normalize_player_name)
    counts = logs.groupby(["player", "team"]).size().reset_index(name="n").sort_values(["player", "n"], ascending=[True, False])
    return counts.drop_duplicates("player").set_index("player")["team"].astype(str).to_dict()


def _repair_teams_with_game_logs(frame: pd.DataFrame, player_team_map: dict[str, str]) -> pd.DataFrame:
    if frame.empty or not player_team_map:
        frame["team_repaired_from_game_logs"] = False
        return frame
    frame = frame.copy()
    frame["team_original"] = frame["team"]
    mapped = frame["player"].map(player_team_map)
    repair_mask = mapped.notna() & mapped.ne(frame["team"])
    frame.loc[repair_mask, "team"] = mapped[repair_mask]
    frame["team_repaired_from_game_logs"] = repair_mask
    return frame


def parse_report_lines(lines: list[str], snapshot_et: datetime, source_url: str) -> pd.DataFrame:
    rows: list[dict] = []
    idx = 0
    current_date = None
    current_time = None
    current_matchup = None
    current_team = None
    current_abbr = None
    while idx < len(lines):
        line = lines[idx]
        if DATE_RE.match(line) and idx + 3 < len(lines) and TIME_RE.match(lines[idx + 1]):
            current_date = line
            current_time = lines[idx + 1]
            if idx + 2 < len(lines) and lines[idx + 2] == "(ET)":
                idx += 1
            current_matchup = lines[idx + 2]
            idx += 3
            continue
        if MATCHUP_RE.match(line):
            current_matchup = line
            idx += 1
            continue
        team_name, abbr, consumed = _match_team(lines, idx)
        if abbr:
            current_team = team_name
            current_abbr = abbr
            idx += consumed
            continue
        if not (current_date and current_time and current_matchup and current_abbr):
            idx += 1
            continue

        status_idx = _find_status(lines, idx)
        if status_idx is None or not any("," in token for token in lines[idx:status_idx]):
            idx += 1
            continue
        status = lines[status_idx]
        end = status_idx + 1
        while end < len(lines) and not _looks_like_next_row(lines, end):
            end += 1
        player = _player_to_first_last(lines[idx:status_idx])
        reason = " ".join(lines[status_idx + 1:end])
        rows.append({
            "snapshot_time": _to_utc_iso(snapshot_et),
            "game_start_time": _game_start_iso(current_date, current_time),
            "date": datetime.strptime(current_date, "%m/%d/%Y").date().isoformat(),
            "team": current_abbr,
            "team_name": current_team,
            "player": player,
            "status": status.lower(),
            "out_probability": STATUS_PROBABILITY[status],
            "availability_confidence": STATUS_CONFIDENCE[status],
            "source": "nba_official_injury_report",
            "matchup": current_matchup,
            "reason": reason,
            "source_url": source_url,
        })
        idx = end
    return pd.DataFrame(rows)


def _date_range(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"))


def parse_times(values: str) -> list[time]:
    out = []
    for raw in values.split(","):
        hour, minute = raw.strip().split(":")
        out.append(time(int(hour), int(minute)))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official historical NBA pre-lock availability snapshots")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--report-times-et", default="11:30,13:30,15:30,17:30,18:30,19:30")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "availability" / "nba" / "historical_prelock_availability_snapshots.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "availability" / "nba" / "raw_official_reports")
    parser.add_argument("--game-logs", type=Path, default=ROOT / "data copy" / "raw" / "nba_enrichment" / "season=2026" / "player_game_logs.parquet")
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    attempts = []
    for date in _date_range(args.start, args.end):
        for report_time in parse_times(args.report_times_et):
            ts_et = datetime.combine(date.date(), report_time, tzinfo=ET)
            try:
                frame, url = _fetch_pdf_frame(ts_et, args.timeout)
                if not frame.empty:
                    frames.append(frame)
                attempts.append({"timestamp_et": ts_et.isoformat(), "status": "ok", "rows": int(len(frame)), "url": url})
            except Exception as exc:
                attempts.append({"timestamp_et": ts_et.isoformat(), "status": "failed", "error": str(exc), "url": _report_url(ts_et)})
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    player_team_map = _load_game_log_team_map(args.game_logs)
    if not combined.empty:
        combined = _repair_teams_with_game_logs(combined, player_team_map)
        combined["snapshot_ts"] = pd.to_datetime(combined["snapshot_time"], utc=True)
        combined["game_start_ts"] = pd.to_datetime(combined["game_start_time"], utc=True)
        combined = combined[combined["snapshot_ts"] < combined["game_start_ts"]].copy()
        combined = combined.sort_values(["date", "team", "player", "game_start_ts", "snapshot_ts"])
        combined = combined.drop_duplicates(["date", "team", "player", "game_start_time"], keep="last")
        combined = combined.drop(columns=["snapshot_ts", "game_start_ts"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    report = {
        "status": "built_official_historical_prelock_availability",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": args.start,
        "end": args.end,
        "output": str(args.output),
        "rows": int(len(combined)),
        "players": int(combined["player"].nunique()) if not combined.empty else 0,
        "teams": int(combined["team"].nunique()) if not combined.empty else 0,
        "attempts": len(attempts),
        "successful_attempts": sum(1 for a in attempts if a["status"] == "ok"),
        "failed_attempts": sum(1 for a in attempts if a["status"] != "ok"),
        "status_counts": combined["status"].value_counts().to_dict() if not combined.empty else {},
        "team_repairs_from_game_logs": int(combined.get("team_repaired_from_game_logs", pd.Series(dtype=bool)).sum()) if not combined.empty else 0,
        "source": "official NBA static injury-report PDFs",
    }
    (args.output.parent / "historical_prelock_availability_manifest.json").write_text(
        json.dumps({**report, "attempt_log": attempts}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
