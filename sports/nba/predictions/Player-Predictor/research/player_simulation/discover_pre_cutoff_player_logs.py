from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.player_simulation.simulate_next_season_player_states import _normalize_logs


DEFAULT_OUTPUT_DIR = (
    PLAYER_PREDICTOR_ROOT.parents[1]
    / "validation"
    / "production_shadow"
    / "player_simulation"
    / "backtests"
    / "2025_preseason"
    / "pre_cutoff_discovery"
)
SOURCE_ROOTS = [
    PLAYER_PREDICTOR_ROOT / "Data-Proc",
    PLAYER_PREDICTOR_ROOT / "model" / "analysis",
    PLAYER_PREDICTOR_ROOT / "data copy" / "raw",
    PLAYER_PREDICTOR_ROOT.parents[1] / "validation",
]


def _candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for suffix in ("*.csv", "*.parquet"):
        for path in root.rglob(suffix):
            lower = str(path).lower()
            if any(token in lower for token in ["player", "game", "log", "processed", "enrichment", "training", "dataset"]):
                files.append(path)
    return sorted(set(files))


def _read_sample(path: Path, nrows: int = 5000) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        return pd.DataFrame()


def _date_col(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if column.lower() in {"date", "game_date", "gamedate", "game date"}:
            return column
    return ""


def _has_any(frame: pd.DataFrame, names: set[str]) -> bool:
    lower = {column.lower() for column in frame.columns}
    return bool(lower.intersection({name.lower() for name in names}))


def _source_row(path: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    frame = _read_sample(path)
    row: dict[str, Any] = {
        "path": str(path),
        "file_size": int(path.stat().st_size) if path.exists() else 0,
        "rows_sampled": int(len(frame)),
        "available_players": 0,
        "min_game_date": "",
        "max_game_date": "",
        "rows_before_cutoff": 0,
        "rows_after_cutoff": 0,
        "usable_pre_cutoff_rows": 0,
        "missing_player_ids": True,
        "missing_date_fields": True,
        "missing_minutes_stat_fields": True,
        "has_required_stats": False,
        "available_seasons": "",
    }
    if frame.empty:
        return row
    date_col = _date_col(frame)
    if not date_col:
        return row
    normalized = _normalize_logs(frame, fallback_player=path.parent.name.replace("_", " "))
    dates = pd.to_datetime(normalized.get("Date"), errors="coerce")
    if dates.notna().sum() == 0:
        return row
    required_stats = _has_any(normalized, {"MIN", "MP", "minutes"}) and _has_any(normalized, {"PTS"}) and _has_any(normalized, {"REB", "TRB"}) and _has_any(normalized, {"AST"})
    pre_mask = dates.lt(cutoff)
    player_col = "Player" if "Player" in normalized.columns else "player" if "player" in normalized.columns else ""
    row.update(
        {
            "available_players": int(normalized[player_col].nunique()) if player_col else 0,
            "min_game_date": dates.min().strftime("%Y-%m-%d"),
            "max_game_date": dates.max().strftime("%Y-%m-%d"),
            "rows_before_cutoff": int(pre_mask.sum()),
            "rows_after_cutoff": int(dates.ge(cutoff).sum()),
            "usable_pre_cutoff_rows": int((pre_mask & normalized["PTS"].notna() & normalized["AST"].notna() & normalized["MIN"].notna() & normalized["REB"].notna()).sum()) if required_stats else 0,
            "missing_player_ids": not _has_any(normalized, {"Player_ID", "player_id"}),
            "missing_date_fields": False,
            "missing_minutes_stat_fields": not required_stats,
            "has_required_stats": bool(required_stats),
            "available_seasons": ",".join(str(year) for year in sorted(dates.dt.year.dropna().astype(int).unique().tolist())),
        }
    )
    return row


def discover_pre_cutoff_player_logs(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cutoff_date: str = "2025-10-01",
    search_root: list[Path] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    roots = search_root or SOURCE_ROOTS
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in _candidate_files(root):
            rows.append(_source_row(path, cutoff))
    df = pd.DataFrame(rows).sort_values(["usable_pre_cutoff_rows", "rows_before_cutoff"], ascending=[False, False]) if rows else pd.DataFrame()
    csv_path = output_dir / "pre_cutoff_available_sources.csv"
    df.to_csv(csv_path, index=False)
    usable = df.loc[df.get("usable_pre_cutoff_rows", pd.Series(dtype=int)).fillna(0).gt(0)] if not df.empty else pd.DataFrame()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "searched_roots": [str(root) for root in roots],
        "source_count": int(len(df)),
        "usable_source_count": int(len(usable)),
        "total_usable_pre_cutoff_rows": int(usable.get("usable_pre_cutoff_rows", pd.Series(dtype=int)).sum()) if not usable.empty else 0,
        "available_seasons": sorted(
            {
                season
                for value in df.get("available_seasons", pd.Series(dtype=str)).fillna("").astype(str).tolist()
                for season in value.split(",")
                if season
            }
        ) if not df.empty else [],
        "top_sources": usable.head(25).to_dict(orient="records") if not usable.empty else [],
        "production_behavior_changed": False,
        "promotion_ready": False,
    }
    (output_dir / "pre_cutoff_data_discovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "pre_cutoff_data_discovery_report.md").write_text(_format_md(report), encoding="utf-8")
    return {"report": report, "sources_csv": str(csv_path)}


def _format_md(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Pre-Cutoff Player Log Discovery",
            "",
            f"- cutoff_date: {report.get('cutoff_date')}",
            f"- source_count: {report.get('source_count')}",
            f"- usable_source_count: {report.get('usable_source_count')}",
            f"- total_usable_pre_cutoff_rows: {report.get('total_usable_pre_cutoff_rows')}",
            f"- promotion_ready: {report.get('promotion_ready')}",
            "",
            "Only local rows with game dates before cutoff and required stat/minutes fields are considered usable.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover local player logs usable for frozen preseason backtests.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-date", default="2025-10-01")
    parser.add_argument("--search-root", type=Path, action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = discover_pre_cutoff_player_logs(
        output_dir=args.output_dir,
        cutoff_date=str(args.cutoff_date),
        search_root=args.search_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
