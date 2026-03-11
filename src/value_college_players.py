"""
value_college_players.py

College valuation pipeline that mirrors NBA valuation outputs while adding
strict validation and compliance checks for the college ingestion process.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils import sanitize_filename
from value_players import PlayerValuator, ValuationResult


COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "player_key": ["player_key"],
    "team_key": ["team_key"],
    "player_name": ["player_name", "player", "name"],
    "team_name": ["team_name", "school", "team"],
    "season": ["season"],
    "position": ["pos", "position", "pos_per_game", "pos_advanced"],
    "class_year": ["class", "class_per_game", "class_advanced"],
    "games": ["g", "games_played", "g_per_game", "g_advanced"],
    "minutes": ["mp", "minutes_per_game", "mp_per_game", "mp_advanced"],
    "plus_minus": ["plus_minus", "bpm", "bpm_advanced", "bpm_per_game", "bpm_advanced_advanced"],
    "usage_rate": ["usage_rate", "usg_pct", "usg_pct_advanced", "usg", "usg_advanced"],
    "wins_share": ["ws", "ws_advanced", "wins_share"],
    "three_pa": ["x3pa", "3pa", "three_point_attempts_per_game", "x3pa_per_game"],
    "fga": ["fga", "field_goal_attempts_per_game", "fga_per_game"],
    "points": ["pts", "points_per_game", "pts_per_game"],
    "assists": ["ast", "assists_per_game", "ast_per_game"],
    "rebounds": ["trb", "rebounds_per_game", "trb_per_game"],
    "steals": ["stl", "steals_per_game", "stl_per_game"],
    "blocks": ["blk", "blocks_per_game", "blk_per_game"],
    "turnovers": ["tov", "turnovers_per_game", "tov_per_game"],
    "source_table": ["source_table"],
    "source_url": ["source_url"],
    "scraped_at_utc": ["scraped_at_utc"],
    "source_site": ["source_site"],
    "conf": ["conf", "conf_per_game", "conf_advanced"],
}


def utc_now_iso() -> str:
    """UTC timestamp in ISO8601 with timezone."""
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into inclusive [lo, hi] range."""
    return max(lo, min(hi, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "":
                return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if value == "":
                return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def slugify(value: str) -> str:
    """Stable slug key for generated identifiers."""
    value = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return value or "unknown"


def read_csv_non_empty(path: Path) -> pd.DataFrame:
    """Read CSV, handling missing/empty cases with explicit errors."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    if path.stat().st_size <= 2:
        raise ValueError(
            f"Input CSV is empty: {path}. Build college data first "
            f"(python src/build_college_player_data.py ...)."
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input CSV has no parseable columns: {path}") from exc


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Case-insensitive column lookup using candidate names."""
    col_lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_lookup:
            return col_lookup[candidate.lower()]
    return None


def resolve_column_map(columns: List[str]) -> Dict[str, Optional[str]]:
    """Resolve logical field -> concrete dataframe column."""
    return {logical: find_column(columns, candidates) for logical, candidates in COLUMN_CANDIDATES.items()}


def normalize_usage_rate(raw_usage: float) -> float:
    """Convert usage to 0-1 scale if needed."""
    if raw_usage <= 0:
        return 0.0
    return raw_usage / 100.0 if raw_usage > 1.5 else raw_usage


def estimate_age_from_class(class_year: str) -> float:
    """Approximate age using class-year bucket if explicit age is unavailable."""
    label = (class_year or "").upper()
    if "FR" in label:
        return 19.0
    if "SO" in label:
        return 20.0
    if "JR" in label:
        return 21.0
    if "SR" in label:
        return 22.0
    if "GR" in label or "GS" in label:
        return 23.0
    return 20.5


def infer_archetype(position: str, usage_rate: float, three_rate: float) -> str:
    """Heuristic archetype mapping into existing NBA valuation archetypes."""
    pos = (position or "").upper()
    has_guard = "G" in pos
    has_big = "C" in pos or "B" in pos
    has_forward = "F" in pos

    if has_guard:
        return "initiator_creator" if usage_rate >= 0.24 else "shooting_specialist"
    if has_big:
        return "connector" if three_rate >= 0.25 else "rim_protector"
    if has_forward:
        return "versatile_wing"
    return "default"


def compute_trust_score(games: int, minutes_per_game: float, coverage_score: float) -> float:
    """Estimate trust score in [0, 1] from sample and metric coverage."""
    games_norm = clamp(games / 40.0, 0.0, 1.0)
    minutes_norm = clamp(minutes_per_game / 30.0, 0.0, 1.0)
    trust = 0.15 + 0.45 * games_norm + 0.25 * minutes_norm + 0.15 * clamp(coverage_score, 0.0, 1.0)
    return clamp(trust, 0.2, 0.95)


def get_row_value(row: pd.Series, resolved_column: Optional[str], default: Any) -> Any:
    """Read row value using resolved column name."""
    if resolved_column is None:
        return default
    value = row.get(resolved_column, default)
    return default if pd.isna(value) else value


def build_card_from_row(row: pd.Series, col_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """Convert one canonical college row into a card compatible with PlayerValuator."""
    player_name = str(get_row_value(row, col_map["player_name"], "Unknown"))
    team_name = str(get_row_value(row, col_map["team_name"], "UNK")).upper()
    season = safe_int(get_row_value(row, col_map["season"], 0), 0)
    position = str(get_row_value(row, col_map["position"], ""))
    class_year = str(get_row_value(row, col_map["class_year"], ""))
    player_key = str(get_row_value(row, col_map["player_key"], ""))

    games = max(0, safe_int(get_row_value(row, col_map["games"], 0), 0))
    minutes_per_game = max(0.0, safe_float(get_row_value(row, col_map["minutes"], 0.0), 0.0))

    raw_usage = safe_float(get_row_value(row, col_map["usage_rate"], 0.0), 0.0)
    usage_rate = normalize_usage_rate(raw_usage)

    plus_minus = safe_float(get_row_value(row, col_map["plus_minus"], float("nan")), float("nan"))
    wins_share = safe_float(get_row_value(row, col_map["wins_share"], float("nan")), float("nan"))

    three_pa = max(0.0, safe_float(get_row_value(row, col_map["three_pa"], 0.0), 0.0))
    fga = max(0.0, safe_float(get_row_value(row, col_map["fga"], 0.0), 0.0))
    three_rate = three_pa / fga if fga > 0 else 0.0

    points = max(0.0, safe_float(get_row_value(row, col_map["points"], 0.0), 0.0))
    assists = max(0.0, safe_float(get_row_value(row, col_map["assists"], 0.0), 0.0))
    rebounds = max(0.0, safe_float(get_row_value(row, col_map["rebounds"], 0.0), 0.0))
    steals = max(0.0, safe_float(get_row_value(row, col_map["steals"], 0.0), 0.0))
    blocks = max(0.0, safe_float(get_row_value(row, col_map["blocks"], 0.0), 0.0))
    turnovers = max(0.0, safe_float(get_row_value(row, col_map["turnovers"], 0.0), 0.0))

    # Keep valuation stable even if BPM is missing.
    if math.isnan(plus_minus):
        if not math.isnan(wins_share) and games > 0:
            plus_minus = wins_share * 30.0 / max((minutes_per_game / 48.0) * games, 1.0)
        else:
            plus_minus = (points / 8.0) + ((steals + blocks) * 0.7) - (turnovers * 0.5)

    coverage_inputs = [usage_rate > 0, not math.isnan(wins_share), position.strip() != "", games > 0]
    coverage_score = sum(1 for flag in coverage_inputs if flag) / len(coverage_inputs)
    trust_score = compute_trust_score(games=games, minutes_per_game=minutes_per_game, coverage_score=coverage_score)
    uncertainty = clamp(1.0 - trust_score, 0.05, 0.95)

    archetype = infer_archetype(position=position, usage_rate=usage_rate, three_rate=three_rate)
    age = estimate_age_from_class(class_year)
    player_id = player_key or f"{slugify(player_name)}_{season}_{slugify(team_name)}"

    return {
        "player": {
            "id": player_id,
            "name": player_name,
            "team": team_name,
            "season": season,
            "position": position,
            "age": age,
        },
        "identity": {
            "primary_archetype": archetype,
            "usage_band": "high" if usage_rate >= 0.28 else ("med" if usage_rate >= 0.22 else "low"),
            "position": position,
        },
        "performance": {
            "traditional": {
                "minutes_per_game": minutes_per_game,
                "games_played": games,
                "points_per_game": points,
                "assists_per_game": assists,
                "rebounds_per_game": rebounds,
                "steals_per_game": steals,
                "blocks_per_game": blocks,
                "turnovers_per_game": turnovers,
                "field_goal_attempts_per_game": fga,
                "three_point_attempts_per_game": three_pa,
            },
            "advanced": {
                "plus_minus": plus_minus,
                "usage_rate": usage_rate,
                "wins_share": wins_share if not math.isnan(wins_share) else None,
            },
        },
        "impact": {
            "net": plus_minus,
            "offensive": points / 10.0,
            "defensive": (steals + blocks) * 0.5,
            "source": "college_proxy",
        },
        "metadata": {
            "games_played": str(games),
            "minutes": minutes_per_game,
            "data_quality": "actual" if games >= 25 and minutes_per_game >= 15 else "limited",
        },
        "v1_1_enhancements": {
            "trust_assessment": {
                "score": trust_score * 100.0,
            },
            "uncertainty_estimates": {
                "overall_uncertainty": uncertainty,
            },
        },
        "trust": {"score": trust_score},
        "uncertainty": {"overall": uncertainty},
    }


def add_check(
    checks: List[Dict[str, Any]],
    name: str,
    passed: bool,
    severity: str,
    detail: str,
    critical: bool = False,
) -> None:
    """Append one verification check result."""
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "critical": bool(critical),
            "detail": detail,
        }
    )


def validate_result_invariants(
    result: ValuationResult,
    valuator: PlayerValuator,
) -> List[str]:
    """Invariant checks to prevent silent valuation logic regressions."""
    issues: List[str] = []
    if not math.isfinite(result.current_wins_added):
        issues.append("current_wins_added is not finite")

    if not (result.trade_value_low <= result.trade_value_base <= result.trade_value_high):
        issues.append("trade value band ordering is invalid (low <= base <= high violated)")

    if set(result.market_value_by_year.keys()) != set(result.salary_by_year.keys()):
        issues.append("market_value_by_year and salary_by_year keys mismatch")
    if set(result.market_value_by_year.keys()) != set(result.surplus_by_year.keys()):
        issues.append("market_value_by_year and surplus_by_year keys mismatch")

    for year in result.market_value_by_year:
        market_val = result.market_value_by_year[year]
        salary_val = result.salary_by_year[year]
        surplus_val = result.surplus_by_year[year]
        if not all(math.isfinite(v) for v in [market_val, salary_val, surplus_val]):
            issues.append(f"non-finite contract values for year {year}")
            continue
        if abs((market_val - salary_val) - surplus_val) > 1e-6:
            issues.append(f"surplus consistency failed for year {year}")

    npv_recalc = valuator.calculate_npv(result.surplus_by_year)
    if not math.isfinite(result.npv_surplus) or abs(npv_recalc - result.npv_surplus) > 1e-6:
        issues.append("npv_surplus does not match discounted surplus cashflows")

    if any((mult < 0.5 or mult > 1.3 or not math.isfinite(mult)) for mult in result.aging_multipliers.values()):
        issues.append("aging multipliers out of expected bounds [0.5, 1.3]")

    return issues


def run_compliance_checks(
    df: pd.DataFrame,
    col_map: Dict[str, Optional[str]],
    build_summary_path: Path,
) -> List[Dict[str, Any]]:
    """Run data+process compliance checks aligned with pipeline expectations."""
    checks: List[Dict[str, Any]] = []

    required_core = ["player_name", "team_name", "season"]
    missing_core = [name for name in required_core if col_map.get(name) is None]
    add_check(
        checks,
        name="core_columns_present",
        passed=not missing_core,
        severity="critical",
        critical=True,
        detail="Missing core columns: none" if not missing_core else f"Missing core columns: {missing_core}",
    )

    required_provenance = ["source_table", "source_url", "scraped_at_utc", "player_key", "team_key"]
    missing_provenance = [name for name in required_provenance if col_map.get(name) is None]
    add_check(
        checks,
        name="provenance_columns_present",
        passed=not missing_provenance,
        severity="critical",
        critical=True,
        detail=(
            "Provenance columns available"
            if not missing_provenance
            else f"Missing provenance columns: {missing_provenance}"
        ),
    )

    if col_map.get("player_key"):
        duplicate_player_keys = int(df[col_map["player_key"]].astype(str).duplicated().sum())
        add_check(
            checks,
            name="player_key_uniqueness",
            passed=duplicate_player_keys == 0,
            severity="critical",
            critical=True,
            detail=f"Duplicate player_key rows: {duplicate_player_keys}",
        )

    if build_summary_path.exists():
        with open(build_summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        strict_robots = bool(summary.get("strict_robots", False))
        add_check(
            checks,
            name="robots_enforcement_enabled",
            passed=strict_robots,
            severity="critical",
            critical=True,
            detail=f"strict_robots={strict_robots}",
        )

        user_agent = str(summary.get("user_agent", "")).strip()
        add_check(
            checks,
            name="user_agent_provided",
            passed=bool(user_agent),
            severity="warning",
            critical=False,
            detail=f"user_agent='{user_agent}'",
        )

        source_errors = summary.get("errors", [])
        add_check(
            checks,
            name="ingestion_errors",
            passed=len(source_errors) == 0,
            severity="warning",
            critical=False,
            detail=f"Source ingestion errors: {len(source_errors)}",
        )
    else:
        add_check(
            checks,
            name="build_summary_exists",
            passed=False,
            severity="warning",
            critical=False,
            detail=f"Missing build summary file: {build_summary_path}",
        )

    return checks


def finalize_verification(
    checks: List[Dict[str, Any]],
    invalid_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build final verification object with pass/warn/fail summary."""
    checks_out = list(checks)
    if invalid_reports:
        checks_out.append(
            {
                "name": "valuation_invariants",
                "passed": False,
                "severity": "critical",
                "critical": True,
                "detail": f"Invalid valuation reports: {len(invalid_reports)}",
            }
        )

    critical_failures = [c for c in checks_out if c["critical"] and not c["passed"]]
    warning_failures = [c for c in checks if (not c["critical"]) and not c["passed"]]

    if critical_failures:
        overall_status = "fail"
    elif warning_failures:
        overall_status = "warn"
    else:
        overall_status = "pass"

    return {
        "run_at_utc": utc_now_iso(),
        "overall_status": overall_status,
        "critical_failures": len(critical_failures),
        "warning_failures": len(warning_failures),
        "checks": checks_out,
        "invalid_report_count": len(invalid_reports),
        "invalid_reports": invalid_reports[:200],
    }


def run_college_valuation(
    input_path: Path,
    output_dir: Path,
    build_summary_path: Path,
    max_players: Optional[int] = None,
    strict_verify: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """End-to-end college valuation with verification output."""
    df = read_csv_non_empty(input_path)
    if max_players is not None:
        df = df.head(max_players).copy()

    if df.empty:
        raise ValueError("Input dataframe has zero rows after filtering.")

    output_dir.mkdir(parents=True, exist_ok=True)
    col_map = resolve_column_map(list(df.columns))
    checks = run_compliance_checks(df=df, col_map=col_map, build_summary_path=build_summary_path)

    critical_failed = any(c["critical"] and not c["passed"] for c in checks)
    if critical_failed and strict_verify:
        verification = finalize_verification(checks=checks, invalid_reports=[])
        verification_path = output_dir / "college_valuation_verification.json"
        with open(verification_path, "w", encoding="utf-8") as handle:
            json.dump(verification, handle, indent=2)
        raise RuntimeError("Critical verification checks failed before valuation. See verification report.")

    valuator = PlayerValuator()
    reports: List[Dict[str, Any]] = []
    invalid_reports: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        card = build_card_from_row(row=row, col_map=col_map)
        result = valuator.valuate_player(card)
        issues = validate_result_invariants(result=result, valuator=valuator)
        report = valuator.generate_report(result)

        report["college_context"] = {
            "team_name": str(get_row_value(row, col_map["team_name"], "")),
            "conference": str(get_row_value(row, col_map["conf"], "")),
            "class_year": str(get_row_value(row, col_map["class_year"], "")),
            "position": str(get_row_value(row, col_map["position"], "")),
            "source_table": str(get_row_value(row, col_map["source_table"], "")),
            "source_url": str(get_row_value(row, col_map["source_url"], "")),
            "scraped_at_utc": str(get_row_value(row, col_map["scraped_at_utc"], "")),
        }

        if issues:
            invalid_reports.append(
                {
                    "player": result.player_name,
                    "season": result.season,
                    "issues": issues,
                }
            )

        player_name = sanitize_filename(result.player_name)
        team_code = sanitize_filename(str(get_row_value(row, col_map["team_name"], "UNK")).upper())
        filename = f"{player_name}_{team_code}_{result.season}_college_valuation.json"
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        reports.append(report)

    add_check(
        checks,
        name="non_empty_valuation_output",
        passed=len(reports) > 0,
        severity="critical",
        critical=True,
        detail=f"Valuation reports written: {len(reports)}",
    )

    summary = {
        "run_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "players_valuated": len(reports),
        "top_surplus_players": sorted(
            reports,
            key=lambda x: x["contract"]["npv_surplus"],
            reverse=True,
        )[:25],
    }

    with open(output_dir / "college_valuation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    verification = finalize_verification(checks=checks, invalid_reports=invalid_reports)
    with open(output_dir / "college_valuation_verification.json", "w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2)

    if strict_verify and verification["overall_status"] == "fail":
        raise RuntimeError("Valuation verification failed. See college_valuation_verification.json for details.")

    return summary, verification


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run NBA-style valuation on college player-season data with compliance verification."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/college/players_season.csv"),
        help="Canonical college player-season CSV input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/valuations/college"),
        help="Output directory for valuation reports.",
    )
    parser.add_argument(
        "--build-summary",
        type=Path,
        default=Path("data/processed/college/build_summary.json"),
        help="College ingestion build summary JSON (for robots/compliance checks).",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Optional max number of players to process (for smoke tests).",
    )
    parser.add_argument(
        "--non-strict-verify",
        action="store_true",
        help="Do not fail the run on verification failures; still write verification report.",
    )
    args = parser.parse_args()

    try:
        summary, verification = run_college_valuation(
            input_path=args.input,
            output_dir=args.output,
            build_summary_path=args.build_summary,
            max_players=args.max_players,
            strict_verify=not args.non_strict_verify,
        )
        print(f"[SUCCESS] Valuated {summary['players_valuated']} college player-seasons")
        print(f"  Summary: {args.output / 'college_valuation_summary.json'}")
        print(f"  Verification: {args.output / 'college_valuation_verification.json'}")
        print(f"  Verification status: {verification['overall_status']}")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
