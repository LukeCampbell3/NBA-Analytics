from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))

from research.market_quality.common import candidate_identity_columns


BADGES = {
    "PRICE_DEFENDABLE",
    "PRICE_DEPENDENT",
    "PRICE_UNTRUSTED",
    "SAFE_STATE_CORE_SHADOW",
    "SAFE_STATE_NEAR_CORE_SHADOW",
    "TRUE_UNSTABLE_REJECTED",
    "NEEDS_MORE_SAMPLE",
    "SIMILAR_STATES_SCATTERED",
    "USAGE_ROLE_SHIFT",
    "MINUTES_VOLATILE",
    "EVENT_START_VERIFIED",
    "PRELOCK_ONLY",
}


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(row: pd.Series, *columns: str, default: str = "") -> str:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return default


def _num(row: pd.Series, *columns: str) -> float | None:
    for column in columns:
        if column in row.index:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def _ensure_candidate_id(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    if "direction" not in out.columns and "side" in out.columns:
        out["direction"] = out["side"]
    if "market_line" not in out.columns and "line" in out.columns:
        out["market_line"] = out["line"]
    return candidate_identity_columns(out)


def _merge_optional(base: pd.DataFrame, extra: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if base.empty or extra.empty:
        return base
    extra = _ensure_candidate_id(extra)
    keep = ["candidate_id", *[column for column in columns if column in extra.columns]]
    merged = base.merge(extra[keep].drop_duplicates("candidate_id"), on="candidate_id", how="left", suffixes=("", "_extra"))
    for col in [column for column in merged.columns if column.endswith("_extra")]:
        original = col.removesuffix("_extra")
        if original in merged.columns:
            base_values = merged[original].astype("object")
            extra_values = merged[col].astype("object")
            merged[original] = base_values.where(base_values.notna() & base_values.astype(str).str.strip().ne(""), extra_values)
        else:
            merged[original] = merged[col]
        merged = merged.drop(columns=[col], errors="ignore")
    return merged


def _badges(row: pd.Series, *, true_unstable_ids: set[str], needs_sample_ids: set[str]) -> list[str]:
    badges: set[str] = set()
    edge_tier = _text(row, "edge_defendability_tier").upper()
    safe_tier = _text(row, "safe_state_tier").upper()
    root = _text(row, "root_cause_primary", "root_cause").upper()
    subtype = _text(row, "gap_subtype", "primary_blocker", "forecastability_gap_primary").upper()
    timestamp_basis = _text(row, "timestamp_safety_basis").upper()
    candidate_id = _text(row, "candidate_id")

    if edge_tier == "EDGE_DEFENDABLE":
        badges.add("PRICE_DEFENDABLE")
    elif edge_tier == "EDGE_PRICE_DEPENDENT":
        badges.add("PRICE_DEPENDENT")
    elif edge_tier:
        badges.add("PRICE_UNTRUSTED")
    if safe_tier == "SAFE_STATE_CORE":
        badges.add("SAFE_STATE_CORE_SHADOW")
    if safe_tier == "SAFE_STATE_NEAR_CORE":
        badges.add("SAFE_STATE_NEAR_CORE_SHADOW")
    if candidate_id in true_unstable_ids or root in {"REAL_MINUTES_VOLATILITY", "REAL_USAGE_VOLATILITY"}:
        badges.add("TRUE_UNSTABLE_REJECTED")
    if candidate_id in needs_sample_ids or root == "INSUFFICIENT_COMPARABLE_STATES":
        badges.add("NEEDS_MORE_SAMPLE")
    if "SCATTER" in subtype:
        badges.add("SIMILAR_STATES_SCATTERED")
    if "USAGE" in subtype or "ROLE_SHIFT" in subtype:
        badges.add("USAGE_ROLE_SHIFT")
    if "MINUTES" in subtype or "REAL_MINUTES" in root:
        badges.add("MINUTES_VOLATILE")
    if timestamp_basis == "EVENT_START_VERIFIED":
        badges.add("EVENT_START_VERIFIED")
    if timestamp_basis == "PRELOCK_RUN_VERIFIED":
        badges.add("PRELOCK_ONLY")
    return [badge for badge in sorted(badges) if badge in BADGES]


def _explanation(row: pd.Series, badges: list[str]) -> str:
    explicit = _text(row, "safe_state_explanation", "explanation", "rejection_reason_if_not_safe")
    if explicit:
        return f"{explicit} This label is shadow-only and does not change production picks."
    player = _text(row, "player", "player_name", default="Candidate")
    edge = _text(row, "edge_defendability_tier", default="edge status unknown")
    safe = _text(row, "safe_state_tier", default="safe-state unclassified")
    blocker = _text(row, "primary_blocker", "forecastability_gap_primary", default="no blocker recorded")
    badge_text = ", ".join(badges) if badges else "no warning badges"
    return f"{player} is classified as {safe} with {edge}; primary blocker: {blocker}. Badges: {badge_text}. Shadow-only evidence."


def _recommended_action(row: pd.Series, badges: list[str]) -> str:
    explicit = _text(row, "recommended_action", "recommended_next_action", "recommended_repair")
    if explicit:
        return explicit
    safe_tier = _text(row, "safe_state_tier").upper()
    edge_tier = _text(row, "edge_defendability_tier").upper()
    if "TRUE_UNSTABLE_REJECTED" in badges:
        return "KEEP_UNSAFE_TRUE_VOLATILITY"
    if "NEEDS_MORE_SAMPLE" in badges:
        return "NEEDS_MORE_SAMPLE"
    if safe_tier in {"SAFE_STATE_CORE", "SAFE_STATE_NEAR_CORE"}:
        return "WATCH_SHADOW_ONLY"
    if edge_tier == "EDGE_PRICE_DEPENDENT":
        return "PRICE_DEPENDENT_RESEARCH_ONLY"
    if edge_tier == "EDGE_UNTRUSTED_PRICE":
        return "PRICE_UNTRUSTED_RESEARCH_ONLY"
    return "SHADOW_MONITOR_ONLY"


def _card_from_row(row: pd.Series, *, true_unstable_ids: set[str], needs_sample_ids: set[str]) -> dict[str, Any]:
    badges = _badges(row, true_unstable_ids=true_unstable_ids, needs_sample_ids=needs_sample_ids)
    return {
        "candidate_id": _text(row, "candidate_id"),
        "player": _text(row, "player", "player_name"),
        "team": _text(row, "team", "market_team"),
        "opponent": _text(row, "opponent", "market_opponent"),
        "market_type": _text(row, "market_type"),
        "side": _text(row, "side", "direction"),
        "line": _num(row, "line", "market_line"),
        "price": _num(row, "market_side_price", "odds_american", "odds"),
        "break_even_probability": _num(row, "market_side_break_even", "break_even_probability"),
        "stress_probability": _num(row, "stress_probability", "p_side_stress", "expected_win_rate"),
        "lcb_probability": _num(row, "lcb_probability"),
        "stress_edge": _num(row, "stress_edge"),
        "lcb_edge": _num(row, "lcb_edge"),
        "edge_defendability_tier": _text(row, "edge_defendability_tier"),
        "forecastability_tier": _text(row, "forecastability_tier"),
        "structural_mispricing_tier": _text(row, "structural_mispricing_tier"),
        "similar_state_reliability_tier": _text(row, "similar_state_reliability_tier"),
        "safe_state_tier": _text(row, "safe_state_tier"),
        "primary_blocker": _text(row, "primary_blocker", "forecastability_gap_primary"),
        "root_cause": _text(row, "root_cause_primary", "root_cause"),
        "recommended_action": _recommended_action(row, badges),
        "settlement_status": _text(row, "settlement_status", default="PENDING"),
        "explanation": _explanation(row, badges),
        "warning_badges": badges,
        "shadow_only": True,
        "promotion_ready": False,
    }


def export_safe_state_site_cards(
    *,
    safe_state_run_dir: Path,
    output_dir: Path,
    run_date: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(safe_state_run_dir / "safe_state_production_shadow_manifest.json")
    annotated = _ensure_candidate_id(_read_csv(safe_state_run_dir / "safe_state_annotated_candidates.csv"))
    production = _ensure_candidate_id(_read_csv(safe_state_run_dir / "production_board_as_is.csv"))
    true_unstable = _ensure_candidate_id(_read_csv(safe_state_run_dir / "true_unstable_shadow_rejections.csv"))
    needs_sample = _ensure_candidate_id(_read_csv(safe_state_run_dir / "needs_more_sample_queue.csv"))
    blockers = _read_csv(safe_state_run_dir / "safe_state_candidate_blockers.csv")
    root_causes = _read_csv(safe_state_run_dir / "forecastability_root_cause_rows.csv")
    settlement = _read_csv(safe_state_run_dir / "safe_state_settlement_status_audit.csv")

    base = annotated if not annotated.empty else production
    if base.empty:
        base = pd.concat([production, true_unstable, needs_sample], ignore_index=True, sort=False)
    base = _ensure_candidate_id(base)
    base = _merge_optional(base, blockers, ["primary_blocker", "secondary_blockers", "missing_features", "evidence_gap_type"])
    base = _merge_optional(base, root_causes, ["root_cause_primary", "root_cause_secondary", "recommended_repair"])
    base = _merge_optional(base, settlement, ["settlement_status", "is_resolved", "resolution_reason", "actual_stat"])

    true_ids = set(true_unstable.get("candidate_id", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    sample_ids = set(needs_sample.get("candidate_id", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    cards = [_card_from_row(row, true_unstable_ids=true_ids, needs_sample_ids=sample_ids) for _, row in base.iterrows()]

    latest_csv = output_dir / "safe_state_latest.csv"
    cards_json = output_dir / "safe_state_cards.json"
    latest_json = output_dir / "safe_state_latest.json"
    pd.DataFrame(cards).to_csv(latest_csv, index=False)
    payload = {
        "run_id": manifest.get("run_id", ""),
        "run_date": run_date or manifest.get("run_date", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff_date": run_date or manifest.get("run_date", ""),
        "provider": manifest.get("provider", "sportsgameodds"),
        "production_behavior_changed": False,
        "promotion_ready": False,
        "shadow_only": True,
        "cards": cards,
    }
    cards_json.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    validation = validate_safe_state_site_cards(payload)
    validation_path = output_dir / "safe_state_site_validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {
        "output_paths": {
            "safe_state_latest_json": str(latest_json),
            "safe_state_latest_csv": str(latest_csv),
            "safe_state_cards_json": str(cards_json),
            "safe_state_site_validation_report": str(validation_path),
        },
        "card_count": int(len(cards)),
        "validation": validation,
        "production_behavior_changed": False,
        "promotion_ready": False,
        "shadow_only": True,
    }


def validate_safe_state_site_cards(payload: dict[str, Any]) -> dict[str, Any]:
    cards = payload.get("cards", [])
    cards = cards if isinstance(cards, list) else []
    missing_shadow = [card.get("candidate_id", "") for card in cards if not card.get("shadow_only")]
    unsafe_safe_labels = [
        card.get("candidate_id", "")
        for card in cards
        if card.get("safe_state_tier") not in {"SAFE_STATE_CORE", "SAFE_STATE_NEAR_CORE"}
        and any(badge in card.get("warning_badges", []) for badge in {"SAFE_STATE_CORE_SHADOW", "SAFE_STATE_NEAR_CORE_SHADOW"})
    ]
    production_behavior_changed = bool(payload.get("production_behavior_changed", False))
    promotion_ready = bool(payload.get("promotion_ready", False))
    return {
        "card_count": int(len(cards)),
        "safe_state_cards_include_shadow_status": not missing_shadow,
        "price_dependent_rows_not_labeled_safe": not unsafe_safe_labels,
        "production_behavior_changed": production_behavior_changed,
        "promotion_claim": False,
        "promotion_ready": promotion_ready,
        "staking_field_enabled": False,
        "validation_passed": not missing_shadow and not unsafe_safe_labels and not production_behavior_changed and not promotion_ready,
        "issues": {
            "missing_shadow_status_candidate_ids": missing_shadow,
            "unsafe_safe_badge_candidate_ids": unsafe_safe_labels,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export safe-state shadow evidence as site-ready cards.")
    parser.add_argument("--safe-state-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_safe_state_site_cards(safe_state_run_dir=args.safe_state_run_dir, output_dir=args.output_dir, run_date=args.run_date)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
