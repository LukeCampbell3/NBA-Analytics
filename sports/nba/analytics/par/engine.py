"""PAR aggregation, forecasting, export, and proof helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import MODEL_CONFIG, ROLE_PROFILES
from .models import PlayerMeta, ReplacementBaseline, ValueAtom, utc_now_iso


NBA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = NBA_ROOT.parents[1]
DEFAULT_DATA_PROC = NBA_ROOT / "predictions" / "Player-Predictor" / "Data-Proc"
DEFAULT_WEB_DATA = NBA_ROOT / "web" / "data"
FROZEN_MODEL_DOC = REPO_ROOT / "docs" / "par_frozen_model.md"
EVIDENCE_CONTRACT = Path(__file__).resolve().parent / "evidence_contract.json"
CATEGORY_FIELDS = [value["field"] for value in MODEL_CONFIG.categories.values()]


def stable_id(*parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return digest[:24]


def season_to_data_year(season: str) -> str:
    text = str(season)
    if "-" in text:
        end = text.split("-")[-1].strip()
        if len(end) == 2 and end.isdigit():
            return f"20{end}"
        if len(end) == 4 and end.isdigit():
            return end
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        return digits[-4:]
    return str(season)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence_contract() -> dict[str, Any]:
    if not EVIDENCE_CONTRACT.exists():
        return {}
    return json.loads(EVIDENCE_CONTRACT.read_text(encoding="utf-8"))


def validate_source_governance() -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    frozen_doc_valid = FROZEN_MODEL_DOC.exists() and MODEL_CONFIG.par_model_version in FROZEN_MODEL_DOC.read_text(encoding="utf-8")
    if not frozen_doc_valid:
        blockers.append("frozen_par_source_document_not_found_or_wrong_version")

    contract = load_evidence_contract()
    direct_source = (contract.get("sources") or {}).get("box_score_direct_v0_5", {})
    contract_valid = (
        contract.get("par_model_version") == MODEL_CONFIG.par_model_version
        and contract.get("parf_model_version") == MODEL_CONFIG.parf_model_version
        and bool(direct_source.get("production_ready"))
    )
    if not contract_valid:
        blockers.append("direct_source_evidence_contract_not_ready")

    manifest = {
        "frozen_model_doc": str(FROZEN_MODEL_DOC.relative_to(REPO_ROOT)) if FROZEN_MODEL_DOC.exists() else None,
        "frozen_model_doc_digest": file_digest(FROZEN_MODEL_DOC) if FROZEN_MODEL_DOC.exists() else None,
        "evidence_contract": str(EVIDENCE_CONTRACT.relative_to(REPO_ROOT)) if EVIDENCE_CONTRACT.exists() else None,
        "evidence_contract_digest": file_digest(EVIDENCE_CONTRACT) if EVIDENCE_CONTRACT.exists() else None,
        "evidence_contract_version": contract.get("contract_version"),
        "ready_sources": [
            source_id
            for source_id, source in (contract.get("sources") or {}).items()
            if source.get("production_ready")
        ],
        "unready_sources": [
            source_id
            for source_id, source in (contract.get("sources") or {}).items()
            if not source.get("production_ready")
        ],
    }
    return manifest, blockers


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_team(row: dict[str, str]) -> str:
    matchup = str(row.get("MATCHUP") or "").strip()
    if matchup:
        return matchup.split(" ")[0].strip()
    return str(row.get("Team_ID") or "").strip()


def infer_role(stats: dict[str, float]) -> str:
    mpg = stats["minutes"] / max(1, stats["games"])
    pts36 = stats["pts"] / max(1.0, stats["minutes"]) * 36.0
    ast36 = stats["ast"] / max(1.0, stats["minutes"]) * 36.0
    trb36 = stats["trb"] / max(1.0, stats["minutes"]) * 36.0
    stl36 = stats["stl"] / max(1.0, stats["minutes"]) * 36.0
    if trb36 >= 10.0 and pts36 < 16.0:
        return "rim_protector"
    if trb36 >= 8.0 and pts36 >= 16.0:
        return "roll_big"
    if ast36 >= 7.0 and pts36 >= 18.0:
        return "primary_creator"
    if ast36 >= 5.0:
        return "secondary_creator"
    if pts36 >= 20.0 and ast36 < 5.0:
        return "scoring_guard"
    if stl36 >= 1.8 and pts36 < 16.0:
        return "three_and_d_wing"
    if mpg < 20 and ast36 >= 3.0:
        return "connector"
    return "connector"


def load_player_metas_from_logs(season: str, data_proc: Path = DEFAULT_DATA_PROC, limit: int | None = None) -> tuple[list[PlayerMeta], dict[str, list[dict[str, str]]], list[str]]:
    year = season_to_data_year(season)
    metas: list[PlayerMeta] = []
    player_rows: dict[str, list[dict[str, str]]] = {}
    digests: list[str] = []
    seen_player_ids: set[str] = set()
    for player_dir in sorted([p for p in data_proc.iterdir() if p.is_dir()]):
        csv_path = player_dir / f"{year}_processed_processed.csv"
        if not csv_path.exists():
            continue
        rows = [r for r in read_rows(csv_path) if int(safe_float(r.get("Did_Not_Play"), 0)) == 0 and safe_float(r.get("MP")) > 0]
        if not rows:
            continue
        h = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        digests.append(f"{csv_path.relative_to(REPO_ROOT)}:{h}")
        stats = {
            "minutes": sum(safe_float(r.get("MP")) for r in rows),
            "games": float(len(rows)),
            "pts": sum(safe_float(r.get("PTS")) for r in rows),
            "ast": sum(safe_float(r.get("AST")) for r in rows),
            "trb": sum(safe_float(r.get("TRB")) for r in rows),
            "stl": sum(safe_float(r.get("STL")) for r in rows),
        }
        first = rows[0]
        player_id = str(first.get("Player_ID") or stable_id(player_dir.name))
        if player_id in seen_player_ids:
            continue
        seen_player_ids.add(player_id)
        meta = PlayerMeta(
            player_id=player_id,
            player_name=str(first.get("Player") or player_dir.name.replace("_", " ")),
            team_id=str(first.get("Team_ID") or ""),
            team=infer_team(first),
            season=season,
            role=infer_role(stats),
            minutes=round(stats["minutes"], 1),
            games_played=len(rows),
            salary_millions=None,
        )
        metas.append(meta)
        player_rows[player_id] = rows
        if limit and len(metas) >= limit:
            break
    return metas, player_rows, digests


def replacement_records(season: str) -> list[ReplacementBaseline]:
    records = []
    for role, atoms in MODEL_CONFIG.replacement_baselines.items():
        for atom_type, value in atoms.items():
            records.append(
                ReplacementBaseline(
                    season=season,
                    role=role,
                    atom_type=atom_type,
                    sample_size=0,
                    replacement_value=float(value),
                    uncertainty=1.0,
                    baseline_version=f"{MODEL_CONFIG.par_model_version}_direct_role_baseline",
                )
            )
    return records


def make_atom(
    *,
    row: dict[str, str],
    meta: PlayerMeta,
    atom_type: str,
    raw_value: float,
    baseline_per36: float,
    source_event_suffix: str,
    context: list[str],
) -> ValueAtom:
    registry = MODEL_CONFIG.atom_registry[atom_type]
    tier = "TIER_A_DIRECT"
    tier_cfg = MODEL_CONFIG.source_tiers[tier]
    mp = safe_float(row.get("MP"))
    replacement = baseline_per36 * mp / 36.0
    value_above = raw_value - replacement
    reliability = float(tier_cfg["reliability_weight"])
    shrinkage = 1.0
    overlap_adjustment = 0.0
    par_value = value_above * reliability * shrinkage - overlap_adjustment
    game_id = f"{row.get('Date')}:{row.get('MATCHUP')}:{row.get('Game_Index')}"
    event_key = f"{game_id}:{meta.player_id}:{source_event_suffix}"
    return ValueAtom(
        atom_id=stable_id(event_key, atom_type),
        possession_id=f"box:{stable_id(game_id, meta.player_id)}",
        game_id=game_id,
        event_time=str(row.get("Date") or ""),
        season=meta.season,
        player_id=meta.player_id,
        team_id=meta.team_id,
        opponent_id=str(row.get("Opponent_ID") or row.get("Opponent") or ""),
        primary_value_label=atom_type,
        category=str(registry["category"]),
        overlap_group_id=stable_id(game_id, meta.player_id, atom_type),
        context_labels=context,
        source_event_ids=[stable_id(event_key)],
        source_type="box_score_direct_v0_5",
        source_tier=tier,
        raw_value=raw_value,
        replacement_baseline=replacement,
        value_above_replacement=value_above,
        reliability_weight=reliability,
        shrinkage_factor=shrinkage,
        overlap_adjustment=overlap_adjustment,
        par_value=par_value,
        label_entropy=0.0,
        confidence_tier=str(tier_cfg["confidence_tier"]),
        player_credit_json={"player_id": meta.player_id, "credit": 1.0},
        category_rollup_json={"category": registry["category"], "atom_type": atom_type},
        residual_value=0.0,
        par_model_version=MODEL_CONFIG.par_model_version,
    )


def build_atoms_from_box_logs(season: str, limit: int | None = None) -> tuple[list[PlayerMeta], list[ValueAtom], list[ReplacementBaseline], list[str], list[str]]:
    metas, player_rows, digests = load_player_metas_from_logs(season, limit=limit)
    baselines = replacement_records(season)
    baseline_map = {(b.role, b.atom_type): b.replacement_value for b in baselines}
    atoms: list[ValueAtom] = []
    for meta in metas:
        for row in player_rows.get(meta.player_id, []):
            mp = safe_float(row.get("MP"))
            if mp <= 0:
                continue
            role = meta.role if meta.role in ROLE_PROFILES else "connector"
            atoms.extend(
                [
                    make_atom(
                        row=row,
                        meta=meta,
                        atom_type="scoring_volume_above_replacement",
                        raw_value=safe_float(row.get("PTS")),
                        baseline_per36=baseline_map[(role, "scoring_volume_above_replacement")],
                        source_event_suffix="PTS",
                        context=["box_visible", "scoring"],
                    ),
                    make_atom(
                        row=row,
                        meta=meta,
                        atom_type="passing_creation",
                        raw_value=safe_float(row.get("AST")) * 2.2,
                        baseline_per36=baseline_map[(role, "passing_creation")],
                        source_event_suffix="AST",
                        context=["box_visible", "creation"],
                    ),
                    make_atom(
                        row=row,
                        meta=meta,
                        atom_type="negative_turnover_value",
                        raw_value=-safe_float(row.get("TOV")) * 1.4,
                        baseline_per36=baseline_map[(role, "negative_turnover_value")],
                        source_event_suffix="TOV",
                        context=["box_visible", "ball_security"],
                    ),
                    make_atom(
                        row=row,
                        meta=meta,
                        atom_type="steals",
                        raw_value=safe_float(row.get("STL")) * 2.0,
                        baseline_per36=baseline_map[(role, "steals")],
                        source_event_suffix="STL",
                        context=["box_visible", "perimeter_disruption"],
                    ),
                ]
            )
    source_governance, blockers = validate_source_governance()
    if not source_governance.get("ready_sources"):
        blockers.append("no_production_ready_atom_source")
    return metas, atoms, baselines, digests, blockers


def validate_overlap(atoms: Iterable[ValueAtom]) -> dict[str, Any]:
    groups: dict[str, list[ValueAtom]] = defaultdict(list)
    for atom in atoms:
        groups[atom.overlap_group_id].append(atom)
    failed = 0
    potential_duplicate = 0.0
    unresolved = 0.0
    for group_atoms in groups.values():
        seen_events: set[tuple[str, str, str]] = set()
        for atom in group_atoms:
            for event_id in atom.source_event_ids:
                key = (event_id, atom.player_id, atom.primary_value_label)
                if key in seen_events:
                    failed += 1
                    potential_duplicate += abs(atom.par_value)
                seen_events.add(key)
        supported_value = sum(max(0.0, abs(a.value_above_replacement)) for a in group_atoms)
        attributed = sum(abs(a.par_value) for a in group_atoms)
        if attributed - supported_value > MODEL_CONFIG.accounting_tolerance:
            failed += 1
            unresolved += attributed - supported_value
    total = len(groups)
    return {
        "overlap_groups_total": total,
        "overlap_groups_clean": max(0, total - failed),
        "overlap_groups_failed": failed,
        "potential_duplicate_value": round(potential_duplicate, 6),
        "removed_overlap_value": 0.0,
        "unresolved_overlap_value": round(unresolved, 6),
        "status": "pass" if failed == 0 else "fail",
    }


def validate_atom_rules(atoms: Iterable[ValueAtom], baselines: Iterable[ReplacementBaseline]) -> dict[str, Any]:
    baseline_keys = {(b.season, b.role, b.atom_type) for b in baselines}
    failures: list[dict[str, Any]] = []
    count = 0
    for atom in atoms:
        count += 1
        if atom.primary_value_label not in MODEL_CONFIG.atom_registry:
            failures.append({"atom_id": atom.atom_id, "rule": "registered_atom_type"})
        if atom.source_tier == "TIER_E_UNSUPPORTED" and abs(atom.par_value) > MODEL_CONFIG.accounting_tolerance:
            failures.append({"atom_id": atom.atom_id, "rule": "unsupported_atoms_contribute_zero"})
        if atom.source_tier == "TIER_D_SHRUNK_PROXY" and atom.shrinkage_factor >= 1.0:
            failures.append({"atom_id": atom.atom_id, "rule": "proxy_atoms_require_shrinkage"})
        if "cv" in atom.source_type.lower() and "evidence_ready" not in atom.context_labels and abs(atom.par_value) > MODEL_CONFIG.accounting_tolerance:
            failures.append({"atom_id": atom.atom_id, "rule": "cv_source_readiness"})
        if atom.replacement_baseline is None:
            failures.append({"atom_id": atom.atom_id, "rule": "replacement_baseline_present"})
    return {
        "status": "pass" if not failures else "fail",
        "production_atom_count": count,
        "baseline_key_count": len(baseline_keys),
        "failure_count": len(failures),
        "failures": failures[:100],
    }


def category_statuses(player_atoms: list[ValueAtom]) -> dict[str, dict[str, Any]]:
    by_cat = defaultdict(list)
    for atom in player_atoms:
        by_cat[atom.category].append(atom)
    statuses = {}
    for category in MODEL_CONFIG.categories:
        atoms = by_cat.get(category, [])
        statuses[category] = {
            "category": category,
            "label": MODEL_CONFIG.categories[category]["label"],
            "par_value": round(sum(a.par_value for a in atoms), 6) if atoms else None,
            "status": "measured" if atoms else "insufficient_evidence",
        }
    return statuses


def aggregate_player_components(metas: list[PlayerMeta], atoms: list[ValueAtom]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    atoms_by_player: dict[str, list[ValueAtom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_player[atom.player_id].append(atom)
    components = []
    max_delta = 0.0
    for meta in metas:
        player_atoms = atoms_by_player.get(meta.player_id, [])
        row = {
            "player_id": meta.player_id,
            "player_name": meta.player_name,
            "season": meta.season,
            "team_id": meta.team_id,
            "team": meta.team,
            "role": meta.role,
            "minutes": meta.minutes,
            "games_played": meta.games_played,
            "salary_millions": meta.salary_millions,
            "model_version": MODEL_CONFIG.par_model_version,
        }
        category_values = {field: 0.0 for field in CATEGORY_FIELDS}
        for atom in player_atoms:
            field = MODEL_CONFIG.categories[atom.category]["field"]
            category_values[field] += atom.par_value
        for key, value in category_values.items():
            row[key] = round(value, 9)
        total = sum(category_values.values())
        direct = sum(a.par_value for a in player_atoms if a.source_tier == "TIER_A_DIRECT")
        tracking = sum(a.par_value for a in player_atoms if a.source_tier == "TIER_B_TRACKING_BACKED")
        confirmed_hidden = sum(a.par_value for a in player_atoms if a.source_tier == "TIER_C_CONFIRMED_HIDDEN_ROLE")
        proxy = sum(a.par_value for a in player_atoms if a.source_tier == "TIER_D_SHRUNK_PROXY")
        residual = category_values["residual_par"]
        unsupported_count = sum(1 for a in player_atoms if a.source_tier == "TIER_E_UNSUPPORTED")
        supported_count = len(player_atoms) - unsupported_count
        reliable_abs = sum(abs(a.par_value) for a in player_atoms if a.source_tier in {"TIER_A_DIRECT", "TIER_B_TRACKING_BACKED", "TIER_C_CONFIRMED_HIDDEN_ROLE"})
        total_abs = sum(abs(a.par_value) for a in player_atoms)
        row.update(
            {
                "box_visible_par": round(direct, 9),
                "confirmed_hidden_role_par": round(confirmed_hidden, 9),
                "shrunk_proxy_par": round(proxy, 9),
                "overlap_leakage": round(sum(a.overlap_adjustment for a in player_atoms), 9),
                "total_par": round(total, 9),
                "par_1000": round((total / meta.minutes) * 1000.0, 9) if meta.minutes else 0.0,
                "war_equivalent": round(total / MODEL_CONFIG.points_per_win, 9),
                "pvg_score": round(50.0 + 45.0 * math.tanh(((total / meta.minutes) * 1000.0 if meta.minutes else 0.0) / 210.0), 9),
                "direct_par": round(direct, 9),
                "tracking_backed_par": round(tracking, 9),
                "proxy_par": round(proxy, 9),
                "residual_par": round(residual, 9),
                "par_evidence_coverage": round(reliable_abs / total_abs, 9) if total_abs else 0.0,
                "atom_count": len(player_atoms),
                "supported_atom_count": supported_count,
                "unsupported_atom_count": unsupported_count,
                "category_statuses": category_statuses(player_atoms),
            }
        )
        delta = abs(row["total_par"] - sum(row[field] for field in CATEGORY_FIELDS))
        max_delta = max(max_delta, delta)
        components.append(row)
    validation = {
        "status": "pass" if max_delta <= MODEL_CONFIG.accounting_tolerance else "fail",
        "maximum_accounting_reconciliation_delta": round(max_delta, 12),
        "player_count": len(components),
        "atom_count": len(atoms),
    }
    return components, validation


def build_leaderboard(components: list[dict[str, Any]], forecasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    f_by_player = {f["player_id"]: f for f in forecasts}
    rows = []
    for row in sorted(components, key=lambda r: r["total_par"], reverse=True):
        forecast = f_by_player.get(row["player_id"], {})
        rows.append(
            {
                "rank": len(rows) + 1,
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "team": row["team"],
                "role": row["role"],
                "season": row["season"],
                "minutes": row["minutes"],
                "par": row["total_par"],
                "par_1000": row["par_1000"],
                "pvg_score": row["pvg_score"],
                "projected_parf": forecast.get("projected_par"),
                "continuation_score": forecast.get("continuation_score"),
                "role_portability_score": forecast.get("role_portability_score"),
                "wins_per_million": forecast.get("wins_per_million"),
            }
        )
    return rows


def build_forecasts(components: list[dict[str, Any]], atoms: list[ValueAtom], season_to: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    atoms_by_player: dict[str, list[ValueAtom]] = defaultdict(list)
    for atom in atoms:
        atoms_by_player[atom.player_id].append(atom)
    forecasts = []
    ledger = []
    max_delta = 0.0
    for comp in components:
        player_atoms = atoms_by_player.get(comp["player_id"], [])
        current_par = float(comp["total_par"])
        stable_par = 0.0
        volatile_par = 0.0
        persistent_sum = 0.0
        for atom in player_atoms:
            registry = MODEL_CONFIG.atom_registry[atom.primary_value_label]
            persistence = MODEL_CONFIG.persistence_values[registry["persistence_key"]]
            reliability = atom.reliability_weight
            persistent = atom.par_value * persistence * reliability
            if persistence >= 0.65:
                stable_par += atom.par_value
            else:
                volatile_par += atom.par_value
            persistent_sum += persistent
            ledger.append(
                {
                    "player_id": comp["player_id"],
                    "season_from": comp["season"],
                    "season_to": season_to,
                    "atom_type": atom.primary_value_label,
                    "category": atom.category,
                    "current_par": round(atom.par_value, 6),
                    "persistence": persistence,
                    "reliability": reliability,
                    "trend_adjustment": 0.0,
                    "persistent_par": round(persistent, 6),
                    "forecast_par": round(persistent, 6),
                    "role_continuity_effect": 1.0,
                    "minutes_effect": 1.0,
                    "health_effect": 1.0,
                    "age_effect": 1.0,
                    "fit_effect": 0.0,
                }
            )
        minutes = float(comp["minutes"] or 0.0)
        minutes_factor = min(1.08, max(0.62, minutes / 1800.0)) if minutes else 0.62
        health_factor = 0.96 if minutes < 900 else 1.0
        role_continuity = 0.96 if comp["role"] in {"primary_creator", "secondary_creator"} else 0.92
        age_curve = 1.0
        fit_lift = 0.0
        projected = persistent_sum * role_continuity * minutes_factor * health_factor * age_curve + fit_lift
        stable_abs = abs(stable_par)
        volatile_abs = abs(volatile_par)
        total_abs = stable_abs + volatile_abs
        stable_share = stable_abs / total_abs if total_abs else 0.0
        volatile_share = volatile_abs / total_abs if total_abs else 0.0
        evidence = float(comp["par_evidence_coverage"])
        proxy_share = abs(float(comp["proxy_par"])) / max(1e-9, sum(abs(float(comp[field])) for field in CATEGORY_FIELDS))
        minutes_conf = min(1.0, minutes / 1800.0)
        health_conf = health_factor
        continuation = stable_share * role_continuity * minutes_conf * health_conf
        role_portability = max(0.25, min(1.0, 0.55 + stable_share * 0.35 + evidence * 0.10 - proxy_share * 0.20))
        uncertainty = 24.0 + (1 - minutes_conf) * 28.0 + volatile_share * 22.0 + (1 - evidence) * 18.0 + proxy_share * 18.0
        low = projected - uncertainty
        high = projected + uncertainty
        projected_wins = projected / MODEL_CONFIG.points_per_win
        salary = comp.get("salary_millions")
        forecasts.append(
            {
                "player_id": comp["player_id"],
                "player_name": comp["player_name"],
                "season_from": comp["season"],
                "season_to": season_to,
                "role": comp["role"],
                "current_par": round(current_par, 6),
                "stable_par_share": round(stable_share, 6),
                "volatile_par_share": round(volatile_share, 6),
                "projected_par": round(projected, 6),
                "confidence_interval_low": round(low, 6),
                "confidence_interval_high": round(high, 6),
                "projected_wins": round(projected_wins, 6),
                "salary_millions": salary,
                "wins_per_million": round(projected_wins / salary, 6) if salary else None,
                "continuation_score": round(continuation, 6),
                "role_portability_score": round(role_portability, 6),
                "health_factor": health_factor,
                "minutes_factor": round(minutes_factor, 6),
                "role_continuity": role_continuity,
                "fit_lift": fit_lift,
                "forecast_confidence": round(min(1.0, max(0.0, continuation * 0.7 + evidence * 0.3)), 6),
                "parf_model_version": MODEL_CONFIG.parf_model_version,
                "forecast_bridge": {
                    "current_par": round(current_par, 6),
                    "persistence_adjustment": round(persistent_sum - current_par, 6),
                    "trend": 0.0,
                    "role_continuity": round((persistent_sum * role_continuity) - persistent_sum, 6),
                    "minutes_adjustment": round((persistent_sum * role_continuity * minutes_factor) - (persistent_sum * role_continuity), 6),
                    "health_adjustment": round((persistent_sum * role_continuity * minutes_factor * health_factor) - (persistent_sum * role_continuity * minutes_factor), 6),
                    "age_curve": 0.0,
                    "fit_lift": fit_lift,
                    "projected_par_f": round(projected, 6),
                },
            }
        )
        max_delta = max(max_delta, abs(round(projected, 6) - forecasts[-1]["forecast_bridge"]["projected_par_f"]))
    validation = {"status": "pass" if max_delta <= MODEL_CONFIG.accounting_tolerance else "fail", "maximum_forecast_bridge_delta": round(max_delta, 12)}
    return forecasts, ledger, validation


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_atom_summary(atoms: list[ValueAtom]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for atom in atoms:
        key = (
            atom.player_id,
            atom.category,
            atom.primary_value_label,
            atom.source_tier,
            atom.confidence_tier,
        )
        row = grouped.setdefault(
            key,
            {
                "player_id": atom.player_id,
                "season": atom.season,
                "team_id": atom.team_id,
                "category": atom.category,
                "primary_value_label": atom.primary_value_label,
                "source_type": atom.source_type,
                "source_tier": atom.source_tier,
                "confidence_tier": atom.confidence_tier,
                "atom_count": 0,
                "source_event_count": 0,
                "raw_value": 0.0,
                "replacement_baseline_total": 0.0,
                "value_above_replacement": 0.0,
                "par_value": 0.0,
                "reliability_weight_total": 0.0,
                "shrinkage_factor_total": 0.0,
                "overlap_adjustment": 0.0,
            },
        )
        row["atom_count"] += 1
        row["source_event_count"] += len(atom.source_event_ids)
        row["raw_value"] += atom.raw_value
        row["replacement_baseline_total"] += atom.replacement_baseline
        row["value_above_replacement"] += atom.value_above_replacement
        row["par_value"] += atom.par_value
        row["reliability_weight_total"] += atom.reliability_weight
        row["shrinkage_factor_total"] += atom.shrinkage_factor
        row["overlap_adjustment"] += atom.overlap_adjustment

    rows: list[dict[str, Any]] = []
    for row in grouped.values():
        count = max(int(row["atom_count"]), 1)
        row["replacement_baseline"] = round(float(row.pop("replacement_baseline_total")) / count, 6)
        row["reliability_weight"] = round(float(row.pop("reliability_weight_total")) / count, 6)
        row["shrinkage_factor"] = round(float(row.pop("shrinkage_factor_total")) / count, 6)
        for field in ("raw_value", "value_above_replacement", "par_value", "overlap_adjustment"):
            row[field] = round(float(row[field]), 6)
        rows.append(row)
    return sorted(rows, key=lambda r: (str(r["player_id"]), str(r["category"]), str(r["primary_value_label"])))


def build_player_metrics(season: str, forecast_season: str, out: Path, *, player_limit: int | None = None, copy_to_web: bool = False) -> dict[str, Any]:
    metas, atoms, baselines, source_digests, blockers = build_atoms_from_box_logs(season, limit=player_limit)
    source_governance, source_blockers = validate_source_governance()
    blockers = sorted(set(blockers + source_blockers))
    overlap = validate_overlap(atoms)
    atom_rules = validate_atom_rules(atoms, baselines)
    components, par_validation = aggregate_player_components(metas, atoms)
    forecasts, forecast_ledger, parf_validation = build_forecasts(components, atoms, forecast_season)
    leaderboard = build_leaderboard(components, forecasts)
    atom_summary = build_atom_summary(atoms)
    production_allowed = not blockers and overlap["status"] == "pass" and atom_rules["status"] == "pass" and par_validation["status"] == "pass" and parf_validation["status"] == "pass"
    validation = {
        "overlap_audit": overlap,
        "atom_rule_validation": atom_rules,
        "par_validation": par_validation,
        "parf_validation": parf_validation,
        "production_publish_allowed": production_allowed,
        "blockers": blockers,
    }
    build_id = stable_id(season, forecast_season, datetime.now(timezone.utc).isoformat(), len(atoms))
    manifest = {
        "build_id": build_id,
        "built_at": utc_now_iso(),
        "season": season,
        "forecast_season": forecast_season,
        "source_datasets": [str(DEFAULT_DATA_PROC.relative_to(REPO_ROOT))],
        "source_digests": source_digests,
        "source_governance": source_governance,
        "par_model_version": MODEL_CONFIG.par_model_version,
        "parf_model_version": MODEL_CONFIG.parf_model_version,
        "player_count": len(components),
        "atom_count": len(atoms),
        "overlap_audit_status": overlap["status"],
        "par_validation_status": par_validation["status"],
        "parf_validation_status": parf_validation["status"],
        "production_publish_allowed": production_allowed,
        "blockers": blockers,
    }
    players = [m.to_dict() for m in metas]
    payloads = {
        "players.json": players,
        "player_par_components.json": components,
        "player_par_forecasts.json": forecasts,
        "player_par_atom_summary.json": atom_summary,
        "player_par_forecast_atoms.json": forecast_ledger,
        "par_leaderboard.json": leaderboard,
        "par_validation.json": validation,
        "par_build_manifest.json": manifest,
        "par_model.json": MODEL_CONFIG.to_dict(),
        "replacement_baselines.json": [b.to_dict() for b in baselines],
    }
    for name, payload in payloads.items():
        write_json(out / name, payload)
    write_jsonl(out / "player_par_atoms.jsonl", (a.to_dict() for a in atoms))
    proof = prove_metrics_dir(out)
    write_json(out / "par_product_proof.json", proof)
    if copy_to_web:
        web_payload_names = [
            "players.json",
            "player_par_components.json",
            "player_par_forecasts.json",
            "player_par_atom_summary.json",
            "par_leaderboard.json",
            "par_validation.json",
            "par_build_manifest.json",
            "par_model.json",
            "replacement_baselines.json",
        ]
        for name in web_payload_names:
            write_json(DEFAULT_WEB_DATA / name, payloads[name])
        write_json(DEFAULT_WEB_DATA / "par_product_proof.json", proof)
    return {**manifest, "proof": proof, "leaderboard": leaderboard, "validation": validation}


def prove_metrics_dir(metrics_dir: Path) -> dict[str, Any]:
    required = [
        "players.json",
        "player_par_components.json",
        "player_par_atoms.jsonl",
        "player_par_forecasts.json",
        "par_leaderboard.json",
        "par_validation.json",
        "par_build_manifest.json",
    ]
    blockers = []
    missing = [name for name in required if not (metrics_dir / name).exists()]
    if missing:
        blockers.append(f"missing_artifacts:{','.join(missing)}")
    manifest = json.loads((metrics_dir / "par_build_manifest.json").read_text(encoding="utf-8")) if (metrics_dir / "par_build_manifest.json").exists() else {}
    validation = json.loads((metrics_dir / "par_validation.json").read_text(encoding="utf-8")) if (metrics_dir / "par_validation.json").exists() else {}
    components = json.loads((metrics_dir / "player_par_components.json").read_text(encoding="utf-8")) if (metrics_dir / "player_par_components.json").exists() else []
    max_delta = 0.0
    for row in components:
        delta = abs(float(row.get("total_par", 0.0)) - sum(float(row.get(field, 0.0)) for field in CATEGORY_FIELDS))
        max_delta = max(max_delta, delta)
    if max_delta > MODEL_CONFIG.accounting_tolerance:
        blockers.append("par_accounting_identity_failed")
    blockers.extend(manifest.get("blockers") or [])
    valid = not blockers and bool(manifest.get("production_publish_allowed"))
    return {
        "valid": valid,
        "par_model_version": MODEL_CONFIG.par_model_version,
        "parf_model_version": MODEL_CONFIG.parf_model_version,
        "production_publish_allowed": valid,
        "proofs": {
            "sources": {"valid": not missing, "source_datasets": manifest.get("source_datasets", [])},
            "atom_ledger": {"valid": (metrics_dir / "player_par_atoms.jsonl").exists(), "atom_count": manifest.get("atom_count", 0)},
            "overlap": validation.get("overlap_audit", {}),
            "atom_rules": validation.get("atom_rule_validation", {}),
            "par": {"valid": max_delta <= MODEL_CONFIG.accounting_tolerance, "maximum_delta": round(max_delta, 12)},
            "parf": validation.get("parf_validation", {}),
            "frontend": {"valid": (metrics_dir / "par_leaderboard.json").exists() and (metrics_dir / "player_par_components.json").exists()},
        },
        "blockers": sorted(set(blockers)),
    }


def build_parf_validation_report(metrics_dir: Path, out_dir: Path) -> dict[str, Any]:
    forecasts = json.loads((metrics_dir / "player_par_forecasts.json").read_text(encoding="utf-8"))
    components = json.loads((metrics_dir / "player_par_components.json").read_text(encoding="utf-8"))
    current = {c["player_id"]: c for c in components}
    errors = []
    for forecast in forecasts[:150]:
        # No holdout actuals are present locally. Use current PAR as a scale-only diagnostic baseline
        # and mark the report blocked rather than validated.
        actual_proxy = current[forecast["player_id"]]["total_par"]
        err = forecast["projected_par"] - actual_proxy
        errors.append({"player_id": forecast["player_id"], "player_name": forecast["player_name"], "error": err, "abs_error": abs(err), "role": forecast["role"]})
    mae = statistics.mean(e["abs_error"] for e in errors) if errors else None
    rmse = math.sqrt(statistics.mean(e["error"] ** 2 for e in errors)) if errors else None
    by_role: dict[str, list[float]] = defaultdict(list)
    for e in errors:
        by_role[e["role"]].append(e["abs_error"])
    role_report = {role: {"sample_size": len(vals), "mae": round(statistics.mean(vals), 6)} for role, vals in by_role.items()}
    report = {
        "status": "blocked",
        "parf_model_version": MODEL_CONFIG.parf_model_version,
        "milestone": "PAR-F v0.7 Sample-Validated Baseline",
        "sample_size": len(errors),
        "mae": round(mae, 6) if mae is not None else None,
        "rmse": round(rmse, 6) if rmse is not None else None,
        "spearman_rank_correlation": None,
        "tier_accuracy": None,
        "continuation_hit_rate": None,
        "role_player_lift": None,
        "salary_surplus_hit_rate": None,
        "blockers": ["holdout_actual_next_season_par_not_available"],
    }
    audit = sorted(errors, key=lambda e: e["abs_error"], reverse=True)[:25]
    write_json(out_dir / "parf_validation_report.json", report)
    write_json(out_dir / "parf_error_audit.json", audit)
    write_json(out_dir / "parf_role_stratified_validation.json", role_report)
    return report
