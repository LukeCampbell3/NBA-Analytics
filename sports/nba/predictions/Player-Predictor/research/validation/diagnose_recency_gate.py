from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PLAYER_PREDICTOR_ROOT = Path(__file__).resolve().parents[2]
if str(PLAYER_PREDICTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_counts(payload: dict[str, Any]) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    counts = payload.get("pipeline_stage_counts")
    if isinstance(counts, dict):
        return {str(k): int(v) for k, v in counts.items() if pd.notna(v)}
    diagnostics = payload.get("board_diagnostics", {})
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("stage_counts"), dict):
        return {str(k): int(v) for k, v in diagnostics["stage_counts"].items() if pd.notna(v)}
    return {}


def _policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("policy", {}) if isinstance(payload, dict) else {}
    return policy if isinstance(policy, dict) else {}


def _describe_numeric(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "min": np.nan, "p25": np.nan, "median": np.nan, "p75": np.nan, "max": np.nan}
    return {
        "count": int(len(numeric)),
        "min": float(numeric.min()),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
    }


def _date_summary(series: pd.Series) -> dict[str, Any]:
    dates = pd.to_datetime(series, errors="coerce")
    valid = dates.dropna()
    if valid.empty:
        return {"count": 0, "min": None, "max": None, "unique_count": 0}
    return {
        "count": int(len(valid)),
        "min": str(valid.min().date()),
        "max": str(valid.max().date()),
        "unique_count": int(valid.nunique()),
        "top_values": {
            str(key.date() if hasattr(key, "date") else key): int(value)
            for key, value in valid.value_counts().head(10).to_dict().items()
        },
    }


def _with_staleness(selector_rows: pd.DataFrame) -> pd.DataFrame:
    out = selector_rows.copy()
    market_dates = pd.to_datetime(out.get("market_date"), errors="coerce")
    history_dates = pd.to_datetime(out.get("last_history_date"), errors="coerce")
    if "history_staleness_days" not in out.columns:
        out["history_staleness_days"] = (market_dates - history_dates).dt.days
    else:
        explicit = pd.to_numeric(out["history_staleness_days"], errors="coerce")
        out["history_staleness_days"] = explicit.where(explicit.notna(), (market_dates - history_dates).dt.days)
    return out


def _root_cause(
    *,
    rows_before_recency: int,
    rows_after_recency: int,
    selector_rows: pd.DataFrame,
    min_recency_factor: float,
    max_history_staleness_days: int,
) -> tuple[str, str, bool, bool]:
    if selector_rows.empty:
        return ("empty_selector_pool", "verify upstream candidate generation before recency analysis", True, False)
    market_dates = pd.to_datetime(selector_rows.get("market_date"), errors="coerce")
    history_dates = pd.to_datetime(selector_rows.get("last_history_date"), errors="coerce")
    if market_dates.isna().mean() > 0.50:
        return ("market_date_parsing_issue", "repair market date normalization before rerunning production profile", True, False)
    if (history_dates.notna() & market_dates.notna() & (history_dates > market_dates)).mean() > 0.20:
        return ("market_date_or_history_date_ordering_issue", "audit market_date and last_history_date normalization", True, False)

    staleness = pd.to_numeric(selector_rows.get("history_staleness_days"), errors="coerce")
    stale_share = float((staleness > float(max_history_staleness_days)).mean()) if max_history_staleness_days > 0 and not staleness.empty else 0.0
    recency = pd.to_numeric(selector_rows.get("recency_factor"), errors="coerce")
    low_recency_share = float((recency < float(min_recency_factor)).mean()) if min_recency_factor > 0 and not recency.empty else 0.0
    if rows_before_recency > 0 and rows_after_recency == 0 and stale_share >= 0.80:
        return (
            "player_game_logs_stale",
            "refresh current-season Data-Proc/player game logs before the market date, then rerun production profile",
            True,
            False,
        )
    if rows_before_recency > 0 and rows_after_recency == 0 and low_recency_share >= 0.80:
        return (
            "recency_factor_below_production_threshold",
            "do not relax production recency by default; investigate history freshness or run a separate replay-validated fail-open study",
            False,
            True,
        )
    if rows_after_recency == 0:
        return ("no_usable_recent_history", "correctly produce no board until usable recent history exists", True, False)
    return ("recency_gate_not_global_failure", "no repair required from recency diagnosis", True, False)


def diagnose_recency_gate(
    *,
    selector_csv: Path,
    final_json: Path,
    slate_csv: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.resolve().mkdir(parents=True, exist_ok=True)
    selector_rows = _with_staleness(_read_csv(selector_csv))
    slate_rows = _read_csv(slate_csv)
    final_payload = _read_json(final_json)
    counts = _stage_counts(final_payload)
    policy = _policy(final_payload)

    min_recency = float(policy.get("min_recency_factor", 0.0) or 0.0)
    max_staleness = int(policy.get("max_history_staleness_days", 0) or 0)
    rows_before_recency = int(counts.get("after_initial_pool_gate", counts.get("selector_rows", len(selector_rows))))
    rows_after_recency = int(counts.get("after_recency", 0))

    recency = pd.to_numeric(selector_rows.get("recency_factor"), errors="coerce")
    staleness = pd.to_numeric(selector_rows.get("history_staleness_days"), errors="coerce")
    recency_removed_mask = pd.Series(False, index=selector_rows.index, dtype=bool)
    if min_recency > 0 and "recency_factor" in selector_rows.columns:
        recency_removed_mask |= recency < min_recency
    if max_staleness > 0 and "history_staleness_days" in selector_rows.columns:
        recency_removed_mask |= staleness > float(max_staleness)
    removed = selector_rows.loc[recency_removed_mask].copy()

    root_cause, repair, production_safe, requires_replay = _root_cause(
        rows_before_recency=rows_before_recency,
        rows_after_recency=rows_after_recency,
        selector_rows=selector_rows,
        min_recency_factor=min_recency,
        max_history_staleness_days=max_staleness,
    )

    market_dates = pd.to_datetime(selector_rows.get("market_date"), errors="coerce")
    history_dates = pd.to_datetime(selector_rows.get("last_history_date"), errors="coerce")
    stale_history_share = float((staleness > float(max_staleness)).mean()) if max_staleness > 0 and len(staleness) else 0.0
    diagnosis = {
        "input_paths": {
            "selector_csv": str(selector_csv),
            "final_json": str(final_json),
            "slate_csv": str(slate_csv) if slate_csv is not None else "",
        },
        "output_paths": {
            "recency_gate_diagnosis_json": str(output_dir / "recency_gate_diagnosis.json"),
            "recency_gate_diagnosis_md": str(output_dir / "recency_gate_diagnosis.md"),
            "recency_removed_rows_csv": str(output_dir / "recency_removed_rows.csv"),
        },
        "rows_before_recency": rows_before_recency,
        "rows_after_recency": rows_after_recency,
        "selector_rows": int(len(selector_rows)),
        "slate_rows": int(len(slate_rows)),
        "final_board_rows": int(counts.get("final_board_rows", 0)),
        "recency_factor_distribution": _describe_numeric(selector_rows.get("recency_factor", pd.Series(dtype=float))),
        "history_staleness_days_distribution": _describe_numeric(selector_rows.get("history_staleness_days", pd.Series(dtype=float))),
        "last_history_date_distribution": _date_summary(selector_rows.get("last_history_date", pd.Series(dtype=object))),
        "market_date_distribution": _date_summary(selector_rows.get("market_date", pd.Series(dtype=object))),
        "market_date": str(market_dates.dropna().max().date()) if market_dates.notna().any() else None,
        "min_recency_factor": min_recency,
        "max_history_staleness_days": max_staleness,
        "players_removed_by_recency": sorted(selector_rows.loc[recency_removed_mask, "player"].dropna().astype(str).unique().tolist())
        if "player" in selector_rows.columns
        else [],
        "targets_removed_by_recency": {
            str(key): int(value)
            for key, value in selector_rows.loc[recency_removed_mask, "target"].fillna("").astype(str).value_counts().to_dict().items()
        }
        if "target" in selector_rows.columns
        else {},
        "stale_history_share": stale_history_share,
        "stale_history_share_caused_global_failure": bool(rows_before_recency > 0 and rows_after_recency == 0 and stale_history_share >= 0.80),
        "data_pipeline_history_freshness_is_stale": bool(stale_history_share >= 0.80),
        "current_season_data_proc_missing_recent_games": bool(
            history_dates.notna().any()
            and market_dates.notna().any()
            and (market_dates.max() - history_dates.max()).days > max(1, max_staleness)
        ),
        "market_date_parsing_wrong": bool(market_dates.isna().mean() > 0.50) if len(market_dates) else False,
        "playoff_or_schedule_date_mismatch_possible": bool(
            history_dates.notna().any()
            and market_dates.notna().any()
            and (market_dates.max() - history_dates.max()).days >= 14
        ),
        "freshness_root_cause": root_cause,
        "recommended_repair": repair,
        "production_safe": bool(production_safe),
        "requires_replay_validation": bool(requires_replay),
        "production_gate_relaxed": False,
    }

    removed_csv = output_dir / "recency_removed_rows.csv"
    removed.to_csv(removed_csv, index=False)
    (output_dir / "recency_gate_diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    md = [
        "# Recency Gate Diagnosis",
        "",
        f"- Rows before recency: {rows_before_recency}",
        f"- Rows after recency: {rows_after_recency}",
        f"- Final board rows: {diagnosis['final_board_rows']}",
        f"- Root cause: {root_cause}",
        f"- Recommended repair: {repair}",
        f"- Production safe repair: {bool(production_safe)}",
        f"- Requires replay validation: {bool(requires_replay)}",
        f"- Stale history share: {stale_history_share:.3f}",
        f"- Last history date range: {diagnosis['last_history_date_distribution'].get('min')} to {diagnosis['last_history_date_distribution'].get('max')}",
        "",
        "No recency threshold was relaxed by this diagnostic.",
    ]
    (output_dir / "recency_gate_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return diagnosis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose production-profile recency gate empty-board failures.")
    parser.add_argument("--selector-csv", type=Path, required=True)
    parser.add_argument("--final-json", type=Path, required=True)
    parser.add_argument("--slate-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnosis = diagnose_recency_gate(
        selector_csv=args.selector_csv,
        final_json=args.final_json,
        slate_csv=args.slate_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
