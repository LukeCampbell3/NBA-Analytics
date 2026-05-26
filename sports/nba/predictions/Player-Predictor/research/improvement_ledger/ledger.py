from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.common import append_jsonl, utc_now_iso


DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent / "improvement_ledger.jsonl"

REQUIRED_FIELDS = [
    "improvement_id",
    "failure_mode_id",
    "intervention_id",
    "author_or_run_id",
    "hypothesis",
    "implementation_files",
    "validation_windows",
    "metrics_before",
    "metrics_after",
    "segment_results",
    "promotion_status",
    "blocked_reasons",
    "rollback_rule",
    "final_decision",
]


def normalize_improvement_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    payload.setdefault("created_at", utc_now_iso())
    for field in REQUIRED_FIELDS:
        if field in {"implementation_files", "validation_windows", "blocked_reasons"}:
            payload.setdefault(field, [])
        elif field in {"metrics_before", "metrics_after", "segment_results"}:
            payload.setdefault(field, {})
        else:
            payload.setdefault(field, "")
    payload["implementation_files"] = [str(value) for value in payload.get("implementation_files", [])]
    payload["validation_windows"] = list(payload.get("validation_windows", []))
    payload["blocked_reasons"] = list(payload.get("blocked_reasons", []))
    payload["rollback_rule"] = str(payload.get("rollback_rule", "")).strip()
    if not payload["rollback_rule"]:
        raise ValueError("Every improvement ledger entry must include a rollback_rule.")
    return payload


def append_improvement_entry(entry: dict[str, Any], *, ledger_path: Path | None = None) -> dict[str, Any]:
    normalized = normalize_improvement_entry(entry)
    append_jsonl(ledger_path or DEFAULT_LEDGER_PATH, normalized)
    return normalized


def load_improvement_ledger(ledger_path: Path | None = None) -> pd.DataFrame:
    path = (ledger_path or DEFAULT_LEDGER_PATH).resolve()
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return pd.DataFrame(rows)
