"""Feature registry builder (mission spec section 8).

Audits the real existing pregame datasets identified in
``reports/INVENTORY.md`` -- it does NOT hand-author feature names. Every
column classified here was read directly from
``historical_pool_universe_2026.csv`` (MLB) and ``backtest_rows.csv`` (NFL).

Classification is deliberately conservative: any column whose provenance
inspection did not clearly establish it as a pregame-only, non-circular
signal is marked UNUSABLE or POSTGAME_FORBIDDEN rather than guessed into
USABLE. In particular, columns that are themselves an *existing per-sport
model's* prediction, selection, or validation-error metadata (MLB
``Prediction``/``Edge``/``Model_Selected``/``Model_Members``/
``Model_Val_MAE``/``Model_Val_RMSE``; NFL ``prediction``/
``current_prediction``/``challenger_prediction``) are excluded as UNUSABLE:
feeding an incumbent model's own output into the universal model as an
input feature would make any later "universal vs. existing predictor"
comparison (spec section 52) circular.

Run as a script to regenerate ``manifests/feature_registry.json``:
    python -m sports.universal_model.data.feature_registry
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"

ALLOWED_CLASSES = {
    "UNIVERSAL",
    "SPORT_SPECIFIC",
    "MARKET",
    "TARGET",
    "POSTGAME_FORBIDDEN",
    "IDENTIFIER_ONLY",
    "UNUSABLE",
}


@dataclass(frozen=True)
class FeatureRecord:
    feature_name: str
    sport: str
    semantic_family: str
    dtype: str
    source: str
    availability_timestamp: str
    missing_rate: float
    unique_count: int
    normalization: str
    allowed_for_training: bool
    classification: str
    reason: str


# (source_column, sport, semantic_family, classification, normalization, availability_timestamp, reason)
_MLB_COLUMN_SPEC: list[tuple[str, str, str, str, str]] = [
    ("Prediction_Run_Date", "temporal_state", "IDENTIFIER_ONLY", "none", "run metadata, not a sporting-event feature"),
    ("Game_Date", "temporal_state", "IDENTIFIER_ONLY", "none", "used as split key, not fed as a raw input"),
    ("Commence_Time_UTC", "temporal_state", "UNIVERSAL", "calendar_embedding", "real scheduled start time, known well pregame"),
    ("Game_ID", "identity", "IDENTIFIER_ONLY", "none", "event identity key"),
    ("Game_Status_Code", "environment_state", "POSTGAME_FORBIDDEN", "none", "status such as 'Final' is only true after the event; unsafe without a verified pregame-only snapshot"),
    ("Game_Status_Detail", "environment_state", "POSTGAME_FORBIDDEN", "none", "same reasoning as Game_Status_Code"),
    ("Player", "identity", "IDENTIFIER_ONLY", "none", "entity display name"),
    ("Player_ID", "identity", "IDENTIFIER_ONLY", "none", "entity identity key"),
    ("Player_Type", "role_state", "SPORT_SPECIFIC", "categorical_embedding", "batter/pitcher role, known pregame"),
    ("Team", "identity", "IDENTIFIER_ONLY", "none", "captured structurally via UniversalEvent.team_id"),
    ("Opponent", "identity", "IDENTIFIER_ONLY", "none", "captured structurally via UniversalEvent.opponent_id"),
    ("Is_Home", "role_state", "UNIVERSAL", "boolean", "known pregame from the schedule"),
    ("Target", "identity", "TARGET", "none", "target/stat identity, mapped to UniversalEvent.target"),
    ("Prediction", "meta_model", "UNUSABLE", "none", "existing MLB model's own prediction output; circular input for a model later compared against that same predictor"),
    ("Market_Line", "market_state", "MARKET", "market_scaling", "real pregame quoted line"),
    ("Market_Source", "market_state", "MARKET", "categorical_embedding", "odds provider identity"),
    ("Market_Books", "market_state", "MARKET", "categorical_embedding", "books contributing to the quote"),
    ("Market_Line_Std", "market_state", "MARKET", "market_scaling", "standardized variant of Market_Line"),
    ("Market_Over_Price", "market_state", "MARKET", "market_scaling", "real pregame quoted price"),
    ("Market_Under_Price", "market_state", "MARKET", "market_scaling", "real pregame quoted price"),
    ("Market_Over_Book_Key", "market_state", "MARKET", "categorical_embedding", "book identity"),
    ("Market_Under_Book_Key", "market_state", "MARKET", "categorical_embedding", "book identity"),
    ("Market_Over_Book", "market_state", "MARKET", "categorical_embedding", "book identity"),
    ("Market_Under_Book", "market_state", "MARKET", "categorical_embedding", "book identity"),
    ("Market_Over_Price_Time", "market_state", "MARKET", "none", "quote timestamp, used to compute quote age"),
    ("Market_Under_Price_Time", "market_state", "MARKET", "none", "quote timestamp, used to compute quote age"),
    ("Edge", "meta_model", "UNUSABLE", "none", "derived from the existing model's own Prediction vs. market; same circularity as Prediction"),
    ("History_Rows", "support", "UNIVERSAL", "log_transform", "real pregame sample-support count for this entity/target"),
    ("Last_History_Date", "recency", "UNIVERSAL", "relative_time", "converted to days-since-cutoff at feature-build time, never used as an absolute date"),
    ("Model_Selected", "meta_model", "UNUSABLE", "none", "identifies which existing sub-model was used; pipeline metadata about a different model"),
    ("Model_Members", "meta_model", "UNUSABLE", "none", "same reasoning as Model_Selected"),
    ("Model_Val_MAE", "meta_model", "UNUSABLE", "none", "the incumbent model's own historical validation error; circularity + undisclosed refit timing risk"),
    ("Model_Val_RMSE", "meta_model", "UNUSABLE", "none", "same reasoning as Model_Val_MAE"),
    ("Result", "target", "TARGET", "none", "settlement outcome, mapped to UniversalEvent.binary_result"),
    ("Actual", "target", "TARGET", "none", "settled real value, mapped to UniversalEvent.actual_value"),
]

_NFL_COLUMN_SPEC: list[tuple[str, str, str, str, str]] = [
    ("player_id", "identity", "IDENTIFIER_ONLY", "none", "entity identity key"),
    ("player_display_name", "identity", "IDENTIFIER_ONLY", "none", "entity display name"),
    ("position", "role_state", "SPORT_SPECIFIC", "categorical_embedding", "known pregame from roster"),
    ("recent_team", "identity", "IDENTIFIER_ONLY", "none", "captured structurally via UniversalEvent.team_id"),
    ("opponent_team", "identity", "IDENTIFIER_ONLY", "none", "captured structurally via UniversalEvent.opponent_id"),
    ("season", "temporal_state", "UNIVERSAL", "ordinal", "known pregame"),
    ("week", "temporal_state", "UNIVERSAL", "ordinal", "known pregame"),
    ("target", "identity", "TARGET", "none", "target/stat identity"),
    ("actual", "target", "TARGET", "none", "settled real value"),
    ("baseline", "recency", "UNIVERSAL", "robust_scaling", "verified in sports/nfl/fantasy/accuracy.py: shift(1)-based prior-games rolling average, strictly causal"),
    ("prediction", "meta_model", "UNUSABLE", "none", "existing NFL model's own prediction; circular, same reasoning as MLB Prediction"),
    ("absolute_error", "meta_model", "POSTGAME_FORBIDDEN", "none", "computed from actual; only knowable after settlement"),
    ("current_prediction", "meta_model", "UNUSABLE", "none", "existing model output"),
    ("challenger_prediction", "meta_model", "UNUSABLE", "none", "existing model output"),
]

_SOURCES = {
    "mlb": (
        "sports/mlb/data/predictions/calibration/historical_pool_universe_2026.csv",
        _MLB_COLUMN_SPEC,
        "mlb_historical_pool_universe_2026",
    ),
    "nfl": (
        "sports/nfl/data/evaluation/backtest_rows.csv",
        _NFL_COLUMN_SPEC,
        "nfl_backtest_rows",
    ),
}


def _dtype_str(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def build_registry() -> list[FeatureRecord]:
    records: list[FeatureRecord] = []
    for sport, (rel_path, column_spec, source_name) in _SOURCES.items():
        path = REPO_ROOT / rel_path
        df = pd.read_csv(path, low_memory=False)
        n = len(df)
        for column, family, classification, normalization, reason in column_spec:
            if column not in df.columns:
                raise KeyError(f"{sport}: expected column '{column}' not found in {rel_path}")
            series = df[column]
            missing_rate = float(series.isna().mean()) if n else 1.0
            unique_count = int(series.nunique(dropna=True))
            allowed = classification in {"UNIVERSAL", "SPORT_SPECIFIC", "MARKET"}
            namespace = "universal" if classification == "UNIVERSAL" else sport
            feature_name = column if not allowed else f"{namespace}.{_slug(column)}"
            records.append(
                FeatureRecord(
                    feature_name=feature_name if allowed else f"{sport}.{_slug(column)}",
                    sport=sport,
                    semantic_family=family,
                    dtype=_dtype_str(series),
                    source=source_name,
                    availability_timestamp="prediction_cutoff_time" if allowed else "n/a",
                    missing_rate=round(missing_rate, 6),
                    unique_count=unique_count,
                    normalization=normalization,
                    allowed_for_training=allowed,
                    classification=classification,
                    reason=reason,
                )
            )
    return records


def _slug(column: str) -> str:
    return column.strip().lower().replace(" ", "_")


def write_registry(path: Optional[Path] = None) -> Path:
    path = path or (MANIFESTS_DIR / "feature_registry.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = build_registry()
    payload = {
        "generated_by": "sports.universal_model.data.feature_registry",
        "classification_vocabulary": sorted(ALLOWED_CLASSES),
        "feature_count": len(records),
        "allowed_for_training_count": sum(1 for r in records if r.allowed_for_training),
        "features": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return path


if __name__ == "__main__":
    out = write_registry()
    payload = json.loads(out.read_text())
    print(f"wrote {out} ({payload['feature_count']} features, {payload['allowed_for_training_count']} allowed_for_training)")
