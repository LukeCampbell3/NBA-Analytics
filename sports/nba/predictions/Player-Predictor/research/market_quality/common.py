from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.common import build_candidate_id, safe_bool, safe_float, series_numeric, series_text
from research.market_quality.price_provenance_schema import (
    annotate_price_provenance_frame,
    derive_snapshot_id,
    load_market_snapshot_manifest,
)
from research.market_quality.priced_event_ledger import build_priced_event_ledger_frame


DEFAULT_BREAK_EVEN_PROB = 110.0 / 210.0
DEFAULT_STALE_MINUTES = 180.0


@dataclass(frozen=True)
class SnapshotBundle:
    snapshot_rows: pd.DataFrame
    source_label: str
    provider: str = ""
    snapshot_id: str = ""


def american_odds_to_decimal(odds: Any) -> float:
    numeric = safe_float(odds, default=np.nan)
    if not np.isfinite(numeric) or numeric == 0.0:
        return np.nan
    if numeric > 0.0:
        return float(1.0 + (numeric / 100.0))
    return float(1.0 + (100.0 / abs(numeric)))


def american_odds_to_break_even(odds: Any) -> float:
    numeric = safe_float(odds, default=np.nan)
    if not np.isfinite(numeric) or numeric == 0.0:
        return np.nan
    if numeric > 0.0:
        return float(100.0 / (numeric + 100.0))
    return float(abs(numeric) / (abs(numeric) + 100.0))


def decimal_odds_to_break_even(decimal_odds: Any) -> float:
    numeric = safe_float(decimal_odds, default=np.nan)
    if not np.isfinite(numeric) or numeric <= 1.0:
        return np.nan
    return float(1.0 / numeric)


def normalize_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.NaT
    return parsed


def normalize_market_date(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed) and len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        return text[:10]
    return parsed.strftime("%Y-%m-%d")


def candidate_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "market_date" not in out.columns and "run_date" in out.columns:
        out["market_date"] = series_text(out, "run_date")
    out["market_date"] = out.get("market_date", pd.Series("", index=out.index)).map(normalize_market_date)
    if "candidate_id" not in out.columns:
        out["candidate_id"] = build_candidate_id(out)
    out["market_type"] = coerce_market_type(out)
    out["market_family"] = out["market_type"].str.split("_").str[0]
    out["player_key"] = series_text(out, "market_player_raw").replace("", np.nan).fillna(series_text(out, "player")).str.strip()
    out["player_key"] = out["player_key"].str.replace(" ", "_", regex=False)
    return out


def coerce_market_type(frame: pd.DataFrame) -> pd.Series:
    if "market_id" in frame.columns:
        return series_text(frame, "market_id").str.upper().str.strip()
    return series_text(frame, "target").str.upper().str.strip() + "_" + series_text(frame, "direction").str.upper().str.strip()


def infer_selector_profile_from_path(path_text: str) -> str:
    text = str(path_text).replace("\\", "/")
    if "/shadow/" in text:
        tail = text.split("/shadow/", 1)[1]
        return tail.split("/", 1)[0] or "shadow"
    if "production" in text.lower():
        return "production"
    return "baseline"


def infer_selector_mode(frame: pd.DataFrame) -> pd.Series:
    if "mode" in frame.columns:
        return series_text(frame, "mode")
    if "ranking_mode" in frame.columns:
        return series_text(frame, "ranking_mode")
    return pd.Series("selector_pool", index=frame.index, dtype="object")


def compute_model_probability(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "selected_board_prob_raw",
        "predicted_probability",
        "board_play_win_prob",
        "selector_expected_win_rate",
        "p_selector",
        "p_base",
        "expected_win_rate",
    ]
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            candidate = pd.to_numeric(frame[column], errors="coerce")
            out = out.fillna(candidate)
    return out


def compute_stress_probability(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "stress_probability",
        "p_side_stress",
        "p_calibrated",
        "board_play_win_prob",
        "expected_win_rate",
        "selector_expected_win_rate",
    ]
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column in frame.columns:
            candidate = pd.to_numeric(frame[column], errors="coerce")
            out = out.fillna(candidate)
    return out


def compute_uncertainty_penalty(frame: pd.DataFrame) -> pd.Series:
    if "belief_uncertainty" in frame.columns:
        penalty = pd.to_numeric(frame["belief_uncertainty"], errors="coerce").fillna(0.0) * 0.05
        return penalty.clip(lower=0.0, upper=0.10)
    return pd.Series(0.0, index=frame.index, dtype="float64")


def compute_no_vig_probabilities(over_prices: pd.Series, under_prices: pd.Series) -> pd.Series:
    over_break_even = over_prices.map(american_odds_to_break_even)
    under_break_even = under_prices.map(american_odds_to_break_even)
    denominator = over_break_even + under_break_even
    with np.errstate(divide="ignore", invalid="ignore"):
        fair_over = over_break_even / denominator
    fair_over = fair_over.where(np.isfinite(fair_over), np.nan)
    return fair_over


def price_is_invalid(odds: Any) -> bool:
    numeric = safe_float(odds, default=np.nan)
    if not np.isfinite(numeric):
        return False
    if numeric == 0.0:
        return True
    return abs(numeric) < 50.0 or abs(numeric) > 2000.0


def _snapshot_long_frame(
    snapshot: pd.DataFrame,
    *,
    source_label: str,
    provider: str = "",
    snapshot_id: str = "",
) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame()
    working = snapshot.copy()
    working["player_key"] = series_text(working, "Market_Player_Raw").replace("", np.nan).fillna(series_text(working, "Player")).str.strip()
    working["player_key"] = working["player_key"].str.replace(" ", "_", regex=False)
    market_date = pd.to_datetime(working.get("Market_Date"), errors="coerce", utc=True)
    if market_date.notna().any():
        working["market_date"] = market_date.dt.tz_convert(None).dt.strftime("%Y-%m-%d")
    else:
        working["market_date"] = series_text(working, "Market_Date").map(normalize_market_date)
    working["odds_snapshot_time"] = working.get("Market_Fetched_At_UTC", pd.Series(pd.NaT, index=working.index)).map(normalize_timestamp)
    snapshot_provider_series = working.get("Market_Provider", pd.Series(str(provider or ""), index=working.index)).fillna(str(provider or "")).astype(str)
    snapshot_book_series = working.get("Market_Book", pd.Series("aggregate_market_snapshot", index=working.index)).fillna("aggregate_market_snapshot").astype(str)
    snapshot_source_series = working.get("Market_Price_Source", pd.Series(str(source_label or ""), index=working.index)).fillna(str(source_label or "")).astype(str)
    snapshot_source_type_series = working.get("Market_Price_Source_Type", pd.Series("ARCHIVED_ENTRY" if str(provider or "").strip() else "UNKNOWN", index=working.index)).fillna(
        "ARCHIVED_ENTRY" if str(provider or "").strip() else "UNKNOWN"
    ).astype(str)
    snapshot_id_series = working.get("Market_Snapshot_ID", pd.Series(str(snapshot_id or ""), index=working.index)).fillna(str(snapshot_id or "")).astype(str)
    commence_time_series = working.get("market_commence_time_utc", working.get("Market_Commence_Time_UTC", pd.Series(pd.NaT, index=working.index))).map(normalize_timestamp)
    event_time_source_series = working.get("event_time_source", pd.Series("MISSING", index=working.index)).fillna("MISSING").astype(str)
    event_time_confidence_series = working.get("event_time_confidence", pd.Series("missing", index=working.index)).fillna("missing").astype(str)
    event_time_reason_series = working.get("event_time_resolution_reason", pd.Series("", index=working.index)).fillna("").astype(str)
    event_time_warning_series = working.get("event_time_resolution_warning", pd.Series("", index=working.index)).fillna("").astype(str)
    frames: list[pd.DataFrame] = []
    for target in ("PTS", "TRB", "AST"):
        target_frame = pd.DataFrame(
            {
                "player_key": working["player_key"],
                "market_date": working["market_date"],
                "target": target,
                "snapshot_market_line": pd.to_numeric(working.get(f"Market_{target}"), errors="coerce"),
                "snapshot_market_books": pd.to_numeric(working.get(f"Market_{target}_books"), errors="coerce"),
                "snapshot_over_price": pd.to_numeric(working.get(f"Market_{target}_over_price"), errors="coerce"),
                "snapshot_under_price": pd.to_numeric(working.get(f"Market_{target}_under_price"), errors="coerce"),
                "snapshot_line_std": pd.to_numeric(working.get(f"Market_{target}_line_std"), errors="coerce"),
                "odds_snapshot_time": working["odds_snapshot_time"],
                "snapshot_source": snapshot_source_series,
                "snapshot_provider": snapshot_provider_series,
                "snapshot_price_source_type": snapshot_source_type_series,
                "snapshot_id": snapshot_id_series,
                "book": snapshot_book_series,
                "market_commence_time_utc": commence_time_series,
                "event_time_source": event_time_source_series,
                "event_time_confidence": event_time_confidence_series,
                "event_time_resolution_reason": event_time_reason_series,
                "event_time_resolution_warning": event_time_warning_series,
            }
        )
        frames.append(target_frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["player_key", "market_date", "target"], keep="first").reset_index(drop=True)
    return out


def load_snapshot_bundle(snapshot_path: Path) -> SnapshotBundle:
    if not snapshot_path.exists():
        return SnapshotBundle(snapshot_rows=pd.DataFrame(), source_label="")
    snapshot = pd.read_parquet(snapshot_path)
    source_label = str(snapshot_path.resolve())
    manifest = load_market_snapshot_manifest(snapshot_path)
    provider = str(manifest.get("provider", "snapshot")).strip() or "snapshot"
    fetched_at = ""
    if "Market_Fetched_At_UTC" in snapshot.columns:
        fetched_series = snapshot["Market_Fetched_At_UTC"].dropna().astype(str)
        if not fetched_series.empty:
            fetched_at = str(fetched_series.iloc[0]).strip()
    bundle_snapshot_id = derive_snapshot_id(provider=provider, odds_snapshot_time=fetched_at, fallback_label=snapshot_path.stem)
    return SnapshotBundle(
        snapshot_rows=_snapshot_long_frame(snapshot, source_label=source_label, provider=provider, snapshot_id=bundle_snapshot_id),
        source_label=source_label,
        provider=provider,
        snapshot_id=bundle_snapshot_id,
    )


def snapshot_path_for_candidate_source(candidate_source: str) -> Path | None:
    path = Path(candidate_source)
    if not path.exists():
        return None
    directory = path.parent
    matches = sorted(directory.glob("current_market_snapshot_*.parquet"))
    if matches:
        return matches[0]
    return None


def manifest_path_for_candidate_source(candidate_source: str) -> Path | None:
    path = Path(candidate_source)
    if not path.exists():
        return None
    directory = path.parent
    matches = sorted(directory.glob("daily_market_pipeline_manifest_*.json"))
    if matches:
        return matches[0]
    return None


def _coalesce_snapshot_suffix_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for base_column in [
        "odds_snapshot_time",
        "book",
        "snapshot_id",
        "market_commence_time_utc",
        "event_time_source",
        "event_time_confidence",
        "event_time_resolution_reason",
        "event_time_resolution_warning",
    ]:
        suffixed = [column for column in [base_column, f"{base_column}_x", f"{base_column}_y"] if column in out.columns]
        if not suffixed:
            continue
        values = pd.Series(pd.NA, index=out.index, dtype="object")
        for column in suffixed:
            candidate = out[column].astype("object")
            values = values.where(values.notna() & values.astype(str).str.strip().ne(""), candidate)
        out[base_column] = values
        drop_columns = [column for column in [f"{base_column}_x", f"{base_column}_y"] if column in out.columns]
        out = out.drop(columns=drop_columns, errors="ignore")
    return out


def augment_with_snapshot_prices(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    working = _coalesce_snapshot_suffix_columns(candidate_identity_columns(rows))
    if "source_candidate_pool_csv" not in working.columns:
        return working
    cache: dict[str, pd.DataFrame] = {}
    merged_frames: list[pd.DataFrame] = []
    for source_path, group in working.groupby(series_text(working, "source_candidate_pool_csv"), dropna=False):
        source_text = str(source_path).strip()
        if not source_text:
            merged_frames.append(group.copy())
            continue
        if source_text not in cache:
            snapshot_path = snapshot_path_for_candidate_source(source_text)
            cache[source_text] = load_snapshot_bundle(snapshot_path).snapshot_rows if snapshot_path is not None else pd.DataFrame()
        snapshot_rows = cache[source_text]
        if snapshot_rows.empty:
            merged_frames.append(group.copy())
            continue
        merge_keys = ["player_key", "market_date", "target"]
        merged = _coalesce_snapshot_suffix_columns(group).merge(
            snapshot_rows,
            on=merge_keys,
            how="left",
            suffixes=("", "_snapshot"),
        )
        merged_frames.append(merged)
    out = pd.concat(merged_frames, ignore_index=True) if merged_frames else working
    for base_column in [
        "odds_snapshot_time",
        "book",
        "snapshot_id",
        "market_commence_time_utc",
        "event_time_source",
        "event_time_confidence",
        "event_time_resolution_reason",
        "event_time_resolution_warning",
    ]:
        snapshot_column = f"{base_column}_snapshot"
        if snapshot_column in out.columns:
            if base_column not in out.columns:
                out[base_column] = out[snapshot_column]
            else:
                base_values = out[base_column].astype("object")
                snapshot_values = out[snapshot_column].astype("object")
                out[base_column] = base_values.where(
                    base_values.notna() & base_values.astype(str).str.strip().ne(""),
                    snapshot_values,
                )
            out = out.drop(columns=[snapshot_column], errors="ignore")
    for base_column in [
        "odds_snapshot_time",
        "book",
        "snapshot_id",
        "market_commence_time_utc",
        "event_time_source",
        "event_time_confidence",
        "event_time_resolution_reason",
        "event_time_resolution_warning",
    ]:
        left_column = f"{base_column}_x"
        right_column = f"{base_column}_y"
        if base_column not in out.columns and (left_column in out.columns or right_column in out.columns):
            out[base_column] = pd.NA
        if base_column in out.columns:
            base_values = out[base_column].astype("object")
            if left_column in out.columns:
                left_values = out[left_column].astype("object")
                base_values = base_values.where(base_values.notna() & base_values.astype(str).str.strip().ne(""), left_values)
            if right_column in out.columns:
                right_values = out[right_column].astype("object")
                base_values = base_values.where(base_values.notna() & base_values.astype(str).str.strip().ne(""), right_values)
            out[base_column] = base_values
    snapshot_over_price = series_numeric(out, "snapshot_over_price")
    snapshot_under_price = series_numeric(out, "snapshot_under_price")
    out["snapshot_market_side_price"] = np.where(
        series_text(out, "direction").str.upper().eq("OVER"),
        snapshot_over_price,
        snapshot_under_price,
    )
    out["snapshot_opposite_side_price"] = np.where(
        series_text(out, "direction").str.upper().eq("OVER"),
        snapshot_under_price,
        snapshot_over_price,
    )
    out["snapshot_market_side_break_even"] = out["snapshot_market_side_price"].map(american_odds_to_break_even)
    out["snapshot_no_vig_probability"] = compute_no_vig_probabilities(
        snapshot_over_price,
        snapshot_under_price,
    )
    if "provider" not in out.columns:
        out["provider"] = series_text(out, "snapshot_provider")
    else:
        out["provider"] = series_text(out, "provider").replace("", np.nan).fillna(series_text(out, "snapshot_provider"))
    if "book" not in out.columns:
        out["book"] = "aggregate_market_snapshot"
    if "price_source" not in out.columns:
        out["price_source"] = np.where(series_text(out, "snapshot_source").ne(""), "current_market_snapshot_pre_event", "")
    if "price_source_type" not in out.columns:
        out["price_source_type"] = series_text(out, "snapshot_price_source_type").replace("", "UNKNOWN")
    return out


def merge_selected_with_candidate_pool(selected_rows: pd.DataFrame, candidate_pool_rows: pd.DataFrame) -> pd.DataFrame:
    if selected_rows.empty:
        return selected_rows.copy()
    selected = candidate_identity_columns(selected_rows)
    if candidate_pool_rows is None or candidate_pool_rows.empty:
        return selected
    pool = augment_with_snapshot_prices(candidate_pool_rows)
    pool_join = pool.drop_duplicates(subset=["candidate_id"], keep="first").copy()
    selected = selected.merge(
        pool_join.drop(columns=[column for column in ["result", "units", "actual", "actual_matched_date"] if column in pool_join.columns], errors="ignore"),
        on="candidate_id",
        how="left",
        suffixes=("", "_pool"),
    )
    for pool_column in [column for column in selected.columns if column.endswith("_pool")]:
        base_column = pool_column[: -len("_pool")]
        if base_column == "candidate_id":
            continue
        if base_column not in selected.columns:
            selected[base_column] = selected[pool_column]
            continue
        base_numeric = pd.to_numeric(selected[base_column], errors="coerce")
        pool_numeric = pd.to_numeric(selected[pool_column], errors="coerce")
        if base_numeric.notna().any() or pool_numeric.notna().any():
            selected[base_column] = base_numeric.where(base_numeric.notna(), pool_numeric)
        else:
            base_text = selected[base_column].astype("object")
            pool_text = selected[pool_column].astype("object")
            selected[base_column] = base_text.where(base_text.notna() & base_text.astype(str).ne(""), pool_text)
    return selected


def compute_price_quality_frame(rows: pd.DataFrame, *, record_scope: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    working = candidate_identity_columns(rows)
    working["record_scope"] = str(record_scope)
    working["selected_on_board"] = working["record_scope"].eq("selected")
    working["game_date"] = series_text(working, "market_date")
    working["snapshot_time"] = working.get("odds_snapshot_time", pd.Series(pd.NaT, index=working.index))
    working["book"] = series_text(working, "book").replace("", "aggregate_market_snapshot")
    working["selector_mode"] = infer_selector_mode(working)
    working["selector_profile"] = series_text(working, "source_candidate_pool_csv").map(infer_selector_profile_from_path)
    working["model_probability"] = compute_model_probability(working)
    working["stress_probability"] = compute_stress_probability(working)
    working["expected_win_rate"] = pd.to_numeric(working.get("expected_win_rate"), errors="coerce").fillna(working["stress_probability"])
    working["existing_market_side_price"] = pd.to_numeric(
        working.get("market_side_price", working.get("odds_american", working.get("odds"))),
        errors="coerce",
    )
    working["existing_market_side_break_even"] = pd.to_numeric(
        working.get("market_side_break_even", working.get("break_even_prob")),
        errors="coerce",
    )
    working["odds_snapshot_time"] = working.get("odds_snapshot_time", pd.Series(pd.NaT, index=working.index)).map(normalize_timestamp)
    working["prediction_snapshot_time"] = working.get(
        "prediction_snapshot_time",
        pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns, UTC]"),
    )
    working["selector_run_time"] = working.get(
        "selector_run_time",
        pd.Series(pd.NaT, index=working.index, dtype="datetime64[ns, UTC]"),
    )
    working["line_at_prediction"] = pd.to_numeric(working.get("market_line"), errors="coerce")
    working["line_at_odds_snapshot"] = pd.to_numeric(working.get("snapshot_market_line"), errors="coerce")
    working["line_moved_since_prediction"] = pd.to_numeric(working.get("line_moved_since_prediction"), errors="coerce")
    working["line_moved_since_prediction"] = working["line_moved_since_prediction"].where(
        working["line_moved_since_prediction"].notna(),
        working["line_at_odds_snapshot"] - working["line_at_prediction"],
    )
    working["odds_moved_since_prediction"] = pd.to_numeric(working.get("odds_moved_since_prediction"), errors="coerce")
    working["odds_moved_since_prediction"] = working["odds_moved_since_prediction"].where(
        working["odds_moved_since_prediction"].notna(),
        pd.to_numeric(working.get("snapshot_market_side_price"), errors="coerce") - working["existing_market_side_price"],
    )
    working["provider"] = series_text(working, "provider").replace("", np.nan).fillna(series_text(working, "snapshot_provider"))
    existing_price_source = series_text(working, "price_source")
    derived_price_source = pd.Series(
        np.where(
            series_text(working, "snapshot_source").ne(""),
            "current_market_snapshot_pre_event",
            np.where(working["existing_market_side_price"].notna(), "selector_embedded_unknown_time", ""),
        ),
        index=working.index,
        dtype="object",
    )
    working["price_source"] = existing_price_source.where(existing_price_source.ne(""), derived_price_source)
    existing_source_type = series_text(working, "price_source_type")
    snapshot_source_type = series_text(working, "snapshot_price_source_type")
    working["price_source_type"] = existing_source_type.where(existing_source_type.ne(""), snapshot_source_type)
    working = annotate_price_provenance_frame(working, stale_seconds_threshold=float(DEFAULT_STALE_MINUTES) * 60.0)
    working = build_priced_event_ledger_frame(working, record_scope=str(record_scope))
    working["missing_price_flag"] = working["price_validity_status"].eq("MISSING_PRICE")
    working["invalid_price_flag"] = working["price_validity_status"].eq("INVALID_PRICE")
    working["stale_price_flag"] = working["price_validity_status"].eq("STALE_PRICE")
    working["stale_line_flag"] = pd.to_numeric(working["line_moved_since_prediction"], errors="coerce").abs().fillna(0.0) >= 0.25
    return working


def summarize_price_quality(audit_rows: pd.DataFrame) -> dict[str, Any]:
    frame = audit_rows.copy()
    if frame.empty:
        return {
            "total_candidate_rows": 0,
            "total_selected_rows": 0,
            "rows_with_usable_price": 0,
            "rows_with_missing_price": 0,
            "rows_with_invalid_price": 0,
            "rows_with_stale_price": 0,
            "rows_with_break_even_computable": 0,
            "rows_with_untrusted_edge": 0,
        }
    usable = frame.get("price_validity_status", pd.Series("", index=frame.index)).astype(str).eq("PRICE_VALID")
    validity = frame.get("price_validity_status", pd.Series("", index=frame.index)).astype(str)
    summary = {
        "total_candidate_rows": int(frame["record_scope"].eq("candidate").sum()),
        "total_selected_rows": int(frame["record_scope"].eq("selected").sum()),
        "rows_with_usable_price": int(usable.sum()),
        "rows_with_missing_price": int(frame["missing_price_flag"].sum()),
        "rows_with_invalid_price": int(frame["invalid_price_flag"].sum()),
        "rows_with_stale_price": int(frame["stale_price_flag"].sum()),
        "rows_with_break_even_computable": int(frame["market_side_break_even"].notna().sum()),
        "rows_with_untrusted_edge": int(validity.ne("PRICE_VALID").sum()),
        "availability_by_date": frame.groupby("game_date", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            usable_price_rows=("market_side_price", lambda s: int((pd.to_numeric(s, errors="coerce").notna()).sum())),
            stale_price_rows=("stale_price_flag", "sum"),
        ).reset_index().to_dict(orient="records"),
        "availability_by_market_type": frame.groupby("market_type", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            usable_price_rows=("market_side_price", lambda s: int((pd.to_numeric(s, errors="coerce").notna()).sum())),
            missing_price_rows=("missing_price_flag", "sum"),
        ).reset_index().to_dict(orient="records"),
        "availability_by_book": frame.groupby("book", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            usable_price_rows=("market_side_price", lambda s: int((pd.to_numeric(s, errors="coerce").notna()).sum())),
            stale_price_rows=("stale_price_flag", "sum"),
        ).reset_index().to_dict(orient="records"),
        "availability_by_selector_profile": frame.groupby("selector_profile", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            usable_price_rows=("market_side_price", lambda s: int((pd.to_numeric(s, errors="coerce").notna()).sum())),
            missing_price_rows=("missing_price_flag", "sum"),
        ).reset_index().to_dict(orient="records"),
    }
    return summary


def summarize_price_provenance(audit_rows: pd.DataFrame) -> dict[str, Any]:
    frame = audit_rows.copy()
    if frame.empty:
        return {
            "total_candidate_rows": 0,
            "total_selected_rows": 0,
            "percent_with_valid_timestamp_safe_market_side_price": 0.0,
            "percent_with_market_side_break_even": 0.0,
            "percent_with_odds_snapshot_time": 0.0,
            "percent_with_price_source": 0.0,
        }
    total_rows = max(int(len(frame)), 1)
    selected = frame.loc[frame["record_scope"].astype(str).eq("selected")].copy()
    valid_entry = frame["price_validity_status"].astype(str).eq("PRICE_VALID") & frame["timestamp_safe_flag"].astype(bool)
    selected_invalid = selected.loc[~(selected["price_validity_status"].astype(str).eq("PRICE_VALID") & selected["timestamp_safe_flag"].astype(bool))].copy()
    summary = {
        "total_candidate_rows": int(frame["record_scope"].eq("candidate").sum()),
        "total_selected_rows": int(frame["record_scope"].eq("selected").sum()),
        "percent_with_valid_timestamp_safe_market_side_price": float(valid_entry.mean()),
        "percent_with_market_side_break_even": float(pd.to_numeric(frame["market_side_break_even"], errors="coerce").notna().mean()),
        "percent_with_odds_snapshot_time": float(pd.to_datetime(frame["odds_snapshot_time"], errors="coerce", utc=True).notna().mean()),
        "percent_with_price_source": float(frame["price_source"].fillna("").astype(str).str.strip().ne("").mean()),
        "stale_price_count": int(frame["price_validity_status"].astype(str).eq("STALE_PRICE").sum()),
        "missing_price_count": int(frame["price_validity_status"].astype(str).eq("MISSING_PRICE").sum()),
        "diagnostic_only_count": int(frame["price_validity_status"].astype(str).eq("DIAGNOSTIC_ONLY").sum()),
        "unknown_source_count": int(frame["price_validity_status"].astype(str).eq("PRICE_SOURCE_UNKNOWN").sum()),
        "timestamp_safety_basis_counts": {
            str(key): int(value)
            for key, value in frame.get("timestamp_safety_basis", pd.Series("", index=frame.index)).fillna("").astype(str).value_counts().to_dict().items()
        },
        "event_time_source_counts": {
            str(key): int(value)
            for key, value in frame.get("event_time_source", pd.Series("", index=frame.index)).fillna("").astype(str).value_counts().to_dict().items()
        },
        "selected_rows_without_valid_entry_price": int(len(selected_invalid)),
        "rows_where_edge_computed_without_valid_price": int(
            (
                pd.to_numeric(frame.get("edge"), errors="coerce").notna()
                & frame["edge_price_untrusted_flag"].astype(bool)
            ).sum()
        ),
        "availability_by_date": frame.groupby("game_date", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            valid_entry_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("PRICE_VALID").sum())),
            missing_price_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("MISSING_PRICE").sum())),
        ).reset_index().to_dict(orient="records"),
        "availability_by_book": frame.groupby("book", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            valid_entry_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("PRICE_VALID").sum())),
            diagnostic_only_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("DIAGNOSTIC_ONLY").sum())),
        ).reset_index().to_dict(orient="records"),
        "availability_by_market_type": frame.groupby("market_type", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            valid_entry_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("PRICE_VALID").sum())),
            stale_price_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("STALE_PRICE").sum())),
        ).reset_index().to_dict(orient="records"),
        "availability_by_selector_profile": frame.groupby("selector_profile", dropna=False).agg(
            total_rows=("candidate_id", "count"),
            valid_entry_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("PRICE_VALID").sum())),
            missing_price_rows=("price_validity_status", lambda s: int(pd.Series(s).astype(str).eq("MISSING_PRICE").sum())),
        ).reset_index().to_dict(orient="records"),
    }
    return summary
