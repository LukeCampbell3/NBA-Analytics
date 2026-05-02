from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGETS = ("PTS", "TRB", "AST")


def _coerce_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _result_for_edges(pred_edge: float, actual_edge: float) -> str | None:
    if pd.isna(pred_edge) or pd.isna(actual_edge):
        return None
    if actual_edge == 0.0:
        return "push"
    if pred_edge == 0.0:
        return None
    return "win" if ((pred_edge > 0.0 and actual_edge > 0.0) or (pred_edge < 0.0 and actual_edge < 0.0)) else "loss"


def backtest_rows_to_decisions(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        source = data.copy()
    else:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Backtest rows CSV not found: {path}")
        source = pd.read_csv(path)

    if source.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for target_index, target in enumerate(TARGETS):
        market = _coerce_numeric(source, f"market_{target}")
        prediction = _coerce_numeric(source, f"pred_{target}")
        actual = _coerce_numeric(source, f"actual_{target}")
        pred_edge = _coerce_numeric(source, f"pred_minus_market_{target}")
        actual_edge = _coerce_numeric(source, f"actual_minus_market_{target}")

        valid = market.notna() & prediction.notna() & actual.notna() & pred_edge.notna() & actual_edge.notna()
        if not valid.any():
            continue

        target_df = pd.DataFrame(
            {
                "player": source.loc[valid, "player"].astype(str),
                "target": target,
                "prediction": prediction.loc[valid],
                "market_line": market.loc[valid],
                "actual": actual.loc[valid],
                "target_date": pd.to_datetime(source.loc[valid, "date"], errors="coerce"),
                "belief_uncertainty": _coerce_numeric(source.loc[valid], "belief_uncertainty"),
                "feasibility": _coerce_numeric(source.loc[valid], "feasibility"),
                "fallback_blend": _coerce_numeric(source.loc[valid], "fallback_blend"),
                "history_rows": _coerce_numeric(source.loc[valid], "history_rows"),
                "market_books": _coerce_numeric(source.loc[valid], f"market_books_{target}"),
                "baseline": _coerce_numeric(source.loc[valid], f"baseline_{target}"),
                "actual_minus_market": actual_edge.loc[valid],
                "target_index": int(target_index),
            }
        )
        target_df["edge"] = target_df["prediction"] - target_df["market_line"]
        target_df["abs_edge"] = target_df["edge"].abs()
        target_df["direction"] = target_df["edge"].map(lambda value: "OVER" if value > 0.0 else ("UNDER" if value < 0.0 else "PUSH"))
        target_df["result"] = [
            _result_for_edges(pred_value, actual_value)
            for pred_value, actual_value in zip(pred_edge.loc[valid], actual_edge.loc[valid])
        ]
        rows.append(target_df)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if out.empty:
        return out
    out = out.dropna(subset=["target_date", "result"]).reset_index(drop=True)
    return out
