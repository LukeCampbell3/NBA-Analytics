from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isnan(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def american_odds_to_decimal(odds: Any) -> float:
    numeric = _safe_float(odds, default=np.nan)
    if not np.isfinite(numeric) or numeric == 0.0:
        return np.nan
    if numeric > 0.0:
        return float(1.0 + numeric / 100.0)
    return float(1.0 + 100.0 / abs(numeric))


def decimal_odds_to_break_even(decimal_odds: Any) -> float:
    numeric = _safe_float(decimal_odds, default=np.nan)
    if not np.isfinite(numeric) or numeric <= 1.0:
        return np.nan
    return float(1.0 / numeric)


def american_odds_to_break_even(odds: Any) -> float:
    numeric = _safe_float(odds, default=np.nan)
    if not np.isfinite(numeric) or numeric == 0.0:
        return np.nan
    if numeric > 0.0:
        return float(100.0 / (numeric + 100.0))
    return float(abs(numeric) / (abs(numeric) + 100.0))


def break_even_to_decimal_odds(probability: Any, *, push_probability: Any = 0.0) -> float:
    win_probability = _safe_float(probability, default=np.nan)
    push = _safe_float(push_probability, default=0.0)
    if not np.isfinite(win_probability) or win_probability <= 0.0:
        return np.nan
    push = float(np.clip(push, 0.0, 0.99))
    available = 1.0 - push
    if available <= 0.0 or win_probability >= available:
        return np.nan
    return float(available / win_probability)


def decimal_odds_to_american(decimal_odds: Any) -> float:
    numeric = _safe_float(decimal_odds, default=np.nan)
    if not np.isfinite(numeric) or numeric <= 1.0:
        return np.nan
    if numeric >= 2.0:
        return float((numeric - 1.0) * 100.0)
    return float(-100.0 / max(numeric - 1.0, 1e-9))


def break_even_to_american_odds(probability: Any, *, push_probability: Any = 0.0) -> float:
    return decimal_odds_to_american(
        break_even_to_decimal_odds(probability, push_probability=push_probability)
    )


def price_is_invalid(odds: Any) -> bool:
    numeric = _safe_float(odds, default=np.nan)
    if not np.isfinite(numeric):
        return False
    if numeric == 0.0:
        return True
    return bool(abs(numeric) < 50.0 or abs(numeric) > 2000.0)


def select_side_specific_price(
    side: Any,
    *,
    explicit_price: Any = np.nan,
    over_price: Any = np.nan,
    under_price: Any = np.nan,
) -> float:
    explicit = _safe_float(explicit_price, default=np.nan)
    if np.isfinite(explicit):
        return explicit
    side_text = str(side or "").strip().upper()
    if side_text == "OVER":
        return _safe_float(over_price, default=np.nan)
    if side_text == "UNDER":
        return _safe_float(under_price, default=np.nan)
    return np.nan


def compute_no_vig_probabilities(
    over_prices: pd.Series | Any,
    under_prices: pd.Series | Any,
) -> tuple[pd.Series | float, pd.Series | float]:
    if isinstance(over_prices, pd.Series) or isinstance(under_prices, pd.Series):
        over_series = over_prices if isinstance(over_prices, pd.Series) else pd.Series(over_prices)
        under_series = under_prices if isinstance(under_prices, pd.Series) else pd.Series(under_prices, index=over_series.index)
        over_break_even = over_series.map(american_odds_to_break_even)
        under_break_even = under_series.map(american_odds_to_break_even)
        denominator = over_break_even + under_break_even
        with np.errstate(divide="ignore", invalid="ignore"):
            fair_over = over_break_even / denominator
            fair_under = under_break_even / denominator
        fair_over = fair_over.where(np.isfinite(fair_over), np.nan)
        fair_under = fair_under.where(np.isfinite(fair_under), np.nan)
        return fair_over, fair_under
    over_break_even = american_odds_to_break_even(over_prices)
    under_break_even = american_odds_to_break_even(under_prices)
    if not np.isfinite(over_break_even) or not np.isfinite(under_break_even):
        return np.nan, np.nan
    denominator = over_break_even + under_break_even
    if denominator <= 0.0:
        return np.nan, np.nan
    return float(over_break_even / denominator), float(under_break_even / denominator)


def timestamp_safe_mask(
    odds_snapshot_time: pd.Series,
    prediction_snapshot_time: pd.Series,
    *,
    game_date: pd.Series | None = None,
    close_only_flag: pd.Series | None = None,
    diagnostic_only_flag: pd.Series | None = None,
) -> pd.Series:
    odds_ts = pd.to_datetime(odds_snapshot_time, errors="coerce", utc=True)
    prediction_ts = pd.to_datetime(prediction_snapshot_time, errors="coerce", utc=True)
    if game_date is not None:
        game_ts = pd.to_datetime(game_date, errors="coerce", utc=True)
        postevent = odds_ts.notna() & game_ts.notna() & (odds_ts.dt.normalize() > game_ts.dt.normalize())
    else:
        postevent = pd.Series(False, index=odds_ts.index, dtype=bool)
    close_flag = close_only_flag if close_only_flag is not None else pd.Series(False, index=odds_ts.index, dtype=bool)
    diagnostic_flag = diagnostic_only_flag if diagnostic_only_flag is not None else pd.Series(False, index=odds_ts.index, dtype=bool)
    return (
        odds_ts.notna()
        & prediction_ts.notna()
        & (odds_ts <= prediction_ts)
        & ~postevent
        & ~pd.Series(close_flag, index=odds_ts.index).astype(bool)
        & ~pd.Series(diagnostic_flag, index=odds_ts.index).astype(bool)
    )
