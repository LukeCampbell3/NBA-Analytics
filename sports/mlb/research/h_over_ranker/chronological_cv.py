from __future__ import annotations

"""Day-grouped, expanding-window chronological cross-validation.

Never splits rows from the same date across train/validation, and never lets
a validation date's rows influence a fold whose training window is
chronologically before it. This is a walk-forward leave-one-day-out scheme:
fold k trains on every date strictly before val date k, validates on val
date k alone.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_dates: tuple[str, ...]
    val_date: str


def expanding_day_folds(dates: list[str], min_train_days: int) -> list[Fold]:
    """One fold per date once at least `min_train_days` prior dates exist.

    `dates` must already be the set of *distinct* dates with eligible rows,
    in chronological order.
    """
    ordered = sorted(dates)
    folds: list[Fold] = []
    for i in range(min_train_days, len(ordered)):
        folds.append(Fold(index=len(folds), train_dates=tuple(ordered[:i]), val_date=ordered[i]))
    return folds


def split(frame: pd.DataFrame, fold: Fold, date_column: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame[date_column].isin(fold.train_dates)]
    val = frame[frame[date_column] == fold.val_date]
    assert_no_leakage(train, val, date_column=date_column)
    return train, val


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, date_column: str = "date") -> None:
    """Raise if train/val share a date, or if any train date is >= the val date.

    val is expected to hold exactly one date (leave-one-day-out), but this
    also tolerates a multi-date val block as long as ordering holds.
    """
    train_dates = set(train[date_column].unique())
    val_dates = set(val[date_column].unique())
    overlap = train_dates & val_dates
    if overlap:
        raise AssertionError(f"train/val share dates (row-level date leakage): {sorted(overlap)}")
    if train_dates and val_dates and max(train_dates) >= min(val_dates):
        raise AssertionError(
            f"a training date ({max(train_dates)}) is not strictly before "
            f"the validation date(s) ({sorted(val_dates)}) -- future information "
            f"would leak into training"
        )
