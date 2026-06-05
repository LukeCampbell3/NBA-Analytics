from __future__ import annotations

import re
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize(value: float, lower: float, upper: float) -> float:
    if upper == lower:
        return 0.0
    return clamp((value - lower) / (upper - lower), 0.0, 1.0)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
