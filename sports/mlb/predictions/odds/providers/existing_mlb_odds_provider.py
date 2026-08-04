#!/usr/bin/env python3
"""Adapter for an existing provider-neutral CSV or parquet observation store."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from odds_contract import ensure_contract


class ExistingMlbOddsProvider:
    def __init__(self, input_path: Path | str | None = None):
        configured = input_path if input_path is not None else os.environ.get("MLB_ODDS_EXISTING_PROVIDER_PATH", "")
        self.input_path = Path(configured) if configured else None

    def validate_config(self) -> dict[str, Any]:
        if self.input_path is None:
            return {"status": "missing_source", "message": "MLB_ODDS_EXISTING_PROVIDER_PATH not set"}
        if not self.input_path.is_file():
            return {"status": "missing_source", "message": f"Existing odds file not found: {self.input_path}"}
        return {"status": "ok"}

    def collect_player_props(self) -> dict[str, Any]:
        config = self.validate_config()
        if config["status"] != "ok":
            return config
        try:
            frame = pd.read_parquet(self.input_path) if self.input_path.suffix.lower() == ".parquet" else pd.read_csv(self.input_path)
        except (OSError, ValueError) as exc:
            return {"status": "source_invalid_data", "message": str(exc)[:200]}
        if frame.empty:
            return {"status": "no_props", "message": "Existing provider store is empty"}
        return {"status": "success", "odds": frame.to_dict(orient="records")}

    def normalize(self, raw_odds: list[dict[str, Any]]) -> pd.DataFrame:
        return ensure_contract(
            pd.DataFrame(raw_odds),
            source="existing_provider",
            acquisition_method="existing_provider",
            source_endpoint=str(self.input_path or ""),
            parser_version="existing-provider-adapter-v1",
        )
