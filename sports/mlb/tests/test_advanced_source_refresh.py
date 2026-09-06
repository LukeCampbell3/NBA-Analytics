from __future__ import annotations

from datetime import date

import pandas as pd

from sports.mlb.advanced import production_refresh
from sports.mlb.advanced.schema import ADVANCED_SCHEMA_VERSION


def test_fangraphs_status_is_explicit_on_failure():
    def fail(*args, **kwargs):
        raise RuntimeError("upstream unavailable")

    mapping, status = production_refresh._fangraphs_map_with_status(fail, 2026)

    assert mapping == {}
    assert status["status"] == "UNAVAILABLE"
    assert status["rows"] == 0
    assert "upstream unavailable" in status["error"]
    assert status["required_for_base_statcast_model"] is False


def test_fangraphs_status_reports_available_advanced_fields():
    frame = pd.DataFrame(
        [
            {"Name": "Pitcher One", "ERA": 3.1, "FIP": 3.3, "xFIP": 3.4, "SIERA": 3.5},
            {"Name": "Pitcher Two", "ERA": 4.1, "FIP": 4.0, "xFIP": 4.2, "SIERA": 4.0},
        ]
    )

    mapping, status = production_refresh._fangraphs_map_with_status(lambda *a, **k: frame, 2026)

    assert status["status"] == "SUCCESS"
    assert status["rows"] == 2
    assert {"ERA", "FIP", "xFIP", "SIERA"}.issubset(set(status["available_fields"]))
    assert mapping["pitcher one"]["xFIP"] == 3.4


def test_same_day_cache_requires_exact_effective_as_of_date():
    run_day = date(2026, 9, 5)
    valid = {
        "schema_version": ADVANCED_SCHEMA_VERSION,
        "run_date": "2026-09-05",
        "effective_as_of_date": "2026-09-04",
    }
    future_tainted = {**valid, "effective_as_of_date": "2026-09-05"}
    prior_day = {**valid, "run_date": "2026-09-04", "effective_as_of_date": "2026-09-03"}

    assert production_refresh._valid_same_day_partition(valid, run_day) is True
    assert production_refresh._valid_same_day_partition(future_tainted, run_day) is False
    assert production_refresh._valid_same_day_partition(prior_day, run_day) is False
