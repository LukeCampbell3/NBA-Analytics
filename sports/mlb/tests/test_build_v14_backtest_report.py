from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

MLB_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import build_v14_backtest_report as report_builder  # noqa: E402
import select_high_precision_predictions as shp  # noqa: E402


def _candidate(**overrides) -> SimpleNamespace:
    base = dict(
        player="Real Player",
        game_id="824970",
        target="TB",
        direction="UNDER",
        market_line=1.5,
        run_date=date(2026, 8, 10),
        selected_side_price=-176.0,
        final_hit_probability=0.75,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_report_grades_real_picks_and_computes_summary(tmp_path, monkeypatch):
    win = _candidate()  # UNDER 1.5, actual 1.0 -> win
    loss = _candidate(player="Other Player", game_id="999", final_hit_probability=0.71)

    pool_csv = tmp_path / "20260810" / "daily_prediction_pool_20260810.csv"
    pool_csv.parent.mkdir(parents=True)
    pool_csv.write_text("pool", encoding="utf-8")

    monkeypatch.setattr(report_builder, "find_raw_pool_csvs", lambda root: [pool_csv])
    monkeypatch.setattr(report_builder, "parse_v11_args", lambda csv: SimpleNamespace())
    monkeypatch.setattr(shp, "prepare_and_filter_candidates", lambda args: ([win, loss], None))
    monkeypatch.setattr(shp, "select_top_candidates", lambda eligible, args: eligible)
    monkeypatch.setattr(
        report_builder,
        "build_actual_lookup",
        lambda root: {
            ("2026-08-10", "real_player", "TB", "824970"): 1.0,
            ("2026-08-10", "other_player", "TB", "999"): 3.0,
        },
    )

    result = report_builder.build_report(daily_runs_root=tmp_path, processed_root=tmp_path)

    assert result["is_live_board"] is False
    assert result["dates_scanned"] == ["20260810"]
    assert len(result["picks"]) == 2
    win_pick = next(p for p in result["picks"] if p["player"] == "Real Player")
    loss_pick = next(p for p in result["picks"] if p["player"] == "Other Player")
    assert win_pick["result"] == "win"
    assert loss_pick["result"] == "loss"
    assert result["summary"]["picks"] == 2
    assert result["summary"]["settled"] == 2
    assert result["summary"]["wins"] == 1
    assert result["summary"]["hit_rate"] == 0.5
    # win profit = 100/176, loss = -1.0 -> roi = (100/176 - 1) / 2
    assert result["summary"]["roi"] == (100 / 176 - 1.0) / 2


def test_build_report_skips_dates_with_no_selected_picks(tmp_path, monkeypatch):
    pool_csv = tmp_path / "20260811" / "daily_prediction_pool_20260811.csv"
    pool_csv.parent.mkdir(parents=True)
    pool_csv.write_text("pool", encoding="utf-8")

    monkeypatch.setattr(report_builder, "find_raw_pool_csvs", lambda root: [pool_csv])
    monkeypatch.setattr(report_builder, "parse_v11_args", lambda csv: SimpleNamespace())
    monkeypatch.setattr(shp, "prepare_and_filter_candidates", lambda args: ([], None))
    monkeypatch.setattr(shp, "select_top_candidates", lambda eligible, args: [])
    monkeypatch.setattr(report_builder, "build_actual_lookup", lambda root: {})

    result = report_builder.build_report(daily_runs_root=tmp_path, processed_root=tmp_path)

    assert result["dates_scanned"] == []
    assert result["picks"] == []
    assert result["summary"] == {"picks": 0, "settled": 0, "wins": 0, "hit_rate": None, "roi": None}


def test_build_report_records_a_load_error_without_aborting(tmp_path, monkeypatch):
    good_csv = tmp_path / "20260810" / "daily_prediction_pool_20260810.csv"
    good_csv.parent.mkdir(parents=True)
    good_csv.write_text("pool", encoding="utf-8")
    bad_csv = tmp_path / "20260811" / "daily_prediction_pool_20260811.csv"
    bad_csv.parent.mkdir(parents=True)
    bad_csv.write_text("pool", encoding="utf-8")

    win = _candidate()

    def fake_prepare(args):
        if "20260811" in str(args.pool_csv):
            raise RuntimeError("boom")
        return [win], None

    monkeypatch.setattr(report_builder, "find_raw_pool_csvs", lambda root: [good_csv, bad_csv])
    monkeypatch.setattr(report_builder, "parse_v11_args", lambda csv: SimpleNamespace(pool_csv=csv))
    monkeypatch.setattr(shp, "prepare_and_filter_candidates", fake_prepare)
    monkeypatch.setattr(shp, "select_top_candidates", lambda eligible, args: eligible)
    monkeypatch.setattr(
        report_builder, "build_actual_lookup", lambda root: {("2026-08-10", "real_player", "TB", "824970"): 1.0}
    )

    result = report_builder.build_report(daily_runs_root=tmp_path, processed_root=tmp_path)

    assert result["dates_scanned"] == ["20260810"]
    assert "20260811" in result["dates_with_load_errors"]
    assert len(result["picks"]) == 1
