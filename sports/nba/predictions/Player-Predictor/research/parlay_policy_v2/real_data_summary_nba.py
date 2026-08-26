"""Summarize the real, already-computed current-NBA-strategy backtest output.

sports/nba/web/data/history/*.json are real daily production exports. Each
carries a `parlay_validation` field, computed by
sports/parlay_analysis.py::evaluate_historical_parlays against real settled
NBA prop results, produced by the production pipeline itself (not by this
script) from a leg-level history CSV that is NOT committed to this repo
(`source_history_csv` in the output below points at a path on the machine
that generated it). This script only aggregates what's already there.

There is no equivalent NBA run of parlay_policy_v2 here: this repo has no
committed NBA leg-level dataset (probability + real settled result + real
price) to feed it, unlike MLB -- see real_data_backtest_mlb.py and
REPORT.md's "What would be needed" section.

Run: python3 sports/nba/predictions/Player-Predictor/research/parlay_policy_v2/real_data_summary_nba.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
HISTORY_DIR = REPO_ROOT / "sports" / "nba" / "web" / "data" / "history"


def main() -> dict:
    snapshots = []
    for path in sorted(HISTORY_DIR.glob("2026-*.json")):
        data = json.loads(path.read_text())
        pv = data.get("parlay_validation")
        if not pv or not pv.get("available"):
            continue
        sel = pv.get("selected", {})
        base = pv.get("baseline_all_pairs", {})
        snapshots.append(
            {
                "snapshot_file": path.name,
                "history_row_count": pv.get("history_row_count"),
                "sample_dates": pv.get("sample_dates"),
                "selected_hit": sel.get("hit_pair_count"),
                "selected_graded": sel.get("graded_pair_count"),
                "selected_hit_rate": sel.get("pair_hit_rate"),
                "baseline_hit": base.get("hit_pair_count"),
                "baseline_graded": base.get("graded_pair_count"),
                "baseline_hit_rate": base.get("pair_hit_rate"),
                "source_history_csv_on_generating_machine": pv.get("source_history_csv"),
            }
        )
    return {
        "note": (
            "Each row is a separate real production run's cumulative validation "
            "as of that day; sample sizes are small (selected picks max out at "
            "3 graded parlays in any single snapshot) and are NOT to be summed "
            "across rows -- they are overlapping/restated cumulative views, not "
            "independent samples."
        ),
        "snapshots": snapshots,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
