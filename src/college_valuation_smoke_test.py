"""
college_valuation_smoke_test.py

Tracked smoke test for college valuation + verification.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd

from value_college_players import run_college_valuation
from validate_college_metric_parity import run_parity_validation


def run_test() -> None:
    tmp = Path(tempfile.mkdtemp())
    input_csv = tmp / "players_season.csv"
    build_summary = tmp / "build_summary.json"
    out_dir = tmp / "college_valuations"

    sample = pd.DataFrame(
        [
            {
                "player_key": "alpha_guard_2025_team_a",
                "team_key": "team_a_2025",
                "player_name": "Alpha Guard",
                "team_name": "Team A",
                "season": 2025,
                "pos": "G",
                "class": "FR",
                "g": 34,
                "mp": 31.2,
                "usg_pct": 27.5,
                "bpm": 8.4,
                "ws": 6.2,
                "pts": 19.4,
                "ast": 5.6,
                "trb": 4.1,
                "stl": 1.8,
                "blk": 0.3,
                "tov": 2.7,
                "x3pa": 6.8,
                "fga": 14.7,
                "source_table": "players_per_game_raw",
                "source_url": "https://www.sports-reference.com/cbb/seasons/2025-per-game.html",
                "scraped_at_utc": "2026-03-10T22:00:00+00:00",
            },
            {
                "player_key": "beta_big_2025_team_b",
                "team_key": "team_b_2025",
                "player_name": "Beta Big",
                "team_name": "Team B",
                "season": 2025,
                "pos": "C",
                "class": "JR",
                "g": 33,
                "mp": 29.7,
                "usg_pct": 22.1,
                "bpm": 6.1,
                "ws": 5.3,
                "pts": 16.2,
                "ast": 2.1,
                "trb": 10.4,
                "stl": 0.9,
                "blk": 2.0,
                "tov": 2.5,
                "x3pa": 0.7,
                "fga": 11.9,
                "source_table": "players_per_game_raw",
                "source_url": "https://www.sports-reference.com/cbb/seasons/2025-per-game.html",
                "scraped_at_utc": "2026-03-10T22:00:00+00:00",
            },
        ]
    )
    sample.to_csv(input_csv, index=False)

    with open(build_summary, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "strict_robots": True,
                "user_agent": "NBAAnalyticsCollegePipeline/1.0 (+local research use)",
                "errors": [],
            },
            handle,
            indent=2,
        )

    summary, verification = run_college_valuation(
        input_path=input_csv,
        output_dir=out_dir,
        build_summary_path=build_summary,
        strict_verify=True,
    )

    assert summary["players_valuated"] == 2, f"Expected 2 valuations, got {summary['players_valuated']}"
    assert verification["overall_status"] == "pass", f"Expected pass verification, got {verification['overall_status']}"

    valuation_files = list(out_dir.glob("*_college_valuation.json"))
    assert len(valuation_files) == 2, f"Expected 2 output valuation files, got {len(valuation_files)}"
    assert (out_dir / "college_valuation_summary.json").exists(), "Missing valuation summary output"
    assert (out_dir / "college_valuation_verification.json").exists(), "Missing verification output"

    parity = run_parity_validation(
        input_path=input_csv,
        build_summary_path=build_summary,
        max_players=None,
        coverage_threshold=0.95,
    )
    assert parity["overall_status"] == "pass", f"Expected pass parity, got {parity['overall_status']}"

    print("[PASS] College valuation smoke test passed.")


if __name__ == "__main__":
    run_test()
