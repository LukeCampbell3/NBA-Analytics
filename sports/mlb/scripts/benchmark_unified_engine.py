#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.mlb.unified.trajectory import simulate_team_runs


def main() -> int:
    results = []
    for trials in (5_000, 10_000, 50_000):
        tracemalloc.start()
        started = time.perf_counter()
        batch = simulate_team_runs(4.5, 4.1, trials=trials, seed=17)
        probability = float(((batch.home_runs + batch.away_runs) > 8.5).mean())
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({"trials": trials, "seconds_per_game": elapsed, "peak_memory_mb": peak/1024/1024, "game_over_8_5_probability": probability})
    tracemalloc.start()
    started = time.perf_counter()
    for game_index in range(15):
        simulate_team_runs(4.0 + game_index / 20, 4.2, trials=10_000, seed=17 + game_index)
    full_slate_seconds = time.perf_counter() - started
    _, full_slate_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output = ROOT / "sports/mlb/unified/PERFORMANCE_REPORT.json"
    convergence_delta = abs(results[-1]["game_over_8_5_probability"] - results[-2]["game_over_8_5_probability"])
    output.write_text(json.dumps({"evidence_state":"DEVELOPMENT", "seed":17, "results":results,
        "convergence_10000_to_50000": convergence_delta,
        "full_slate_15_games_10000_trials_seconds": full_slate_seconds,
        "full_slate_peak_memory_mb": full_slate_peak/1024/1024}, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
