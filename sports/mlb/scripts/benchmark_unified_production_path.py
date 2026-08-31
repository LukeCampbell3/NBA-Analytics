#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.mlb.unified.decision import DecisionPolicy, select
from sports.mlb.unified.parlay import construct_all_ticket_classes
from sports.mlb.unified.pipeline import collect_candidates, export_payload, run
from sports.mlb.unified.production_state import atomic_write_json


def measured(callable_, repetitions=20):
    samples=[]
    tracemalloc.start()
    result=None
    for _ in range(repetitions):
        started=time.perf_counter(); result=callable_(); samples.append(time.perf_counter()-started)
    _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return result,{"median_seconds":statistics.median(samples),"maximum_seconds":max(samples),"peak_memory_mb":peak/1024/1024,"repetitions":repetitions}


def main():
    data_dir=ROOT/'sports/mlb/web/data'
    (collected,status),normalization=measured(lambda:collect_candidates(data_dir))
    (accepted,rejected),selection=measured(lambda:select(collect_candidates(data_dir)[0],DecisionPolicy()))
    tickets,parlay=measured(lambda:construct_all_ticket_classes(accepted))
    result,full=measured(lambda:run(data_dir))
    payload=export_payload(result,run_date=json.loads((data_dir/'daily_predictions.json').read_text()).get('run_date'),repo_root=ROOT)
    report={"evidence_state":"DEVELOPMENT","candidate_count":len(collected),"accepted_count":len(accepted),"normalization":normalization,"selection":selection,"parlay_search":parlay,"full_core_path":full,"artifact_size_bytes":len(json.dumps(payload).encode()),"model_runtime":"NOT_SEPARATELY_MEASURABLE_COMPATIBILITY_INPUTS_PRECOMPUTED","calibration_runtime":"INCLUDED_IN_SOURCE_COMPATIBILITY_ARTIFACT"}
    atomic_write_json(ROOT/'artifacts/mlb_unified_runtime_benchmark.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
