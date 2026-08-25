"""Compute benchmarking (spec sections 25/56).

No GPU is available in this environment (see reports/INVENTORY.md) -- GPU
utilization/VRAM are reported as N/A rather than fabricated. Everything
else (dataset compile throughput, training throughput, inference latency,
checkpoint size) is measured for real on CPU.

Run: python -m sports.universal_model.validation.benchmark
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.train.checkpoints import load_checkpoint

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
CHECKPOINTS_DIR = MANIFESTS_DIR / "checkpoints"

CHECKPOINT_NAMES = ["dense_baseline", "switch_baseline", "top2_moe", "drm_final"]


@torch.no_grad()
def _inference_latency(model, dataset: UniversalDataset, batch_size: int, n_batches: int = 20) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    it = iter(loader)
    # warmup
    for _ in range(3):
        model(next(it))
    times = []
    for _ in range(n_batches):
        batch = next(it)
        t0 = time.perf_counter()
        model(batch)
        times.append((time.perf_counter() - t0) * 1000)
    model.train()
    return {
        "batch_size": batch_size,
        "mean_ms_per_batch": sum(times) / len(times),
        "mean_ms_per_example": (sum(times) / len(times)) / batch_size,
    }


def main() -> None:
    select = UniversalDataset(split="SELECT")
    report = {"gpu": "not available in this environment (torch.cuda.is_available()==False, CPU only)", "per_model": {}}

    for name in CHECKPOINT_NAMES:
        path = CHECKPOINTS_DIR / f"{name}.pt"
        if not path.exists():
            continue
        model, _ = load_checkpoint(path)
        latency_1 = _inference_latency(model, select, batch_size=1)
        latency_64 = _inference_latency(model, select, batch_size=64)
        report["per_model"][name] = {
            "total_params": model.total_parameters(),
            "active_params": model.active_parameters_per_token(),
            "checkpoint_size_bytes": path.stat().st_size,
            "single_example_latency_ms": latency_1["mean_ms_per_example"],
            "batch64_latency_ms_per_example": latency_64["mean_ms_per_example"],
        }

    # Training throughput: pulled from the already-saved baseline result
    # reports rather than re-run (those numbers ARE the real measurement).
    for name in ["dense_baseline_results", "switch_baseline_results", "top2_moe_results"]:
        p = REPORTS_DIR / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            report.setdefault("training_throughput", {})[name.replace("_results", "")] = {
                "examples_per_sec": data["examples_per_sec"],
                "wall_time_sec": data["wall_time_sec"],
            }

    # Dataset compile throughput: re-time the real compiler on the real sources.
    t0 = time.time()
    from sports.universal_model.data.compiler import collect_sufficient_sports

    events_by_sport, _, _, _ = collect_sufficient_sports()
    elapsed = time.time() - t0
    total_rows = sum(len(v) for v in events_by_sport.values())
    report["dataset_compile_throughput"] = {
        "elapsed_sec": elapsed,
        "rows": total_rows,
        "rows_per_sec": total_rows / elapsed if elapsed > 0 else None,
    }

    (REPORTS_DIR / "compute_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
