"""Universal daily inference entrypoint (spec section 39): the SAME
checkpoint runs every sport. No sport-specific trained checkpoint exists;
only the sport's thin adapter differs.

    python -m sports.universal_model.inference.predict --sport mlb --date 2026-08-05
    python -m sports.universal_model.inference.predict --sport nfl --date 2025-12-20

For a sport whose adapter reports sufficient_for_training=False (nba,
golf, f1), this still runs -- it builds real observations from that
adapter (there may be zero, honestly, per reports/INVENTORY.md) and
produces the same universal output schema. It does NOT fabricate rows to
make an unsupported sport "work": zero real observations for that date
produces zero predictions, reported as such.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sports.universal_model.adapters.registry import build_adapter
from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.inference.payload import build_payload
from sports.universal_model.train.checkpoints import load_checkpoint

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
CHECKPOINTS_DIR = MANIFESTS_DIR / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "drm_final.pt"


def predict_for_date(sport: str, date: str, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> list[dict]:
    model, payload_meta = load_checkpoint(checkpoint_path)
    model.eval()

    adapter = build_adapter(sport)
    events, coverage = adapter.build_observations()
    day_events = [e for e in events if e.event_time[:10] == date]
    if not day_events:
        return []

    # Reuse the same compiled-dataset feature pipeline so inference-time
    # normalization/vocab lookups are identical to training (spec section
    # 39: one universal output schema, same checkpoint).
    ds = UniversalDataset.__new__(UniversalDataset)  # bypass split-file loading
    import pandas as pd

    from sports.universal_model.data.compiler import _to_wide_frame

    ds.norm = json.loads((MANIFESTS_DIR / "normalization_manifest.json").read_text())
    features = adapter.map_universal_features(day_events) + adapter.map_namespaced_features(day_events)
    ds.frame = _to_wide_frame(day_events, features)

    outputs = []
    with torch.no_grad():
        for i in range(len(ds.frame)):
            item = ds.__getitem__(i)
            batch = {k: v.unsqueeze(0) for k, v in item.items()}
            out = model(batch)
            row = ds.frame.iloc[i]
            outputs.append(build_payload(row, out, checkpoint_path.name))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    results = predict_for_date(args.sport, args.date, Path(args.checkpoint))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
