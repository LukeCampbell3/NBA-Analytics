"""Checkpoint save/load (spec section 38): must contain enough to
reproduce inference without any mutable global config file."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from sports.universal_model.data.dataset import current_signature
from sports.universal_model.model.universal_model import UniversalModel


def save_checkpoint(path: Path, model: UniversalModel, optimizer, config, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = current_signature()
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_config": model.config,
        "train_config": config.to_dict() if hasattr(config, "to_dict") else config,
        "dataset_signature": sig.__dict__,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> tuple[UniversalModel, dict]:
    payload = torch.load(path, weights_only=False)
    model = UniversalModel(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    current = current_signature()
    saved = payload["dataset_signature"]
    if saved != current.__dict__:
        payload["extra"]["signature_mismatch_warning"] = (
            f"checkpoint dataset signature {saved} does not match current manifests {current.__dict__}"
        )
    return model, payload
