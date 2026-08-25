"""Within-sport and global chronological validation (spec section 11.A/B)
-- thin wrappers around trainer.evaluate() making explicit which split_kind
was used, so a report can never conflate the two questions."""
from __future__ import annotations

from sports.universal_model.data.dataset import UniversalDataset
from sports.universal_model.model.universal_model import UniversalModel
from sports.universal_model.train.trainer import evaluate


def within_sport_chronological(model: UniversalModel, split: str = "TEST") -> dict:
    ds = UniversalDataset(split=split, split_kind="per_sport")
    return {"split_kind": "per_sport", "split": split, **evaluate(model, ds)}


def global_chronological(model: UniversalModel, split: str = "TEST") -> dict:
    ds = UniversalDataset(split=split, split_kind="global")
    return {"split_kind": "global", "split": split, **evaluate(model, ds)}
