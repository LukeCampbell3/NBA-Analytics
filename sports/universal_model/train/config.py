"""Training configuration (spec section 37: every run stores its config)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    name: str
    hidden_dim: int = 64
    n_heads: int = 4
    block_types: list[str] = field(default_factory=lambda: ["dense", "dense", "dense", "dense"])
    n_experts: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 64
    steps: int = 1500
    eval_every: int = 250
    alpha: float = 0.5  # temperature sampling exponent, spec section 12
    seed: int = 1234
    lambda_load_balance: float = 0.01
    lambda_z_loss: float = 0.001
    lambda_prob: float = 1.0
    lambda_reg: float = 0.5
    use_scheduler: bool = False  # tested and rejected as a default -- see trainer.py

    def to_dict(self) -> dict:
        return asdict(self)
