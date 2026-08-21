from __future__ import annotations

"""PROGRAM-LEVEL MULTIPLICITY (mission section 13) -- a research-level
prospective alpha budget so that repeatedly freezing new policy versions
(after an earlier one fails or is demoted) does not create hidden
"try again until something clears" multiplicity.

    alpha_program = 0.05  (example; see manifest.ALPHA_PROGRAM)
    sum(alpha_policy_k for k in tested policy versions) <= alpha_program
    alpha_C + alpha_L + alpha_V <= alpha_policy_k          (within one policy)

Historical DEVELOPMENT-only policy versions (never frozen for prospective
confirmation) do NOT consume program alpha -- only versions that actually
entered FROZEN_PROSPECTIVE_INCONCLUSIVE (i.e. were tested as confirmatory
policies, per state_machine.PolicyStatus) draw from the budget. The
ledger is append-only and never resets a spent budget back to fresh on
failure -- see `spend` below.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROGRAM_ALPHA_LEDGER_VERSION = "PROGRAM_ALPHA_LEDGER_V1"


@dataclass(frozen=True)
class AlphaSpend:
    policy_version: str
    alpha_policy: float
    reason: str  # e.g. "frozen_for_prospective_confirmation"
    recorded_at_utc: str


class ProgramAlphaLedger:
    """Append-only JSON ledger. One file per research program (e.g. one
    per product surface -- here, the MLB 2-leg parlay program)."""

    def __init__(self, path: Path, alpha_program: float):
        self.path = Path(path)
        self.alpha_program = float(alpha_program)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return json.load(f)

    def _write(self, rows: list[dict]) -> None:
        with open(self.path, "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)

    def total_spent(self) -> float:
        return float(sum(row["alpha_policy"] for row in self._read()))

    def remaining(self) -> float:
        return float(self.alpha_program - self.total_spent())

    def already_spent_for(self, policy_version: str) -> bool:
        return any(row["policy_version"] == policy_version for row in self._read())

    def spend(self, spend: AlphaSpend) -> None:
        """Records alpha spend for a policy version entering prospective
        confirmation. Raises if this would exceed alpha_program, or if
        this policy_version already has a recorded spend (idempotent --
        a policy version's spend is recorded exactly once, at the moment
        it is frozen for prospective use, never re-spent on every
        evaluation). Never resets total_spent() on a demotion/failure --
        that would let a failed policy's alpha be silently reused."""
        if self.already_spent_for(spend.policy_version):
            return  # idempotent no-op -- already recorded, not a fresh spend
        rows = self._read()
        projected_total = sum(row["alpha_policy"] for row in rows) + spend.alpha_policy
        if projected_total > self.alpha_program + 1e-12:
            raise ValueError(
                f"recording alpha_policy={spend.alpha_policy} for {spend.policy_version!r} would bring "
                f"total spend to {projected_total} > alpha_program={self.alpha_program}"
            )
        rows.append(asdict(spend))
        self._write(rows)
