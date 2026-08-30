from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from validate_daily_publication import validate_mlb_v4_shadow  # noqa: E402


def _payload(*, deeplink: str | None) -> dict:
    play = {
        "authorization_status": "SHADOW_ONLY",
        "sportsbook": "fanduel",
    }
    if deeplink is not None:
        play["sportsbook_deeplink"] = deeplink
    return {
        "v4_singles_shadow": {
            "status": "INSUFFICIENT_PRIOR_SLATES",
            "publication_authority": False,
            "eligible_count": 1,
            "plays": [play],
        }
    }


def test_v4_every_slate_contract_accepts_real_fanduel_selection_link() -> None:
    payload = _payload(
        deeplink="https://sportsbook.fanduel.com/addToBetslip?marketId=734.1&selectionId=22"
    )

    validate_mlb_v4_shadow(payload, label="test")


def test_v4_every_slate_contract_rejects_unavailable_report() -> None:
    payload = _payload(deeplink=None)
    payload["v4_singles_shadow"]["status"] = "UNAVAILABLE"

    with pytest.raises(ValueError, match="without a completed V4 singles score"):
        validate_mlb_v4_shadow(payload, label="test")


def test_v4_every_slate_contract_rejects_fanduel_play_without_link() -> None:
    with pytest.raises(ValueError, match="missing its real deep link"):
        validate_mlb_v4_shadow(_payload(deeplink=None), label="test")


def test_v4_every_slate_contract_allows_honest_zero_pick_abstention() -> None:
    payload = {
        "v4_singles_shadow": {
            "status": "SHADOW_ONLY",
            "publication_authority": False,
            "eligible_count": 0,
            "plays": [],
        }
    }

    validate_mlb_v4_shadow(payload, label="test")
