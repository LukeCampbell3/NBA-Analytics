from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_ODDS_PROVIDERS_ROOT = REPO_ROOT / "sports" / "mlb" / "predictions" / "odds" / "providers"
sys.path.insert(0, str(MLB_ODDS_PROVIDERS_ROOT))

from fanduel_public_mlb_provider import FanduelPublicMlbProvider  # noqa: E402
from fanduel_regions import FANDUEL_LICENSED_STATES, STATE_NAMES  # noqa: E402


def test_region_constructor_param_overrides_env_var(monkeypatch) -> None:
    """Real bug this guards against: build_multi_region_odds_indexes()
    constructs one real provider per state in the same process -- if the
    constructor silently kept reading only the env var, every one of
    those instances would collapse to the same single region."""
    monkeypatch.setenv("MLB_FANDUEL_REGION", "NJ")

    assert FanduelPublicMlbProvider(region="NY").region == "NY"
    assert FanduelPublicMlbProvider().region == "NJ"  # env var still respected when region is omitted
    assert FanduelPublicMlbProvider(region="pa").region == "PA"  # normalized uppercase, same as the env-var path


def test_fanduel_licensed_states_list_is_internally_consistent() -> None:
    """Real, disclosed data-integrity check on the verified state list
    itself: no duplicates, every code has a real display name, and no
    orphaned display name for a state not actually in the list."""
    assert len(FANDUEL_LICENSED_STATES) == len(set(FANDUEL_LICENSED_STATES))
    assert set(FANDUEL_LICENSED_STATES) == set(STATE_NAMES)
    assert all(len(code) == 2 and code.isupper() for code in FANDUEL_LICENSED_STATES)
