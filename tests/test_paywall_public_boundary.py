from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPO_ROOT / "dist"


def test_protected_route_prefixes_are_absent_from_public_bundle() -> None:
    forbidden = ("app", "member", "private-data", "downloads")
    leaked = [name for name in forbidden if (PUBLIC_ROOT / name).exists()]
    assert not leaked, f"protected route prefixes exist in dist/: {leaked}"


def test_sport_boards_and_prediction_payloads_are_published() -> None:
    missing = []
    for sport in ("nba", "mlb", "nfl"):
        for relative in ("predictions/index.html", "data/daily_predictions.json"):
            path = PUBLIC_ROOT / sport / relative
            if not path.is_file():
                missing.append(str(path.relative_to(PUBLIC_ROOT)))
    assert not missing, f"public prediction assets are missing: {missing}"


def test_legacy_private_route_prefix_is_absent_from_public_manifest() -> None:
    manifest = (PUBLIC_ROOT / "data" / "sports.json").read_text(encoding="utf-8")
    assert '"/app/' not in manifest
