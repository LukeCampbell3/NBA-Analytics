from __future__ import annotations

import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPO_ROOT / "dist"
PRIVATE_ROOT = REPO_ROOT / "paywall" / "private-content"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protected_route_prefixes_are_absent_from_public_bundle() -> None:
    forbidden = ("app", "member", "private-data", "downloads")
    leaked = [name for name in forbidden if (PUBLIC_ROOT / name).exists()]
    assert not leaked, f"protected route prefixes exist in dist/: {leaked}"


def test_sport_boards_and_prediction_payloads_are_absent_from_public_bundle() -> None:
    forbidden_roots = ("nba", "mlb", "nfl")
    forbidden_files = {"daily_predictions.json", "predictions.js", "prediction-about.js"}
    leaked_roots = [name for name in forbidden_roots if (PUBLIC_ROOT / name).exists()]
    leaked_files = [
        str(path.relative_to(PUBLIC_ROOT))
        for path in PUBLIC_ROOT.rglob("*")
        if path.is_file() and path.name in forbidden_files
    ]
    assert not leaked_roots, f"protected sport roots exist in dist/: {leaked_roots}"
    assert not leaked_files, f"prediction payloads/scripts exist in dist/: {leaked_files}"


def test_private_source_bytes_are_absent_from_public_bundle() -> None:
    private_files = [
        path
        for path in PRIVATE_ROOT.rglob("*")
        if path.is_file() and path.name.lower() != "readme.md"
    ]
    if not private_files:
        return

    public_by_digest: dict[str, list[Path]] = {}
    for path in PUBLIC_ROOT.rglob("*"):
        if path.is_file():
            public_by_digest.setdefault(_digest(path), []).append(path)

    leaks: list[str] = []
    for private_path in private_files:
        for public_path in public_by_digest.get(_digest(private_path), []):
            leaks.append(
                f"{private_path.relative_to(REPO_ROOT)} -> "
                f"{public_path.relative_to(REPO_ROOT)}"
            )
    assert not leaks, "private content copied into dist/: " + ", ".join(leaks)
