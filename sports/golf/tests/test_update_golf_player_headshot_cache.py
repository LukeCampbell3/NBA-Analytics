from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLF_SCRIPTS_ROOT = REPO_ROOT / "sports" / "golf" / "scripts"
sys.path.insert(0, str(GOLF_SCRIPTS_ROOT))

import update_golf_player_headshot_cache as cache_script  # noqa: E402


GOLF_URL = "https://a.espncdn.com/i/headshots/golf/players/full/9478.png"


def _payload() -> dict:
    return {
        "candidates": [
            {"player_name": "Scottie Scheffler", "player_headshot_url": GOLF_URL},
        ],
        "top_10": [
            {"player_name": "Scottie Scheffler", "headshot_url": GOLF_URL},  # not bettable -- must not be cached
        ],
    }


def test_find_headshot_dicts_only_looks_at_candidates_not_top10():
    found = cache_script.find_headshot_dicts(_payload())
    assert len(found) == 1
    assert found[0]["player_name"] == "Scottie Scheffler"


def test_sync_and_rewrite_points_at_local_cache(tmp_path: Path):
    daily_path = tmp_path / "daily_predictions.json"
    daily_path.write_text(json.dumps(_payload()), encoding="utf-8")
    cache_dir = tmp_path / "headshots"
    manifest_path = cache_dir / "manifest.json"

    def fake_fetch(url: str):
        return b"real-image-bytes", "image/png"

    summary = cache_script.sync_and_rewrite([daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fake_fetch)

    assert summary["sync"]["downloaded"] == 1
    written = json.loads(daily_path.read_text(encoding="utf-8"))
    assert written["candidates"][0]["player_headshot_url"] == "data/headshots/9478.png"
    assert written["candidates"][0]["player_headshot_fallback_url"] == GOLF_URL
    # top_10's own headshot_url field is untouched -- out of scope for this cache
    assert written["top_10"][0]["headshot_url"] == GOLF_URL
