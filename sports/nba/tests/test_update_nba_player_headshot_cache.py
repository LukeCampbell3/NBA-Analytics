from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NBA_SCRIPTS_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor" / "scripts"
sys.path.insert(0, str(NBA_SCRIPTS_ROOT))

import update_nba_player_headshot_cache as cache_script  # noqa: E402


NBA_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/1629029.png"


def _payload() -> dict:
    return {
        "plays": [
            {"player_display_name": "Luka Doncic", "player_headshot_url": NBA_URL},
            {"player_display_name": "No Photo Player"},  # no headshot -- must be skipped, not fabricated
        ],
    }


def test_find_headshot_dicts_skips_plays_without_a_real_headshot_url():
    found = cache_script.find_headshot_dicts(_payload())
    assert len(found) == 1
    assert found[0]["player_display_name"] == "Luka Doncic"


def test_player_id_parsed_from_real_nba_cdn_url_pattern():
    assert cache_script._player_id_from_url(NBA_URL) == "1629029"
    assert cache_script._player_id_from_url("https://example.com/x.png") == ""


def test_sync_and_rewrite_points_at_local_cache_and_keeps_real_remote_as_fallback(tmp_path: Path):
    daily_path = tmp_path / "daily_predictions.json"
    daily_path.write_text(json.dumps(_payload()), encoding="utf-8")
    cache_dir = tmp_path / "headshots"
    manifest_path = cache_dir / "manifest.json"

    def fake_fetch(url: str):
        return b"real-image-bytes", "image/png"

    summary = cache_script.sync_and_rewrite([daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fake_fetch)

    assert summary["unique_players_seen"] == 1
    assert summary["sync"]["downloaded"] == 1

    written = json.loads(daily_path.read_text(encoding="utf-8"))
    play = written["plays"][0]
    assert play["player_headshot_url"] == "data/headshots/1629029.png"
    assert play["player_headshot_fallback_url"] == NBA_URL
    assert "player_headshot_url" not in written["plays"][1]


def test_sync_and_rewrite_never_fabricates_local_path_when_fetch_fails(tmp_path: Path):
    daily_path = tmp_path / "daily_predictions.json"
    daily_path.write_text(json.dumps(_payload()), encoding="utf-8")
    cache_dir = tmp_path / "headshots"
    manifest_path = cache_dir / "manifest.json"

    def failing_fetch(url: str):
        raise OSError("network down")

    cache_script.sync_and_rewrite([daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=failing_fetch)

    written = json.loads(daily_path.read_text(encoding="utf-8"))
    assert written["plays"][0]["player_headshot_url"] == NBA_URL
