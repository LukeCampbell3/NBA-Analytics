from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MLB_SCRIPTS_ROOT = REPO_ROOT / "sports" / "mlb" / "scripts"
sys.path.insert(0, str(MLB_SCRIPTS_ROOT))

import update_mlb_player_headshot_cache as cache_script  # noqa: E402


MLB_URL = "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/624413/headshot/67/current"
MLB_FALLBACK = "https://midfield.mlbstatic.com/v1/people/624413/headshot/67/current"


def _payload_with_all_leg_shapes() -> dict:
    return {
        "plays": [
            {"player_display_name": "Pete Alonso", "player_headshot_url": MLB_URL, "player_headshot_fallback_url": MLB_FALLBACK},
        ],
        "daily_parlay": {
            "selected_ticket": {
                "legs": [
                    {"player_display_name": "CJ Abrams", "player_headshot_url": "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/682928/headshot/67/current"},
                ],
            },
        },
        "parlays": {
            "shadow_candidate": {
                "leg_1": {"player": "Pete Crow-Armstrong", "player_headshot_url": "https://img.mlbstatic.com/mlb-photos/image/upload/w_180,q_auto:best/v1/people/691718/headshot/67/current"},
            },
        },
    }


def test_find_headshot_dicts_covers_plays_legacy_ticket_and_v2_legs():
    payload = _payload_with_all_leg_shapes()
    found = cache_script.find_headshot_dicts(payload)
    assert len(found) == 3


def test_person_id_parsed_from_real_url_pattern():
    assert cache_script._person_id_from_url(MLB_URL) == "624413"
    assert cache_script._person_id_from_url("not a url") == ""


def test_collect_headshot_entries_dedupes_across_multiple_payloads():
    payload_a = _payload_with_all_leg_shapes()
    payload_b = {"plays": [{"player_headshot_url": MLB_URL}]}  # same player_id 624413 again
    entries = cache_script.collect_headshot_entries([payload_a, payload_b])
    ids = [e.id for e in entries]
    assert ids.count("624413") == 2  # collect doesn't dedupe itself -- sync_headshot_cache does
    assert "682928" in ids
    assert "691718" in ids


def test_sync_and_rewrite_points_at_local_cache_and_keeps_real_remote_as_fallback(tmp_path: Path):
    daily_path = tmp_path / "daily_predictions.json"
    daily_path.write_text(json.dumps(_payload_with_all_leg_shapes()), encoding="utf-8")
    cache_dir = tmp_path / "headshots"
    manifest_path = cache_dir / "manifest.json"

    def fake_fetch(url: str):
        return b"real-image-bytes", "image/jpeg"

    summary = cache_script.sync_and_rewrite(
        [daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=fake_fetch,
    )

    assert summary["unique_players_seen"] == 3
    assert summary["sync"]["downloaded"] == 3

    written = json.loads(daily_path.read_text(encoding="utf-8"))
    play = written["plays"][0]
    assert play["player_headshot_url"] == "data/headshots/624413.jpg"
    assert play["player_headshot_fallback_url"] == MLB_URL  # real remote kept as safety net

    leg = written["daily_parlay"]["selected_ticket"]["legs"][0]
    assert leg["player_headshot_url"] == "data/headshots/682928.jpg"


def test_sync_and_rewrite_is_incremental_on_second_run(tmp_path: Path):
    daily_path = tmp_path / "daily_predictions.json"
    daily_path.write_text(json.dumps(_payload_with_all_leg_shapes()), encoding="utf-8")
    cache_dir = tmp_path / "headshots"
    manifest_path = cache_dir / "manifest.json"
    calls: list[str] = []

    def counting_fetch(url: str):
        calls.append(url)
        return b"real-image-bytes", "image/jpeg"

    cache_script.sync_and_rewrite([daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch)
    # Second run: payload's player_headshot_url is now the LOCAL path (no
    # /people/<id>/ pattern to re-parse), so nothing new should be fetched.
    summary2 = cache_script.sync_and_rewrite([daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=counting_fetch)

    assert len(calls) == 3  # only the first run's real fetches
    assert summary2["sync"]["downloaded"] == 0


def test_sync_and_rewrite_leaves_unresolved_headshot_untouched_when_fetch_fails(tmp_path: Path):
    daily_path = tmp_path / "daily_predictions.json"
    daily_path.write_text(json.dumps(_payload_with_all_leg_shapes()), encoding="utf-8")
    cache_dir = tmp_path / "headshots"
    manifest_path = cache_dir / "manifest.json"

    def failing_fetch(url: str):
        raise OSError("network down")

    cache_script.sync_and_rewrite([daily_path], cache_dir=cache_dir, manifest_path=manifest_path, fetch_fn=failing_fetch)

    written = json.loads(daily_path.read_text(encoding="utf-8"))
    play = written["plays"][0]
    assert play["player_headshot_url"] == MLB_URL  # unchanged -- never a broken/fake local path
