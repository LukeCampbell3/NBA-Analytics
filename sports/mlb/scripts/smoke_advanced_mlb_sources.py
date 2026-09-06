#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.mlb.advanced.data_layer import load_profile_partition
from sports.mlb.advanced.production_refresh import refresh_advanced_profiles_incremental

DEFAULT_BOARD = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/mlb_advanced_source_smoke.json"


def _first_real_hitter(board: dict) -> dict:
    pools = [
        board.get("plays") or [],
        (board.get("tight_quality_overlay") or {}).get("rejections") or [],
        (board.get("v4_singles_shadow") or {}).get("plays") or [],
    ]
    for rows in pools:
        for raw in rows:
            row = raw.get("play") if isinstance(raw, dict) and isinstance(raw.get("play"), dict) else raw
            if not isinstance(row, dict):
                continue
            target = str(row.get("target") or row.get("Target") or "").upper()
            ptype = str(row.get("player_type") or row.get("Player_Type") or "hitter").lower()
            batter_id = row.get("player_mlbam_id") or row.get("player_id") or row.get("Player_MLBAM_ID")
            pitcher_id = row.get("opposing_pitcher_id") or row.get("Opposing_Pitcher_ID")
            game_id = row.get("game_id") or row.get("Game_ID")
            if target in {"H", "TB"} and ptype == "hitter" and batter_id and pitcher_id and game_id:
                return {
                    "Target": target,
                    "Player_Type": "hitter",
                    "Market_Source": "real",
                    "Game_ID": str(game_id),
                    "Player": row.get("player") or row.get("player_display_name") or row.get("Player"),
                    "Team": row.get("team") or row.get("Team"),
                    "Player_MLBAM_ID": int(batter_id),
                    "Opposing_Pitcher_ID": int(pitcher_id),
                    "Opposing_Pitcher": row.get("opposing_pitcher") or row.get("Opposing_Pitcher") or "",
                }
    raise RuntimeError("no current real-market H/TB candidate with batter/pitcher MLBAM IDs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove live Baseball Savant/FanGraphs connectivity on one current H/TB matchup.")
    parser.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback-days", type=int, default=45)
    args = parser.parse_args()

    board = json.loads(args.board.read_text(encoding="utf-8"))
    run_date = str(board.get("run_date") or "")
    if not run_date:
        raise RuntimeError("daily board has no run_date")
    candidate = _first_real_hitter(board)

    with tempfile.TemporaryDirectory(prefix="mlb-advanced-smoke-") as tmp:
        tmp_root = Path(tmp)
        pool_csv = tmp_root / "smoke_pool.csv"
        advanced_root = tmp_root / "advanced"
        pd.DataFrame([candidate]).to_csv(pool_csv, index=False)
        manifest = refresh_advanced_profiles_incremental(
            pool_csv=pool_csv,
            run_date=run_date,
            advanced_root=advanced_root,
            lookback_days=args.lookback_days,
            max_candidates=1,
        )
        batter_payload, pitcher_payload, matchup_payload, loaded_manifest = load_profile_partition(advanced_root, run_date)
        manifest = loaded_manifest or manifest
        batter = (batter_payload.get("profiles") or {}).get(str(candidate["Player_MLBAM_ID"]))
        pitcher = (pitcher_payload.get("profiles") or {}).get(str(candidate["Opposing_Pitcher_ID"]))
        direct = (matchup_payload.get("matchups") or {}).get(
            f"{candidate['Player_MLBAM_ID']}:{candidate['Opposing_Pitcher_ID']}"
        )

    source_status = manifest.get("source_status") or {}
    savant = source_status.get("baseball_savant_statcast") or {}
    fangraphs = source_status.get("fangraphs") or {}
    if savant.get("status") != "SUCCESS":
        raise RuntimeError(f"Baseball Savant smoke failed: {json.dumps(savant, sort_keys=True)}")
    if not batter or not pitcher:
        raise RuntimeError("Baseball Savant smoke did not produce both batter and pitcher profiles")

    result = {
        "schema_version": "mlb_advanced_source_smoke_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "candidate": candidate,
        "effective_as_of_date": manifest.get("effective_as_of_date"),
        "source_status": source_status,
        "baseball_savant": {
            "status": savant.get("status"),
            "batter_profile_present": True,
            "pitcher_profile_present": True,
            "direct_bvp_present": direct is not None,
            "batter_sample_pa": batter.get("pa"),
            "pitcher_sample_pa": pitcher.get("pa"),
            "batter_fields": {
                key: batter.get(key)
                for key in ("woba", "xwoba", "ba", "xba", "slg", "xslg", "avg_ev", "hard_hit_rate", "barrel_rate", "sweet_spot_rate", "k_rate", "bb_rate", "contact_rate", "whiff_rate", "chase_rate")
            },
            "pitcher_fields": {
                key: pitcher.get(key)
                for key in ("era", "fip", "xfip", "siera", "xera", "xwoba_allowed", "xba_allowed", "xslg_allowed", "avg_ev_allowed", "hard_hit_rate_allowed", "barrel_rate_allowed", "whiff_rate", "csw_rate")
            },
            "arsenal_pitch_types": sorted((pitcher.get("arsenal") or {}).keys()),
        },
        "fangraphs": {
            "status": fangraphs.get("status"),
            "rows": fangraphs.get("rows"),
            "available_fields": fangraphs.get("available_fields") or [],
            "error": fangraphs.get("error"),
            "pitcher_xfip": pitcher.get("xfip"),
            "pitcher_siera": pitcher.get("siera"),
        },
        "failures": manifest.get("failures") or [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
