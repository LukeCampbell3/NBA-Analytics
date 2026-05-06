#!/usr/bin/env python3
"""
Daily prediction runner — NBA + MLB parlays and singles.

This is the single script to run every day.  It:
  1. Refreshes market data and processed player files
  2. Generates prediction pools for NBA and MLB
  3. Trains the direction classifier on recent history
  4. Applies precision enhancements (recency, instability, market signals)
  5. Builds parlay tickets and singles boards
  6. Updates the web frontend JSON payloads
  7. Prints the actionable daily board

Usage:
    python run_daily.py                          # run everything for today
    python run_daily.py --run-date 2026-05-04    # specific date
    python run_daily.py --skip-nba               # MLB only
    python run_daily.py --skip-mlb               # NBA only
    python run_daily.py --skip-refresh           # skip data refresh, use existing pools
    python run_daily.py --bankroll 500           # custom bankroll
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ET = ZoneInfo("America/New_York")

# --- NBA paths ---
NBA_PREDICTOR = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
NBA_DAILY_RUNNER = NBA_PREDICTOR / "scripts" / "run_daily_market_pipeline.py"
NBA_DAILY_RUNS = NBA_PREDICTOR / "model" / "analysis" / "daily_runs"
NBA_VALIDATION_DIR = REPO_ROOT / "sports" / "validation"
NBA_WEB_JSON = REPO_ROOT / "sports" / "nba" / "web" / "data" / "daily_predictions.json"

# --- MLB paths ---
MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
MLB_SELECTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_high_precision_predictions.py"
MLB_PARLAY_BUILDER = REPO_ROOT / "sports" / "mlb" / "scripts" / "build_daily_parlay_board.py"
MLB_DAILY_RUNS = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
MLB_DATA_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
MLB_MANIFEST = MLB_DATA_DIR / "update_manifest_2026.json"
MLB_DATA_UPDATER = REPO_ROOT / "Player-Predictor" / "scripts" / "update_mlb_processed_data.py"
MLB_MARKET_FETCHER = REPO_ROOT / "Player-Predictor" / "scripts" / "fetch_mlb_market_props.py"
MLB_WEB_JSON = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily prediction runner - NBA + MLB parlays and singles.")
    p.add_argument("--run-date", type=str, default=None, help="YYYY-MM-DD run date (default: today ET).")
    p.add_argument("--skip-nba", action="store_true", help="Skip NBA predictions.")
    p.add_argument("--skip-mlb", action="store_true", help="Skip MLB predictions.")
    p.add_argument("--skip-refresh", action="store_true", help="Skip data refresh, use existing pools.")
    p.add_argument("--skip-model-train", action="store_true", help="Skip direction model retraining.")
    p.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for stake sizing.")
    p.add_argument("--nba-policy", type=str, default="production_board_objective_b12", help="NBA policy profile.")
    return p.parse_args()


def run_step(label: str, cmd: list[str], allow_fail: bool = False) -> bool:
    print(f"\n  --- {label} ---")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        if allow_fail:
            print(f"  [warning] {label} failed: {e}")
            return False
        raise


def resolve_run_date(raw: str | None) -> str:
    if raw:
        return raw.replace("/", "-")
    return datetime.now(ET).strftime("%Y-%m-%d")


def run_date_stamp(d: str) -> str:
    return d.replace("-", "")


# ─────────────────────────────────────────────────────────
# Direction Model Training
# ─────────────────────────────────────────────────────────

def train_nba_direction_model():
    """Train the direction classifier on the latest validation history."""
    try:
        sys.path.insert(0, str(NBA_PREDICTOR))
        from decision_engine.direction_model import train_direction_model, DirectionModelConfig
        import pandas as pd

        # Find latest validation history
        hist_files = sorted(NBA_VALIDATION_DIR.glob("validation_recent_pool_selector_*_rows.csv"),
                           key=lambda p: p.stat().st_size, reverse=True)
        if not hist_files:
            print("  [info] No validation history found, skipping model training.")
            return None

        hist = pd.read_csv(hist_files[0])
        print(f"  Training direction model on {len(hist)} rows from {hist_files[0].name}")

        cfg = DirectionModelConfig()
        payload = train_direction_model(hist, config=cfg)

        if payload.get("model"):
            m = payload["metrics"]
            print(f"  Model trained: accuracy={m['accuracy']:.3f}, brier={m['brier_score']:.4f}")
            return payload
        else:
            print(f"  [warning] Model training failed: {payload.get('error')}")
            return None
    except Exception as e:
        print(f"  [warning] Direction model training failed: {e}")
        return None


# ─────────────────────────────────────────────────────────
# NBA
# ─────────────────────────────────────────────────────────

def run_nba(run_date: str, args: argparse.Namespace, direction_model=None) -> None:
    stamp = run_date_stamp(run_date)
    print(f"\n{'=' * 70}")
    print(f"  NBA PREDICTIONS - {run_date}")
    print(f"{'=' * 70}")

    if not args.skip_refresh:
        cmd = [
            PYTHON, str(NBA_DAILY_RUNNER),
            "--run-date", run_date,
            "--policy-profile", args.nba_policy,
            "--allow-heuristic-fallback",
        ]
        run_step("NBA Daily Pipeline", cmd, allow_fail=True)

    # Find the selector CSV
    date_dir = NBA_DAILY_RUNS / stamp
    selector_csvs = sorted(date_dir.glob("upcoming_market_play_selector_*.csv")) if date_dir.exists() else []

    if not selector_csvs:
        for d in sorted(NBA_DAILY_RUNS.iterdir(), reverse=True):
            if d.is_dir() and d.name.isdigit():
                selector_csvs = sorted(d.glob("upcoming_market_play_selector_*.csv"))
                if selector_csvs:
                    print(f"  [info] Using latest NBA selector from {d.name}")
                    break

    if not selector_csvs:
        print("  [warning] No NBA selector CSV found. Skipping NBA board.")
        return

    # Build NBA parlay board with enhancements
    try:
        sys.path.insert(0, str(NBA_PREDICTOR))
        from decision_engine.parlay_builder import build_daily_board, format_daily_board, ParlayConfig
        from decision_engine.precision_enhancements import annotate_precision_enhancements
        import pandas as pd

        selector = pd.read_csv(selector_csvs[-1])
        print(f"  Loaded {len(selector)} candidates")

        # Apply precision enhancements
        selector = annotate_precision_enhancements(selector)
        enhanced_count = (selector["precision_enhancement_adj"].abs() > 0.001).sum()
        print(f"  Precision enhancements applied: {enhanced_count} picks adjusted")

        # Apply direction model predictions if available
        if direction_model is not None:
            try:
                from decision_engine.direction_model import predict_win_probability
                model_probs = predict_win_probability(selector, direction_model)
                selector["direction_model_prob"] = model_probs
                # Use model probability to filter: penalize picks where model disagrees
                model_disagrees = (model_probs < 0.48) & (selector["expected_win_rate"] > 0.52)
                if model_disagrees.any():
                    print(f"  Direction model flagged {model_disagrees.sum()} picks as risky")
            except Exception as e:
                print(f"  [info] Direction model prediction skipped: {e}")

        # Build the board
        board = build_daily_board(selector, config=ParlayConfig())

        print(f"\n{'=' * 70}")
        print("  NBA DAILY BOARD")
        print(f"{'=' * 70}")
        try:
            print(format_daily_board(board, bankroll=args.bankroll))
        except UnicodeEncodeError:
            # Fallback for terminals that can't handle box-drawing chars
            diag = board.diagnostics
            print(f"  Parlays: {diag.get('primary_parlays', 0)} | Singles: {diag.get('singles', 0)}")
            for p in board.primary_parlays:
                legs = p.get("legs", [])
                print(f"  PARLAY ({len(legs)}-leg, joint={p.get('joint_prob', 0):.1%}):")
                for leg in legs:
                    print(f"    {leg.get('player', '?')} {leg.get('target', '?')} {leg.get('direction', '?')} {leg.get('market_line', '?')}")
            for i, s in enumerate(board.singles[:6], 1):
                print(f"  {i}. {s.get('player', '?')} {s.get('target', '?')} {s.get('direction', '?')} {s.get('market_line', '?')}")

        # Fetch bet slip links from The Odds API
        betslip_links = _fetch_betslip_links("basketball_nba")

        # Inject into web JSON
        _inject_nba_parlay_json(board, betslip_links=betslip_links)

    except Exception as e:
        print(f"  [warning] NBA parlay builder failed: {e}")
        import traceback
        traceback.print_exc()


def _fetch_betslip_links(sport: str) -> list[dict]:
    """Fetch FanDuel bet slip links. Falls back to constructing search URLs."""
    links = []

    # Try The Odds API first (requires paid plan for player prop links)
    try:
        sys.path.insert(0, str(NBA_PREDICTOR / "scripts"))
        from fetch_betslip_links import fetch_player_prop_links, NBA_MARKET_MAP, MLB_MARKET_MAP

        api_key = None
        for env_key in ("THE_ODDS_API_KEY", "ODDS_API_KEY"):
            api_key = os.environ.get(env_key)
            if api_key:
                break
        if not api_key:
            config_path = REPO_ROOT / "config.local.yaml"
            if config_path.exists():
                text = config_path.read_text(encoding="utf-8")
                match = re.search(r'api_key:\s*["\']?([a-f0-9]+)["\']?', text)
                if match:
                    api_key = match.group(1)

        if api_key:
            markets = list(NBA_MARKET_MAP.values()) if "nba" in sport else list(MLB_MARKET_MAP.values())
            try:
                links = fetch_player_prop_links(sport, markets, api_key)
                if links:
                    print(f"  Found {len(links)} FanDuel deep links")
                    return links
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: no deep links available, return empty
    # The frontend will construct FanDuel search URLs client-side
    return []


def _inject_nba_parlay_json(board, betslip_links=None) -> None:
    """Inject parlay board AND singles into the NBA web JSON."""
    if not NBA_WEB_JSON.exists():
        return

    # Build betslip link lookup
    link_lookup = {}
    if betslip_links:
        for item in betslip_links:
            player_key = str(item.get("player", "")).lower().strip()
            direction = str(item.get("direction", "")).upper()
            line = float(item.get("line", 0))
            market = str(item.get("market_key", ""))
            link_lookup[(player_key, market, direction, line)] = item

    def _find_link(name, target, direction, line):
        from fetch_betslip_links import NBA_MARKET_MAP
        market_key = NBA_MARKET_MAP.get(target, "")
        if not market_key:
            return None, None
        name_lower = name.lower().strip()
        # Exact match
        key = (name_lower, market_key, direction, line)
        if key in link_lookup:
            item = link_lookup[key]
            return item.get("link"), item.get("odds_american")
        # Fuzzy match
        for lk, lv in link_lookup.items():
            if lk[1] == market_key and lk[2] == direction and lk[3] == line:
                if name_lower in lk[0] or lk[0] in name_lower:
                    return lv.get("link"), lv.get("odds_american")
        return None, None

    try:
        payload = json.loads(NBA_WEB_JSON.read_text(encoding="utf-8"))

        # Build player ID lookup from cards.json for headshots
        player_id_lookup = {}
        cards_path = REPO_ROOT / "sports" / "nba" / "web" / "data" / "cards.json"
        if cards_path.exists():
            try:
                cards = json.loads(cards_path.read_text(encoding="utf-8"))
                for card in cards:
                    p = card.get("player", {})
                    name = str(p.get("name", "")).strip().lower()
                    pid = p.get("id")
                    if name and pid:
                        player_id_lookup[name] = int(pid)
            except Exception:
                pass

        def _find_player_id(name: str) -> int | None:
            n = name.strip().lower()
            if n in player_id_lookup:
                return player_id_lookup[n]
            # Try partial match
            for k, v in player_id_lookup.items():
                if n in k or k in n:
                    return v
            return None
        parlays_out = []
        for p in board.primary_parlays:
            legs_out = []
            for leg in p.get("legs", []):
                # Get full name from csv path
                csv_path = str(leg.get("csv", ""))
                full_name = str(leg.get("player", "")).replace("_", " ")
                if csv_path:
                    parts = csv_path.replace("\\", "/").split("/")
                    for j, part in enumerate(parts):
                        if part == "Data-Proc" and j + 1 < len(parts):
                            full_name = parts[j + 1].replace("_", " ")
                            break

                leg_odds = -110
                leg_decimal = 1.0 + (100.0 / 110.0)  # 1.909
                # Try to find FanDuel deep link
                leg_link, leg_real_odds = (None, None)
                if betslip_links:
                    try:
                        leg_link, leg_real_odds = _find_link(full_name, str(leg.get("target", "")), str(leg.get("direction", "")).upper(), float(leg.get("market_line", 0) or 0))
                    except Exception:
                        pass
                if leg_real_odds:
                    leg_odds = leg_real_odds
                    leg_decimal = (1.0 + 100.0 / abs(leg_odds)) if leg_odds < 0 else (1.0 + leg_odds / 100.0)

                legs_out.append({
                    "player": full_name,
                    "player_display_name": full_name,
                    "player_id": leg.get("player_id", None) or _find_player_id(full_name),
                    "target": str(leg.get("target", "")),
                    "direction": str(leg.get("direction", "")),
                    "market_line": float(leg.get("market_line", 0) or 0),
                    "win_rate": float(leg.get("expected_win_rate", 0) or 0),
                    "abs_edge": float(leg.get("abs_edge", 0) or 0),
                    "odds_american": leg_odds,
                    "odds_decimal": round(leg_decimal, 3),
                    "betslip_link": leg_link,
                })
            # Parlay odds = product of decimal odds for each leg
            n_legs = len(legs_out)
            parlay_decimal = 1.0
            for leg in legs_out:
                parlay_decimal *= leg["odds_decimal"]
            parlay_decimal = round(parlay_decimal, 2)
            # Convert to American
            if parlay_decimal >= 2.0:
                parlay_american = int(round((parlay_decimal - 1) * 100))
            else:
                parlay_american = int(round(-100 / (parlay_decimal - 1)))
            parlay_payout_per_dollar = round(parlay_decimal - 1, 2)

            parlays_out.append({
                "type": p.get("type", "primary"),
                "leg_count": n_legs,
                "joint_probability": float(p.get("joint_prob", 0) or 0),
                "adjusted_probability": float(p.get("adjusted_prob", 0) or 0),
                "avg_win_rate": float(p.get("avg_win_rate", 0) or 0),
                "n_games": p.get("n_games", 0),
                "odds_american": f"+{parlay_american}" if parlay_american > 0 else str(parlay_american),
                "odds_decimal": parlay_decimal,
                "payout_per_dollar": parlay_payout_per_dollar,
                "legs": legs_out,
            })
        payload["parlay_board"] = {"parlays": parlays_out, "diagnostics": board.diagnostics}

        # Singles with odds
        plays_out = []
        for i, pick in enumerate(board.singles, 1):
            pick_odds = -110
            pick_decimal = round(1.0 + (100.0 / 110.0), 3)
            # Try to find FanDuel deep link
            pick_link, pick_real_odds = (None, None)
            if betslip_links:
                try:
                    pick_link, pick_real_odds = _find_link(full_name, str(pick.get("target", "")), str(pick.get("direction", "")).upper(), float(pick.get("market_line", 0) or 0))
                except Exception:
                    pass
            if pick_real_odds:
                pick_odds = pick_real_odds
                pick_decimal = round((1.0 + 100.0 / abs(pick_odds)) if pick_odds < 0 else (1.0 + pick_odds / 100.0), 3)

            # Get full player name from csv path
            csv_path = str(pick.get("csv", ""))
            full_name = str(pick.get("player", "")).replace("_", " ")
            if csv_path:
                parts = csv_path.replace("\\", "/").split("/")
                for j, part in enumerate(parts):
                    if part == "Data-Proc" and j + 1 < len(parts):
                        full_name = parts[j + 1].replace("_", " ")
                        break

            # Get player_id from market_player_raw or construct headshot URL
            player_id = pick.get("player_id", None)
            market_player_raw = str(pick.get("market_player_raw", ""))

            plays_out.append({
                "rank": i,
                "player": full_name,
                "player_display_name": full_name,
                "player_id": player_id or _find_player_id(full_name),
                "market_player_raw": market_player_raw,
                "target": str(pick.get("target", "")),
                "direction": str(pick.get("direction", "")),
                "market_line": float(pick.get("market_line", 0) or 0),
                "prediction": float(pick.get("prediction", 0) or 0),
                "ev": float(pick.get("ev", 0) or 0),
                "abs_edge": float(pick.get("abs_edge", 0) or 0),
                "expected_win_rate": float(pick.get("expected_win_rate", 0) or 0),
                "recommendation": str(pick.get("pf_tier", pick.get("recommendation", "consider"))),
                "market_home_team": str(pick.get("market_home_team", "")),
                "market_away_team": str(pick.get("market_away_team", "")),
                "market_date": str(pick.get("market_date", "")),
                "stake_tier": str(pick.get("stake_tier", "")),
                "stake_fraction": float(pick.get("stake_fraction", 0) or 0),
                "odds_american": pick_odds,
                "odds_decimal": pick_decimal,
                "betslip_link": pick_link,
            })
        payload["plays"] = plays_out

        NBA_WEB_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  Web JSON updated: {NBA_WEB_JSON.name}")
    except Exception as e:
        print(f"  [warning] Failed to update NBA web JSON: {e}")


# ─────────────────────────────────────────────────────────
# MLB
# ─────────────────────────────────────────────────────────

def run_mlb(run_date: str, args: argparse.Namespace) -> None:
    stamp = run_date_stamp(run_date)
    print(f"\n{'=' * 70}")
    print(f"  MLB PREDICTIONS - {run_date}")
    print(f"{'=' * 70}")

    pool_csv = None
    selected_csv = None

    if not args.skip_refresh:
        if MLB_MARKET_FETCHER.exists():
            run_step("Fetch MLB Market Props", [
                PYTHON, str(MLB_MARKET_FETCHER),
                "--provider", "rotowire",
                "--event-date", run_date,
            ], allow_fail=True)

        if MLB_DATA_UPDATER.exists():
            run_step("Update MLB Processed Data", [
                PYTHON, str(MLB_DATA_UPDATER),
                "--through-date", run_date,
            ], allow_fail=True)

        if MLB_GENERATOR.exists():
            run_step("Generate MLB Prediction Pool", [
                PYTHON, str(MLB_GENERATOR),
                "--run-date", run_date,
                "--data-dir", str(MLB_DATA_DIR),
                "--manifest", str(MLB_MANIFEST),
                "--daily-runs-root", str(MLB_DAILY_RUNS),
                "--fallback-policy", "exact_or_latest",
            ], allow_fail=True)

    # Find pool CSV
    date_dir = MLB_DAILY_RUNS / stamp
    if date_dir.exists():
        pool_csvs = sorted(date_dir.glob(f"daily_prediction_pool_{stamp}.csv"))
        if pool_csvs:
            pool_csv = pool_csvs[-1]

    if pool_csv is None:
        for d in sorted(MLB_DAILY_RUNS.iterdir(), reverse=True):
            if d.is_dir() and d.name.isdigit():
                csvs = sorted(d.glob("daily_prediction_pool_*.csv"))
                csvs = [c for c in csvs if "high_precision" not in c.name and "best_" not in c.name]
                if csvs:
                    pool_csv = csvs[-1]
                    print(f"  [info] Using latest MLB pool from {d.name}")
                    break

    if pool_csv is None:
        print("  [warning] No MLB pool CSV found. Skipping MLB board.")
        return

    # Run high-precision selector
    selected_csv = pool_csv.parent / f"{pool_csv.stem}_high_precision_predictions.csv"
    summary_json = pool_csv.parent / f"{pool_csv.stem}_high_precision_predictions_summary.json"

    if MLB_SELECTOR.exists():
        run_step("Select MLB High-Precision Predictions", [
            PYTHON, str(MLB_SELECTOR),
            "--pool-csv", str(pool_csv),
            "--out-csv", str(selected_csv),
            "--summary-json", str(summary_json),
        ], allow_fail=True)

    if not selected_csv.exists():
        for d in sorted(MLB_DAILY_RUNS.iterdir(), reverse=True):
            if d.is_dir():
                hp = sorted(d.glob("*_high_precision_predictions.csv"))
                if hp:
                    selected_csv = hp[-1]
                    print(f"  [info] Using latest MLB selected from {d.name}")
                    break

    if selected_csv is None or not selected_csv.exists():
        print("  [warning] No MLB high-precision CSV found. Skipping MLB board.")
        return

    # Build MLB parlay board and inject into web JSON — all inline
    _build_and_inject_mlb(selected_csv, args.bankroll)


def _build_and_inject_mlb(selected_csv: Path, bankroll: float) -> None:
    """Build MLB parlay board and update web JSON inline."""
    try:
        import pandas as pd
        import sys as _sys

        # Register a fake module so dataclass resolution works
        import types
        mlb_mod_name = "mlb_decision_engine_parlay_builder"
        mlb_pb = types.ModuleType(mlb_mod_name)
        mlb_pb.__file__ = str(REPO_ROOT / "sports" / "mlb" / "decision_engine" / "parlay_builder.py")
        mlb_pb.__package__ = mlb_mod_name
        _sys.modules[mlb_mod_name] = mlb_pb

        source = (REPO_ROOT / "sports" / "mlb" / "decision_engine" / "parlay_builder.py").read_text(encoding="utf-8")
        code = compile(source, mlb_pb.__file__, "exec")
        exec(code, mlb_pb.__dict__)

        df = pd.read_csv(selected_csv)
        board = mlb_pb.build_mlb_daily_board(df)

        # Print the board
        print(f"\n  MLB PARLAY BOARD")
        print(f"  Parlays: {len(board.primary_parlays)} | Singles: {len(board.singles)}")
        for p in board.primary_parlays:
            legs = p.get("legs", [])
            print(f"  PARLAY ({len(legs)}-leg, joint={p.get('joint_prob', 0):.1%}):")
            for leg in legs:
                player = leg.get("Player", leg.get("player", "?"))
                target = leg.get("Target", leg.get("target", "?"))
                direction = leg.get("Direction", leg.get("direction", "?"))
                line = leg.get("Market_Line", leg.get("market_line", "?"))
                print(f"    {player} {target} {direction} {line}")
        for i, s in enumerate(board.singles[:5], 1):
            player = s.get("Player", s.get("player", "?"))
            target = s.get("Target", s.get("target", "?"))
            direction = s.get("Direction", s.get("direction", "?"))
            line = s.get("Market_Line", s.get("market_line", "?"))
            print(f"  {i}. {player} {target} {direction} {line}")

        # Inject into MLB web JSON
        if MLB_WEB_JSON.exists():
            payload = json.loads(MLB_WEB_JSON.read_text(encoding="utf-8"))
            parlays_out = []
            for p in board.primary_parlays:
                legs_out = []
                for leg in p.get("legs", []):
                    leg_odds = -110
                    leg_decimal = round(1.0 + (100.0 / 110.0), 3)
                    legs_out.append({
                        "player": str(leg.get("Player", leg.get("player", ""))),
                        "team": str(leg.get("Team", leg.get("team", ""))),
                        "opponent": str(leg.get("Opponent", leg.get("opponent", ""))),
                        "target": str(leg.get("Target", leg.get("target", ""))),
                        "direction": str(leg.get("Direction", leg.get("direction", ""))),
                        "market_line": float(leg.get("Market_Line", leg.get("market_line", 0)) or 0),
                        "hit_probability": float(leg.get("Estimated_Hit_Probability", leg.get("calibrated_hit_probability", 0)) or 0),
                        "abs_edge": float(leg.get("Abs_Edge", leg.get("abs_edge", 0)) or 0),
                        "confidence_tier": str(leg.get("Confidence_Tier", leg.get("confidence_tier", ""))),
                        "odds_american": leg_odds,
                        "odds_decimal": leg_decimal,
                    })
                n_legs = len(legs_out)
                parlay_decimal = 1.0
                for leg in legs_out:
                    parlay_decimal *= leg["odds_decimal"]
                parlay_decimal = round(parlay_decimal, 2)
                parlay_american = int(round((parlay_decimal - 1) * 100)) if parlay_decimal >= 2.0 else int(round(-100 / (parlay_decimal - 1)))
                parlays_out.append({
                    "type": p.get("type", "primary"),
                    "leg_count": n_legs,
                    "joint_probability": float(p.get("joint_prob", 0) or 0),
                    "adjusted_probability": float(p.get("adjusted_prob", 0) or 0),
                    "avg_hit_probability": float(p.get("avg_hit_prob", 0) or 0),
                    "n_games": p.get("n_games", 0),
                    "n_teams": p.get("n_teams", 0),
                    "odds_american": f"+{parlay_american}" if parlay_american > 0 else str(parlay_american),
                    "odds_decimal": parlay_decimal,
                    "payout_per_dollar": round(parlay_decimal - 1, 2),
                    "legs": legs_out,
                })
            payload["parlay_board"] = {"parlays": parlays_out, "diagnostics": board.diagnostics}
            MLB_WEB_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            print(f"  MLB Web JSON updated: {MLB_WEB_JSON.name}")

        # Clean up
        del _sys.modules[mlb_mod_name]

    except Exception as e:
        print(f"  [warning] MLB parlay board failed: {e}")


# ─────────────────────────────────────────────────────────
# Main
        print(f"  Web JSON updated: {MLB_WEB_JSON.name}")
    except Exception as e:
        print(f"  [warning] Failed to update MLB web JSON: {e}")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run_date = resolve_run_date(args.run_date)

    print("=" * 70)
    print(f"  DAILY PREDICTIONS - {run_date}")
    print(f"  Bankroll: ${args.bankroll:,.0f}")
    print("=" * 70)

    # Train direction model (shared across NBA picks)
    direction_model = None
    if not args.skip_nba and not args.skip_model_train:
        print("\n  --- Direction Model Training ---")
        direction_model = train_nba_direction_model()

    if not args.skip_nba:
        try:
            run_nba(run_date, args, direction_model=direction_model)
        except Exception as e:
            print(f"\n  [error] NBA pipeline failed: {e}")

    if not args.skip_mlb:
        try:
            run_mlb(run_date, args)
        except Exception as e:
            print(f"\n  [error] MLB pipeline failed: {e}")

    # Build the static site so dist/ is up to date
    print(f"\n  --- Building static site ---")
    build_script = REPO_ROOT / "sports" / "site" / "pipeline" / "build_static_site.py"
    if build_script.exists():
        try:
            subprocess.run([PYTHON, str(build_script)], check=True, capture_output=True)
            print(f"  Site built to dist/")
        except Exception as e:
            print(f"  [warning] Site build failed: {e}")

    print(f"\n{'=' * 70}")
    print("  DAILY RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Local preview: python -m http.server 8080 --directory dist/nba")
    print(f"  To deploy: git add dist/ && git commit -m 'daily update' && git push")


if __name__ == "__main__":
    main()
