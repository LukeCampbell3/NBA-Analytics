#!/usr/bin/env python3
"""Select one adaptive, OVER-only MLB parlay from the broader daily pool."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sports.parlay_analysis import score_candidate_parlays

try:
    from . import select_high_precision_predictions as selector
    from .parlay_hit_survival_model import (
        EVIDENCE_LABEL as HIT_SURVIVAL_EVIDENCE_LABEL,
        MODEL_VERSION as HIT_SURVIVAL_MODEL_VERSION,
        candidate_features as hit_survival_candidate_features,
        fit_hit_survival_model,
    )
    from .latent_parlay_model import (
        DEFAULT_ARTIFACT_PATH as LATENT_PARLAY_ARTIFACT_PATH,
        LatentParlayBundle,
        candidate_features as latent_candidate_features,
        market_residual_probability,
    )
except ImportError:
    import select_high_precision_predictions as selector
    from parlay_hit_survival_model import (
        EVIDENCE_LABEL as HIT_SURVIVAL_EVIDENCE_LABEL,
        MODEL_VERSION as HIT_SURVIVAL_MODEL_VERSION,
        candidate_features as hit_survival_candidate_features,
        fit_hit_survival_model,
    )
    from latent_parlay_model import (
        DEFAULT_ARTIFACT_PATH as LATENT_PARLAY_ARTIFACT_PATH,
        LatentParlayBundle,
        candidate_features as latent_candidate_features,
        market_residual_probability,
    )


SCRIPT_PATH = Path(__file__).resolve()
SPORT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
CALIBRATION_ROOT = SPORT_ROOT / "data" / "predictions" / "calibration"
DEFAULT_PROVIDER_OBSERVATIONS = (
    SPORT_ROOT / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io" / "latest_provider_observations.csv"
)
POLICY_VERSION = "mlb_fanduel_public_betslip_parlay_v9"
ALLOWED_TARGETS = ("H", "TB", "R", "RBI", "K")
CONSISTENCY_TARGETS = {"H", "TB", "RBI"}
MIN_LEGS = 2
MAX_LEGS = 4
MIN_LEG_PROBABILITY = 0.62
MIN_TICKET_PROBABILITY = 0.40
MIN_COMBINED_DECIMAL_PRICE = 1.8
MIN_EXPECTED_RETURN = 0.0
MAX_CANDIDATES_PER_BOOK = 12
TICKET_PROBABILITY_FLOORS = {2: 0.40, 3: 0.25, 4: 0.18}
TICKET_RELIABILITY_PROBABILITY_FLOORS = {2: 0.42, 3: 0.40, 4: 0.30}
TICKET_RELIABILITY_MIN_LEG_PROBABILITY = {2: 0.64, 3: 0.66, 4: 0.68}
TICKET_MAX_DECIMAL_PRICES = {2: 6.0, 3: 10.0, 4: 18.0}
TICKET_TIERS = {2: "consistency", 3: "balanced", 4: "extended"}
TARGET_MARKET_TYPES = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "RBI": "batter_rbis",
    "K": "pitcher_strikeouts",
}
ALT_LINE_MAX_INCREMENTS = {"H": 1.0, "TB": 1.0, "R": 1.0, "RBI": 1.0, "K": 2.0}
PROFIT_BOOST_MIN_LEG_PROBABILITY = 0.18
PROFIT_BOOST_MIN_TICKET_PROBABILITY = 0.10
PROFIT_BOOST_MIN_DECIMAL_PRICE = 4.0
PROFIT_BOOST_MAX_DECIMAL_PRICE = 15.0
PROFIT_BOOST_MIN_EXPECTED_RETURN = 0.05
HIT_SURVIVAL_MIN_PROBABILITY = 0.58
HIT_SURVIVAL_MIN_CONSENSUS = 0.62
HIT_SURVIVAL_MAX_DISAGREEMENT = 0.16
MLB_LIVE_FEED_ROOT = "https://statsapi.mlb.com/api/v1.1/game"
FANDUEL_SPORTSBOOK_KEY = "fanduel"
FANDUEL_BETSLIP_ENDPOINT = "https://account.sportsbook.fanduel.com/sportsbook/addToBetslip"
FANDUEL_DEEPLINK_HOSTS = {"account.sportsbook.fanduel.com", "sportsbook.fanduel.com"}
FANDUEL_ID_PATTERN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an adaptive OVER-only MLB consistency parlay.")
    parser.add_argument("--pool-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--history-dir", type=Path, default=selector.DEFAULT_HISTORY_DIR)
    parser.add_argument("--provider-observations", type=Path, default=DEFAULT_PROVIDER_OBSERVATIONS)
    parser.add_argument("--min-legs", type=int, default=MIN_LEGS)
    parser.add_argument("--max-legs", type=int, default=MAX_LEGS)
    parser.add_argument("--min-leg-probability", type=float, default=MIN_LEG_PROBABILITY)
    parser.add_argument("--min-ticket-probability", type=float, default=MIN_TICKET_PROBABILITY)
    parser.add_argument("--min-combined-decimal-price", type=float, default=MIN_COMBINED_DECIMAL_PRICE)
    parser.add_argument("--min-expected-return", type=float, default=MIN_EXPECTED_RETURN)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _wilson_interval(wins: int, rows: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if rows <= 0:
        return None, None
    probability = wins / rows
    denominator = 1.0 + (z * z / rows)
    center = (probability + z * z / (2.0 * rows)) / denominator
    margin = z * math.sqrt((probability * (1.0 - probability) / rows) + (z * z / (4.0 * rows * rows))) / denominator
    return center - margin, center + margin


def _candidate_probability(candidate: selector.Candidate) -> float:
    latent_probability = _safe_probability(candidate.raw.get("Parlay_Leg_Probability"))
    if latent_probability is not None:
        return latent_probability
    return float(candidate.calibrated_graded_hit_rate)


def _safe_probability(value: object) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _safe_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_to_play(candidate: selector.Candidate) -> dict[str, Any]:
    probability = _candidate_probability(candidate)
    market_date = str(candidate.raw.get("Game_Date", candidate.run_date.isoformat()))
    return {
        "play_key": "|".join(
            (market_date, candidate.game_id, candidate.player, candidate.target, candidate.direction)
        ),
        "player": candidate.player,
        "player_display_name": candidate.player.replace("_", " "),
        "player_id": candidate.player_id,
        "player_type": str(candidate.raw.get("Player_Type", "")),
        "team": candidate.team,
        "opponent": str(candidate.raw.get("Opponent", "")),
        "game_id": candidate.game_id,
        "market_date": market_date,
        "commence_time_utc": str(candidate.raw.get("Commence_Time_UTC", "")),
        "game_status_code": candidate.game_status_code,
        "target": candidate.target,
        "direction": candidate.direction,
        "prediction": candidate.prediction,
        "market_line": candidate.market_line,
        "market_source": candidate.market_source,
        "market_bucket": candidate.market_bucket,
        "historical_bucket_key": candidate.historical_bucket_key,
        "estimated_graded_hit_rate": probability,
        "model_hit_probability": candidate.model_hit_probability,
        "market_implied_probability": candidate.market_implied_probability,
        "final_pool_quality_score": candidate.selection_score,
        "expected_value_per_unit": candidate.expected_value_per_unit,
        "selected_side_price": candidate.selected_side_price,
        "selected_sportsbook_key": candidate.selected_sportsbook_key,
        "selected_sportsbook": candidate.selected_sportsbook,
        "price_confirmed": candidate.price_confirmed,
        "market_books": candidate.market_books,
        "market_common_books": candidate.market_common_books,
        "market_book_keys": [value for value in candidate.market_book_keys.split("|") if value],
        "market_common_book_keys": [value for value in candidate.market_common_book_keys.split("|") if value],
        "history_rows": candidate.history_rows,
        "days_since_history": candidate.days_since_history,
        "selection_score": candidate.selection_score,
        "parlay_precision_eligible": True,
        "line_variant": "main",
        "hit_survival_probability": _safe_probability(candidate.raw.get("Hit_Survival_Probability")),
        "hit_survival_raw_probability": _safe_probability(candidate.raw.get("Hit_Survival_Raw_Probability")),
        "hit_survival_model_version": str(candidate.raw.get("Hit_Survival_Model_Version", "")),
        "hit_survival_evidence_label": str(candidate.raw.get("Hit_Survival_Evidence_Label", "")),
        "hit_survival_batting_order": _safe_number(candidate.raw.get("Hit_Survival_Batting_Order")),
        "hit_survival_batting_order_source": str(candidate.raw.get("Hit_Survival_Batting_Order_Source", "")),
        "latent_probability_disagreement": selector.to_float(
            candidate.raw.get("Parlay_Probability_Disagreement"), default=0.0
        ),
        "parlay_leg_probability": probability,
        "latent_leg_probability": _safe_probability(candidate.raw.get("Latent_Leg_Probability")),
        "latent_leg_raw_probability": _safe_probability(candidate.raw.get("Latent_Leg_Raw_Probability")),
        "latent_leg_ensemble_std": _safe_number(candidate.raw.get("Latent_Leg_Ensemble_Std")),
        "latent_support_fraction": _safe_number(candidate.raw.get("Latent_Support_Fraction")),
        "latent_in_support": bool(candidate.raw.get("Latent_In_Support", False)),
        "latent_model_version": str(candidate.raw.get("Latent_Model_Version", "")),
        "latent_evidence_label": str(candidate.raw.get("Latent_Evidence_Label", "")),
        "latent_numeric_features": candidate.raw.get("Latent_Numeric_Features"),
        "latent_categorical_features": candidate.raw.get("Latent_Categorical_Features"),
    }


def _american_to_decimal(price: object) -> float | None:
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or abs(value) < 100.0:
        return None
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def parse_fanduel_selection_deeplink(value: object) -> tuple[str, str] | None:
    """Extract provider-issued FanDuel IDs without accepting arbitrary redirect URLs."""
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return None
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() not in FANDUEL_DEEPLINK_HOSTS
        or not parsed.path.lower().endswith("/addtobetslip")
    ):
        return None
    query = parse_qs(parsed.query, keep_blank_values=True)
    market_ids = query.get("marketId") or query.get("marketId[0]") or []
    selection_ids = query.get("selectionId") or query.get("selectionId[0]") or []
    if len(market_ids) != 1 or len(selection_ids) != 1:
        return None
    market_id = str(market_ids[0]).strip()
    selection_id = str(selection_ids[0]).strip()
    if not FANDUEL_ID_PATTERN.fullmatch(market_id) or not FANDUEL_ID_PATTERN.fullmatch(selection_id):
        return None
    return market_id, selection_id


def build_fanduel_betslip_url(legs: list[dict[str, Any]]) -> str | None:
    selections: list[tuple[str, str]] = []
    for leg in legs:
        if str(leg.get("selected_sportsbook_key") or "").strip().lower() != FANDUEL_SPORTSBOOK_KEY:
            return None
        selection = parse_fanduel_selection_deeplink(leg.get("sportsbook_deeplink"))
        if selection is None or selection in selections:
            return None
        selections.append(selection)
    if len(selections) < MIN_LEGS:
        return None
    params: list[tuple[str, str]] = []
    for index, (market_id, selection_id) in enumerate(selections):
        params.extend(
            [
                (f"marketId[{index}]", market_id),
                (f"selectionId[{index}]", selection_id),
            ]
        )
    return f"{FANDUEL_BETSLIP_ENDPOINT}?{urlencode(params)}"


def attach_fanduel_betslip(ticket: dict[str, Any]) -> None:
    url = build_fanduel_betslip_url(ticket.get("legs") or [])
    if url is None:
        ticket["betslip"] = {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
            "sportsbook": "FanDuel",
            "status": "unavailable",
            "reason": "complete_provider_selection_links_unavailable",
        }
        ticket.pop("betslip_url", None)
        return
    ticket["betslip"] = {
        "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
        "sportsbook": "FanDuel",
        "status": "ready",
        "leg_count": len(ticket.get("legs") or []),
        "url": url,
        "source": "direct_fanduel_public_market_ids",
    }
    ticket["betslip_url"] = url


def _load_provider_observations(provider_observations: Path) -> pd.DataFrame:
    if not provider_observations.exists():
        return pd.DataFrame()
    try:
        observations = pd.read_csv(provider_observations, low_memory=False)
    except Exception:
        return pd.DataFrame()
    required = {
        "player_name", "market_type", "side", "line", "price_american", "sportsbook",
        "home_team", "away_team", "game_start_utc",
    }
    if observations.empty or not required.issubset(observations.columns):
        return pd.DataFrame()
    observations = observations.copy()
    observations["_player"] = observations["player_name"].map(_normalize_name)
    observations["_market"] = observations["market_type"].astype(str).str.strip().str.lower()
    observations["_side"] = observations["side"].astype(str).str.strip().str.lower()
    observations["_book"] = observations["sportsbook"].astype(str).str.strip().str.lower()
    observations["_line"] = pd.to_numeric(observations["line"], errors="coerce")
    observations["_price"] = pd.to_numeric(observations["price_american"], errors="coerce")
    observations["_start"] = pd.to_datetime(observations["game_start_utc"], utc=True, errors="coerce")
    observations["_market_date"] = observations["_start"].dt.tz_convert("America/New_York").dt.date.astype(str)
    observations["_home"] = observations["home_team"].map(_normalize_name)
    observations["_away"] = observations["away_team"].map(_normalize_name)
    if "validation_status" in observations:
        observations = observations.loc[
            observations["validation_status"].astype(str).str.strip().str.upper().eq("VALID")
        ]
    return observations


def _candidate_event_mask(observations: pd.DataFrame, candidate: selector.Candidate) -> pd.Series:
    candidate_date = str(candidate.raw.get("Game_Date") or candidate.run_date.isoformat())
    mask = observations["_market_date"].eq(candidate_date)
    candidate_start = pd.to_datetime(candidate.raw.get("Commence_Time_UTC"), utc=True, errors="coerce")
    if not pd.isna(candidate_start):
        mask &= observations["_start"].notna() & (
            (observations["_start"] - candidate_start).abs().dt.total_seconds() <= 5400
        )
    return mask


def build_fanduel_main_line_plays(
    candidates: list[selector.Candidate],
    provider_observations: Path,
) -> list[dict[str, Any]]:
    observations = _load_provider_observations(provider_observations)
    if observations.empty or "sportsbook_deeplink" not in observations:
        return []
    plays: list[dict[str, Any]] = []
    for candidate in candidates:
        market_type = TARGET_MARKET_TYPES.get(candidate.target)
        if market_type is None:
            continue
        rows = observations.loc[
            _candidate_event_mask(observations, candidate)
            & observations["_player"].eq(_normalize_name(candidate.player))
            & observations["_market"].eq(market_type)
            & observations["_side"].eq("over")
            & observations["_book"].eq(FANDUEL_SPORTSBOOK_KEY)
            & observations["_line"].sub(float(candidate.market_line)).abs().le(1e-9)
        ].copy()
        if rows.empty:
            continue
        rows["_selection"] = rows["sportsbook_deeplink"].map(parse_fanduel_selection_deeplink)
        rows = rows.loc[rows["_selection"].notna()]
        rows = rows.loc[rows["_price"].between(-250.0, 125.0, inclusive="both")]
        if rows.empty:
            continue
        if "observed_at_utc" in rows:
            rows["_observed"] = pd.to_datetime(rows["observed_at_utc"], utc=True, errors="coerce")
            rows = rows.sort_values("_observed", ascending=False, kind="stable")
        row = rows.iloc[0]
        price = float(row["_price"])
        decimal_price = _american_to_decimal(price)
        probability = _candidate_probability(candidate)
        if decimal_price is None or probability * decimal_price - 1.0 < -0.03:
            continue
        play = _candidate_to_play(candidate)
        play.update(
            {
                "selected_side_price": price,
                "selected_sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
                "selected_sportsbook": "FanDuel",
                "expected_value_per_unit": probability * decimal_price - 1.0,
                "sportsbook_deeplink": str(row["sportsbook_deeplink"]),
                "provider_source_market_id": str(row.get("source_market_id") or ""),
                "sportsbook_deeplink_observed_at_utc": str(row.get("observed_at_utc") or ""),
                "sportsbook_deeplink_source": str(row.get("source") or row.get("provider_name") or "provider"),
            }
        )
        plays.append(play)
    return plays


def build_alternate_line_plays(
    candidates: list[selector.Candidate],
    provider_observations: Path,
) -> list[dict[str, Any]]:
    observations = _load_provider_observations(provider_observations)
    if observations.empty or "sportsbook_deeplink" not in observations:
        return []

    plays: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, float]] = set()
    for candidate in candidates:
        market_type = TARGET_MARKET_TYPES.get(candidate.target)
        max_increment = ALT_LINE_MAX_INCREMENTS.get(candidate.target)
        if market_type is None or max_increment is None:
            continue
        same_event = _candidate_event_mask(observations, candidate)
        rows = observations.loc[
            same_event
            & observations["_player"].eq(_normalize_name(candidate.player))
            & observations["_market"].eq(market_type)
            & observations["_side"].eq("over")
            & observations["_book"].eq(FANDUEL_SPORTSBOOK_KEY)
            & observations["_line"].gt(float(candidate.market_line) + 1e-9)
            & observations["_line"].le(float(candidate.market_line) + float(max_increment) + 1e-9)
        ].copy()
        rows["_selection"] = rows["sportsbook_deeplink"].map(parse_fanduel_selection_deeplink)
        rows = rows.loc[rows["_selection"].notna()]
        for _, row in rows.iterrows():
            line = float(row["_line"])
            price = float(row["_price"])
            book = str(row["_book"])
            if not book or not math.isfinite(price) or not (100.0 <= price <= 600.0):
                continue
            raw_alt_probability = selector.estimate_count_hit_probabilities(
                max(0.0, float(candidate.prediction)), line, "OVER"
            )[2]
            base_model_probability = max(1e-6, float(candidate.model_hit_probability))
            calibration_ratio = min(1.0, float(candidate.calibrated_graded_hit_rate) / base_model_probability)
            probability = min(
                float(candidate.calibrated_graded_hit_rate),
                raw_alt_probability * calibration_ratio * 0.90,
            )
            decimal_price = _american_to_decimal(price)
            if probability < PROFIT_BOOST_MIN_LEG_PROBABILITY or decimal_price is None:
                continue
            expected_value = probability * decimal_price - 1.0
            if expected_value < 0.03:
                continue
            line_rows = rows.loc[rows["_line"].eq(line)]
            alternate_books = sorted({str(value) for value in line_rows["_book"] if str(value)})
            key = (candidate.player_id, candidate.target, book, line)
            if key in seen:
                continue
            seen.add(key)
            play = _candidate_to_play(candidate)
            play.update(
                {
                    "play_key": "|".join(
                        (play["market_date"], candidate.game_id, candidate.player, candidate.target, "OVER", f"ALT_{line:g}")
                    ),
                    "market_line": line,
                    "base_market_line": float(candidate.market_line),
                    "line_increment": line - float(candidate.market_line),
                    "line_variant": "alternate",
                    "market_bucket": f"{candidate.target}|OVER|{line:g}|ALT",
                    "historical_bucket_key": f"{candidate.target}|OVER|{line:g}|ALT",
                    "estimated_graded_hit_rate": probability,
                    "model_hit_probability": raw_alt_probability,
                    "expected_value_per_unit": expected_value,
                    "selected_side_price": price,
                    "selected_sportsbook_key": book,
                    "selected_sportsbook": "FanDuel",
                    "price_confirmed": True,
                    "sportsbook_deeplink": str(row.get("sportsbook_deeplink") or ""),
                    "alternate_line_price_source": str(row.get("source") or row.get("provider_name") or "provider"),
                    "provider_source_market_id": str(row.get("source_market_id") or ""),
                    "alternate_line_observed_at_utc": str(row.get("observed_at_utc") or ""),
                    "alternate_line_books": len(alternate_books),
                    "alternate_line_book_keys": alternate_books,
                    "parlay_precision_eligible": True,
                }
            )
            plays.append(play)
    return plays


def select_profit_boost_ticket(alternate_plays: list[dict[str, Any]]) -> dict[str, Any] | None:
    tickets = score_candidate_parlays(
        alternate_plays,
        sport="mlb",
        probability_field="estimated_graded_hit_rate",
        eligibility_field="parlay_precision_eligible",
        min_leg_probability=PROFIT_BOOST_MIN_LEG_PROBABILITY,
        min_pair_probability=PROFIT_BOOST_MIN_TICKET_PROBABILITY,
        min_legs_per_parlay=2,
        max_legs_per_parlay=2,
        forbid_same_market_bucket_parlay=False,
        min_expected_return_per_unit=PROFIT_BOOST_MIN_EXPECTED_RETURN,
    )
    executable: list[dict[str, Any]] = []
    for source in tickets:
        combined_price = float(source.get("combined_decimal_price") or 0.0)
        probability = float(source.get("projected_probability") or 0.0)
        expected_return = float(source.get("expected_return_per_unit") or -999.0)
        if not PROFIT_BOOST_MIN_DECIMAL_PRICE <= combined_price <= PROFIT_BOOST_MAX_DECIMAL_PRICE:
            continue
        if probability < PROFIT_BOOST_MIN_TICKET_PROBABILITY or expected_return < PROFIT_BOOST_MIN_EXPECTED_RETURN:
            continue
        if bool(source.get("same_game")) or bool(source.get("same_player")):
            continue
        ticket = dict(source)
        ticket["legs"] = [alternate_plays[int(index)] for index in ticket["leg_indices"]]
        ticket["ticket_tier"] = "profit_boost"
        ticket["probability_floor"] = PROFIT_BOOST_MIN_TICKET_PROBABILITY
        ticket["maximum_decimal_price"] = PROFIT_BOOST_MAX_DECIMAL_PRICE
        ticket["risk_adjusted_profit_score"] = expected_return * math.sqrt(probability)
        ticket["evidence_status"] = "SHADOW_ALT_LINE_PRICE_CAPTURE"
        executable.append(ticket)
    executable.sort(
        key=lambda row: (
            float(row.get("risk_adjusted_profit_score") or -999.0),
            float(row.get("projected_probability") or 0.0),
            float(row.get("expected_return_per_unit") or -999.0),
        ),
        reverse=True,
    )
    return executable[0] if executable else None


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii").lower()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def fetch_official_game_contexts(candidates: list[selector.Candidate]) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for game_id in sorted({candidate.game_id for candidate in candidates if candidate.game_id}):
        request = Request(f"{MLB_LIVE_FEED_ROOT}/{game_id}/feed/live", headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        game_data = payload.get("gameData") or {}
        status = game_data.get("status") or {}
        boxscore_teams = ((payload.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        game_teams = game_data.get("teams") or {}
        lineups: dict[str, set[str]] = {}
        batting_orders: dict[str, dict[str, int]] = {}
        for side in ("away", "home"):
            team = str((game_teams.get(side) or {}).get("abbreviation", "")).strip().upper()
            names: set[str] = set()
            orders: dict[str, int] = {}
            for player in ((boxscore_teams.get(side) or {}).get("players") or {}).values():
                raw_order = str(player.get("battingOrder", "")).strip()
                if not raw_order:
                    continue
                player_name = _normalize_name((player.get("person") or {}).get("fullName", ""))
                if not player_name:
                    continue
                names.add(player_name)
                try:
                    orders[player_name] = max(1, min(9, int(raw_order) // 100))
                except ValueError:
                    continue
            if team and names:
                lineups[team] = names
                batting_orders[team] = orders
        contexts[game_id] = {
            "state": str(status.get("abstractGameState", "")).strip().lower(),
            "lineups": lineups,
            "batting_orders": batting_orders,
        }
    return contexts


def apply_hit_survival_gate(
    candidates: list[selector.Candidate],
    *,
    bundle: Any,
    official_contexts: dict[str, dict[str, Any]],
    latent_bundle: LatentParlayBundle | None = None,
) -> tuple[list[selector.Candidate], Counter[str]]:
    """Apply an independently learned role score to supported H OVER 0.5 legs."""
    kept: list[selector.Candidate] = []
    rejected: Counter[str] = Counter()
    for candidate in candidates:
        if not (candidate.target == "H" and candidate.direction == "OVER" and abs(candidate.market_line - 0.5) <= 1e-9):
            kept.append(candidate)
            continue
        if bundle is None:
            rejected["hit_survival_model_unavailable"] += 1
            continue
        context = official_contexts.get(candidate.game_id, {})
        team_orders = (context.get("batting_orders") or {}).get(candidate.team.upper(), {})
        confirmed_order = team_orders.get(_normalize_name(candidate.player))
        features, order_source = hit_survival_candidate_features(
            candidate,
            bundle,
            confirmed_batting_order=float(confirmed_order) if confirmed_order is not None else None,
        )
        raw_probability, survival_probability = bundle.predict(features)
        if latent_bundle is not None:
            latent_numeric, latent_categories = latent_candidate_features(
                candidate,
                last_hits=features["last_hits"],
                batting_order=features["batting_order"],
            )
            latent_prediction = latent_bundle.predict_leg(latent_numeric, latent_categories)
            candidate.raw.update(
                {
                    "Latent_Leg_Probability": latent_prediction.probability,
                    "Latent_Leg_Raw_Probability": latent_prediction.raw_probability,
                    "Latent_Leg_Ensemble_Std": latent_prediction.ensemble_std,
                    "Latent_Support_Fraction": latent_prediction.support_fraction,
                    "Latent_In_Support": latent_prediction.in_support,
                    "Latent_Model_Version": latent_bundle.model_version,
                    "Latent_Evidence_Label": latent_bundle.evidence_label,
                    "Latent_Numeric_Features": latent_numeric,
                    "Latent_Categorical_Features": latent_categories,
                }
            )
        market_probability = candidate.market_implied_probability
        if market_probability is None:
            rejected["hit_survival_market_probability_unavailable"] += 1
            continue
        component_probabilities = [
            float(survival_probability),
            float(market_probability),
            float(candidate.calibrated_graded_hit_rate),
        ]
        disagreement = max(component_probabilities) - min(component_probabilities)
        consensus = (
            (0.45 * survival_probability)
            + (0.35 * market_probability)
            + (0.20 * candidate.calibrated_graded_hit_rate)
            - (0.30 * disagreement)
        )
        if confirmed_order is None:
            consensus -= 0.01
        consensus = max(0.01, min(0.99, float(consensus)))
        candidate.raw.update(
            {
                "Hit_Survival_Probability": survival_probability,
                "Hit_Survival_Raw_Probability": raw_probability,
                "Hit_Survival_Model_Version": HIT_SURVIVAL_MODEL_VERSION,
                "Hit_Survival_Evidence_Label": HIT_SURVIVAL_EVIDENCE_LABEL,
                "Hit_Survival_Batting_Order": features["batting_order"],
                "Hit_Survival_Batting_Order_Source": order_source,
                "Parlay_Probability_Disagreement": disagreement,
                "Parlay_Leg_Probability": consensus,
            }
        )
        if survival_probability < HIT_SURVIVAL_MIN_PROBABILITY:
            rejected["hit_survival_probability_too_low"] += 1
            continue
        if disagreement > HIT_SURVIVAL_MAX_DISAGREEMENT:
            rejected["hit_survival_probability_disagreement"] += 1
            continue
        if consensus < HIT_SURVIVAL_MIN_CONSENSUS:
            rejected["hit_survival_consensus_too_low"] += 1
            continue
        kept.append(candidate)
    return kept, rejected


def filter_anchor_candidates(candidates: list[selector.Candidate], *, min_leg_probability: float) -> tuple[list[selector.Candidate], Counter[str]]:
    kept: list[selector.Candidate] = []
    rejected: Counter[str] = Counter()
    for candidate in candidates:
        if candidate.direction != "OVER":
            rejected["not_over"] += 1
            continue
        if candidate.target not in ALLOWED_TARGETS:
            rejected["unsupported_anchor_target"] += 1
            continue
        if str(candidate.model_selected).strip().lower() == "baseline":
            rejected["baseline_model"] += 1
            continue
        if candidate.market_source != "real" or not candidate.price_confirmed:
            rejected["unconfirmed_market_price"] += 1
            continue
        if candidate.market_books < 1 or candidate.market_common_books < 1:
            rejected["insufficient_book_coverage"] += 1
            continue
        if not selector.is_standard_bettable_line(candidate.target, candidate.market_line):
            rejected["nonstandard_line"] += 1
            continue
        if candidate.history_rows < 35:
            rejected["history_too_short"] += 1
            continue
        if candidate.days_since_history is None or candidate.days_since_history > 4:
            rejected["history_too_stale"] += 1
            continue
        if not selector.is_upcoming_status(candidate.game_status_code, str(candidate.raw.get("Game_Status_Detail", ""))):
            rejected["game_not_upcoming"] += 1
            continue
        if candidate.selected_side_price is None or not (-250.0 <= candidate.selected_side_price <= 125.0):
            rejected["price_outside_playable_range"] += 1
            continue
        if _candidate_probability(candidate) < float(min_leg_probability):
            rejected["leg_probability_too_low"] += 1
            continue
        if candidate.expected_value_per_unit is None or candidate.expected_value_per_unit < -0.03:
            rejected["leg_value_too_low"] += 1
            continue
        kept.append(candidate)
    return kept, rejected


def select_ticket(
    candidates: list[selector.Candidate],
    *,
    min_legs: int,
    max_legs: int,
    min_leg_probability: float,
    min_ticket_probability: float,
    min_combined_decimal_price: float,
    min_expected_return: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    ladder, limited = select_ticket_ladder(
        candidates,
        min_legs=min_legs,
        max_legs=max_legs,
        min_leg_probability=min_leg_probability,
        base_min_ticket_probability=min_ticket_probability,
        min_combined_decimal_price=min_combined_decimal_price,
        min_expected_return=min_expected_return,
    )
    return (ladder[0] if ladder else None), limited


def _limited_plays_by_book(candidates: list[selector.Candidate]) -> dict[str, list[dict[str, Any]]]:
    return _limited_plays_by_book_from_plays([_candidate_to_play(candidate) for candidate in candidates])


def _limited_plays_by_book_from_plays(plays: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_book: dict[str, list[dict[str, Any]]] = {}
    for play in plays:
        by_book.setdefault(str(play["selected_sportsbook_key"]), []).append(play)
    for book, book_plays in by_book.items():
        book_plays.sort(
            key=lambda row: (
                float(row["estimated_graded_hit_rate"]),
                float(row.get("expected_value_per_unit") or -999.0),
                float(row.get("selection_score") or 0.0),
            ),
            reverse=True,
        )
        by_book[book] = book_plays[:MAX_CANDIDATES_PER_BOOK]
    return by_book


def _latent_set_profile(
    legs: list[dict[str, Any]],
    projected_probability: float,
    latent_bundle: LatentParlayBundle | None = None,
) -> dict[str, Any]:
    probabilities = [float(leg.get("parlay_leg_probability") or leg.get("estimated_graded_hit_rate") or 0.0) for leg in legs]
    disagreements = [float(leg.get("latent_probability_disagreement") or 0.0) for leg in legs]
    target_counts = Counter(str(leg.get("target") or "") for leg in legs)
    largest_target_cluster = max(target_counts.values(), default=0)
    prior_order_proxies = sum(
        str(leg.get("hit_survival_batting_order_source") or "") == "prior_start_proxy" for leg in legs
    )
    concentration_excess = max(0, largest_target_cluster - 1)
    maximum_disagreement = max(disagreements, default=0.0)
    uncertainty_penalty = min(
        0.20,
        (0.25 * maximum_disagreement)
        + (0.02 * concentration_excess)
        + (0.01 * prior_order_proxies),
    )
    consistency_score = float(projected_probability) * (1.0 - uncertainty_penalty)
    profile = {
        "representation": "permutation_invariant_leg_aggregate_v1",
        "minimum_leg_probability": min(probabilities, default=0.0),
        "mean_leg_probability": sum(probabilities) / len(probabilities) if probabilities else 0.0,
        "maximum_probability_disagreement": maximum_disagreement,
        "distinct_targets": len(target_counts),
        "largest_target_cluster": largest_target_cluster,
        "prior_order_proxy_legs": prior_order_proxies,
        "uncertainty_penalty": uncertainty_penalty,
        "set_consistency_score": consistency_score,
    }
    latent_inputs = [
        (leg.get("latent_numeric_features"), leg.get("latent_categorical_features"))
        for leg in legs
    ]
    if (
        latent_bundle is not None
        and all(isinstance(numeric, dict) and isinstance(categories, dict) for numeric, categories in latent_inputs)
    ):
        prediction = latent_bundle.predict_ticket(latent_inputs)
        independent_leg_product = math.prod(
            float(leg.get("latent_leg_probability") or 0.0) for leg in legs
        )
        market_leg_probabilities = [
            float(leg.get("market_implied_probability") or 0.0) for leg in legs
        ]
        hybrid_leg_probabilities = [
            market_residual_probability(
                leg.get("latent_leg_probability"),
                market_probability,
                leg.get("latent_leg_ensemble_std"),
            )
            for leg, market_probability in zip(legs, market_leg_probabilities)
        ]
        profile.update(
            {
                "shadow_representation": "gpu_set_attention_market_residual_v2",
                "shadow_joint_probability": prediction.probability,
                "shadow_independent_leg_product": independent_leg_product,
                "shadow_market_leg_product": math.prod(market_leg_probabilities),
                "shadow_hybrid_leg_product": math.prod(hybrid_leg_probabilities),
                "shadow_raw_joint_probability": prediction.raw_probability,
                "shadow_ensemble_std": prediction.ensemble_std,
                "shadow_support_fraction": prediction.support_fraction,
                "shadow_in_support": prediction.in_support,
                "shadow_model_version": latent_bundle.model_version,
                "shadow_evidence_label": latent_bundle.evidence_label,
                "shadow_authorization": "diagnostic_only",
            }
        )
    return profile


def _ticket_reliability_gate(ticket: dict[str, Any], leg_count: int) -> tuple[bool, dict[str, Any]]:
    legs = list(ticket.get("legs") or [])
    leg_probabilities = [
        float(leg.get("parlay_leg_probability") or leg.get("estimated_graded_hit_rate") or 0.0)
        for leg in legs
    ]
    targets = {str(leg.get("target") or "") for leg in legs}
    projected_probability = float(ticket.get("projected_probability") or 0.0)
    minimum_leg_probability = min(leg_probabilities, default=0.0)
    probability_floor = TICKET_RELIABILITY_PROBABILITY_FLOORS.get(leg_count, 1.0)
    leg_floor = TICKET_RELIABILITY_MIN_LEG_PROBABILITY.get(leg_count, 1.0)
    failures: list[str] = []
    if not targets.issubset(CONSISTENCY_TARGETS):
        failures.append("target_scope")
    if minimum_leg_probability < leg_floor:
        failures.append("minimum_leg_probability")
    if projected_probability < probability_floor:
        failures.append("projected_ticket_probability")
    profile = {
        "policy": "reliability_first_over_parlay_v1",
        "allowed_targets": sorted(CONSISTENCY_TARGETS),
        "probability_floor": probability_floor,
        "minimum_leg_probability_floor": leg_floor,
        "projected_probability": projected_probability,
        "minimum_leg_probability": minimum_leg_probability,
        "target_scope": sorted(targets),
        "status": "pass" if not failures else "withheld",
        "failures": failures,
    }
    return not failures, profile


def _best_ticket_for_leg_count(
    by_book: dict[str, list[dict[str, Any]]],
    *,
    leg_count: int,
    min_leg_probability: float,
    min_ticket_probability: float,
    min_combined_decimal_price: float,
    max_combined_decimal_price: float,
    min_expected_return: float,
    latent_bundle: LatentParlayBundle | None = None,
    rank_by_latent_shadow: bool = False,
) -> dict[str, Any] | None:
    executable: list[dict[str, Any]] = []
    for book_plays in by_book.values():
        tickets = score_candidate_parlays(
            book_plays,
            sport="mlb",
            probability_field="estimated_graded_hit_rate",
            eligibility_field="parlay_precision_eligible",
            min_leg_probability=min_leg_probability,
            min_pair_probability=min_ticket_probability,
            min_legs_per_parlay=leg_count,
            max_legs_per_parlay=leg_count,
            forbid_same_market_bucket_parlay=False,
            min_expected_return_per_unit=min_expected_return,
        )
        for source in tickets:
            combined_price = float(source.get("combined_decimal_price") or 0.0)
            if not min_combined_decimal_price <= combined_price <= max_combined_decimal_price:
                continue
            if float(source.get("expected_return_per_unit") or -999.0) < min_expected_return:
                continue
            if bool(source.get("same_game")) or bool(source.get("same_player")):
                continue
            ticket = dict(source)
            ticket["legs"] = [book_plays[int(index)] for index in ticket["leg_indices"]]
            latent_set = _latent_set_profile(
                ticket["legs"],
                float(ticket["projected_probability"]),
                latent_bundle,
            )
            ticket["latent_set"] = latent_set
            ticket["set_consistency_score"] = latent_set["set_consistency_score"]
            ticket["ticket_tier"] = TICKET_TIERS[leg_count]
            ticket["ticket_id"] = f"{TICKET_TIERS[leg_count]}_{leg_count}_leg"
            ticket["probability_floor"] = min_ticket_probability
            ticket["maximum_decimal_price"] = max_combined_decimal_price
            passes_reliability, reliability_profile = _ticket_reliability_gate(ticket, leg_count)
            ticket["reliability_profile"] = reliability_profile
            if not passes_reliability:
                continue
            executable.append(ticket)

    if rank_by_latent_shadow:
        executable = [
            row
            for row in executable
            if bool((row.get("latent_set") or {}).get("shadow_in_support", False))
        ]
        def shadow_rank(row: dict[str, Any]) -> tuple[float, float, float]:
            profile = row.get("latent_set") or {}
            joint = float(profile.get("shadow_joint_probability") or 0.0)
            independent = float(profile.get("shadow_independent_leg_product") or 0.0)
            hybrid = float(profile.get("shadow_hybrid_leg_product") or 0.0)
            ensemble_std = float(profile.get("shadow_ensemble_std") or 1.0)
            if leg_count == 2:
                score = joint
                method = "bounded_attention_joint_probability"
            else:
                score = hybrid
                method = "market_residual_uncertainty_penalized_product"
            profile["shadow_ranking_method"] = method
            profile["shadow_ranking_score"] = score
            return score, -ensemble_std, float(row.get("set_consistency_score") or 0.0)

        executable.sort(key=shadow_rank, reverse=True)
    else:
        executable.sort(
            key=lambda row: (
                float(row.get("set_consistency_score") or 0.0),
                float((row.get("latent_set") or {}).get("minimum_leg_probability") or 0.0),
                float(row.get("expected_return_per_unit") or -999.0),
                float(row.get("avg_leg_quality") or 0.0),
            ),
            reverse=True,
        )
    return executable[0] if executable else None


def select_ticket_ladder(
    candidates: list[selector.Candidate],
    *,
    min_legs: int,
    max_legs: int,
    min_leg_probability: float,
    base_min_ticket_probability: float,
    min_combined_decimal_price: float,
    min_expected_return: float,
    plays_override: list[dict[str, Any]] | None = None,
    latent_bundle: LatentParlayBundle | None = None,
    rank_by_latent_shadow: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_book = (
        _limited_plays_by_book_from_plays(plays_override)
        if plays_override is not None
        else _limited_plays_by_book(candidates)
    )
    limited = [play for book_plays in by_book.values() for play in book_plays]
    ladder: list[dict[str, Any]] = []
    for leg_count in range(min_legs, max_legs + 1):
        probability_floor = min(
            float(base_min_ticket_probability),
            float(TICKET_PROBABILITY_FLOORS[leg_count]),
        )
        ticket = _best_ticket_for_leg_count(
            by_book,
            leg_count=leg_count,
            min_leg_probability=min_leg_probability,
            min_ticket_probability=probability_floor,
            min_combined_decimal_price=min_combined_decimal_price,
            max_combined_decimal_price=TICKET_MAX_DECIMAL_PRICES[leg_count],
            min_expected_return=min_expected_return,
            latent_bundle=latent_bundle,
            rank_by_latent_shadow=rank_by_latent_shadow,
        )
        if ticket is not None:
            ladder.append(ticket)
    return ladder, limited


def _historical_over_events(history_dir: Path, season: int, before_date: date, min_leg_probability: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(history_dir.glob(f"*/{season}_processed_processed.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Date" not in frame:
            continue
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.loc[frame["Date"].dt.date < before_date].sort_values("Date")
        for history_rows, (_, row) in enumerate(frame.iterrows(), start=1):
            if history_rows < 35:
                continue
            for target in ALLOWED_TARGETS:
                actual = selector.to_float(row.get(target), default=float("nan"))
                line = selector.to_float(row.get(f"Market_{target}"), default=float("nan"))
                gap = selector.to_float(row.get(f"{target}_market_gap"), default=float("nan"))
                source = str(row.get(f"Market_Source_{target}", "")).strip().lower()
                books = selector.to_int(row.get(f"Market_{target}_books"), 0)
                over_price = selector.to_float(row.get(f"Market_{target}_over_price"), default=float("nan"))
                if not all(math.isfinite(value) for value in (actual, line, gap)) or gap <= 0.0:
                    continue
                if source != "real" or books < 5 or not math.isfinite(over_price) or abs(over_price) < 100.0:
                    continue
                if abs(actual - line) <= 1e-9:
                    continue
                if not selector.is_standard_bettable_line(target, line):
                    continue
                probability = selector.estimate_count_hit_probabilities(max(0.0, line + gap), line, "OVER")[2]
                if probability < min_leg_probability:
                    continue
                records.append(
                    {
                        "date": row["Date"].date().isoformat(),
                        "player": str(row.get("Player", path.parent.name)),
                        "game_id": str(row.get("Game_ID", "")).removesuffix(".0"),
                        "target": target,
                        "probability": probability,
                        "win": int(actual > line),
                        "line": line,
                        "actual": actual,
                        "plus_one_win": int(actual > line + 1.0),
                        "plus_two_win": int(actual > line + 2.0),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.drop_duplicates(["date", "player", "game_id", "target"], keep="last")


def _grade_leg_count(events: pd.DataFrame, leg_count: int) -> dict[str, Any]:
    tickets: list[dict[str, Any]] = []
    for slate_date, frame in events.groupby("date", sort=True):
        frame = frame.sort_values("probability", ascending=False).drop_duplicates("player", keep="first")
        selected: list[pd.Series] = []
        games: set[str] = set()
        for _, row in frame.iterrows():
            game_id = str(row["game_id"])
            if not game_id or game_id in games:
                continue
            selected.append(row)
            games.add(game_id)
            if len(selected) == leg_count:
                break
        if len(selected) != leg_count:
            continue
        projected = math.prod(float(row["probability"]) for row in selected)
        tickets.append(
            {
                "date": slate_date,
                "projected_probability": projected,
                "hit": int(all(int(row["win"]) == 1 for row in selected)),
            }
        )
    if not tickets:
        return {"leg_count": leg_count, "tickets": 0, "dates": 0, "hits": 0, "hit_rate": None}
    frame = pd.DataFrame(tickets)
    holdout_size = min(20, max(1, len(frame) // 5))
    development = frame.iloc[:-holdout_size]
    holdout = frame.iloc[-holdout_size:]

    def metrics(part: pd.DataFrame) -> dict[str, Any]:
        if part.empty:
            return {"tickets": 0, "hits": 0, "hit_rate": None}
        wins = int(part["hit"].sum())
        low, high = _wilson_interval(wins, len(part))
        return {
            "tickets": int(len(part)),
            "dates": int(part["date"].nunique()),
            "hits": wins,
            "hit_rate": float(part["hit"].mean()),
            "win_rate_wilson_95_low": low,
            "win_rate_wilson_95_high": high,
            "mean_projected_probability": float(part["projected_probability"].mean()),
            "brier_score": float(((part["hit"] - part["projected_probability"]) ** 2).mean()),
        }

    return {
        "leg_count": leg_count,
        "all_dates": metrics(frame),
        "development": metrics(development),
        "fixed_recent_holdout": metrics(holdout),
    }


def build_validation(history_dir: Path, season: int, before_date: date, min_leg_probability: float) -> dict[str, Any]:
    events = _historical_over_events(history_dir, season, before_date, min_leg_probability)
    base_winners = events.loc[events["win"].eq(1)] if not events.empty else events
    by_target = {}
    if not events.empty:
        for target, frame in events.groupby("target", sort=True):
            target_winners = frame.loc[frame["win"].eq(1)]
            by_target[str(target)] = {
                "rows": int(len(frame)),
                "base_hit_rate": float(frame["win"].mean()),
                "one_unit_higher_hit_rate": float(frame["plus_one_win"].mean()),
                "winner_margin_retention_one_unit": (
                    float(target_winners["plus_one_win"].mean()) if not target_winners.empty else None
                ),
            }
    return {
        "method": "stored_pre_event_projection_real_price_confirmed_grading",
        "price_validation": "real market, at least five books, selected OVER price confirmed; no parlay ROI claim",
        "history_before_date": before_date.isoformat(),
        "event_rows": int(len(events)),
        "dates": int(events["date"].nunique()) if not events.empty else 0,
        "by_leg_count": [_grade_leg_count(events, leg_count) for leg_count in range(MIN_LEGS, MAX_LEGS + 1)],
        "alternate_line_margin_audit": {
            "method": "synthetic_event_grading_without_alternate_price_history",
            "claim_scope": "hit-rate diagnostic only; does not establish alternate-line ROI",
            "rows": int(len(events)),
            "base_hit_rate": float(events["win"].mean()) if not events.empty else None,
            "one_unit_higher_hit_rate": float(events["plus_one_win"].mean()) if not events.empty else None,
            "two_units_higher_hit_rate": float(events["plus_two_win"].mean()) if not events.empty else None,
            "winner_margin_retention_one_unit": (
                float(base_winners["plus_one_win"].mean()) if not base_winners.empty else None
            ),
            "by_target": by_target,
        },
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    pool_csv = args.pool_csv.resolve()
    run_date = selector.infer_pool_run_date(pool_csv)
    if run_date is None:
        raise ValueError(f"Could not infer run date from {pool_csv}")
    season = selector.infer_history_season(pool_csv, None)
    calibration = _load_json(selector.default_history_cache_path(season))
    bet_profiles = _load_json(selector.default_bet_profile_cache_path(season))
    live_calibration = _load_json(selector.default_live_confidence_cache_path(season))
    survival_model = _load_json(selector.default_pick_survival_cache_path(season))
    candidates = selector.load_candidates(
        pool_csv,
        calibration=calibration,
        bet_profile_priors=bet_profiles,
        live_confidence_calibration=live_calibration,
        pick_survival_model=survival_model,
        min_history_bucket_rows=50,
        max_history_prior_weight=0.35,
        history_prior_strength=400.0,
        min_bet_profile_rows=12,
        max_bet_profile_prior_weight=0.25,
        bet_profile_prior_strength=80.0,
        min_market_availability_rows=12,
    )
    anchors, rejected = filter_anchor_candidates(candidates, min_leg_probability=args.min_leg_probability)
    official_contexts = fetch_official_game_contexts(anchors)
    open_anchors: list[selector.Candidate] = []
    for candidate in anchors:
        context = official_contexts.get(candidate.game_id, {})
        if context.get("state") in {"live", "final"}:
            rejected["official_game_closed"] += 1
            continue
        team_lineup = (context.get("lineups") or {}).get(candidate.team.upper())
        if team_lineup and _normalize_name(candidate.player) not in team_lineup:
            rejected["not_in_posted_lineup"] += 1
            continue
        open_anchors.append(candidate)
    anchors = open_anchors
    hit_survival_bundle = fit_hit_survival_model(args.history_dir.resolve(), before_date=run_date)
    latent_parlay_bundle = LatentParlayBundle.load(LATENT_PARLAY_ARTIFACT_PATH)
    anchors, hit_survival_rejected = apply_hit_survival_gate(
        anchors,
        bundle=hit_survival_bundle,
        official_contexts=official_contexts,
        latent_bundle=latent_parlay_bundle,
    )
    rejected.update(hit_survival_rejected)
    min_legs = max(MIN_LEGS, int(args.min_legs))
    max_legs = min(MAX_LEGS, max(int(args.min_legs), int(args.max_legs)))
    fanduel_plays = build_fanduel_main_line_plays(anchors, args.provider_observations.resolve())
    ticket_ladder, considered = select_ticket_ladder(
        anchors,
        min_legs=min_legs,
        max_legs=max_legs,
        min_leg_probability=float(args.min_leg_probability),
        base_min_ticket_probability=float(args.min_ticket_probability),
        min_combined_decimal_price=float(args.min_combined_decimal_price),
        min_expected_return=float(args.min_expected_return),
        plays_override=fanduel_plays if len(fanduel_plays) >= MIN_LEGS else None,
        latent_bundle=latent_parlay_bundle,
    )
    latent_shadow_ladder: list[dict[str, Any]] = []
    if latent_parlay_bundle is not None:
        latent_shadow_ladder, _ = select_ticket_ladder(
            anchors,
            min_legs=min_legs,
            max_legs=max_legs,
            min_leg_probability=float(args.min_leg_probability),
            base_min_ticket_probability=float(args.min_ticket_probability),
            min_combined_decimal_price=float(args.min_combined_decimal_price),
            min_expected_return=float(args.min_expected_return),
            plays_override=fanduel_plays if len(fanduel_plays) >= MIN_LEGS else None,
            latent_bundle=latent_parlay_bundle,
            rank_by_latent_shadow=True,
        )
        for shadow_ticket in latent_shadow_ladder:
            shadow_ticket["authorization"] = "shadow_only"
    # Real, disclosed fix: `ticket` (selected_ticket -- the featured
    # single-pick surface every frontend consumer treats as "the" daily
    # parlay) is decided from the real consistency-tier ladder ONLY,
    # before profit_boost is appended to it. profit_boost's own real
    # thresholds (min_leg_probability=0.18) are deliberately far below a
    # real single-pick bar -- it exists as a permanently-shadow,
    # alternate-line bonus product (validate_daily_publication.py
    # enforces it can never be candidate_authorized), never a fallback
    # for "nothing real cleared the consistency bar today". Appending it
    # to ticket_ladder AFTER `ticket` is chosen keeps it available via
    # its own real ticket_ladder/profit_boost_ticket fields for a
    # consumer that explicitly wants it, without silently promoting a
    # sub-50%-probability longshot into the primary single-pick slot on
    # a day the real consistency ladder is empty.
    ticket = ticket_ladder[0] if ticket_ladder else None
    alternate_plays = build_alternate_line_plays(anchors, args.provider_observations.resolve())
    profit_boost_ticket = select_profit_boost_ticket(alternate_plays)
    if profit_boost_ticket is not None:
        profit_boost_ticket["ticket_id"] = "profit_boost_2_leg"
        ticket_ladder.append(profit_boost_ticket)
    for ladder_ticket in ticket_ladder:
        attach_fanduel_betslip(ladder_ticket)
    validation = build_validation(args.history_dir.resolve(), season, run_date, float(args.min_leg_probability))
    return {
        "schema_version": 4,
        "policy_version": POLICY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date.isoformat(),
        "status": "ready" if ticket is not None else "withheld",
        "objective": "exact_set_search_with_role_survival_consensus_and_shadow_positive_ev_alternates",
        "direction_policy": "OVER_ONLY",
        "betslip_policy": {
            "sportsbook_key": FANDUEL_SPORTSBOOK_KEY,
            "sportsbook": "FanDuel",
            "status": "ready" if any(
                (item.get("betslip") or {}).get("status") == "ready" for item in ticket_ladder
            ) else "unavailable",
            "construction": "direct_fanduel_public_market_ids_only",
            "linked_ticket_count": sum(
                1 for item in ticket_ladder if (item.get("betslip") or {}).get("status") == "ready"
            ),
        },
        "allowed_leg_counts": list(range(min_legs, max_legs + 1)),
        "gates": {
            "min_leg_probability": float(args.min_leg_probability),
            "min_ticket_probability": float(args.min_ticket_probability),
            "ticket_probability_floors": {
                str(leg_count): min(float(args.min_ticket_probability), TICKET_PROBABILITY_FLOORS[leg_count])
                for leg_count in range(min_legs, max_legs + 1)
            },
            "ticket_max_decimal_prices": {
                str(leg_count): TICKET_MAX_DECIMAL_PRICES[leg_count]
                for leg_count in range(min_legs, max_legs + 1)
            },
            "min_combined_decimal_price": float(args.min_combined_decimal_price),
            "min_expected_return_per_unit": float(args.min_expected_return),
            "min_market_books": 1,
            "min_common_market_books": 1,
            "single_book_scope": "exact_fanduel_quote_with_provider_market_and_selection_ids",
            "same_sportsbook_required": True,
            "linked_sportsbook": "fanduel",
            "direct_fanduel_public_selection_link_required_for_betslip": True,
            "distinct_games_required": True,
            "hit_survival": {
                "model_version": HIT_SURVIVAL_MODEL_VERSION,
                "status": "development_shadow",
                "evidence_label": HIT_SURVIVAL_EVIDENCE_LABEL,
                "minimum_probability": HIT_SURVIVAL_MIN_PROBABILITY,
                "minimum_consensus_probability": HIT_SURVIVAL_MIN_CONSENSUS,
                "maximum_probability_disagreement": HIT_SURVIVAL_MAX_DISAGREEMENT,
                "probability_components": [
                    "role_survival_model",
                    "no_vig_market_probability",
                    "existing_calibrated_probability",
                ],
                "roi_claim": False,
            },
            "latent_parlay": {
                "model_version": latent_parlay_bundle.model_version if latent_parlay_bundle is not None else None,
                "status": "prospective_shadow" if latent_parlay_bundle is not None else "unavailable",
                "evidence_label": latent_parlay_bundle.evidence_label if latent_parlay_bundle is not None else None,
                "ranking_effect": "none_on_authorized_ladder",
                "shadow_ranking_policy": {
                    "2": "bounded_attention_joint_probability",
                    "3": "market_residual_uncertainty_penalized_product",
                    "4": "market_residual_uncertainty_penalized_product",
                },
                "maximum_legs": 4,
                "in_support_required": True,
                "roi_claim": False,
            },
            "profit_boost": {
                "status": "shadow_only",
                "min_leg_probability": PROFIT_BOOST_MIN_LEG_PROBABILITY,
                "min_ticket_probability": PROFIT_BOOST_MIN_TICKET_PROBABILITY,
                "min_combined_decimal_price": PROFIT_BOOST_MIN_DECIMAL_PRICE,
                "max_combined_decimal_price": PROFIT_BOOST_MAX_DECIMAL_PRICE,
                "min_expected_return_per_unit": PROFIT_BOOST_MIN_EXPECTED_RETURN,
                "exact_sportsbook_alternate_price_required": True,
            },
        },
        "pool_candidate_count": int(len(candidates)),
        "eligible_anchor_count": int(len(anchors)),
        "considered_anchor_count": int(len(considered)),
        "alternate_line_candidate_count": int(len(alternate_plays)),
        "filter_rejections": dict(sorted(rejected.items())),
        "selected_ticket": ticket,
        "profit_boost_ticket": profit_boost_ticket,
        "ticket_ladder": ticket_ladder,
        "latent_shadow_ticket_ladder": latent_shadow_ladder,
        "latent_parlay_validation": (
            latent_parlay_bundle.artifact.get("validation")
            if latent_parlay_bundle is not None
            else {
                "model_version": None,
                "status": "unavailable",
                "evidence_label": "NO_LATENT_ARTIFACT",
            }
        ),
        "hit_survival_validation": (
            hit_survival_bundle.report
            if hit_survival_bundle is not None
            else {
                "model_version": HIT_SURVIVAL_MODEL_VERSION,
                "status": "unavailable",
                "evidence_label": HIT_SURVIVAL_EVIDENCE_LABEL,
            }
        ),
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    output = args.out_json or args.pool_csv.with_name(f"{args.pool_csv.stem}_daily_parlay.json")
    payload = build_payload(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ticket = payload.get("selected_ticket") or {}
    print(f"Daily MLB parlay: {output}")
    print(f"Status: {payload['status']}; anchors={payload['eligible_anchor_count']}")
    if ticket:
        names = ", ".join(str(value) for value in ticket.get("leg_names", []))
        print(
            f"Selected {ticket.get('leg_count')} legs at {ticket.get('sportsbook')}: "
            f"p={float(ticket.get('projected_probability') or 0.0):.3f}, "
            f"EV={float(ticket.get('expected_return_per_unit') or 0.0):+.3f} ({names})"
        )
    if len(payload.get("ticket_ladder", [])) > 1:
        alternatives = ", ".join(
            f"{item['leg_count']}-leg {float(item['projected_probability']):.3f}"
            for item in payload["ticket_ladder"][1:]
        )
        print(f"Longer alternatives: {alternatives}")


if __name__ == "__main__":
    main()
