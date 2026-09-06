from __future__ import annotations

import json
import math
import re
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import (
    ADVANCED_SCHEMA_VERSION,
    BatterProcessProfile,
    DirectMatchupProcess,
    PitcherProcessProfile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADVANCED_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "advanced"
SOURCE_STATCAST = "baseball_savant_statcast_via_pybaseball_2_2_7"
SOURCE_FANGRAPHS = "fangraphs_via_pybaseball_2_2_7"
PROFILE_LOOKBACK_DAYS = 120
DIRECT_BVP_SHRINKAGE_PA = 24.0
MIN_FRESH_PROFILE_PA = 20
MAX_PROFILE_AGE_DAYS = 3

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "missed_bunt",
    "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _quantile(frame: pd.DataFrame, column: str, q: float) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else None


def _rate(mask: pd.Series, denominator: int, default: float) -> float:
    return float(mask.sum()) / denominator if denominator > 0 else float(default)


def _normalize_name(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower())
    return " ".join(text.split())


def _pa_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "events" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["events"].notna()].copy()


def _bbe_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    if "launch_speed" not in frame.columns:
        return frame.iloc[0:0].copy()
    launch_speed = pd.to_numeric(frame["launch_speed"], errors="coerce")
    return frame.loc[launch_speed.notna()].copy()


def _event_rate(pa: pd.DataFrame, names: set[str], default: float) -> float:
    if pa.empty:
        return default
    events = pa["events"].astype(str).str.lower()
    return _rate(events.isin(names), len(pa), default)


def _swing_whiff_rates(frame: pd.DataFrame) -> tuple[float, float | None, float | None]:
    if frame.empty or "description" not in frame.columns:
        return 0.225, None, None
    descriptions = frame["description"].astype(str).str.lower()
    swing = descriptions.isin(SWING_DESCRIPTIONS)
    whiff = descriptions.isin(WHIFF_DESCRIPTIONS)
    swings = int(swing.sum())
    whiffs = int(whiff.sum())
    whiff_rate = whiffs / swings if swings else 0.225
    contact_rate = 1.0 - whiff_rate

    chase_rate = None
    if "zone" in frame.columns:
        zone = pd.to_numeric(frame["zone"], errors="coerce")
        out_zone = zone.gt(9)
        out_zone_pitches = int(out_zone.sum())
        if out_zone_pitches:
            chase_rate = float((swing & out_zone).sum()) / out_zone_pitches
    return _clip(whiff_rate, 0.0, 1.0), _clip(contact_rate, 0.0, 1.0), chase_rate


def _csw_rate(frame: pd.DataFrame) -> float | None:
    if frame.empty or "description" not in frame.columns:
        return None
    descriptions = frame["description"].astype(str).str.lower()
    if len(descriptions) == 0:
        return None
    csw = descriptions.isin(WHIFF_DESCRIPTIONS | CALLED_STRIKE_DESCRIPTIONS)
    return float(csw.mean())


def _contact_rates(bbe: pd.DataFrame) -> dict[str, float | None]:
    if bbe.empty:
        return {
            "xba": None,
            "xwoba": None,
            "xslg": None,
            "avg_ev": None,
            "ev90": None,
            "hard_hit_rate": None,
            "barrel_rate": None,
            "sweet_spot_rate": None,
            "gb_rate": None,
            "ld_rate": None,
            "fb_rate": None,
        }
    ev = pd.to_numeric(bbe.get("launch_speed"), errors="coerce")
    la = pd.to_numeric(bbe.get("launch_angle"), errors="coerce")
    hard = ev.ge(95.0)
    sweet = la.between(8.0, 32.0, inclusive="both")
    barrel = pd.Series(False, index=bbe.index)
    if "launch_speed_angle" in bbe.columns:
        barrel = pd.to_numeric(bbe["launch_speed_angle"], errors="coerce").eq(6)
    elif ev.notna().any() and la.notna().any():
        # Statcast's official barrel definition is more detailed than this;
        # this fallback is explicitly a conservative process proxy used only
        # when launch_speed_angle is absent.
        barrel = ev.ge(98.0) & la.between(26.0, 30.0, inclusive="both")

    bb_type = bbe.get("bb_type", pd.Series("", index=bbe.index)).astype(str).str.lower()
    denom = max(1, int(len(bbe)))
    return {
        "xba": _mean(bbe, "estimated_ba_using_speedangle"),
        "xwoba": _mean(bbe, "estimated_woba_using_speedangle"),
        "xslg": _mean(bbe, "estimated_slg_using_speedangle"),
        "avg_ev": float(ev.mean()) if ev.notna().any() else None,
        "ev90": float(ev.quantile(0.90)) if ev.notna().any() else None,
        "hard_hit_rate": float(hard.sum()) / denom,
        "barrel_rate": float(barrel.sum()) / denom,
        "sweet_spot_rate": float(sweet.sum()) / denom if la.notna().any() else None,
        "gb_rate": float(bb_type.eq("ground_ball").sum()) / denom,
        "ld_rate": float(bb_type.eq("line_drive").sum()) / denom,
        "fb_rate": float(bb_type.isin({"fly_ball", "popup"}).sum()) / denom,
    }


def _actual_woba(pa: pd.DataFrame) -> float | None:
    if pa.empty or "woba_value" not in pa.columns:
        return None
    values = pd.to_numeric(pa["woba_value"], errors="coerce")
    denom = pd.to_numeric(pa.get("woba_denom", 1), errors="coerce").fillna(0)
    valid = values.notna() & denom.gt(0)
    return float(values.loc[valid].mean()) if valid.any() else None


def _non_hr_hit_shares(pa: pd.DataFrame) -> tuple[float, float, float]:
    if pa.empty:
        return 0.70, 0.27, 0.03
    events = pa["events"].astype(str).str.lower()
    counts = [int(events.eq("single").sum()), int(events.eq("double").sum()), int(events.eq("triple").sum())]
    total = sum(counts)
    if total < 5:
        return 0.70, 0.27, 0.03
    shares = [value / total for value in counts]
    return float(shares[0]), float(shares[1]), float(shares[2])


def _rolling_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    pa = _pa_rows(frame).sort_values([column for column in ["game_date", "at_bat_number", "pitch_number"] if column in frame.columns])
    out: dict[str, dict[str, float | int | None]] = {}
    for label, count in (("last_15", 15), ("last_30", 30), ("last_60", 60)):
        recent_pa = pa.tail(count)
        if recent_pa.empty:
            continue
        event_frame = frame.loc[frame.index.isin(recent_pa.index)] if len(frame.index) else frame
        bbe = _bbe_rows(event_frame)
        contact = _contact_rates(bbe)
        whiff, _, _ = _swing_whiff_rates(event_frame)
        events = recent_pa["events"].astype(str).str.lower()
        out[label] = {
            "pa": int(len(recent_pa)),
            "k_rate": float(events.isin({"strikeout", "strikeout_double_play"}).mean()),
            "bb_rate": float(events.isin({"walk", "intent_walk"}).mean()),
            "hr_rate": float(events.eq("home_run").mean()),
            "xwoba_contact": contact["xwoba"],
            "xba_contact": contact["xba"],
            "xslg_contact": contact["xslg"],
            "hard_hit_rate": contact["hard_hit_rate"],
            "barrel_rate": contact["barrel_rate"],
            "whiff_rate": whiff,
        }
    return out


def _pitch_type_summaries(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float | int | None]]]:
    xwoba: dict[str, float] = {}
    whiffs: dict[str, float] = {}
    arsenal: dict[str, dict[str, float | int | None]] = {}
    if frame.empty or "pitch_type" not in frame.columns:
        return xwoba, whiffs, arsenal
    total = len(frame)
    for pitch_type, group in frame.groupby(frame["pitch_type"].astype(str)):
        pitch = str(pitch_type or "").strip()
        if not pitch or pitch.lower() == "nan":
            continue
        bbe = _bbe_rows(group)
        contact = _contact_rates(bbe)
        whiff, _, _ = _swing_whiff_rates(group)
        if contact["xwoba"] is not None:
            xwoba[pitch] = float(contact["xwoba"])
        whiffs[pitch] = float(whiff)
        arsenal[pitch] = {
            "pitches": int(len(group)),
            "usage": float(len(group) / max(1, total)),
            "velocity": _mean(group, "release_speed"),
            "pfx_x": _mean(group, "pfx_x"),
            "pfx_z": _mean(group, "pfx_z"),
            "whiff_rate": float(whiff),
            "xwoba_allowed_contact": contact["xwoba"],
            "hard_hit_rate_allowed": contact["hard_hit_rate"],
        }
    return xwoba, whiffs, arsenal


def build_batter_profile(frame: pd.DataFrame, *, player_id: int, player_name: str, as_of_date: str) -> BatterProcessProfile:
    pa = _pa_rows(frame)
    bbe = _bbe_rows(frame)
    events = pa.get("events", pd.Series(dtype=str)).astype(str).str.lower()
    whiff_rate, contact_rate, chase_rate = _swing_whiff_rates(frame)
    contact = _contact_rates(bbe)
    singles, doubles, triples = _non_hr_hit_shares(pa)
    pitch_xwoba, pitch_whiff, _ = _pitch_type_summaries(frame)
    k_rate = float(events.isin({"strikeout", "strikeout_double_play"}).mean()) if len(events) else 0.225
    bb_rate = float(events.isin({"walk", "intent_walk"}).mean()) if len(events) else 0.085
    hbp_rate = float(events.eq("hit_by_pitch").mean()) if len(events) else 0.012
    hr_rate = float(events.eq("home_run").mean()) if len(events) else 0.030
    stand = ""
    if "stand" in frame.columns and frame["stand"].notna().any():
        stand = str(frame["stand"].dropna().mode().iloc[0])
    support = _clip(min(len(pa) / 150.0, len(bbe) / 90.0 if len(bbe) else 0.0), 0.0, 1.0)
    return BatterProcessProfile(
        player_id=int(player_id), player_name=player_name, as_of_date=as_of_date,
        sample_pa=int(len(pa)), sample_bbe=int(len(bbe)), handedness=stand,
        k_rate=_clip(k_rate, 0.02, 0.60), bb_rate=_clip(bb_rate, 0.01, 0.30),
        hbp_rate=_clip(hbp_rate, 0.0, 0.08), hr_rate=_clip(hr_rate, 0.0, 0.20),
        contact_rate=float(contact_rate if contact_rate is not None else 1.0 - whiff_rate),
        whiff_rate=float(whiff_rate), chase_rate=chase_rate, woba=_actual_woba(pa),
        xwoba=contact["xwoba"], xba=contact["xba"], xslg=contact["xslg"],
        avg_ev=contact["avg_ev"], ev90=contact["ev90"], hard_hit_rate=contact["hard_hit_rate"],
        barrel_rate=contact["barrel_rate"], sweet_spot_rate=contact["sweet_spot_rate"],
        gb_rate=contact["gb_rate"], ld_rate=contact["ld_rate"], fb_rate=contact["fb_rate"],
        single_share_non_hr_hits=singles, double_share_non_hr_hits=doubles,
        triple_share_non_hr_hits=triples, rolling=_rolling_summary(frame),
        pitch_type_xwoba=pitch_xwoba, pitch_type_whiff_rate=pitch_whiff, support=support,
    )


def build_pitcher_profile(frame: pd.DataFrame, *, player_id: int, player_name: str, as_of_date: str) -> PitcherProcessProfile:
    pa = _pa_rows(frame)
    bbe = _bbe_rows(frame)
    events = pa.get("events", pd.Series(dtype=str)).astype(str).str.lower()
    whiff_rate, _, _ = _swing_whiff_rates(frame)
    contact = _contact_rates(bbe)
    _, _, arsenal = _pitch_type_summaries(frame)
    k_rate = float(events.isin({"strikeout", "strikeout_double_play"}).mean()) if len(events) else 0.225
    bb_rate = float(events.isin({"walk", "intent_walk"}).mean()) if len(events) else 0.085
    hbp_rate = float(events.eq("hit_by_pitch").mean()) if len(events) else 0.012
    hr_rate = float(events.eq("home_run").mean()) if len(events) else 0.030
    p_throws = ""
    if "p_throws" in frame.columns and frame["p_throws"].notna().any():
        p_throws = str(frame["p_throws"].dropna().mode().iloc[0])
    support = _clip(min(len(pa) / 180.0, len(bbe) / 100.0 if len(bbe) else 0.0), 0.0, 1.0)
    return PitcherProcessProfile(
        player_id=int(player_id), player_name=player_name, as_of_date=as_of_date,
        sample_pa=int(len(pa)), sample_bbe=int(len(bbe)), handedness=p_throws,
        k_rate=_clip(k_rate, 0.02, 0.60), bb_rate=_clip(bb_rate, 0.01, 0.30),
        hbp_rate=_clip(hbp_rate, 0.0, 0.08), hr_rate=_clip(hr_rate, 0.0, 0.20),
        k_minus_bb_rate=_clip(k_rate - bb_rate, -0.10, 0.45), whiff_rate=float(whiff_rate),
        csw_rate=_csw_rate(frame), xwoba_allowed=contact["xwoba"], xba_allowed=contact["xba"],
        xslg_allowed=contact["xslg"], avg_ev_allowed=contact["avg_ev"],
        hard_hit_rate_allowed=contact["hard_hit_rate"], barrel_rate_allowed=contact["barrel_rate"],
        sweet_spot_rate_allowed=contact["sweet_spot_rate"], gb_rate=contact["gb_rate"],
        fb_rate=contact["fb_rate"], arsenal=arsenal, rolling=_rolling_summary(frame), support=support,
    )


def build_direct_matchup(frame: pd.DataFrame, *, batter_id: int, pitcher_id: int) -> DirectMatchupProcess | None:
    if frame.empty or "pitcher" not in frame.columns:
        return None
    pitcher_ids = pd.to_numeric(frame["pitcher"], errors="coerce").fillna(0).astype(int)
    matched = frame.loc[pitcher_ids.eq(int(pitcher_id))].copy()
    pa = _pa_rows(matched)
    if pa.empty:
        return None
    bbe = _bbe_rows(matched)
    events = pa["events"].astype(str).str.lower()
    ev = pd.to_numeric(bbe.get("launch_speed"), errors="coerce") if not bbe.empty else pd.Series(dtype=float)
    hard = int(ev.ge(95.0).sum()) if not ev.empty else 0
    weak = int(ev.lt(80.0).sum()) if not ev.empty else 0
    contact = _contact_rates(bbe)
    whiff, _, _ = _swing_whiff_rates(matched)
    weight = len(pa) / (len(pa) + DIRECT_BVP_SHRINKAGE_PA)
    return DirectMatchupProcess(
        batter_id=int(batter_id), pitcher_id=int(pitcher_id), pa=int(len(pa)),
        strikeouts=int(events.isin({"strikeout", "strikeout_double_play"}).sum()),
        walks=int(events.isin({"walk", "intent_walk"}).sum()), hbp=int(events.eq("hit_by_pitch").sum()),
        home_runs=int(events.eq("home_run").sum()), non_hr_contacts=int(len(bbe) - events.eq("home_run").sum()),
        hard_contacts=hard, weak_contacts=weak, xwoba_contact=contact["xwoba"],
        xba_contact=contact["xba"], xslg_contact=contact["xslg"], avg_ev=contact["avg_ev"],
        barrel_rate=contact["barrel_rate"], whiff_rate=whiff, shrinkage_weight=float(weight),
    )


def _attach_fangraphs_pitching(profile: PitcherProcessProfile, row: dict[str, Any] | None) -> PitcherProcessProfile:
    if not row:
        return profile
    def pick(*names: str) -> float | None:
        for name in names:
            if name in row:
                value = _finite(row.get(name))
                if value is not None:
                    return value
        return None
    return replace(
        profile,
        era=pick("ERA"), fip=pick("FIP"), xfip=pick("xFIP", "XFIP"),
        siera=pick("SIERA"), xera=pick("xERA", "XERA"),
    )


def _load_pybaseball() -> dict[str, Any]:
    try:
        from pybaseball import cache, pitching_stats, statcast_batter, statcast_pitcher
    except Exception as exc:  # pragma: no cover - exercised in workflow/network integration
        raise RuntimeError(f"pybaseball unavailable: {exc}") from exc
    try:
        cache.enable()
    except Exception:
        pass
    return {
        "pitching_stats": pitching_stats,
        "statcast_batter": statcast_batter,
        "statcast_pitcher": statcast_pitcher,
    }


def _safe_statcast_fetch(fn: Any, start_date: str, end_date: str, player_id: int, *, retries: int = 2) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frame = fn(start_date, end_date, player_id=int(player_id))
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - network path
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Statcast fetch failed for {player_id}: {last_error}")


def _fangraphs_pitching_map(fn: Any, season: int) -> dict[str, dict[str, Any]]:
    try:
        frame = fn(season, season, qual=0)
    except Exception:  # pragma: no cover - network path
        return {}
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Name" not in frame.columns:
        return {}
    return {_normalize_name(row.get("Name")): row.to_dict() for _, row in frame.iterrows()}


def read_pool_candidate_identities(pool_csv: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(pool_csv)
    if frame.empty:
        return []
    target = frame.get("Target", pd.Series("", index=frame.index)).astype(str).str.upper()
    player_type = frame.get("Player_Type", pd.Series("", index=frame.index)).astype(str).str.lower()
    market_source = frame.get("Market_Source", pd.Series("", index=frame.index)).astype(str).str.lower()
    subset = frame.loc[target.isin({"H", "TB"}) & player_type.eq("hitter") & market_source.eq("real")].copy()
    if subset.empty:
        return []
    identities: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        batter_id = int(_finite(row.get("Player_MLBAM_ID"), 0) or _finite(row.get("Player_ID_MLBAM"), 0) or 0)
        if batter_id <= 0:
            raw = str(row.get("Player_ID") or "")
            if raw.isdigit():
                batter_id = int(raw)
        pitcher_id = int(_finite(row.get("Opposing_Pitcher_ID"), 0) or 0)
        identities.append({
            "game_id": str(row.get("Game_ID") or ""),
            "batter_id": batter_id,
            "batter_name": str(row.get("Player") or "").strip(),
            "pitcher_id": pitcher_id,
            "pitcher_name": str(row.get("Opposing_Pitcher") or "").strip(),
        })
    unique: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in identities:
        key = (row["game_id"], row["batter_id"], row["pitcher_id"], _normalize_name(row["batter_name"]))
        unique[key] = row
    return list(unique.values())


def refresh_advanced_profiles(
    *,
    pool_csv: Path,
    run_date: str,
    advanced_root: Path = DEFAULT_ADVANCED_ROOT,
    lookback_days: int = PROFILE_LOOKBACK_DAYS,
    max_candidates: int = 80,
) -> dict[str, Any]:
    run_day = date.fromisoformat(run_date)
    start_day = max(date(run_day.year, 3, 1), run_day - timedelta(days=max(30, int(lookback_days))))
    partition = advanced_root / run_date.replace("-", "")
    partition.mkdir(parents=True, exist_ok=True)
    batter_path = partition / "batter_profiles.json"
    pitcher_path = partition / "pitcher_profiles.json"
    matchup_path = partition / "bvp_process.json"
    manifest_path = partition / "manifest.json"

    identities = read_pool_candidate_identities(pool_csv)[: max(1, int(max_candidates))]
    tools = _load_pybaseball()
    fg_pitchers = _fangraphs_pitching_map(tools["pitching_stats"], run_day.year)

    batters: dict[str, dict[str, Any]] = {}
    pitchers: dict[str, dict[str, Any]] = {}
    matchups: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    batter_frames: dict[int, pd.DataFrame] = {}

    for identity in identities:
        batter_id = int(identity["batter_id"] or 0)
        pitcher_id = int(identity["pitcher_id"] or 0)
        if batter_id > 0 and batter_id not in batter_frames:
            try:
                batter_frames[batter_id] = _safe_statcast_fetch(
                    tools["statcast_batter"], start_day.isoformat(), (run_day - timedelta(days=1)).isoformat(), batter_id
                )
                profile = build_batter_profile(
                    batter_frames[batter_id], player_id=batter_id, player_name=identity["batter_name"], as_of_date=run_date
                )
                batters[str(batter_id)] = profile.to_dict()
            except Exception as exc:
                failures.append({"entity": "batter", "player_id": batter_id, "error": str(exc)})

        if pitcher_id > 0 and str(pitcher_id) not in pitchers:
            try:
                pitcher_frame = _safe_statcast_fetch(
                    tools["statcast_pitcher"], start_day.isoformat(), (run_day - timedelta(days=1)).isoformat(), pitcher_id
                )
                profile = build_pitcher_profile(
                    pitcher_frame, player_id=pitcher_id, player_name=identity["pitcher_name"], as_of_date=run_date
                )
                profile = _attach_fangraphs_pitching(profile, fg_pitchers.get(_normalize_name(identity["pitcher_name"])))
                pitchers[str(pitcher_id)] = profile.to_dict()
            except Exception as exc:
                failures.append({"entity": "pitcher", "player_id": pitcher_id, "error": str(exc)})

        if batter_id > 0 and pitcher_id > 0 and batter_id in batter_frames:
            direct = build_direct_matchup(batter_frames[batter_id], batter_id=batter_id, pitcher_id=pitcher_id)
            if direct is not None:
                matchups[f"{batter_id}:{pitcher_id}"] = direct.to_dict()

    fetched_at = datetime.now(timezone.utc).isoformat()
    batter_payload = {"schema_version": ADVANCED_SCHEMA_VERSION, "source": SOURCE_STATCAST, "run_date": run_date, "fetched_at_utc": fetched_at, "profiles": batters}
    pitcher_payload = {"schema_version": ADVANCED_SCHEMA_VERSION, "source": [SOURCE_STATCAST, SOURCE_FANGRAPHS], "run_date": run_date, "fetched_at_utc": fetched_at, "profiles": pitchers}
    matchup_payload = {"schema_version": ADVANCED_SCHEMA_VERSION, "source": SOURCE_STATCAST, "run_date": run_date, "fetched_at_utc": fetched_at, "matchups": matchups}
    batter_path.write_text(json.dumps(batter_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pitcher_path.write_text(json.dumps(pitcher_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matchup_path.write_text(json.dumps(matchup_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": ADVANCED_SCHEMA_VERSION,
        "run_date": run_date,
        "effective_as_of_date": (run_day - timedelta(days=1)).isoformat(),
        "fetched_at_utc": fetched_at,
        "lookback_start_date": start_day.isoformat(),
        "sources": [SOURCE_STATCAST, SOURCE_FANGRAPHS],
        "candidate_identities": len(identities),
        "batter_profiles": len(batters),
        "pitcher_profiles": len(pitchers),
        "direct_matchups": len(matchups),
        "failures": failures,
        "paths": {
            "batter_profiles": str(batter_path.relative_to(REPO_ROOT)),
            "pitcher_profiles": str(pitcher_path.relative_to(REPO_ROOT)),
            "bvp_process": str(matchup_path.relative_to(REPO_ROOT)),
        },
        "freshness_policy": {
            "max_profile_age_days": MAX_PROFILE_AGE_DAYS,
            "minimum_fresh_profile_pa": MIN_FRESH_PROFILE_PA,
            "stale_profiles_may_not_silently_authorize": True,
        },
        "defense_layer": {
            "status": "AVERAGE_CONTEXT_RESIDUAL_ONLY_UNTIL_SPECIFIC_OAA_IS_AVAILABLE",
            "default_residual": 0.0,
            "fabricated_oaa_allowed": False,
        },
        "sprint_speed": {
            "status": "NOT_APPLIED_AS_SEPARATE_RESIDUAL_WHEN_STATCAST_EXPECTED_METRICS_ARE_USED",
            "double_count_prevention": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_profile_partition(advanced_root: Path, run_date: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    partition = advanced_root / run_date.replace("-", "")
    def read(name: str) -> dict[str, Any]:
        path = partition / name
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
    return read("batter_profiles.json"), read("pitcher_profiles.json"), read("bvp_process.json"), read("manifest.json")
