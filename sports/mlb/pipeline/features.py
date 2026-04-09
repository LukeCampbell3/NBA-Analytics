from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, normalize_player_name, write_csv, write_json


def _safe_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(float(default))


def _rolling_shifted_mean(frame: pd.DataFrame, by: str, col: str, window: int = 5) -> pd.Series:
    return frame.groupby(by, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )


def _lag1(frame: pd.DataFrame, by: str, col: str) -> pd.Series:
    return frame.groupby(by, sort=False)[col].shift(1)


def _rest_days(frame: pd.DataFrame, by: str = "Player") -> pd.Series:
    return (
        frame.groupby(by, sort=False)["Date"]
        .transform(lambda s: (s.diff().dt.days - 1).clip(lower=0).fillna(2.0))
        .astype(float)
    )


def _merge_asof_by_team(
    base: pd.DataFrame,
    team_hist: pd.DataFrame,
    *,
    base_team_col: str,
    hist_team_col: str,
    hist_date_col: str = "Date",
    base_date_col: str = "Date",
    cols: list[str],
) -> pd.DataFrame:
    out = base.copy()
    team_hist = team_hist.sort_values([hist_team_col, hist_date_col]).copy()
    out = out.sort_values([base_team_col, base_date_col]).copy()

    chunks = []
    for team_id, grp in out.groupby(base_team_col, sort=False):
        hist = team_hist.loc[team_hist[hist_team_col] == team_id, [hist_date_col] + cols].sort_values(hist_date_col)
        if hist.empty:
            grp = grp.copy()
            for col in cols:
                if col not in grp.columns:
                    grp[col] = np.nan
            chunks.append(grp)
            continue
        merged = pd.merge_asof(
            grp.sort_values(base_date_col),
            hist.rename(columns={hist_date_col: "__hist_date"}),
            left_on=base_date_col,
            right_on="__hist_date",
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["__hist_date"])
        chunks.append(merged)
    out = pd.concat(chunks, ignore_index=True)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _parse_market_fetched_at(date_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series, errors="coerce").dt.strftime("%Y-%m-%dT16:00:00Z")


def _apply_market_columns(
    df: pd.DataFrame,
    *,
    targets: list[str],
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    key_cols = ["Date", "Player", "Player_Type"]
    if market_df is not None and not market_df.empty:
        md = market_df.copy()
        md["Date"] = pd.to_datetime(md["Date"], errors="coerce")
        md["Player"] = md["Player"].astype(str)
        md["Player_Type"] = md["Player_Type"].astype(str).str.lower()
        out = out.merge(md, on=key_cols, how="left", suffixes=("", "_marketfile"))

    for target in targets:
        market_col = f"Market_{target}"
        synth_col = f"Synthetic_Market_{target}"
        source_col = f"Market_Source_{target}"
        books_col = f"Market_{target}_books"
        over_col = f"Market_{target}_over_price"
        under_col = f"Market_{target}_under_price"
        std_col = f"Market_{target}_line_std"
        base_col = f"{target}_rolling_avg"

        if market_col not in out.columns:
            out[market_col] = np.nan
        if synth_col not in out.columns:
            out[synth_col] = np.nan
        market = pd.to_numeric(out[market_col], errors="coerce")
        synthetic = pd.to_numeric(out[synth_col], errors="coerce")
        baseline = pd.to_numeric(out.get(base_col), errors="coerce")
        lag1 = pd.to_numeric(out.get(f"{target}_lag1"), errors="coerce")
        # Synthetic market fallbacks must be pregame-safe and never read same-game targets.
        fallback = baseline.fillna(lag1).fillna(0.0)
        market_filled = market.fillna(fallback)
        synthetic_filled = synthetic.fillna(fallback)

        out[market_col] = market_filled
        out[synth_col] = synthetic_filled
        out[source_col] = np.where(market.notna(), "real", "synthetic")
        out[books_col] = _safe_num(out.get(books_col, pd.Series(np.nan, index=out.index)), 0.0)
        out[over_col] = _safe_num(out.get(over_col, pd.Series(np.nan, index=out.index)), -110.0)
        out[under_col] = _safe_num(out.get(under_col, pd.Series(np.nan, index=out.index)), -110.0)
        out[std_col] = _safe_num(out.get(std_col, pd.Series(np.nan, index=out.index)), 0.0).clip(lower=0.0)
        out[f"{target}_market_gap"] = out[market_col] - _safe_num(out[base_col], 0.0)

    if "Market_Fetched_At_UTC" not in out.columns:
        out["Market_Fetched_At_UTC"] = _parse_market_fetched_at(out["Date"])
    else:
        fetched = pd.to_datetime(out["Market_Fetched_At_UTC"], errors="coerce", utc=True)
        default = _parse_market_fetched_at(out["Date"])
        out["Market_Fetched_At_UTC"] = fetched.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna(default)
    return out


def _woba(df: pd.DataFrame) -> pd.Series:
    singles = (df["H"] - df["2B"] - df["3B"] - df["HR"]).clip(lower=0.0)
    denom = (df["AB"] + df["BB"] - df["IBB"] + df["SF"] + df["HBP"]).replace(0, np.nan)
    numer = (
        0.69 * (df["BB"] - df["IBB"])
        + 0.72 * df["HBP"]
        + 0.88 * singles
        + 1.247 * df["2B"]
        + 1.578 * df["3B"]
        + 2.031 * df["HR"]
    )
    return (numer / denom).replace([np.inf, -np.inf], np.nan)


def _build_hitter_features(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = hitters.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Player"]).sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)

    for col in ["H", "HR", "RBI", "TB", "PA", "AB", "BB", "SO", "SB", "R", "2B", "3B", "HBP", "IBB", "SF", "SH"]:
        df[col] = _safe_num(df[col], 0.0)

    team_pa = df.groupby(["Game_ID", "Team_ID"], sort=False)["PA"].transform("sum").replace(0.0, np.nan)
    df["Team_PA_share"] = (df["PA"] / team_pa).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["wOBA"] = _woba(df).fillna(0.0)
    avg = (df["H"] / df["AB"].replace(0, np.nan)).fillna(0.0)
    slg = (df["TB"] / df["AB"].replace(0, np.nan)).fillna(0.0)
    df["ISO"] = (slg - avg).fillna(0.0)
    df["xwOBA"] = df.groupby("Player", sort=False)["wOBA"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.320)
    df["Barrel%"] = df.groupby("Player", sort=False)["ISO"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.0) * 25.0
    df["HardHit%"] = df.groupby("Player", sort=False)["xwOBA"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.0) * 100.0
    df["Batting_Order"] = _safe_num(df.get("Batting_Order", 9.0), 9.0).clip(lower=1.0, upper=9.0)

    # Opponent pitching context from prior games only.
    p = pitchers.copy()
    if not p.empty:
        p["Date"] = pd.to_datetime(p["Date"], errors="coerce")
        p["IP"] = _safe_num(p["IP"], 0.0)
        p["ER"] = _safe_num(p["ER"], 0.0)
        p["K"] = _safe_num(p["K"], 0.0)
        p["Is_Starter"] = _safe_num(p.get("Is_Starter", 0.0), 0.0)

        team_pitch = (
            p.groupby(["Date", "Game_ID", "Team_ID"], as_index=False)
            .agg(IP=("IP", "sum"), ER=("ER", "sum"), K=("K", "sum"))
            .sort_values(["Team_ID", "Date"])
        )
        team_pitch["Opp_Pitcher_ERA_game"] = 9.0 * team_pitch["ER"] / team_pitch["IP"].replace(0, np.nan)
        team_pitch["Opp_Pitcher_K9_game"] = 9.0 * team_pitch["K"] / team_pitch["IP"].replace(0, np.nan)
        team_pitch["Opp_Pitcher_ERA_3"] = team_pitch.groupby("Team_ID", sort=False)["Opp_Pitcher_ERA_game"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        team_pitch["Opp_Pitcher_K9_3"] = team_pitch.groupby("Team_ID", sort=False)["Opp_Pitcher_K9_game"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

        bullpen_rows = []
        for (_date, game_id, team_id), grp in p.groupby(["Date", "Game_ID", "Team_ID"], sort=False):
            grp = grp.sort_values("IP", ascending=False).reset_index(drop=True)
            pen = grp.iloc[1:].copy() if len(grp) > 1 else grp.iloc[0:0].copy()
            pen_ip = float(pen["IP"].sum()) if not pen.empty else 0.0
            pen_er = float(pen["ER"].sum()) if not pen.empty else 0.0
            bullpen_rows.append({"Date": _date, "Game_ID": game_id, "Team_ID": team_id, "bullpen_ip": pen_ip, "bullpen_er": pen_er})
        bullpen = pd.DataFrame.from_records(bullpen_rows).sort_values(["Team_ID", "Date"])
        bullpen["bullpen_era_game"] = 9.0 * bullpen["bullpen_er"] / bullpen["bullpen_ip"].replace(0, np.nan)
        bullpen["Opp_Bullpen_ERA_7"] = bullpen.groupby("Team_ID", sort=False)["bullpen_era_game"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=1).mean()
        )

        team_ctx = team_pitch.merge(
            bullpen[["Date", "Game_ID", "Team_ID", "Opp_Bullpen_ERA_7"]],
            on=["Date", "Game_ID", "Team_ID"],
            how="left",
        )
        df = _merge_asof_by_team(
            df,
            team_ctx[["Date", "Team_ID", "Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3", "Opp_Bullpen_ERA_7"]],
            base_team_col="Opponent_ID",
            hist_team_col="Team_ID",
            cols=["Opp_Pitcher_ERA_3", "Opp_Pitcher_K9_3", "Opp_Bullpen_ERA_7"],
        )
    else:
        df["Opp_Pitcher_ERA_3"] = np.nan
        df["Opp_Pitcher_K9_3"] = np.nan
        df["Opp_Bullpen_ERA_7"] = np.nan

    park_factors = pd.Series(1.0, index=df.index)
    park_path = Path(__file__).resolve().parent.parent / "data" / "static" / "park_factors.csv"
    if park_path.exists():
        pf = pd.read_csv(park_path)
        if {"Team", "Park_Factor"}.issubset(pf.columns):
            park_map = pf.set_index("Team")["Park_Factor"].to_dict()
            park_factors = df["Team"].map(park_map).astype(float)
    df["Park_Factor"] = pd.to_numeric(park_factors, errors="coerce").fillna(1.0)
    df["Wind_Out_MPH"] = _safe_num(df.get("Wind_Out_MPH", 0.0), 0.0)
    df["Temp_F"] = _safe_num(df.get("Temp_F", 0.0), 0.0)

    df = df.sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)
    df["Did_Not_Play"] = 0
    df["Rest_Days"] = _rest_days(df, by="Player")
    df["Game_Index"] = df.groupby("Player", sort=False).cumcount().astype(int)
    df["Month_sin"] = np.sin(2 * np.pi * (df["Date"].dt.month.fillna(1) / 12.0))
    df["Month_cos"] = np.cos(2 * np.pi * (df["Date"].dt.month.fillna(1) / 12.0))
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * (df["Date"].dt.dayofweek.fillna(0) / 7.0))
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * (df["Date"].dt.dayofweek.fillna(0) / 7.0))

    for target in ["H", "HR", "RBI"]:
        roll_col = f"{target}_rolling_avg"
        lag_col = f"{target}_lag1"
        df[roll_col] = _rolling_shifted_mean(df, by="Player", col=target, window=5)
        df[lag_col] = _lag1(df, by="Player", col=target)
        df[roll_col] = df[roll_col].fillna(df.groupby("Player", sort=False)[target].transform("mean")).fillna(0.0)
        df[lag_col] = df[lag_col].fillna(df[roll_col]).fillna(0.0)

    df = _apply_market_columns(df, targets=["H", "HR", "RBI"], market_df=market_df)

    required_fill = [
        "Opp_Pitcher_ERA_3",
        "Opp_Pitcher_K9_3",
        "Opp_Bullpen_ERA_7",
    ]
    for col in required_fill:
        df[col] = _safe_num(df[col], 0.0)

    # Keep a stable column order with required contract columns first.
    lead = [
        "Date",
        "Player",
        "Player_Type",
        "Team",
        "Opponent",
        "Season",
        "Game_ID",
        "Game_Index",
        "Team_ID",
        "Opponent_ID",
        "Is_Home",
        "H",
        "HR",
        "RBI",
        "TB",
        "R",
        "PA",
        "AB",
        "BB",
        "SO",
        "SB",
        "Batting_Order",
        "Team_PA_share",
        "wOBA",
        "xwOBA",
        "ISO",
        "Barrel%",
        "HardHit%",
        "Opp_Pitcher_ERA_3",
        "Opp_Pitcher_K9_3",
        "Opp_Bullpen_ERA_7",
        "Park_Factor",
        "Wind_Out_MPH",
        "Temp_F",
        "Did_Not_Play",
        "Rest_Days",
        "Month_sin",
        "Month_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
    ]
    market_cols = []
    for t in ["H", "HR", "RBI"]:
        market_cols.extend(
            [
                f"Market_{t}",
                f"Synthetic_Market_{t}",
                f"Market_Source_{t}",
                f"Market_{t}_books",
                f"Market_{t}_over_price",
                f"Market_{t}_under_price",
                f"Market_{t}_line_std",
            ]
        )
    tail = ["Market_Fetched_At_UTC", "H_market_gap", "HR_market_gap", "RBI_market_gap", "H_rolling_avg", "HR_rolling_avg", "RBI_rolling_avg", "H_lag1", "HR_lag1", "RBI_lag1"]
    ordered = [c for c in lead + market_cols + tail if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining].copy()
    return df


def _build_pitcher_features(
    pitchers: pd.DataFrame,
    hitters: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pitchers.copy()
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Player"]).sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)

    for col in ["K", "ER", "ERA", "IP", "BF", "Pitches", "BB_allowed", "H_allowed", "HR_allowed", "HBP_allowed"]:
        df[col] = _safe_num(df[col], 0.0)

    # FIP/xFIP baseline estimates from box score-level stats.
    ip = df["IP"].replace(0, np.nan)
    df["__FIP_game"] = (
        ((13.0 * df["HR_allowed"]) + (3.0 * (df["BB_allowed"] + df["HBP_allowed"])) - (2.0 * df["K"])) / ip + 3.2
    ).replace([np.inf, -np.inf], np.nan)
    df["FIP"] = df.groupby("Player", sort=False)["__FIP_game"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["xFIP"] = df["FIP"]
    df["FIP"] = pd.to_numeric(df["FIP"], errors="coerce").fillna(4.20)
    df["xFIP"] = pd.to_numeric(df["xFIP"], errors="coerce").fillna(4.20)
    df = df.drop(columns=["__FIP_game"])
    df["CSW%"] = np.nan
    df["Whiff%"] = np.nan

    h = hitters.copy()
    if not h.empty:
        h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
        for col in ["H", "2B", "3B", "HR", "AB", "BB", "IBB", "SF", "HBP", "SO", "PA"]:
            h[col] = _safe_num(h.get(col), 0.0)
        h["wOBA_game_calc"] = _woba(h).fillna(0.0)
        lineup = (
            h.groupby(["Date", "Game_ID", "Team_ID"], as_index=False)
            .agg(PA=("PA", "sum"), SO=("SO", "sum"), team_wOBA=("wOBA_game_calc", "mean"))
            .sort_values(["Team_ID", "Date"])
        )
        lineup["k_rate_game"] = lineup["SO"] / lineup["PA"].replace(0, np.nan)
        lineup["Opp_Lineup_wOBA_3"] = lineup.groupby("Team_ID", sort=False)["team_wOBA"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        lineup["Opp_Lineup_K_rate_3"] = lineup.groupby("Team_ID", sort=False)["k_rate_game"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        df = _merge_asof_by_team(
            df,
            lineup[["Date", "Team_ID", "Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3"]],
            base_team_col="Opponent_ID",
            hist_team_col="Team_ID",
            cols=["Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3"],
        )
    else:
        df["Opp_Lineup_wOBA_3"] = np.nan
        df["Opp_Lineup_K_rate_3"] = np.nan

    park_factors = pd.Series(1.0, index=df.index)
    park_path = Path(__file__).resolve().parent.parent / "data" / "static" / "park_factors.csv"
    if park_path.exists():
        pf = pd.read_csv(park_path)
        if {"Team", "Park_Factor"}.issubset(pf.columns):
            park_map = pf.set_index("Team")["Park_Factor"].to_dict()
            park_factors = df["Team"].map(park_map).astype(float)
    df["Park_Factor"] = pd.to_numeric(park_factors, errors="coerce").fillna(1.0)
    df["Wind_Out_MPH"] = _safe_num(df.get("Wind_Out_MPH", 0.0), 0.0)
    df["Temp_F"] = _safe_num(df.get("Temp_F", 0.0), 0.0)

    df = df.sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)
    df["Did_Not_Play"] = 0
    df["Rest_Days"] = _rest_days(df, by="Player")
    df["Game_Index"] = df.groupby("Player", sort=False).cumcount().astype(int)
    df["Month_sin"] = np.sin(2 * np.pi * (df["Date"].dt.month.fillna(1) / 12.0))
    df["Month_cos"] = np.cos(2 * np.pi * (df["Date"].dt.month.fillna(1) / 12.0))
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * (df["Date"].dt.dayofweek.fillna(0) / 7.0))
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * (df["Date"].dt.dayofweek.fillna(0) / 7.0))

    for target in ["K", "ER", "ERA"]:
        roll_col = f"{target}_rolling_avg"
        lag_col = f"{target}_lag1"
        df[roll_col] = _rolling_shifted_mean(df, by="Player", col=target, window=5)
        df[lag_col] = _lag1(df, by="Player", col=target)
        df[roll_col] = df[roll_col].fillna(df.groupby("Player", sort=False)[target].transform("mean")).fillna(0.0)
        df[lag_col] = df[lag_col].fillna(df[roll_col]).fillna(0.0)

    df = _apply_market_columns(df, targets=["K", "ER", "ERA"], market_df=market_df)
    for col in ["Opp_Lineup_wOBA_3", "Opp_Lineup_K_rate_3", "FIP", "xFIP", "CSW%", "Whiff%"]:
        df[col] = _safe_num(df[col], 0.0)

    lead = [
        "Date",
        "Player",
        "Player_Type",
        "Team",
        "Opponent",
        "Season",
        "Game_ID",
        "Game_Index",
        "Team_ID",
        "Opponent_ID",
        "Is_Home",
        "K",
        "ER",
        "ERA",
        "IP",
        "BF",
        "Pitches",
        "BB_allowed",
        "H_allowed",
        "HR_allowed",
        "FIP",
        "xFIP",
        "CSW%",
        "Whiff%",
        "Opp_Lineup_wOBA_3",
        "Opp_Lineup_K_rate_3",
        "Park_Factor",
        "Wind_Out_MPH",
        "Temp_F",
        "Did_Not_Play",
        "Rest_Days",
        "Month_sin",
        "Month_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
    ]
    market_cols = []
    for t in ["K", "ER", "ERA"]:
        market_cols.extend(
            [
                f"Market_{t}",
                f"Synthetic_Market_{t}",
                f"Market_Source_{t}",
                f"Market_{t}_books",
                f"Market_{t}_over_price",
                f"Market_{t}_under_price",
                f"Market_{t}_line_std",
            ]
        )
    tail = ["Market_Fetched_At_UTC", "K_market_gap", "ER_market_gap", "ERA_market_gap", "K_rolling_avg", "ER_rolling_avg", "ERA_rolling_avg", "K_lag1", "ER_lag1", "ERA_lag1"]
    ordered = [c for c in lead + market_cols + tail if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining].copy()
    return df


@dataclass
class FeatureBuildConfig:
    raw_dir: Path
    processed_dir: Path
    season: int
    market_file: Path | None = None


def _load_market_file(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Player" not in df.columns or "Player_Type" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def build_processed_mlb_features(config: FeatureBuildConfig) -> dict:
    raw_dir = config.raw_dir.resolve()
    processed_dir = ensure_dir(config.processed_dir.resolve())
    hitter_raw_path = raw_dir / "hitter_game_logs.csv"
    pitcher_raw_path = raw_dir / "pitcher_game_logs.csv"

    if not hitter_raw_path.exists() and not pitcher_raw_path.exists():
        raise FileNotFoundError(f"No raw files found in {raw_dir}. Expected hitter_game_logs.csv and/or pitcher_game_logs.csv")

    hitters_raw = pd.read_csv(hitter_raw_path) if hitter_raw_path.exists() else pd.DataFrame()
    pitchers_raw = pd.read_csv(pitcher_raw_path) if pitcher_raw_path.exists() else pd.DataFrame()
    market_df = _load_market_file(config.market_file)

    hitters = _build_hitter_features(hitters_raw, pitchers_raw, market_df=market_df) if not hitters_raw.empty else pd.DataFrame()
    pitchers = _build_pitcher_features(pitchers_raw, hitters_raw, market_df=market_df) if not pitchers_raw.empty else pd.DataFrame()

    all_rows = []
    if not hitters.empty:
        all_rows.append(hitters)
    if not pitchers.empty:
        all_rows.append(pitchers)
    if not all_rows:
        raise RuntimeError("Feature builder produced no rows.")
    combined = pd.concat(all_rows, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values(["Player", "Date", "Game_ID"]).reset_index(drop=True)

    # Per-player-type processed outputs.
    players_written = {}
    for (player_type, player), grp in combined.groupby(["Player_Type", "Player"], sort=False):
        safe_name = normalize_player_name(player)
        if not safe_name:
            continue
        player_dir = ensure_dir(processed_dir / f"{str(player_type).lower()}_{safe_name}")
        out_path = player_dir / f"{int(config.season)}_processed_processed.csv"
        grp_out = grp.sort_values(["Date", "Game_ID"]).reset_index(drop=True).copy()
        grp_out["Game_Index"] = np.arange(len(grp_out), dtype=int)
        grp_out["Rest_Days"] = (grp_out["Date"].diff().dt.days - 1).clip(lower=0).fillna(2.0)
        grp_out["Date"] = pd.to_datetime(grp_out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        write_csv(grp_out, out_path)
        players_written[f"{player_type}:{player}"] = {
            "rows": int(len(grp_out)),
            "path": str(out_path),
            "max_date": str(pd.to_datetime(grp_out["Date"], errors="coerce").max().date()) if not grp_out.empty else None,
        }

    # Aggregate role files for faster trainer loads.
    if not hitters.empty:
        hitters_out = hitters.copy()
        hitters_out["Date"] = pd.to_datetime(hitters_out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        write_csv(hitters_out, processed_dir / f"{int(config.season)}_hitters_processed.csv")
    if not pitchers.empty:
        pitchers_out = pitchers.copy()
        pitchers_out["Date"] = pd.to_datetime(pitchers_out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        write_csv(pitchers_out, processed_dir / f"{int(config.season)}_pitchers_processed.csv")

    manifest = {
        "season": int(config.season),
        "sport": "mlb",
        "model_contract": "mlb_native_player_v1",
        "source_refresh": True,
        "processed_summary": {
            "players": int(len(players_written)),
            "rows": int(len(combined)),
            "hitter_rows": int(len(hitters)),
            "pitcher_rows": int(len(pitchers)),
            "min_date": str(combined["Date"].min().date()) if not combined.empty else None,
            "max_date": str(combined["Date"].max().date()) if not combined.empty else None,
        },
        "market_props_merge": {
            "available": bool(market_df is not None and not market_df.empty),
            "path": str(config.market_file) if config.market_file is not None else None,
            "rows": int(len(market_df)) if market_df is not None else 0,
        },
        "players_written": int(len(players_written)),
        "written": players_written,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(processed_dir.parent / "update_manifest_2026.json", manifest)
    return manifest
