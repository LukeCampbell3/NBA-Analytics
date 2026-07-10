"""Seven-season PAR validation against public box-score metrics.

This harness intentionally treats BPM/VORP/PER/WS/EPM as comparison targets,
not tuning targets. It recomputes the frozen direct-event PAR adapter from
public season totals and evaluates alignment and one-year-ahead behavior.
"""
from __future__ import annotations

import argparse
from io import StringIO
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.stats import pearsonr, spearmanr

from .config import MODEL_CONFIG
from .engine import infer_role


BREF_BASE = "https://www.basketball-reference.com/leagues/NBA_{year}_{table}.html"
DUNKS_EPM_ACTUAL = "https://dunksandthrees.com/epm/actual"
MIN_MINUTES = 500


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "", text.lower())
    return text


def season_label(year: int) -> str:
    return f"{year - 1}-{str(year)[-2:]}"


def clean_bref(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Player"].astype(str) != "Player"].copy()
    for col in df.columns:
        if col not in {"Player", "Team", "Tm", "Pos", "Awards"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    team_col = "Tm" if "Tm" in df.columns else "Team"
    df["player_key"] = df["Player"].map(normalize_name)
    rows = []
    for _, group in df.groupby("player_key", sort=False):
        total = group[group[team_col] == "TOT"]
        if not total.empty:
            rows.append(total.iloc[0])
        else:
            minute_col = "MP" if "MP" in group.columns else "G"
            rows.append(group.sort_values(minute_col, ascending=False).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def read_bref(year: int, table: str, cache_dir: Path) -> pd.DataFrame:
    url = BREF_BASE.format(year=year, table=table)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"basketball_reference_{year}_{table}.html"
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html = response.text
        cache_path.write_text(html, encoding="utf-8")
        time.sleep(1.5)
    df = pd.read_html(StringIO(html))[0]
    return clean_bref(df)


def role_baseline(role: str, atom_type: str) -> float:
    role = role if role in MODEL_CONFIG.replacement_baselines else "connector"
    return float(MODEL_CONFIG.replacement_baselines[role][atom_type])


def compute_par_row(row: pd.Series, year: int) -> dict[str, Any]:
    minutes = float(row.get("MP") or 0.0)
    stats = {
        "minutes": minutes,
        "games": float(row.get("G") or 0.0),
        "pts": float(row.get("PTS") or 0.0),
        "ast": float(row.get("AST") or 0.0),
        "trb": float(row.get("TRB") or row.get("REB") or 0.0),
        "stl": float(row.get("STL") or 0.0),
    }
    role = infer_role(stats)
    atom_raw = {
        "scoring_volume_above_replacement": stats["pts"],
        "passing_creation": float(row.get("AST") or 0.0) * 2.2,
        "negative_turnover_value": -float(row.get("TOV") or 0.0) * 1.4,
        "steals": float(row.get("STL") or 0.0) * 2.0,
    }
    atoms: dict[str, float] = {}
    forecast_atoms: dict[str, float] = {}
    for atom_type, raw_value in atom_raw.items():
        replacement = role_baseline(role, atom_type) * minutes / 36.0
        par_value = raw_value - replacement
        atoms[atom_type] = par_value
        persistence_key = MODEL_CONFIG.atom_registry[atom_type]["persistence_key"]
        forecast_atoms[atom_type] = par_value * MODEL_CONFIG.persistence_values[persistence_key]
    total = sum(atoms.values())
    role_continuity = 0.96 if role in {"primary_creator", "secondary_creator"} else 0.92
    minutes_factor = min(1.08, max(0.62, minutes / 1800.0)) if minutes else 0.62
    health_factor = 0.96 if minutes < 900 else 1.0
    projected = sum(forecast_atoms.values()) * role_continuity * minutes_factor * health_factor
    par_1000 = total / minutes * 1000.0 if minutes else 0.0
    return {
        "season": season_label(year),
        "bref_year": year,
        "player": row["Player"],
        "player_key": row["player_key"],
        "team": row.get("Tm") or row.get("Team"),
        "role": role,
        "games": float(row.get("G") or 0.0),
        "minutes": minutes,
        "par": total,
        "par_1000": par_1000,
        "war_equivalent": total / MODEL_CONFIG.points_per_win,
        "pvg_score": 50.0 + 45.0 * math.tanh(par_1000 / 210.0) if minutes else 50.0,
        "projected_parf": projected,
        "scoring_par": atoms["scoring_volume_above_replacement"],
        "creation_par": atoms["passing_creation"],
        "ball_security_par": atoms["negative_turnover_value"],
        "perimeter_disruption_par": atoms["steals"],
    }


def metric_corr(df: pd.DataFrame, left: str, right: str) -> dict[str, Any]:
    sample = df[[left, right]].replace([math.inf, -math.inf], pd.NA).dropna()
    if len(sample) < 3:
        return {"n": len(sample), "pearson": None, "spearman": None}
    return {
        "n": int(len(sample)),
        "pearson": round(float(pearsonr(sample[left], sample[right]).statistic), 6),
        "spearman": round(float(spearmanr(sample[left], sample[right]).statistic), 6),
    }


def rmse(values: pd.Series) -> float:
    return float(math.sqrt((values.astype(float) ** 2).mean()))


def tier_accuracy(pred: pd.Series, actual: pd.Series) -> float:
    frame = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if len(frame) < 9:
        return float("nan")
    frame["pred_tier"] = pd.qcut(frame["pred"].rank(method="first"), 3, labels=False)
    frame["actual_tier"] = pd.qcut(frame["actual"].rank(method="first"), 3, labels=False)
    return float((frame["pred_tier"] == frame["actual_tier"]).mean())


def parse_current_epm() -> pd.DataFrame:
    html = requests.get(DUNKS_EPM_ACTUAL, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue
        link = cells[0].find("a")
        if link is None:
            continue
        strings = [list(td.stripped_strings) for td in cells[1:]]
        values = [parts[0].replace("−", "-").replace("+", "") if parts else "" for parts in strings]
        try:
            rows.append(
                {
                    "player": link.get_text(strip=True),
                    "player_key": normalize_name(link.get_text(strip=True)),
                    "epm_gp": float(values[0]),
                    "epm_mpg": float(values[1]),
                    "epm_usage": float(values[2]),
                    "epm_off": float(values[3]),
                    "epm_def": float(values[4]),
                    "epm": float(values[5]),
                    "estimated_wins": float(values[6]),
                }
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def build_validation(years: list[int], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "source_cache"
    seasons = []
    for year in years:
        totals = read_bref(year, "totals", cache_dir)
        advanced = read_bref(year, "advanced", cache_dir)
        per_game = read_bref(year, "per_game", cache_dir)
        rows = pd.DataFrame([compute_par_row(row, year) for _, row in totals.iterrows()])
        merged = rows.merge(
            advanced[
                [
                    "player_key",
                    "PER",
                    "TS%",
                    "USG%",
                    "OWS",
                    "DWS",
                    "WS",
                    "WS/48",
                    "OBPM",
                    "DBPM",
                    "BPM",
                    "VORP",
                ]
            ],
            on="player_key",
            how="left",
        ).merge(
            per_game[["player_key", "PTS", "TRB", "AST", "STL", "BLK", "TOV"]],
            on="player_key",
            how="left",
            suffixes=("", "_per_game"),
        )
        merged["box_score_index"] = (
            merged["PTS"].fillna(0)
            + merged["TRB"].fillna(0)
            + merged["AST"].fillna(0)
            + merged["STL"].fillna(0)
            + merged["BLK"].fillna(0)
            - merged["TOV"].fillna(0)
        )
        seasons.append(merged)
    all_rows = pd.concat(seasons, ignore_index=True)
    eligible = all_rows[all_rows["minutes"] >= MIN_MINUTES].copy()
    by_season: dict[str, Any] = {}
    for season, group in eligible.groupby("season"):
        by_season[season] = {
            metric: metric_corr(group, "par", metric)
            for metric in ["BPM", "VORP", "WS", "PER", "box_score_index"]
        }
    pooled = {
        metric: metric_corr(eligible, "par", metric)
        for metric in ["BPM", "VORP", "WS", "PER", "box_score_index"]
    }
    next_rows = all_rows[["player_key", "bref_year", "par", "BPM", "VORP", "WS"]].copy()
    next_rows = next_rows.rename(
        columns={
            "bref_year": "next_bref_year",
            "par": "next_par",
            "BPM": "next_bpm",
            "VORP": "next_vorp",
            "WS": "next_ws",
        }
    )
    next_rows["bref_year"] = next_rows["next_bref_year"] - 1
    forecast = eligible.merge(
        next_rows,
        on=["player_key", "bref_year"],
        how="inner",
    )
    forecast = forecast[forecast["next_par"].notna()].copy()
    forecast["parf_error"] = forecast["projected_parf"] - forecast["next_par"]
    forecast_summary = {
        "sample_size": int(len(forecast)),
        "parf_to_next_par": metric_corr(forecast, "projected_parf", "next_par"),
        "current_par_to_next_par": metric_corr(forecast, "par", "next_par"),
        "parf_mae_next_par": round(float(forecast["parf_error"].abs().mean()), 6),
        "parf_rmse_next_par": round(rmse(forecast["parf_error"]), 6),
        "parf_tier_accuracy_next_par": round(tier_accuracy(forecast["projected_parf"], forecast["next_par"]), 6),
        "par_to_next_bpm": metric_corr(forecast, "par", "next_bpm"),
        "parf_to_next_bpm": metric_corr(forecast, "projected_parf", "next_bpm"),
        "par_to_next_vorp": metric_corr(forecast, "par", "next_vorp"),
        "parf_to_next_vorp": metric_corr(forecast, "projected_parf", "next_vorp"),
        "par_to_next_ws": metric_corr(forecast, "par", "next_ws"),
        "parf_to_next_ws": metric_corr(forecast, "projected_parf", "next_ws"),
    }
    epm_summary: dict[str, Any] = {"status": "not_run"}
    try:
        epm = parse_current_epm()
        current = eligible[eligible["bref_year"] == max(years)].merge(epm, on="player_key", how="inner")
        epm_summary = {
            "status": "pass" if len(current) else "empty_join",
            "source": DUNKS_EPM_ACTUAL,
            "sample_size": int(len(current)),
            "par_to_epm": metric_corr(current, "par", "epm"),
            "par1000_to_epm": metric_corr(current, "par_1000", "epm"),
            "par_to_estimated_wins": metric_corr(current, "par", "estimated_wins"),
            "top_joined_examples": current.sort_values("par", ascending=False)[
                ["player_x", "par", "par_1000", "BPM", "VORP", "epm", "estimated_wins"]
            ]
            .head(15)
            .rename(columns={"player_x": "player"})
            .to_dict(orient="records"),
        }
        epm.to_csv(out / "current_epm_scrape.csv", index=False)
    except Exception as exc:  # pragma: no cover - network/source dependent
        epm_summary = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}", "source": DUNKS_EPM_ACTUAL}
    report = {
        "par_model_version": MODEL_CONFIG.par_model_version,
        "parf_model_version": MODEL_CONFIG.parf_model_version,
        "years": years,
        "seasons": [season_label(year) for year in years],
        "source": "Basketball-Reference season totals/advanced/per-game tables",
        "source_urls": [BREF_BASE.format(year=year, table="{totals,advanced,per_game}") for year in years],
        "minimum_minutes": MIN_MINUTES,
        "player_seasons": int(len(all_rows)),
        "eligible_player_seasons": int(len(eligible)),
        "current_value_correlations": {"pooled": pooled, "by_season": by_season},
        "forward_validation": forecast_summary,
        "epm_comparison": epm_summary,
        "lebron_comparison": {
            "status": "blocked",
            "reason": "Public BBall Index pages expose dashboard shell here, but no downloadable season table was available without credentials.",
        },
        "doctrine_note": "External all-in-one metrics are comparison references only; PAR constants were not tuned to them.",
    }
    all_rows.to_csv(out / "par_seven_season_player_rows.csv", index=False)
    eligible.to_csv(out / "par_seven_season_eligible_rows.csv", index=False)
    forecast.to_csv(out / "parf_forward_validation_rows.csv", index=False)
    (out / "par_seven_season_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run seven-season PAR validation.")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--out", default="out/par_validation_7y")
    args = parser.parse_args(argv)
    years = list(range(args.start_year, args.end_year + 1))
    report = build_validation(years, Path(args.out))
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
