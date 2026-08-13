"""Public Formula 1 data clients and normalizers.

Jolpica supplies the durable race history and schedule. OpenF1 is used only as
an optional, best-effort source for a post-qualifying starting grid.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests


JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
OPENF1_BASE_URL = "https://api.openf1.org/v1"


class JsonClient:
    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "Prediction-Bounties-F1/1.0"})

    def get(self, path: str, params: dict[str, Any] | None = None, max_retries: int = 3) -> Any:
        for attempt in range(max_retries + 1):
            response = self.session.get(f"{self.base_url}/{path.lstrip('/')}", params=params or {}, timeout=45)
            if response.ok:
                return response.json()
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
                raise RuntimeError(f"F1 data provider returned HTTP {response.status_code} for {path}")
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 2**attempt
            time.sleep(min(delay, 20.0))
        raise AssertionError("unreachable")


def _races(payload: Any) -> list[dict[str, Any]]:
    try:
        races = payload["MRData"]["RaceTable"]["Races"]
    except (KeyError, TypeError):
        return []
    return races if isinstance(races, list) else []


def _driver_name(driver: dict[str, Any]) -> str:
    return " ".join(str(driver.get(key) or "").strip() for key in ("givenName", "familyName")).strip()


def normalize_history_races(races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for race in races:
        season = int(race.get("season") or 0)
        round_number = int(race.get("round") or 0)
        circuit = race.get("Circuit") if isinstance(race.get("Circuit"), dict) else {}
        rows: list[dict[str, Any]] = []
        for result in race.get("Results") or []:
            driver = result.get("Driver") if isinstance(result.get("Driver"), dict) else {}
            constructors = result.get("Constructor") if isinstance(result.get("Constructor"), dict) else {}
            status = str(result.get("status") or "")
            try:
                finish = int(result.get("position") or result.get("positionText"))
            except (TypeError, ValueError):
                finish = 22
            try:
                grid = int(result.get("grid") or 0)
            except (TypeError, ValueError):
                grid = 0
            try:
                points = float(result.get("points") or 0.0)
            except (TypeError, ValueError):
                points = 0.0
            rows.append(
                {
                    "driver_id": str(driver.get("driverId") or ""),
                    "driver": _driver_name(driver),
                    "driver_number": str(driver.get("permanentNumber") or result.get("number") or ""),
                    "constructor_id": str(constructors.get("constructorId") or ""),
                    "constructor": str(constructors.get("name") or ""),
                    "grid": grid,
                    "finish": finish,
                    "points": points,
                    "dnf": not (status == "Finished" or status.startswith("+")),
                }
            )
        if rows:
            normalized.append(
                {
                    "season": season,
                    "round": round_number,
                    "race_name": str(race.get("raceName") or ""),
                    "date": str(race.get("date") or ""),
                    "circuit_id": str(circuit.get("circuitId") or ""),
                    "circuit": str(circuit.get("circuitName") or ""),
                    "results": rows,
                }
            )
    return sorted(normalized, key=lambda item: (item["season"], item["round"]))


def fetch_history(
    start_year: int,
    end_year: int,
    *,
    client: JsonClient | None = None,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    active = client or JsonClient(JOLPICA_BASE_URL)
    cached_by_year: dict[int, list[dict[str, Any]]] = {}
    complete_cached_seasons: set[int] = set()
    if cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            for race in cached.get("races", []):
                cached_by_year.setdefault(int(race["season"]), []).append(race)
            if int(cached.get("schema_version") or 0) >= 2:
                complete_cached_seasons = {int(value) for value in cached.get("complete_seasons", [])}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            cached_by_year = {}
            complete_cached_seasons = set()

    all_races: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        # Completed current-season rounds can change after every scheduled run;
        # prior seasons are immutable and safe to reuse from cache.
        if year < end_year and year in complete_cached_seasons and cached_by_year.get(year):
            all_races.extend(cached_by_year[year])
            continue
        offset = 0
        season_races: list[dict[str, Any]] = []
        while True:
            payload = active.get(f"{year}/results.json", {"limit": 100, "offset": offset})
            page = normalize_history_races(_races(payload))
            season_races.extend(page)
            metadata = payload.get("MRData", {}) if isinstance(payload, dict) else {}
            try:
                total = int(metadata.get("total") or len(page))
                page_limit = int(metadata.get("limit") or 100)
            except (TypeError, ValueError):
                total, page_limit = len(page), 100
            offset += max(1, page_limit)
            if offset >= total or not page:
                break
        # Pages are result-row based and can theoretically split a race. Merge
        # those pieces before persisting the completed season.
        merged: dict[tuple[int, int], dict[str, Any]] = {}
        for race in season_races:
            key = (int(race["season"]), int(race["round"]))
            target = merged.setdefault(key, {**race, "results": []})
            existing = {row["driver_id"] for row in target["results"]}
            target["results"].extend(row for row in race["results"] if row["driver_id"] not in existing)
        all_races.extend(merged.values())
        if year < end_year:
            complete_cached_seasons.add(year)

    all_races.sort(key=lambda item: (item["season"], item["round"]))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "complete_seasons": sorted(complete_cached_seasons),
                    "races": all_races,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return all_races


def fetch_schedule(season: int, *, client: JsonClient | None = None) -> list[dict[str, Any]]:
    active = client or JsonClient(JOLPICA_BASE_URL)
    return _races(active.get(f"{season}.json", {"limit": 100}))


def select_next_event(schedule: list[dict[str, Any]], run_date: date) -> dict[str, Any] | None:
    candidates: list[tuple[date, dict[str, Any]]] = []
    for race in schedule:
        try:
            race_day = date.fromisoformat(str(race.get("date")))
        except (TypeError, ValueError):
            continue
        if race_day >= run_date:
            candidates.append((race_day, race))
    if not candidates:
        return None
    _, race = min(candidates, key=lambda item: item[0])
    circuit = race.get("Circuit") if isinstance(race.get("Circuit"), dict) else {}
    location = circuit.get("Location") if isinstance(circuit.get("Location"), dict) else {}
    return {
        "season": int(race.get("season") or run_date.year),
        "round": int(race.get("round") or 0),
        "race_name": str(race.get("raceName") or "Upcoming Grand Prix"),
        "date": str(race.get("date") or ""),
        "time_utc": str(race.get("time") or ""),
        "circuit_id": str(circuit.get("circuitId") or ""),
        "circuit": str(circuit.get("circuitName") or ""),
        "locality": str(location.get("locality") or ""),
        "country": str(location.get("country") or ""),
    }


def fetch_driver_standings(season: int, *, client: JsonClient | None = None) -> list[dict[str, Any]]:
    active = client or JsonClient(JOLPICA_BASE_URL)
    payload = active.get(f"{season}/driverstandings.json", {"limit": 100})
    try:
        lists = payload["MRData"]["StandingsTable"]["StandingsLists"]
    except (KeyError, TypeError):
        return []
    if not lists:
        return []
    entries: list[dict[str, Any]] = []
    for row in lists[-1].get("DriverStandings") or []:
        driver = row.get("Driver") if isinstance(row.get("Driver"), dict) else {}
        constructors = row.get("Constructors") if isinstance(row.get("Constructors"), list) else []
        constructor = constructors[-1] if constructors and isinstance(constructors[-1], dict) else {}
        entries.append(
            {
                "driver_id": str(driver.get("driverId") or ""),
                "driver": _driver_name(driver),
                "driver_number": str(driver.get("permanentNumber") or ""),
                "constructor_id": str(constructor.get("constructorId") or ""),
                "constructor": str(constructor.get("name") or ""),
                "standing_position": int(row.get("position") or len(entries) + 1),
                "championship_points": float(row.get("points") or 0.0),
                "grid": 0,
            }
        )
    return entries


def entries_from_latest_race(history: list[dict[str, Any]], season: int) -> list[dict[str, Any]]:
    season_races = [race for race in history if int(race["season"]) == season]
    source = season_races[-1] if season_races else (history[-1] if history else None)
    if source is None:
        return []
    return [
        {
            "driver_id": row["driver_id"],
            "driver": row["driver"],
            "driver_number": row.get("driver_number", ""),
            "constructor_id": row["constructor_id"],
            "constructor": row["constructor"],
            "standing_position": index + 1,
            "championship_points": 0.0,
            "grid": 0,
        }
        for index, row in enumerate(source["results"])
    ]


def fetch_starting_grid(
    event: dict[str, Any], entries: list[dict[str, Any]], *, client: JsonClient | None = None
) -> dict[str, int]:
    """Return a best-effort driver-id to grid-position mapping."""

    active = client or JsonClient(OPENF1_BASE_URL)
    meetings = active.get("meetings", {"year": event["season"]})
    if not isinstance(meetings, list):
        return {}
    target_date = date.fromisoformat(event["date"])
    matches: list[tuple[int, dict[str, Any]]] = []
    for meeting in meetings:
        try:
            meeting_date = date.fromisoformat(str(meeting.get("date_start") or "")[:10])
        except ValueError:
            continue
        distance = abs((meeting_date - target_date).days)
        if distance <= 5:
            matches.append((distance, meeting))
    if not matches:
        return {}
    meeting = min(matches, key=lambda item: item[0])[1]
    sessions = active.get("sessions", {"meeting_key": meeting.get("meeting_key"), "session_name": "Race"})
    if not isinstance(sessions, list) or not sessions:
        return {}
    session_key = sessions[-1].get("session_key")
    grid = active.get("starting_grid", {"session_key": session_key})
    if not isinstance(grid, list):
        return {}
    number_to_id = {str(entry.get("driver_number") or ""): entry["driver_id"] for entry in entries}
    result: dict[str, int] = {}
    for row in grid:
        driver_id = number_to_id.get(str(row.get("driver_number") or ""))
        try:
            position = int(row.get("position"))
        except (TypeError, ValueError):
            continue
        if driver_id and position > 0:
            result[driver_id] = position
    return result
