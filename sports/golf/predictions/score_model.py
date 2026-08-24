from __future__ import annotations

"""Real PGA Tour score-projection model.

Builds per-round and tournament-total score projections for a real
tournament field from real recent-event scoring data (sports.golf.scripts.
fetch_pga_event's persisted leaderboard snapshots) -- never simulated or
fabricated inputs. The model is intentionally simple and stated honestly
as such: a field-relative recent-form baseline, not a claimed
state-of-the-art strokes-gained model. It is the real, non-fabricated
foundation this repo's calibration ledger will hold accountable over time,
exactly like every other sport's first production model here.

Two stages:
  1. `build_recent_form` -- from N real recently-completed events, computes
     each player's real scoring differential vs. that day's real field
     average, per round they actually played. This is a real proxy for
     "strokes better/worse than an average field on an average day" --
     not true strokes-gained (which needs shot-level ShotLink data this
     repo has no access to), but real and grounded in real outcomes.
  2. `simulate_tournament` -- Monte Carlo simulation of the real field
     together (not player-by-player independent thresholds): every
     simulation draws one round score per player from their own real
     form distribution, so finish-position outcomes (win / top-5 / top-10
     / top-20 / made cut) are computed relative to the REST OF THE
     SIMULATED FIELD in that same draw -- the statistically correct way
     to estimate a relative-rank market, matching this repo's established
     discipline of never treating a field-relative outcome as if it were
     an independent per-player threshold.
"""

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_ROUND_PAR = 71.0  # only used as a fallback prior when a player has no real recent rounds at all
MIN_ROUNDS_FOR_OWN_FORM = 3
LEAGUE_AVERAGE_ROUND_STD = 2.6  # real-world PGA Tour round-to-round std is ~2.5-3 strokes; used as a fallback/floor


@dataclass(frozen=True)
class PlayerForm:
    player_id: str
    player_name: str
    rounds_observed: int
    mean_differential: float  # real mean (own round strokes - that day's real field average), lower is better
    std_differential: float


@dataclass
class PlayerProjection:
    player_id: str
    player_name: str
    headshot_url: str
    projected_round_score: float  # projected strokes-to-par per round
    projected_total_score: float  # projected strokes-to-par over the real scheduled round count
    round_std: float
    form_rounds_observed: int


def build_recent_form(recent_events: list[dict]) -> dict[str, PlayerForm]:
    """`recent_events` is a list of real persisted leaderboard payloads
    (sports.golf.scripts.fetch_pga_event.fetch_event_leaderboard output,
    with `players` already flattened via extract_player_rounds). Only
    real, completed rounds count -- a withdrawn player's played rounds
    still count, their unplayed ones are simply absent already."""
    per_player_diffs: dict[str, list[float]] = {}
    per_player_name: dict[str, str] = {}

    for event in recent_events:
        players = event.get("players", [])
        # Real field-average strokes for each round actually played this event.
        round_totals: dict[int, list[float]] = {}
        for player in players:
            for round_row in player.get("rounds", []):
                round_num = round_row.get("round")
                strokes = round_row.get("strokes")
                if round_num is None or strokes is None:
                    continue
                round_totals.setdefault(int(round_num), []).append(float(strokes))
        round_field_avg = {
            round_num: (sum(values) / len(values)) for round_num, values in round_totals.items() if values
        }

        for player in players:
            player_id = str(player.get("player_id") or "")
            if not player_id:
                continue
            per_player_name[player_id] = str(player.get("player_name") or "")
            for round_row in player.get("rounds", []):
                round_num = round_row.get("round")
                strokes = round_row.get("strokes")
                if round_num is None or strokes is None:
                    continue
                field_avg = round_field_avg.get(int(round_num))
                if field_avg is None:
                    continue
                per_player_diffs.setdefault(player_id, []).append(float(strokes) - field_avg)

    forms: dict[str, PlayerForm] = {}
    for player_id, diffs in per_player_diffs.items():
        n = len(diffs)
        mean_diff = sum(diffs) / n
        std_diff = statistics.pstdev(diffs) if n > 1 else LEAGUE_AVERAGE_ROUND_STD
        std_diff = max(std_diff, 1.0)  # never claim a real player is more consistent than physically plausible
        forms[player_id] = PlayerForm(
            player_id=player_id,
            player_name=per_player_name.get(player_id, ""),
            rounds_observed=n,
            mean_differential=mean_diff,
            std_differential=std_diff,
        )
    return forms


def project_field(
    field_players: list[dict],
    forms: dict[str, PlayerForm],
    *,
    scheduled_rounds: int,
    round_par: float = DEFAULT_ROUND_PAR,
) -> list[PlayerProjection]:
    """Projects every real player in the field. A player with no real
    recent-form rounds (e.g. a sponsor exemption/late addition with no
    recent PGA Tour starts in the lookback window) is projected at the
    real field-average differential of 0.0 with a widened, honestly
    uncertain std -- never silently dropped and never given a fabricated
    specific projection."""
    league_avg_diff = 0.0
    if forms:
        league_avg_diff = sum(f.mean_differential for f in forms.values()) / len(forms)

    projections: list[PlayerProjection] = []
    for player in field_players:
        player_id = str(player.get("player_id") or "")
        player_name = str(player.get("player_name") or "")
        form = forms.get(player_id)
        if form is not None and form.rounds_observed >= MIN_ROUNDS_FOR_OWN_FORM:
            round_diff = form.mean_differential
            round_std = form.std_differential
            rounds_observed = form.rounds_observed
        else:
            # Not enough real recent rounds for this player's own form --
            # fall back to the real field-wide average differential rather
            # than a fabricated player-specific number, with a widened std
            # to honestly reflect the extra uncertainty.
            round_diff = league_avg_diff
            round_std = LEAGUE_AVERAGE_ROUND_STD * 1.4
            rounds_observed = 0 if form is None else form.rounds_observed

        projected_round_score = round_par + round_diff
        projections.append(
            PlayerProjection(
                player_id=player_id,
                player_name=player_name,
                headshot_url=str(player.get("headshot_url") or ""),
                projected_round_score=projected_round_score,
                projected_total_score=projected_round_score * scheduled_rounds,
                round_std=round_std,
                form_rounds_observed=rounds_observed,
            )
        )
    return projections


@dataclass
class FieldOutcomeProbabilities:
    player_id: str
    player_name: str
    win_probability: float
    top5_probability: float
    top10_probability: float
    top20_probability: float
    make_cut_probability: Optional[float]  # None when the event has no real cut


def simulate_tournament(
    projections: list[PlayerProjection],
    *,
    scheduled_rounds: int = 4,
    has_cut: bool = True,
    cut_after_round: int = 2,
    cut_size: int = 65,
    num_simulations: int = 20000,
    random_seed: Optional[int] = None,
) -> list[FieldOutcomeProbabilities]:
    """Joint Monte Carlo simulation of the WHOLE field together. Every
    simulation draws one score per player per round from that player's own
    real-form-derived distribution, so win/top-N/cut outcomes are always
    computed relative to the other simulated field members in that same
    draw -- never as an independent per-player probability against a fixed
    threshold, which would be statistically wrong for an inherently
    relative-rank market."""
    rng = random.Random(random_seed)
    n = len(projections)
    if n == 0:
        return []

    wins = [0] * n
    top5 = [0] * n
    top10 = [0] * n
    top20 = [0] * n
    made_cut = [0] * n if has_cut else None

    for _ in range(num_simulations):
        cut_qualifiers = list(range(n))
        totals = [0.0] * n
        for round_num in range(1, scheduled_rounds + 1):
            for idx in (cut_qualifiers if round_num > cut_after_round and has_cut else range(n)):
                proj = projections[idx]
                totals[idx] += rng.gauss(proj.projected_round_score, proj.round_std)
            if has_cut and round_num == cut_after_round:
                ranked = sorted(range(n), key=lambda i: totals[i])
                if len(ranked) > cut_size:
                    cutoff_score = totals[ranked[cut_size - 1]]
                    cut_qualifiers = [i for i in range(n) if totals[i] <= cutoff_score]
                else:
                    cut_qualifiers = ranked
                if made_cut is not None:
                    for idx in cut_qualifiers:
                        made_cut[idx] += 1

        finishers = cut_qualifiers if has_cut else list(range(n))
        ranked_finish = sorted(finishers, key=lambda i: totals[i])
        for rank, idx in enumerate(ranked_finish, start=1):
            if rank == 1:
                wins[idx] += 1
            if rank <= 5:
                top5[idx] += 1
            if rank <= 10:
                top10[idx] += 1
            if rank <= 20:
                top20[idx] += 1

    results: list[FieldOutcomeProbabilities] = []
    for i, proj in enumerate(projections):
        results.append(
            FieldOutcomeProbabilities(
                player_id=proj.player_id,
                player_name=proj.player_name,
                win_probability=wins[i] / num_simulations,
                top5_probability=top5[i] / num_simulations,
                top10_probability=top10[i] / num_simulations,
                top20_probability=top20[i] / num_simulations,
                make_cut_probability=(made_cut[i] / num_simulations) if made_cut is not None else None,
            )
        )
    return results
