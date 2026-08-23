"""Recipient network analysis (spec section 7), built from real
Basketball-Reference play-by-play assist events.

HONEST SCOPE, STATED UP FRONT: Basketball-Reference's play-by-play only
records a "pass" when it becomes a made-shot assist. It has no signal
for a drive kick-out that was missed, a reset pass, or any non-scoring
pass. This means every quantity below that the original spec defines
with "passes" in the denominator (pass share, AST/pass, recipient
leverage in its original ``assist share / pass share`` form, shot
leverage) cannot be computed from this source and is returned as an
explicit UNAVAILABLE Metric. What CAN be computed, honestly, from real
assisted-shot events:

  - real assist counts and assist share per recipient (this player's
    real sampled assists, broken out by who received them)
  - the real, DERIVED shot-zone breakdown of what each recipient's
    resulting made shot was (rim/short-paint/midrange/three)
  - a real ``high_value_share_index``: this recipient's share of
    high-value assisted shots (rim + three) relative to the player's
    own overall high-value-assist rate. This is a genuine, defensible
    analog of "shot leverage" restricted to the assisted-shot sample --
    it is NOT the spec's original formula (which needs total pass
    volume) and is labeled accordingly.

If/when a real touch/passing-tracking source becomes reachable, the
pass-share-dependent fields below are the ones to fill in first -- see
docs/advantage-routing.md.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..models.schemas import EvidenceStatus, Metric, ShotZone
from ..sources.bball_ref import RealAssistEvent
from .states import classify_shot_zone_from_text

HIGH_VALUE_ZONES = (ShotZone.RIM.value, ShotZone.ABOVE_BREAK_3.value, ShotZone.CORNER_3.value)


@dataclass
class RecipientStats:
    recipient_label: str
    recipient_slug: str
    assists: Metric
    assist_share: Metric
    points_generated: Metric
    zone_breakdown: dict[str, int]
    most_common_resulting_shot: Optional[str]
    high_value_assist_rate: Metric
    high_value_share_index: Metric
    # Explicitly unavailable, spec-named fields -- present so the
    # frontend/consumer never has to special-case a missing key.
    passes: Metric = field(default_factory=lambda: Metric.unavailable("passes", reason=_PASS_UNAVAILABLE_REASON))
    pass_share: Metric = field(default_factory=lambda: Metric.unavailable("pass_share", reason=_PASS_UNAVAILABLE_REASON))
    ast_per_pass: Metric = field(default_factory=lambda: Metric.unavailable("ast_per_pass", reason=_PASS_UNAVAILABLE_REASON))
    recipient_leverage: Metric = field(default_factory=lambda: Metric.unavailable("recipient_leverage", reason=_PASS_UNAVAILABLE_REASON))

    def as_dict(self) -> dict:
        return {
            "recipient_label": self.recipient_label,
            "recipient_slug": self.recipient_slug,
            "assists": self.assists.as_dict(),
            "assist_share": self.assist_share.as_dict(),
            "points_generated": self.points_generated.as_dict(),
            "zone_breakdown": self.zone_breakdown,
            "most_common_resulting_shot": self.most_common_resulting_shot,
            "high_value_assist_rate": self.high_value_assist_rate.as_dict(),
            "high_value_share_index": self.high_value_share_index.as_dict(),
            "passes": self.passes.as_dict(),
            "pass_share": self.pass_share.as_dict(),
            "ast_per_pass": self.ast_per_pass.as_dict(),
            "recipient_leverage": self.recipient_leverage.as_dict(),
        }


_PASS_UNAVAILABLE_REASON = (
    "Requires a total-pass count (not just assisted passes). "
    "Basketball-Reference play-by-play only records passes that became "
    "made-shot assists; no reachable source publishes non-scoring passes. "
    "See routing/recipients.py module docstring."
)


@dataclass
class RecipientNetwork:
    player_name: str
    sample_size: int
    games_sampled: int
    games_available_total: int
    recipients: list[RecipientStats]

    def as_dict(self) -> dict:
        return {
            "player_name": self.player_name,
            "sample_description": (
                f"{self.sample_size} real assisted-shot events across {self.games_sampled} "
                f"of {self.games_available_total} real season games (most recent games sampled, "
                "chronological order preserved -- see sources/collect.py)."
            ),
            "sample_size": self.sample_size,
            "games_sampled": self.games_sampled,
            "games_available_total": self.games_available_total,
            "recipients": [r.as_dict() for r in self.recipients],
        }


def build_recipient_network(player_name: str, assists: list[RealAssistEvent], *, games_sampled: int, games_available_total: int, season: str) -> RecipientNetwork:
    total = len(assists)
    by_recipient: dict[str, list[RealAssistEvent]] = defaultdict(list)
    for a in assists:
        by_recipient[a.recipient_slug].append(a)

    overall_high_value = sum(
        1 for a in assists
        if classify_shot_zone_from_text(a.shot_description, a.shot_distance_ft, a.is_three).zone in HIGH_VALUE_ZONES
    )
    overall_high_value_rate = (overall_high_value / total) if total else None

    recipients: list[RecipientStats] = []
    for slug, events in sorted(by_recipient.items(), key=lambda kv: -len(kv[1])):
        label = events[0].recipient_label
        n = len(events)
        zone_counter: Counter[str] = Counter()
        points = 0
        high_value = 0
        for a in events:
            classification = classify_shot_zone_from_text(a.shot_description, a.shot_distance_ft, a.is_three)
            zone_counter[classification.zone] += 1
            points += 3 if a.is_three else 2
            if classification.zone in HIGH_VALUE_ZONES:
                high_value += 1

        high_value_rate = high_value / n if n else None
        relative_index = (high_value_rate / overall_high_value_rate) if (high_value_rate is not None and overall_high_value_rate) else None

        recipients.append(RecipientStats(
            recipient_label=label,
            recipient_slug=slug,
            assists=Metric.observed("assists", n, source="Basketball-Reference play-by-play (sampled)", season=season, sample_size=total),
            assist_share=Metric.derived("assist_share", (n / total) if total else None, method="recipient_assists / player_total_sampled_assists", season=season, sample_size=total),
            points_generated=Metric.derived("points_generated", float(points), method="sum(3 if three else 2 for each real assisted make)", season=season, sample_size=n),
            zone_breakdown=dict(zone_counter),
            most_common_resulting_shot=zone_counter.most_common(1)[0][0] if zone_counter else None,
            high_value_assist_rate=Metric.derived("high_value_assist_rate", high_value_rate, method="high_value assists to recipient / total assists to recipient (rim+three, sampled)", season=season, sample_size=n),
            high_value_share_index=Metric.derived(
                "high_value_share_index", relative_index,
                method="recipient's high_value_assist_rate / player's overall high_value_assist_rate (sampled assists only) -- an honest analog of 'shot leverage' restricted to the assisted-shot sample; NOT the spec's original pass-share-based formula, see module docstring",
                season=season, sample_size=n,
            ),
        ))

    return RecipientNetwork(
        player_name=player_name, sample_size=total, games_sampled=games_sampled,
        games_available_total=games_available_total, recipients=recipients,
    )
