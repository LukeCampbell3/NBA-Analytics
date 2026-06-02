"""Plan → capability mapping. Enforced server-side only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ACTIVE_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing"})
RESTRICTED_SUBSCRIPTION_STATUSES = frozenset(
    {"past_due", "canceled", "unpaid", "incomplete", "incomplete_expired", "inactive"}
)


@dataclass(frozen=True)
class EntitlementCapabilities:
    plan_id: str
    can_view_full_safe_state: bool = False
    can_view_candidate_pool: bool = False
    can_view_simulation_filters: bool = False
    can_export_csv: bool = False
    can_use_api: bool = False
    max_cards_per_day: int | None = 3
    settlement_delay_hours: int = 24
    api_daily_limit: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "can_view_full_safe_state": self.can_view_full_safe_state,
            "can_view_candidate_pool": self.can_view_candidate_pool,
            "can_view_simulation_filters": self.can_view_simulation_filters,
            "can_export_csv": self.can_export_csv,
            "can_use_api": self.can_use_api,
            "max_cards_per_day": self.max_cards_per_day,
            "settlement_delay_hours": self.settlement_delay_hours,
            "api_daily_limit": self.api_daily_limit,
        }


PLAN_ENTITLEMENTS: dict[str, EntitlementCapabilities] = {
    "free": EntitlementCapabilities(
        plan_id="free",
        can_view_full_safe_state=False,
        can_view_candidate_pool=False,
        can_view_simulation_filters=False,
        can_export_csv=False,
        can_use_api=False,
        max_cards_per_day=3,
        settlement_delay_hours=24,
        api_daily_limit=0,
    ),
    "plus": EntitlementCapabilities(
        plan_id="plus",
        can_view_full_safe_state=True,
        can_view_candidate_pool=False,
        can_view_simulation_filters=False,
        can_export_csv=False,
        can_use_api=False,
        max_cards_per_day=None,
        settlement_delay_hours=0,
        api_daily_limit=0,
    ),
    "pro": EntitlementCapabilities(
        plan_id="pro",
        can_view_full_safe_state=True,
        can_view_candidate_pool=True,
        can_view_simulation_filters=True,
        can_export_csv=True,
        can_use_api=False,
        max_cards_per_day=None,
        settlement_delay_hours=0,
        api_daily_limit=0,
    ),
    "api": EntitlementCapabilities(
        plan_id="api",
        can_view_full_safe_state=True,
        can_view_candidate_pool=True,
        can_view_simulation_filters=True,
        can_export_csv=True,
        can_use_api=True,
        max_cards_per_day=None,
        settlement_delay_hours=0,
        api_daily_limit=5000,
    ),
}


def resolve_plan_id(plan_id: str | None, subscription_status: str | None = None) -> str:
    plan = (plan_id or "free").lower()
    if plan not in PLAN_ENTITLEMENTS:
        plan = "free"
    status = (subscription_status or "inactive").lower()
    if plan != "free" and status not in ACTIVE_SUBSCRIPTION_STATUSES:
        return "free"
    return plan


def entitlements_for_plan(plan_id: str | None, subscription_status: str | None = None) -> EntitlementCapabilities:
    resolved = resolve_plan_id(plan_id, subscription_status)
    return PLAN_ENTITLEMENTS[resolved]


def entitlements_to_row(user_id: str, caps: EntitlementCapabilities) -> dict[str, Any]:
    payload = caps.to_dict()
    payload.pop("plan_id")
    return {"user_id": user_id, "plan_id": caps.plan_id, **payload}
