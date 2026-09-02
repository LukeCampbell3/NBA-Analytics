from __future__ import annotations

from urllib.parse import parse_qs, urlparse


def validate_fanduel_link(url: str | None, *, require_exact_selection: bool) -> tuple[bool, str]:
    if not url:
        return False, "DEEPLINK_MISSING"
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "fanduel"}:
        return False, "DEEPLINK_SCHEME_INVALID"
    if parsed.scheme == "https" and "fanduel" not in parsed.netloc.lower():
        return False, "DEEPLINK_HOST_INVALID"
    if require_exact_selection:
        query = parse_qs(parsed.query)
        selection = query.get("selectionId") or query.get("selectionId[0]")
        market = query.get("marketId") or query.get("marketId[0]")
        if not selection or not market:
            return False, "EXACT_SELECTION_UNAVAILABLE"
    return True, "VALID"
