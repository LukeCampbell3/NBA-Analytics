#!/usr/bin/env python3
"""The real, verified list of US states/territories FanDuel Sportsbook is
legally licensed to operate online sports betting in, as of 2026-08.

FanDuel is a state-by-state licensed operator: each region is a distinct
real sportsbook instance with its own market/selection IDs for the same
real-world game (confirmed directly -- fanduel_public_mlb_provider.py
sends `x-sportsbook-region` on every request, and FanDuel's own API
response supplies a different real marketId/selectionId per region for
the identical player/market/line). A deep link built under one region is
not guaranteed to resolve for a viewer whose real FanDuel account is in a
different region -- this is the real root cause of "the link loads but
doesn't add to my betslip" for any viewer not in the region the site
happened to fetch under.

Source, verified via live web search on 2026-08-27 rather than assumed
(cross-referenced across multiple independent trackers -- sailgp.com,
sharpfootballanalysis.com, footballwhispers.com, ats.io,
worldpopulationreview.com -- all agreeing on this list): FanDuel operates
in 24 US states plus DC. This list changes over time as new states
legalize online sports betting or FanDuel launches in them (most
recently Arkansas, 2026-03-20) -- it is a real, disclosed snapshot, not
a permanent guarantee; a state absent here may simply not have real
online sports betting yet, and this list should be re-verified
periodically rather than assumed evergreen.
"""
from __future__ import annotations

# Two-letter USPS codes; DC included since FanDuel operates there too.
# Ordered roughly by real population (largest real-market states first)
# purely so the default frontend picker option list reads sensibly --
# carries no other meaning.
FANDUEL_LICENSED_STATES: tuple[str, ...] = (
    "NY", "PA", "OH", "IL", "MI", "NC", "VA", "AZ", "TN", "IN",
    "MA", "MD", "MO", "CO", "WV", "LA", "KY", "CT", "IA", "KS",
    "AR", "WY", "VT", "NJ", "DC",
)

STATE_NAMES: dict[str, str] = {
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "MI": "Michigan",
    "MO": "Missouri",
    "NC": "North Carolina",
    "NJ": "New Jersey",
    "NY": "New York",
    "OH": "Ohio",
    "PA": "Pennsylvania",
    "TN": "Tennessee",
    "VA": "Virginia",
    "VT": "Vermont",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

DEFAULT_FALLBACK_REGION = "NJ"  # the region every existing caller already defaulted to
