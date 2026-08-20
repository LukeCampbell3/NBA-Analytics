from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sports.nba.conditional_chain.allocation_path import (
    PathQualityStatus,
    attach_realized_allocations,
    build_allocation_paths,
)
from sports.nba.conditional_chain.chain_resolver import resolve_conditional_chain
from sports.nba.conditional_chain.conditional_extension import (
    ConditionalExtensionModel,
    build_extension_training_ledger,
)
from sports.nba.conditional_chain.confirmation import (
    chronological_confirmation,
    evaluate_improvement_sequence,
)
from sports.nba.conditional_chain.freeze import build_freeze_manifest
from sports.nba.conditional_chain.frozen_selector import (
    score_frozen_selector,
    select_frozen_board,
    selected_probability,
)
from sports.nba.conditional_chain.protocol import (
    ALLOCATION_PATH_PROTOCOL,
    FROZEN_SELECTOR_PROTOCOL,
)
from sports.nba.conditional_chain.synthetic_audit import generate_synthetic_settled_paths


def _selector_history(players: list[str]) -> pd.DataFrame:
    rows = []
    for player in players:
        for day in range(1, 22):
            rows.append(
                {
                    "event_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "player": player,
                    "market": "player_points",
                    "actual": 25.0,
                }
            )
        rows.append(
            {
                "event_date": "2026-02-01",
                "player": player,
                "market": "player_points",
                "actual": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _valid_quotes(*, missing_team: bool = False) -> pd.DataFrame:
    event_start = pd.Timestamp("2026-02-01T00:00:00Z")
    players = ["Alpha", "Beta", "Gamma", "Delta"]
    base_lines = np.asarray([24.0, 20.0, 16.0, 12.0])
    movements = np.asarray([1.0, -0.5, 0.25, -0.25])
    rows = []
    for checkpoint_index, offset in enumerate(ALLOCATION_PATH_PROTOCOL.checkpoints_minutes):
        checkpoint = event_start + pd.Timedelta(minutes=offset)
        for player_index, player in enumerate(players):
            line = base_lines[player_index] + movements[player_index] * checkpoint_index / 4.0
            for book in ("book_a", "book_b"):
                rows.append(
                    {
                        "event_id": "event_1",
                        "event_start_time_utc": event_start,
                        "snapshot_time_utc": checkpoint,
                        "player": player,
                        "team": None if missing_team else "AAA",
                        "market": "player_points",
                        "line": line + (0.1 if book == "book_b" else 0.0),
                        "book": book,
                    }
                )
    rows.append(
        {
            "event_id": "event_1",
            "event_start_time_utc": event_start,
            "snapshot_time_utc": event_start - pd.Timedelta(minutes=4),
            "player": "Alpha",
            "team": None if missing_team else "AAA",
            "market": "player_points",
            "line": 100.0,
            "book": "book_a",
        }
    )
    return pd.DataFrame(rows)


def test_selected_probability_correctly_inverts_unders() -> None:
    result = selected_probability(
        pd.Series([0.72, 0.72]), pd.Series(["OVER", "UNDER"])
    )
    assert result.tolist() == [0.72, 0.28]


def test_frozen_selector_is_date_safe_and_one_prop_per_player() -> None:
    players = ["Alpha", "Beta", "Gamma", "Delta", "Echo"]
    candidates = pd.DataFrame(
        [
            {
                "event_date": "2026-02-01",
                "player": player,
                "market": "player_points",
                "side": "OVER",
                "line": 20.0,
                "p_over": 0.78 - index * 0.01,
            }
            for index, player in enumerate(players)
        ]
        + [
            {
                "event_date": "2026-02-01",
                "player": "Alpha",
                "market": "player_assists",
                "side": "UNDER",
                "line": 6.5,
                "p_over": 0.20,
            }
        ]
    )
    history = _selector_history(players)
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                [
                    {
                        "event_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                        "player": "Alpha",
                        "market": "player_assists",
                        "actual": 8.0,
                    }
                    for day in range(1, 22)
                ]
            ),
        ],
        ignore_index=True,
    )
    scored = score_frozen_selector(candidates, history)
    alpha_points = scored.loc[
        scored["player"].eq("Alpha") & scored["market"].eq("player_points")
    ].iloc[0]
    assert int(alpha_points["state_history_n"]) == 20
    assert float(alpha_points["state_empirical_rate"]) == 1.0

    selection = select_frozen_board(candidates, history)
    assert selection.selector_version == "ROBUST_STATE_INTERSECTION_Q25_V1"
    assert len(selection.control_parlay) == 4
    assert selection.control_parlay["player"].nunique() == 4
    assert selection.published is True


def test_allocation_path_uses_only_fresh_past_quotes() -> None:
    result = build_allocation_paths(_valid_quotes())
    assert result.quality_ledger["status"].tolist() == [PathQualityStatus.VALID.value]
    assert len(result.event_features) == 1
    assert len(result.player_features) == 4
    alpha = result.player_features.loc[result.player_features["player"].eq("Alpha")].iloc[0]
    assert float(alpha["close_line"]) < 30.0
    assert float(result.event_features["max_quote_age_minutes"].iloc[0]) == 0.0
    assert int(result.event_features["minimum_engine_count"].iloc[0]) == 2


def test_allocation_path_rejects_missing_pregame_team_identity() -> None:
    result = build_allocation_paths(_valid_quotes(missing_team=True))
    assert result.player_features.empty
    assert result.quality_ledger["status"].tolist() == [
        PathQualityStatus.TEAM_IDENTITY_MISSING.value
    ]


def test_allocation_path_rejects_stale_final_checkpoint() -> None:
    quotes = _valid_quotes()
    event_start = pd.Timestamp("2026-02-01T00:00:00Z")
    final_checkpoint = event_start - pd.Timedelta(minutes=5)
    quotes = quotes.loc[quotes["snapshot_time_utc"] != final_checkpoint]
    result = build_allocation_paths(quotes)
    assert result.player_features.empty
    assert result.quality_ledger["status"].tolist() == [
        PathQualityStatus.MISSING_CHECKPOINT.value
    ]


def test_book_skins_on_one_engine_do_not_create_false_consensus() -> None:
    result = build_allocation_paths(
        _valid_quotes(), engine_map={"book_a": "shared", "book_b": "shared"}
    )
    assert result.player_features.empty
    assert result.quality_ledger["status"].tolist() == [
        PathQualityStatus.INSUFFICIENT_ENGINES.value
    ]


def test_missing_actual_invalidates_entire_unit() -> None:
    paths = build_allocation_paths(_valid_quotes())
    outcomes = pd.DataFrame(
        [
            {"event_id": "event_1", "team": "AAA", "player": player, "actual": actual}
            for player, actual in [("Alpha", 28), ("Beta", 18), ("Gamma", 17)]
        ]
    )
    settled = attach_realized_allocations(paths.player_features, outcomes)
    assert settled.settled_player_features.empty
    assert settled.quality_ledger["status"].tolist() == [PathQualityStatus.MISSING_ACTUAL.value]


def test_practical_effect_gate_requires_more_than_a_tiny_improvement() -> None:
    fast_protocol = replace(
        ALLOCATION_PATH_PROTOCOL,
        bootstrap_samples=2_000,
        sign_flip_samples=5_000,
    )
    below = evaluate_improvement_sequence(
        np.full(20, 0.0049), protocol=fast_protocol
    )
    above = evaluate_improvement_sequence(
        np.full(20, 0.0060), protocol=fast_protocol
    )
    assert below["passed"] is False
    assert above["passed"] is True


def test_confirmation_is_same_day_safe_and_clusters_teams_by_game() -> None:
    first_team = generate_synthetic_settled_paths(events=28, seed=1234)
    second_team = first_team.copy()
    second_team["team"] = "SYN_B"
    second_team["unit_id"] = second_team["unit_id"].str.replace(
        "::SYN::", "::SYN_B::", regex=False
    )
    result = chronological_confirmation(pd.concat([first_team, second_team], ignore_index=True))
    assert len(result.event_evaluations) == 8
    assert int(result.event_evaluations["training_events"].min()) == 20
    assert result.report["statistical_unit"] == "game_event"


def test_chain_is_blocked_without_path_certificate_and_stays_shadow_after_gate() -> None:
    reservoir = pd.DataFrame(
        [
            {
                "event_id": "event_1",
                "team": "AAA",
                "player": player,
                "market": "player_points",
                "side": "OVER" if index % 2 == 0 else "UNDER",
                "robust_score": 0.80 - index * 0.01,
                "selected_probability": 0.82 - index * 0.01,
            }
            for index, player in enumerate(["Alpha", "Beta", "Gamma", "Delta", "Echo"])
        ]
    )
    path = reservoir[["event_id", "team", "player"]].copy()
    path["delta_share"] = [0.02, -0.02, 0.01, -0.01, 0.005]
    path["player_path_efficiency"] = 0.8
    path["direction_reversals"] = 0

    blocked = resolve_conditional_chain(
        reservoir,
        path,
        {"status": "INSUFFICIENT_REAL_PATH_EVENTS", "path_authorized": False},
    )
    assert blocked.status == "PATH_NOT_CERTIFIED"
    assert blocked.shadow_chain.empty

    shadow = resolve_conditional_chain(
        reservoir,
        path,
        {"status": "PATH_INCREMENTAL_VALUE_SUPPORTED", "path_authorized": True},
    )
    assert shadow.status == "PATH_POLICY_DEVELOPMENT_SHADOW"
    assert len(shadow.shadow_chain) == 4
    assert shadow.shadow_chain["event_id"].nunique() == 1
    assert shadow.publication_authorized is False


def test_conditional_extension_model_uses_only_prefix_surviving_rows() -> None:
    rows = []
    players = ["Alpha", "Beta", "Gamma", "Delta"]
    for decision_index in range(12):
        for index, player in enumerate(players):
            rows.append(
                {
                    "slate_date": f"2026-01-{decision_index + 1:02d}",
                    "decision_id": f"decision_{decision_index}",
                    "leg_order": index + 1,
                    "hit": 0 if index == 3 and decision_index % 3 == 0 else 1,
                    "event_id": f"event_{decision_index}",
                    "team": "AAA",
                    "player": player,
                    "market": "player_points",
                    "side": "OVER" if index % 2 == 0 else "UNDER",
                    "robust_score": 0.82 - index * 0.02,
                    "selected_probability": 0.84 - index * 0.02,
                    "delta_share": 0.02 if index % 2 == 0 else -0.02,
                    "path_support": 0.75 - index * 0.05,
                }
            )
    extension_ledger = build_extension_training_ledger(pd.DataFrame(rows))
    assert len(extension_ledger) == 36
    assert int(extension_ledger["prefix_survived"].sum()) == 36
    model = ConditionalExtensionModel(minimum_training_rows=30).fit(extension_ledger)
    assert model.fitted is True
    assert model.training_rows == 36

    reservoir = pd.DataFrame(rows[:4]).drop(columns=["slate_date", "decision_id", "leg_order", "hit", "path_support", "delta_share"])
    path = pd.DataFrame(rows[:4])[
        ["event_id", "team", "player", "delta_share"]
    ].copy()
    path["player_path_efficiency"] = 0.8
    path["direction_reversals"] = 0
    resolved = resolve_conditional_chain(
        reservoir,
        path,
        {"status": "PATH_INCREMENTAL_VALUE_SUPPORTED", "path_authorized": True},
        extension_model=model,
    )
    assert resolved.status == "CONDITIONAL_EXTENSION_MODEL_SHADOW"
    assert resolved.diagnostics["conditional_extension_model_used"] is True
    assert resolved.publication_authorized is False


def test_executable_freeze_hash_changes_with_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = build_freeze_manifest(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = build_freeze_manifest(tmp_path)
    assert first["protocol_sha256"] == second["protocol_sha256"]
    assert first["executable_bundle_sha256"] != second["executable_bundle_sha256"]
