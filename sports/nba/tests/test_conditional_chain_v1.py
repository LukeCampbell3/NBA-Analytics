from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from sports.nba.conditional_chain.allocation_path import (
    PathQualityStatus,
    attach_realized_allocations,
    build_allocation_paths,
)
from sports.nba.conditional_chain.authorization import (
    QuoteEvidenceStatus,
    assess_quote_evidence,
    authorize_parlay,
)
from sports.nba.conditional_chain.binary_path_audit import (
    exhaustive_intersection_theorem_audit,
    run_binary_path_sensitivity_audit,
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
from sports.nba.conditional_chain.core_policy import (
    fit_rank_reliability_core,
    select_frozen_core,
)
from sports.nba.conditional_chain.freeze import build_freeze_manifest
from sports.nba.conditional_chain.frozen_selector import (
    score_frozen_selector,
    select_frozen_board,
    selected_probability,
)
from sports.nba.conditional_chain.protocol import (
    ALLOCATION_PATH_PROTOCOL,
    BINARY_OUTCOME_SET_PROTOCOL,
    FROZEN_SELECTOR_PROTOCOL,
    SURVIVAL_BUILDER_PROTOCOL,
)
from sports.nba.conditional_chain.outcome_set_backtest import (
    chronological_outcome_set_replay,
)
from sports.nba.conditional_chain.outcome_worlds import (
    apply_candidate_evidence_path,
    apply_joint_world_evidence_path,
    build_binary_outcome_set,
    build_world_distribution,
    certify_perfect_parlay,
    conformal_aps_threshold,
    enumerate_binary_worlds,
    guaranteed_winner_indices,
    search_parlay_proof_frontier,
)
from sports.nba.conditional_chain.proof_trajectory import (
    build_proof_trajectory,
    certificate_world_ceiling,
    minimum_support_contraction_bits,
)
from sports.nba.conditional_chain.research_replay import adapt_master_research_ledger
from sports.nba.conditional_chain.snapshot_ledger import MarketSnapshotLedger
from sports.nba.conditional_chain.survival_backtest import (
    chronological_survival_replay,
    combine_survival_replays,
)
from sports.nba.conditional_chain.survival_builder import (
    build_survival_parlays,
    score_recent_regime_candidates,
)
from sports.nba.conditional_chain.synthetic_audit import (
    generate_synthetic_settled_paths,
)


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
    for checkpoint_index, offset in enumerate(
        ALLOCATION_PATH_PROTOCOL.checkpoints_minutes
    ):
        checkpoint = event_start + pd.Timedelta(minutes=offset)
        for player_index, player in enumerate(players):
            line = (
                base_lines[player_index]
                + movements[player_index] * checkpoint_index / 4.0
            )
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
    result = selected_probability(pd.Series([0.72, 0.72]), pd.Series(["OVER", "UNDER"]))
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
                        "event_date": pd.Timestamp("2026-01-01")
                        + pd.Timedelta(days=day),
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
    alpha = result.player_features.loc[
        result.player_features["player"].eq("Alpha")
    ].iloc[0]
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
    assert settled.quality_ledger["status"].tolist() == [
        PathQualityStatus.MISSING_ACTUAL.value
    ]


def test_practical_effect_gate_requires_more_than_a_tiny_improvement() -> None:
    fast_protocol = replace(
        ALLOCATION_PATH_PROTOCOL,
        bootstrap_samples=2_000,
        sign_flip_samples=5_000,
    )
    below = evaluate_improvement_sequence(np.full(20, 0.0049), protocol=fast_protocol)
    above = evaluate_improvement_sequence(np.full(20, 0.0060), protocol=fast_protocol)
    assert below["passed"] is False
    assert above["passed"] is True


def test_confirmation_is_same_day_safe_and_clusters_teams_by_game() -> None:
    first_team = generate_synthetic_settled_paths(events=28, seed=1234)
    second_team = first_team.copy()
    second_team["team"] = "SYN_B"
    second_team["unit_id"] = second_team["unit_id"].str.replace(
        "::SYN::", "::SYN_B::", regex=False
    )
    result = chronological_confirmation(
        pd.concat([first_team, second_team], ignore_index=True)
    )
    assert len(result.event_evaluations) == 8
    assert int(result.event_evaluations["training_events"].min()) == 20
    assert result.report["statistical_unit"] == "game_event"


def test_chain_is_blocked_without_path_certificate_and_stays_shadow_after_gate() -> (
    None
):
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

    reservoir = pd.DataFrame(rows[:4]).drop(
        columns=[
            "slate_date",
            "decision_id",
            "leg_order",
            "hit",
            "path_support",
            "delta_share",
        ]
    )
    path = pd.DataFrame(rows[:4])[["event_id", "team", "player", "delta_share"]].copy()
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


def _verified_parlay_quotes() -> pd.DataFrame:
    rows = []
    for index, player in enumerate(["Alpha", "Beta", "Gamma", "Delta"]):
        rows.append(
            {
                "event_id": f"event_{index}",
                "event_start_time_utc": "2026-02-01T20:00:00Z",
                "snapshot_time_utc": "2026-02-01T19:45:00Z",
                "player": player,
                "market": "player_points",
                "side": "OVER",
                "line": 10.5 + index,
                "book": "fanduel",
                "decimal_odds": 1.91,
                "source": "provider_a",
                "raw_source_hash": "a" * 64,
                "parser_version": "parser_v1",
                "policy_version": "CHAIN_V1",
                "model_version": "MODEL_V1",
                "path_representation_version": "PATH_REP_V1",
                "feature_cutoff_utc": "2026-02-01T19:40:00Z",
                "lineup_state": "CONFIRMED",
                "player_state": "ACTIVE",
                "identity_status": "MATCHED",
                "support_status": "IN_SUPPORT",
                "exposure_status": "PASS",
                "eligible_by_input_rules": True,
            }
        )
    return pd.DataFrame(rows)


def test_quote_evidence_requires_fresh_auditable_book_price() -> None:
    verified = assess_quote_evidence(
        _verified_parlay_quotes(), qualification_time="2026-02-01T19:50:00Z"
    )
    assert (
        verified["quote_evidence_status"]
        .eq(QuoteEvidenceStatus.VERIFIED_EXECUTABLE_QUOTE.value)
        .all()
    )
    assert verified["odds_validated_as_true"].all()

    synthetic = _verified_parlay_quotes().drop(
        columns=["book", "source", "raw_source_hash", "parser_version"]
    )
    rejected = assess_quote_evidence(
        synthetic, qualification_time="2026-02-01T19:50:00Z"
    )
    assert not rejected["odds_validated_as_true"].any()


def test_parlay_authorization_belongs_to_exact_policy_not_candidate_score() -> None:
    quotes = _verified_parlay_quotes()
    active_certificate = {
        "certificate_id": "CHAIN_V1_PROSPECTIVE_001",
        "certificate_status": "ACTIVE",
        "policy_version": "CHAIN_V1",
        "eligible_for_candidate_authorization": True,
        "scope": {
            "league": "NBA",
            "leg_count": 4,
            "markets": ["player_points"],
            "books": ["fanduel"],
            "minimum_decimal_odds": 1.80,
            "maximum_decimal_odds": 2.10,
            "model_version": "MODEL_V1",
        },
        "evidence": {
            "resolved_action_slates": 50,
            "resolved_selections": 200,
            "slate_coverage": 0.50,
        },
        "evaluation": {
            "anytime_valid_return_lcb": 0.03,
            "deployment_margin": 0.01,
        },
        "support": {"current_status": "IN_SUPPORT"},
        "shift": {"current_status": "TOLERABLE"},
    }
    path_certificate = {
        "status": "PATH_INCREMENTAL_VALUE_SUPPORTED",
        "path_authorized": True,
        "representation_version": "PATH_REP_V1",
    }
    authorized = authorize_parlay(
        quotes,
        qualification_time="2026-02-01T19:50:00Z",
        active_policy_version="CHAIN_V1",
        policy_certificate=active_certificate,
        path_certificate=path_certificate,
    )
    assert authorized.authorized is True
    assert authorized.staking_enabled is False

    core_certificate = {
        **active_certificate,
        "certificate_id": "CHAIN_V1_CORE_PROSPECTIVE_001",
        "scope": {**active_certificate["scope"], "leg_count": 2},
        "evidence": {
            "resolved_action_slates": 50,
            "resolved_selections": 100,
            "slate_coverage": 0.50,
        },
    }
    core_authorized = authorize_parlay(
        quotes.head(2),
        qualification_time="2026-02-01T19:50:00Z",
        active_policy_version="CHAIN_V1",
        policy_certificate=core_certificate,
        path_certificate=path_certificate,
    )
    assert core_authorized.authorized is True

    blocked = authorize_parlay(
        quotes,
        qualification_time="2026-02-01T19:50:00Z",
        active_policy_version="CHAIN_V1",
        policy_certificate=active_certificate,
        path_certificate={"status": "INSUFFICIENT_REAL_PATH_EVENTS"},
    )
    assert blocked.authorized is False
    assert "PATH_INCREMENTAL_VALUE_NOT_CERTIFIED" in blocked.reasons


def test_master_research_adapter_rejects_under_probability_mismatch() -> None:
    row = {
        "date": "2026-01-01",
        "player": "Alpha",
        "target": "PTS",
        "direction": "UNDER",
        "market_line": 20.2,
        "prediction": 0.20,
        "actual": 18.0,
        "source": "training_ledger",
        "edge_kind": "probability",
        "selected_probability": 0.80,
    }
    adapted = adapt_master_research_ledger(pd.DataFrame([row]))
    assert len(adapted) == 1
    row["selected_probability"] = 0.20
    with np.testing.assert_raises(ValueError):
        adapt_master_research_ledger(pd.DataFrame([row]))


def test_market_snapshot_ledger_is_idempotent_and_append_only(tmp_path: Path) -> None:
    ledger = MarketSnapshotLedger(tmp_path / "quotes.sqlite")
    row = {
        "slate_id": "2026-02-01",
        "event_id": "event_1",
        "game_id": "game_1",
        "sport": "NBA",
        "event_start_time_utc": "2026-02-01T20:00:00Z",
        "snapshot_time_utc": "2026-02-01T19:45:00Z",
        "player_id": "player_1",
        "player_name": "Alpha",
        "market": "player_points",
        "line": 20.5,
        "side": "OVER",
        "book": "fanduel",
        "engine": "fanduel",
        "decimal_odds": 1.91,
        "source": "provider_a",
        "raw_source_hash": "b" * 64,
        "parser_version": "parser_v1",
    }
    assert ledger.append([row]) == 1
    assert ledger.append([row]) == 0
    assert len(ledger.rows()) == 1
    with sqlite3.connect(ledger.path) as connection:
        with np.testing.assert_raises(sqlite3.IntegrityError):
            connection.execute("UPDATE market_snapshots SET line = 21.5")


def test_rank_reliability_core_is_date_safe_and_version_locked() -> None:
    rows = []
    for day in range(1, 25):
        for rank in range(1, 5):
            hit = int(
                rank == 1
                or (rank == 4 and day > 2)
                or (rank == 2 and day > 6)
                or (rank == 3 and day > 8)
            )
            rows.append(
                {
                    "event_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
                    "rank": rank,
                    "leg_result": hit,
                    "selector_version": "SELECTOR_V1",
                    "model_version": "MODEL_V1",
                    "player": f"Player_{rank}",
                }
            )
    reservoir = pd.DataFrame(rows)
    policy = fit_rank_reliability_core(
        reservoir,
        source_policy_version="SELECTOR_V1",
        source_model_version="MODEL_V1",
        training_cutoff="2026-02-01",
    )
    assert policy.status == "FROZEN_SHADOW"
    assert policy.ranks == (1, 4)

    today = reservoir.loc[reservoir["event_date"].eq(pd.Timestamp("2026-01-25"))]
    selected = select_frozen_core(today, policy)
    assert selected["rank"].tolist() == [1, 4]
    assert not selected["publication_authorized"].any()

    mismatched = today.assign(selector_version="SELECTOR_V2")
    with np.testing.assert_raises(ValueError):
        select_frozen_core(mismatched, policy)


def _survival_history() -> pd.DataFrame:
    rows = []
    for day in range(1, 22):
        event_date = pd.Timestamp("2026-01-10") + pd.Timedelta(days=day)
        rows.extend(
            [
                {
                    "event_date": event_date,
                    "market": "player_assists",
                    "side": "UNDER",
                    "leg_result": 1.0,
                },
                {
                    "event_date": event_date,
                    "market": "player_points",
                    "side": "OVER",
                    "leg_result": float(day % 3 == 0),
                },
            ]
        )
    return pd.DataFrame(rows)


def _survival_reservoir(event_date: str = "2026-02-01") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_date": event_date,
                "player": "Assist A",
                "market": "player_assists",
                "side": "UNDER",
                "robust_score": 0.72,
                "rank": 1,
                "leg_result": 1.0,
                "model_version": "MODEL_A",
            },
            {
                "event_date": event_date,
                "player": "Assist B",
                "market": "player_assists",
                "side": "UNDER",
                "robust_score": 0.71,
                "rank": 2,
                "leg_result": 1.0,
                "model_version": "MODEL_A",
            },
            {
                "event_date": event_date,
                "player": "Points A",
                "market": "player_points",
                "side": "OVER",
                "robust_score": 0.74,
                "rank": 3,
                "leg_result": 0.0,
                "model_version": "MODEL_A",
            },
            {
                "event_date": event_date,
                "player": "Points B",
                "market": "player_points",
                "side": "OVER",
                "robust_score": 0.73,
                "rank": 4,
                "leg_result": 0.0,
                "model_version": "MODEL_A",
            },
        ]
    )


def test_survival_scoring_is_date_safe_and_recent_regime_aware() -> None:
    reservoir = _survival_reservoir()
    history = _survival_history()
    scored = score_recent_regime_candidates(reservoir, history)
    same_day = reservoir[["event_date", "market", "side", "leg_result"]].copy()
    contaminated = score_recent_regime_candidates(
        reservoir,
        pd.concat([history, same_day], ignore_index=True),
    )
    np.testing.assert_allclose(
        scored["survival_probability"], contaminated["survival_probability"]
    )
    assist_score = scored.loc[
        scored["market"].eq("player_assists"), "survival_probability"
    ].min()
    points_score = scored.loc[
        scored["market"].eq("player_points"), "survival_probability"
    ].max()
    assert assist_score > points_score
    assert scored["regime_history_end_exclusive"].eq(pd.Timestamp("2026-02-01")).all()


def test_survival_builder_is_model_version_invariant_and_rejects_four_legs() -> None:
    reservoir = _survival_reservoir()
    history = _survival_history()
    first = build_survival_parlays(reservoir, history)
    second = build_survival_parlays(reservoir.assign(model_version="MODEL_B"), history)
    assert first.primary_parlay["player"].tolist() == ["Assist A", "Assist B"]
    assert (
        first.primary_parlay["player"].tolist()
        == second.primary_parlay["player"].tolist()
    )
    assert set(first.alternatives) == {2, 3}
    assert 4 not in first.alternatives
    assert (
        first.diagnostics["four_leg_status"] == "REJECTED_NO_CROSS_VERSION_IMPROVEMENT"
    )
    assert first.publication_authorized is False


def test_survival_replay_uses_one_decision_per_slate_after_warmup() -> None:
    rows = []
    for day in range(25):
        event_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        slate = _survival_reservoir(str(event_date.date()))
        slate["leg_result"] = np.where(slate["market"].eq("player_assists"), 1.0, 0.0)
        rows.append(slate)
    replay = chronological_survival_replay(
        pd.concat(rows, ignore_index=True),
        block_label="test",
        warmup_slates=SURVIVAL_BUILDER_PROTOCOL.minimum_warmup_slates,
    )
    assert replay.report["evaluation_slates"] == 5
    assert len(replay.decisions) == 10
    assert replay.decisions.groupby("leg_count")["event_date"].nunique().to_dict() == {
        2: 5,
        3: 5,
    }
    assert replay.report["production_authorizable"] is False
    combined = combine_survival_replays([replay])
    assert combined.report["research_gate"]["status"] == "RESEARCH_GATE_NOT_PASSED"


def test_binary_world_distribution_enumerates_every_joint_state() -> None:
    outcomes = enumerate_binary_worlds(3)
    assert outcomes.shape == (8, 3)
    assert {tuple(row) for row in outcomes} == {
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    }
    distribution = build_world_distribution(["a", "b", "c"], [0.70, 0.60, 0.80])
    np.testing.assert_allclose(distribution.probabilities.sum(), 1.0)
    np.testing.assert_allclose(distribution.marginals, [0.70, 0.60, 0.80])

    full_set = build_binary_outcome_set(
        distribution,
        aps_threshold=1.0,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
    )
    assert full_set.world_count == 8
    assert guaranteed_winner_indices(full_set) == ()


def test_perfect_parlay_exists_exactly_when_world_intersection_is_large_enough() -> (
    None
):
    worlds = enumerate_binary_worlds(3)
    admissible = (worlds[:, 0] == 1) & (worlds[:, 1] == 1)
    distribution = build_world_distribution(
        ["a", "b", "c"],
        [0.70, 0.70, 0.50],
        admissible_world_mask=admissible,
    )
    outcome_set = build_binary_outcome_set(
        distribution,
        aps_threshold=1.0,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "player": ["Alpha", "Beta", "Gamma"],
            "survival_probability": [0.70, 0.70, 0.50],
        }
    )
    path_certificate = {
        "status": "PATH_INCREMENTAL_VALUE_SUPPORTED",
        "path_authorized": True,
    }
    pair = certify_perfect_parlay(
        candidates,
        outcome_set,
        requested_leg_count=2,
        path_certificate=path_certificate,
    )
    triple = certify_perfect_parlay(
        candidates,
        outcome_set,
        requested_leg_count=3,
        path_certificate=path_certificate,
    )
    assert guaranteed_winner_indices(outcome_set) == (0, 1)
    assert pair.logical_implication_proven is True
    assert pair.selected_candidate_ids == ("a", "b")
    assert pair.production_authorized is False
    assert triple.logical_implication_proven is False


def test_market_evidence_path_can_shrink_worlds_without_settlement() -> None:
    prior = build_world_distribution(["a", "b"], [0.50, 0.50])
    path = apply_candidate_evidence_path(
        prior,
        np.asarray([[10.0, 10.0], [10.0, 10.0]]),
        checkpoint_labels=["T-30", "T-5"],
    )
    assert path.distributions[-1].entropy < prior.entropy
    assert (path.distributions[-1].marginals > 0.99).all()
    outcome_set = build_binary_outcome_set(
        path.distributions[-1],
        aps_threshold=0.99,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
    )
    assert guaranteed_winner_indices(outcome_set) == (0, 1)


def test_exact_proof_frontier_exposes_every_remaining_counterexample() -> None:
    distribution = build_world_distribution(
        ["a", "b", "c"],
        [0.80, 0.75, 0.55],
    )
    outcome_set = build_binary_outcome_set(
        distribution,
        aps_threshold=0.90,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "player": ["Alpha", "Beta", "Gamma"],
            "survival_probability": [0.80, 0.75, 0.55],
        }
    )
    frontier = search_parlay_proof_frontier(
        candidates,
        outcome_set,
        requested_leg_count=2,
    )
    assert frontier.selected_candidate_ids == ("a", "b")
    assert frontier.logically_proven is False
    assert frontier.counterexample_world_count > 0
    assert 0.0 < frontier.counterexample_mass_within_set < 1.0
    assert frontier.combinations_evaluated == 3


def test_empty_outcome_set_cannot_create_a_vacuous_parlay_proof() -> None:
    distribution = build_world_distribution(["a", "b"], [0.50, 0.50])
    outcome_set = build_binary_outcome_set(
        distribution,
        aps_threshold=0.90,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "player": ["Alpha", "Beta"],
            "survival_probability": [0.50, 0.50],
        }
    )
    certificate = certify_perfect_parlay(
        candidates,
        outcome_set,
        requested_leg_count=2,
    )
    assert outcome_set.world_count == 0
    assert guaranteed_winner_indices(outcome_set) == ()
    assert certificate.logical_implication_proven is False
    assert certificate.status == "NO_ROBUST_WINNER_INTERSECTION"


def test_joint_world_path_supports_non_factorized_state_evidence() -> None:
    prior = build_world_distribution(["a", "b", "c"], [0.55, 0.55, 0.49])
    shared_state = prior.outcomes[:, 0].astype(bool) & prior.outcomes[:, 1].astype(bool)
    evidence = np.where(shared_state, 5.0, 0.0)
    path = apply_joint_world_evidence_path(
        prior,
        np.vstack([evidence, evidence]),
        checkpoint_labels=["T-30", "T-5"],
    )
    outcome_set = build_binary_outcome_set(
        path.distributions[-1],
        aps_threshold=0.90,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
    )
    assert guaranteed_winner_indices(outcome_set) == (0, 1)


def test_binary_path_sensitivity_and_intersection_theorem_audits_pass() -> None:
    theorem = exhaustive_intersection_theorem_audit(candidate_count=3)
    sensitivity = run_binary_path_sensitivity_audit(aps_threshold=0.90)
    assert theorem["theorem_checks"] == 765
    assert theorem["passed"] is True
    assert sensitivity["mechanism_passed"] is True
    assert sensitivity["scenarios"]["coherent_joint_path"]["pair_logically_proven"]
    assert not sensitivity["scenarios"]["fully_reversed_path"]["pair_logically_proven"]


def test_certificate_cardinality_bounds_are_exact_for_top_ten() -> None:
    assert certificate_world_ceiling(10, 2) == 256
    assert certificate_world_ceiling(10, 3) == 128
    assert certificate_world_ceiling(10, 4) == 64
    assert np.isclose(minimum_support_contraction_bits(430.7, 256), 0.7501, atol=1e-3)


def test_proof_trajectory_tracks_fixed_counterexample_elimination_and_reversal() -> (
    None
):
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "player": ["Alpha", "Beta", "Gamma", "Delta"],
            "survival_probability": [0.61, 0.57, 0.53, 0.49],
        }
    )
    prior = build_world_distribution(
        candidates["candidate_id"], candidates["survival_probability"]
    )
    shared_pair = prior.outcomes[:, 0].astype(bool) & prior.outcomes[:, 1].astype(bool)
    evidence = np.where(shared_pair, 5.0, 0.0)
    path = apply_joint_world_evidence_path(
        prior,
        np.vstack([evidence, evidence, -2.0 * evidence]),
        checkpoint_labels=["T-30", "T-5", "REVERSAL"],
    )
    trajectory = build_proof_trajectory(
        candidates,
        path,
        aps_thresholds=0.90,
        calibration_slates=BINARY_OUTCOME_SET_PROTOCOL.minimum_calibration_slates,
        fixed_targets={2: ("a", "b")},
    )
    rows = trajectory.diagnostics.set_index("checkpoint")
    assert trajectory.threshold_mode == "FIXED_MECHANISM_THRESHOLD"
    assert rows.loc["prior", "retained_world_count"] == 13
    assert rows.loc["T-5", "retained_world_count"] == 3
    assert rows.loc["T-5", "2_leg_fixed_logical_certificate"]
    assert rows.loc["T-5", "2_leg_fixed_counterexample_world_count"] == 0
    assert rows.loc["T-5", "2_leg_minimum_counterexample_world_count"] == 0
    assert rows.loc["T-5", "2_leg_minimum_counterexample_mass"] == 0.0
    assert rows.loc["T-30", "2_leg_fixed_counterexamples_eliminated_since_prior"] > 0
    assert (
        rows.loc["REVERSAL", "2_leg_fixed_counterexamples_eliminated_since_prior"] < 0
    )
    assert not rows.loc["REVERSAL", "2_leg_fixed_logical_certificate"]


def test_joint_outcome_replay_waits_for_prior_calibration_slates() -> None:
    rows = []
    for day in range(25):
        event_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        slate = _survival_reservoir(str(event_date.date()))
        slate["leg_result"] = np.where(slate["market"].eq("player_assists"), 1.0, 0.0)
        rows.append(slate)
    replay = chronological_outcome_set_replay(
        pd.concat(rows, ignore_index=True),
        block_label="test",
    )
    assert int(replay.decisions["evaluated"].sum()) == 5
    assert replay.report["evaluated_slates"] == 5
    assert (
        replay.report["ex_post_oracle_feasibility_by_leg_count"]["2"]["feasible_slates"]
        == 5
    )
    assert (
        replay.report["ex_post_oracle_feasibility_by_leg_count"]["3"]["feasible_slates"]
        == 0
    )
    pair_structure = replay.report["structural_certificate_feasibility_by_leg_count"][
        "2"
    ]
    assert pair_structure["world_ceiling_at_maximum_reservoir"] == 256
    assert pair_structure["necessary_condition"] == "0 < |C| <= 2^(M-n)"
    assert len(replay.calibration_scores) == 25
    threshold = conformal_aps_threshold([0.2] * 20)
    assert np.isclose(threshold, 0.2)
