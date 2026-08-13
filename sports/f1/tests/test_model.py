from __future__ import annotations

from sports.f1.predictions.model import build_training_rows, predict_event, train_and_evaluate


def history(races: int = 18) -> list[dict]:
    drivers = [
        ("alpha", "Alex Alpha", "team_a"),
        ("bravo", "Blake Bravo", "team_a"),
        ("charlie", "Casey Charlie", "team_b"),
        ("delta", "Drew Delta", "team_b"),
        ("echo", "Evan Echo", "team_c"),
        ("foxtrot", "Fran Foxtrot", "team_c"),
        ("golf", "Gray Golf", "team_d"),
        ("hotel", "Hayden Hotel", "team_d"),
        ("india", "Indy India", "team_e"),
        ("juliet", "Jules Juliet", "team_e"),
        ("kilo", "Kai Kilo", "team_f"),
        ("lima", "Lane Lima", "team_f"),
    ]
    rows = []
    for round_number in range(1, races + 1):
        rotated = drivers[(round_number - 1) % len(drivers):] + drivers[:(round_number - 1) % len(drivers)]
        results = []
        for finish, (driver_id, name, constructor_id) in enumerate(rotated, start=1):
            results.append({
                "driver_id": driver_id, "driver": name, "driver_number": str(finish),
                "constructor_id": constructor_id, "constructor": constructor_id,
                "grid": finish, "finish": finish, "points": max(0, 13 - finish), "dnf": finish == 12,
            })
        rows.append({
            "season": 2024 + (round_number > 9), "round": ((round_number - 1) % 9) + 1,
            "race_name": f"Race {round_number}", "date": f"2025-01-{round_number:02d}",
            "circuit_id": f"circuit_{round_number % 4}", "circuit": "Test Circuit", "results": results,
        })
    return rows


def test_features_are_pre_race_and_not_current_result_leakage() -> None:
    x, _, _, _ = build_training_rows(history(18))
    # The first race has no driver history, so all driver average-finish inputs
    # must be the identical prior value even though race results differ.
    assert len(set(x[:12, 3].tolist())) == 1
    assert len(set(x[:12, 4].tolist())) == 1


def test_model_returns_coherent_field_probabilities() -> None:
    races = history(18)
    models, state, metadata = train_and_evaluate(races)
    entries = [{
        "driver_id": row["driver_id"], "driver": row["driver"], "driver_number": row["driver_number"],
        "constructor_id": row["constructor_id"], "constructor": row["constructor"],
        "standing_position": index + 1, "grid": index + 1,
    } for index, row in enumerate(races[-1]["results"])]
    event = {"season": 2026, "round": 1, "circuit_id": "circuit_1"}
    predictions = predict_event(models, state, event, entries)
    assert len(predictions) == 12
    assert abs(sum(row["win_probability"] for row in predictions) - 1.0) < 1e-9
    assert metadata["backtest"]["holdout_races"] > 0
