from datetime import datetime
import importlib.util
from pathlib import Path
from zoneinfo import ZoneInfo


SCRIPT = Path("Player-Predictor/scripts/build_historical_prelock_availability_snapshots.py")
SPEC = importlib.util.spec_from_file_location("build_historical_prelock_availability_snapshots", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_official_pdf_text_lines_to_v95_schema_rows():
    lines = [
        "01/02/2026",
        "07:00",
        "(ET)",
        "BKN@WAS",
        "Brooklyn",
        "Nets",
        "Claxton,",
        "Nic",
        "Out",
        "Personal",
        "Reasons",
        "Mann,",
        "Terance",
        "Probable",
        "Injury/Illness",
        "-",
        "Right",
        "Hip;",
        "Contusion",
        "Washington",
        "Wizards",
        "Kispert,",
        "Corey",
        "Questionable",
        "Injury/Illness",
    ]
    frame = MODULE.parse_report_lines(
        lines,
        datetime(2026, 1, 2, 17, 30, tzinfo=ZoneInfo("America/New_York")),
        "https://example.test/report.pdf",
    )
    assert len(frame) == 3
    assert set(frame["team"]) == {"BKN", "WAS"}
    probs = dict(zip(frame["player"], frame["out_probability"]))
    assert probs["Nic_Claxton"] == 1.0
    assert probs["Terance_Mann"] == 0.15
    assert probs["Corey_Kispert"] == 0.45
    assert frame["snapshot_time"].str.endswith("+00:00").all()
    assert frame["game_start_time"].str.endswith("+00:00").all()
