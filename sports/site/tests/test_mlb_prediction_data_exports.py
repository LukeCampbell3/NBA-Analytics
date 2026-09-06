from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "sports" / "site" / "pipeline" / "build_static_site.py"
spec = spec_from_file_location("build_static_site", MODULE_PATH)
assert spec and spec.loader
build_static_site = module_from_spec(spec)
spec.loader.exec_module(build_static_site)


def test_mlb_authoritative_payloads_are_static_export_assets(tmp_path: Path) -> None:
    sport = tmp_path / "mlb"
    data = sport / "data"
    data.mkdir(parents=True)

    required = {
        "daily_predictions.json",
        "latest_candidates.json",
        "sequential_pa_hitter_predictions.json",
    }
    for name in required:
        (data / name).write_text('{"run_date":"2026-09-06"}\n', encoding="utf-8")
    (data / "internal_only.json").write_text('{}\n', encoding="utf-8")

    build_static_site.prune_non_prediction_assets(sport)

    for name in required:
        assert (data / name).exists(), f"static builder pruned authoritative MLB payload: {name}"
    assert not (data / "internal_only.json").exists()


def test_required_mlb_payloads_are_explicitly_allowlisted() -> None:
    assert {
        "daily_predictions.json",
        "latest_candidates.json",
        "sequential_pa_hitter_predictions.json",
    }.issubset(build_static_site.PREDICTION_DATA_FILES)
