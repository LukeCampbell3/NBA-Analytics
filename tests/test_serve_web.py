from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1] / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from serve_web import MultiPageRequestHandler


def make_handler(root: Path) -> MultiPageRequestHandler:
    handler = object.__new__(MultiPageRequestHandler)
    handler.directory = str(root)
    return handler


def test_trailing_slash_route_prefers_generated_directory_index(tmp_path: Path) -> None:
    route = tmp_path / "mlb" / "predictions"
    route.mkdir(parents=True)
    (route / "index.html").write_text("clean route", encoding="utf-8")
    (tmp_path / "mlb" / "predictions.html").write_text("flat route", encoding="utf-8")

    handler = make_handler(tmp_path)

    assert handler._normalize_clean_route("/mlb/predictions/") == "/mlb/predictions/index.html"
    assert handler._normalize_clean_route("/mlb/predictions") == "/mlb/predictions.html"
