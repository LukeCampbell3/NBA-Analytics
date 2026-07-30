from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = REPO_ROOT / "sports" / "site" / "pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from validate_daily_publication import validate_publication


def write_payload(path: Path, *, run_date: str, status: str = "ready") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_date": run_date,
                "publication_status": status,
                "plays": [],
            }
        ),
        encoding="utf-8",
    )


def build_static_shell(root: Path) -> None:
    for relative_path in ("dist/index.html", "dist/app.js", "dist/styles.css"):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")


def test_validate_publication_accepts_current_payloads(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    for sport in ("nba", "mlb"):
        write_payload(
            tmp_path / f"sports/{sport}/web/data/daily_predictions.json",
            run_date="2026-04-28",
        )
        write_payload(
            tmp_path / f"dist/{sport}/data/daily_predictions.json",
            run_date="2026-04-28",
        )
        route = tmp_path / f"dist/{sport}/predictions/index.html"
        route.parent.mkdir(parents=True, exist_ok=True)
        route.write_text("ok", encoding="utf-8")

    summaries = validate_publication(
        repo_root=tmp_path,
        output_dir=Path("dist"),
        run_date="2026-04-28",
        sports=["nba", "mlb"],
    )

    assert summaries == [
        "NBA: 2026-04-28, status=ready, plays=0",
        "MLB: 2026-04-28, status=ready, plays=0",
    ]


def test_validate_publication_rejects_stale_payload(tmp_path: Path) -> None:
    build_static_shell(tmp_path)
    write_payload(
        tmp_path / "sports/mlb/web/data/daily_predictions.json",
        run_date="2026-04-27",
    )
    write_payload(
        tmp_path / "dist/mlb/data/daily_predictions.json",
        run_date="2026-04-27",
    )
    route = tmp_path / "dist/mlb/predictions/index.html"
    route.parent.mkdir(parents=True, exist_ok=True)
    route.write_text("ok", encoding="utf-8")

    with pytest.raises(ValueError, match="MLB source payload is stale"):
        validate_publication(
            repo_root=tmp_path,
            output_dir=Path("dist"),
            run_date="2026-04-28",
            sports=["mlb"],
        )
