#!/usr/bin/env python3
"""
build_static_site.py - Build a deployable static site bundle.

This script:
1. Copies `web/` into a target output directory.
2. Optionally trims the college payload to a max card count (matches prior serve_web default behavior).
3. Creates clean-route folders (e.g. /about/ -> about/index.html) with base href support.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCRIPT_PATH = Path(__file__).resolve()
NBA_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_SOURCE_DIR = NBA_ROOT / "web"
DEFAULT_OUTPUT_DIR = NBA_ROOT / "dist"


def player_identity_key(record: Dict) -> str:
    """Stable key for mapping cards <-> valuations."""
    player = record.get("player", {}) if isinstance(record, dict) else {}
    pid = str(player.get("id", "")).strip()
    if pid:
        return f"id:{pid}"
    name = str(player.get("name", "")).strip().lower()
    season = str(player.get("season", "")).strip()
    team = str(player.get("team", "")).strip().lower()
    return f"name:{name}|season:{season}|team:{team}"


def card_value_score(card: Dict) -> float:
    """Sortable player value score with safe fallback."""
    if not isinstance(card, dict):
        return 0.0
    metrics = card.get("value_metrics", {})
    try:
        return float(metrics.get("player_value_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else []


def write_json(path: Path, value: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def trim_college_payload(data_dir: Path, limit: Optional[int]) -> None:
    if limit is None or limit <= 0:
        return

    cards_path = data_dir / "college_cards.json"
    vals_path = data_dir / "college_valuations.json"
    if not cards_path.exists():
        return

    cards = read_json_list(cards_path)
    vals = read_json_list(vals_path)
    original_count = len(cards)
    if original_count <= limit:
        print(f"[trim] college_cards.json under limit ({original_count} <= {limit}); no trim needed")
        return

    top_cards = sorted(cards, key=card_value_score, reverse=True)[:limit]
    keep_keys = {player_identity_key(card) for card in top_cards}
    top_vals = [row for row in vals if player_identity_key(row) in keep_keys] if vals else []

    write_json(cards_path, top_cards)
    if vals_path.exists():
        write_json(vals_path, top_vals)

    print(
        f"[trim] college payload trimmed {original_count} -> {len(top_cards)} cards "
        f"(valuations: {len(vals)} -> {len(top_vals)})"
    )


def inject_base_href(html: str, base_href: str = "../") -> str:
    lower = html.lower()
    if "<base " in lower:
        return html

    marker = "<head>"
    idx = lower.find(marker)
    if idx < 0:
        return html

    insert_at = idx + len(marker)
    return f'{html[:insert_at]}\n    <base href="{base_href}">{html[insert_at:]}'


def nonempty_html_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.html")):
        if path.is_file() and path.stat().st_size > 0:
            yield path


def normalize_home_links(directory: Path) -> None:
    """
    Ensure Home anchors always route to root (/), not index.html.
    """
    pattern = re.compile(r'(<a\b[^>]*\bhref=["\'])index\.html(["\'][^>]*>\s*Home\s*</a>)', re.IGNORECASE)
    for html_file in nonempty_html_files(directory):
        src_text = html_file.read_text(encoding="utf-8")
        normalized = pattern.sub(r'\1/\2', src_text)
        if normalized != src_text:
            html_file.write_text(normalized, encoding="utf-8")


def create_clean_routes(out_dir: Path) -> None:
    for html_file in nonempty_html_files(out_dir):
        stem = html_file.stem.lower()
        if stem == "index":
            continue

        route_dir = out_dir / stem
        route_dir.mkdir(parents=True, exist_ok=True)
        route_index = route_dir / "index.html"

        src_text = html_file.read_text(encoding="utf-8")
        route_index.write_text(inject_base_href(src_text), encoding="utf-8")
        print(f"[route] {html_file.name} -> {route_index.relative_to(out_dir)}")


def build_static_site(
    source_dir: Path,
    output_dir: Path,
    college_card_limit: Optional[int],
) -> int:
    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    print(f"[copy] {source_dir} -> {output_dir}")

    trim_college_payload(output_dir / "data", college_card_limit)
    normalize_home_links(output_dir)
    create_clean_routes(output_dir)

    print("\n[SUCCESS] Static site build complete.")
    print(f"Output directory: {output_dir}")
    print("\nQuick local preview:")
    print(f"  python -m http.server 8000 --directory \"{output_dir}\"")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deployable static bundle from sports/nba/web/")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Source web directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for static bundle (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--college-card-limit",
        type=int,
        default=1200,
        help="Max college cards to include in output (default: 1200, use 0 to disable)",
    )
    args = parser.parse_args()

    effective_limit: Optional[int] = args.college_card_limit
    if effective_limit is not None and effective_limit <= 0:
        effective_limit = None

    return build_static_site(
        source_dir=Path(args.source).resolve(),
        output_dir=Path(args.output).resolve(),
        college_card_limit=effective_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
