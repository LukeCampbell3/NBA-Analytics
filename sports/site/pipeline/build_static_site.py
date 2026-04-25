#!/usr/bin/env python3
"""
Build the multi-sport static site bundle.

This builder creates a site root with:
1. A shared landing page from `sports/site/web/`.
2. One subdirectory per sport discovered under `sports/*/web/`.
3. Clean-route folders for every copied HTML page.
4. A generated `data/sports.json` manifest used by the landing page.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
SITE_ROOT = SCRIPT_PATH.parents[1]
SPORTS_ROOT = SITE_ROOT.parent
REPO_ROOT = SPORTS_ROOT.parent
DEFAULT_SOURCE_DIR = SITE_ROOT / "web"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist"


def nonempty_html_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.html")):
        if path.is_file() and path.stat().st_size > 0:
            yield path


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
        print(f"[route] {html_file.relative_to(out_dir)} -> {route_index.relative_to(out_dir)}")


def read_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else []


def write_json(path: Path, value: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def player_identity_key(record: Dict) -> str:
    player = record.get("player", {}) if isinstance(record, dict) else {}
    pid = str(player.get("id", "")).strip()
    if pid:
        return f"id:{pid}"
    name = str(player.get("name", "")).strip().lower()
    season = str(player.get("season", "")).strip()
    team = str(player.get("team", "")).strip().lower()
    return f"name:{name}|season:{season}|team:{team}"


def card_value_score(card: Dict) -> float:
    if not isinstance(card, dict):
        return 0.0
    metrics = card.get("value_metrics", {})
    try:
        return float(metrics.get("player_value_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
        return

    top_cards = sorted(cards, key=card_value_score, reverse=True)[:limit]
    keep_keys = {player_identity_key(card) for card in top_cards}
    top_vals = [row for row in vals if player_identity_key(row) in keep_keys] if vals else []

    write_json(cards_path, top_cards)
    if vals_path.exists():
        write_json(vals_path, top_vals)

    print(
        f"[trim] {data_dir.parent.name} college payload {original_count} -> {len(top_cards)} cards "
        f"(valuations {len(vals)} -> {len(top_vals)})"
    )


def slug_to_title(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").upper()


def titleize_stem(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").title()


def load_site_metadata(sport_dir: Path) -> Dict[str, object]:
    metadata_path = sport_dir / "site.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def discover_pages(slug: str, source_dir: Path, metadata: Dict[str, object]) -> List[Dict[str, str]]:
    route_labels = metadata.get("route_labels", {})
    route_labels = route_labels if isinstance(route_labels, dict) else {}
    pages: List[Dict[str, str]] = []

    html_files = sorted(
        nonempty_html_files(source_dir),
        key=lambda path: (0 if path.stem.lower() == "index" else 1, path.name.lower()),
    )

    for html_file in html_files:
        stem = html_file.stem.lower()
        label = str(route_labels.get(stem) or ("Overview" if stem == "index" else titleize_stem(stem)))
        href = f"/{slug}/" if stem == "index" else f"/{slug}/{stem}/"
        pages.append({
            "slug": stem,
            "label": label,
            "href": href,
        })

    return pages


def discover_sports() -> List[Dict[str, object]]:
    sports: List[Dict[str, object]] = []

    for sport_dir in sorted(SPORTS_ROOT.iterdir()):
        if not sport_dir.is_dir() or sport_dir.name == "site":
            continue

        source_dir = sport_dir / "web"
        index_path = source_dir / "index.html"
        if not index_path.exists():
            continue

        slug = sport_dir.name.lower()
        metadata = load_site_metadata(sport_dir)
        pages = discover_pages(slug, source_dir, metadata)
        sports.append(
            {
                "slug": slug,
                "source_dir": source_dir,
                "title": str(metadata.get("title") or slug_to_title(slug)),
                "tagline": str(metadata.get("tagline") or "Sport workspace"),
                "summary": str(metadata.get("summary") or "Sport pages available in this workspace."),
                "status": str(metadata.get("status") or "planned"),
                "status_label": str(metadata.get("status_label") or str(metadata.get("status") or "planned").title()),
                "accent": str(metadata.get("accent") or "#2563eb"),
                "surface": str(metadata.get("surface") or "#0f172a"),
                "pages": pages,
            }
        )

    return sports


def build_static_site(
    landing_source_dir: Path,
    output_dir: Path,
    college_card_limit: Optional[int],
) -> int:
    if not landing_source_dir.exists():
        print(f"Error: landing source directory not found: {landing_source_dir}")
        return 1

    sports = discover_sports()
    if not sports:
        print("Error: no sport web directories were discovered under sports/*/web/")
        return 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(landing_source_dir, output_dir)
    print(f"[copy] landing {landing_source_dir} -> {output_dir}")

    manifest: List[Dict[str, object]] = []
    for sport in sports:
        slug = str(sport["slug"])
        source_dir = Path(sport["source_dir"])
        sport_output = output_dir / slug
        shutil.copytree(source_dir, sport_output)
        trim_college_payload(sport_output / "data", college_card_limit)
        create_clean_routes(sport_output)
        print(f"[copy] sport {slug}: {source_dir} -> {sport_output}")

        manifest.append(
            {
                "slug": slug,
                "title": sport["title"],
                "tagline": sport["tagline"],
                "summary": sport["summary"],
                "status": sport["status"],
                "status_label": sport["status_label"],
                "accent": sport["accent"],
                "surface": sport["surface"],
                "pages": sport["pages"],
                "entry_href": f"/{slug}/",
            }
        )

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(data_dir / "sports.json", manifest)
    create_clean_routes(output_dir)

    print("\n[SUCCESS] Multi-sport static site build complete.")
    print(f"Output directory: {output_dir}")
    print("Included sports:", ", ".join(item["slug"] for item in manifest))
    print("\nQuick local preview:")
    print(f"  python -m http.server 8000 --directory \"{output_dir}\"")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the multi-sport site bundle from sports/site/web/")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Landing source web directory (default: {DEFAULT_SOURCE_DIR})",
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
        help="Max college cards to include for sports that ship college payloads (default: 1200, use 0 to disable)",
    )
    args = parser.parse_args()

    effective_limit: Optional[int] = args.college_card_limit
    if effective_limit is not None and effective_limit <= 0:
        effective_limit = None

    return build_static_site(
        landing_source_dir=Path(args.source).resolve(),
        output_dir=Path(args.output).resolve(),
        college_card_limit=effective_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
