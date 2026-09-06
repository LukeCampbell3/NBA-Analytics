#!/usr/bin/env python3
"""Repair MLB previous-picks history selection and archive indexing.

This is intentionally narrow:
- Previous Published Picks means the newest archived board strictly before
  today's Eastern date, even when daily_predictions.json itself is stale.
- history/index.json is derived from the actual dated history files so a
  preserved slate cannot become invisible in navigation.
- deployed dist/private copies are kept byte-identical to source.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ROI_SOURCE = REPO_ROOT / "sports/mlb/web/predictions-roi-overrides.js"
PRESERVE_SCRIPT = REPO_ROOT / "sports/mlb/scripts/preserve_published_predictions.py"
PRESERVE_TEST = REPO_ROOT / "sports/mlb/tests/test_preserve_published_predictions.py"
HTML_PATHS = (
    REPO_ROOT / "sports/mlb/web/predictions.html",
    REPO_ROOT / "dist/mlb/predictions.html",
    REPO_ROOT / "paywall/private-content/app/mlb/predictions.html",
)
ROI_DEPLOY_PATHS = (
    REPO_ROOT / "dist/mlb/predictions-roi-overrides.js",
    REPO_ROOT / "paywall/private-content/app/mlb/predictions-roi-overrides.js",
)
INDEX_DEPLOY_PATHS = (
    REPO_ROOT / "dist/mlb/data/history/index.json",
    REPO_ROOT / "paywall/private-content/app/mlb/data/history/index.json",
)

OLD_SELECTOR = '''    const findPreviousBoard = async (current) => {
        const runDate = String(current?.run_date || "").trim();
        if (runDate) {
            try {
                const sameDay = await fetchJson(`data/history/${runDate}.json`);
                if (hasDistinctPicks(sameDay, current)) {
                    return { payload: sameDay, sameDay: true };
                }
            } catch (_) { /* same-day archive may not exist yet */ }
        }

        try {
            const index = await fetchJson("data/history/index.json");
            const dates = Array.isArray(index?.dates) ? index.dates.map(String) : [];
            for (const date of dates) {
                if (!date || date === runDate) continue;
                try {
                    const archived = await fetchJson(`data/history/${date}.json`);
                    if (Array.isArray(archived?.plays) && archived.plays.length) {
                        return { payload: archived, sameDay: false };
                    }
                } catch (_) { /* try the next preserved date */ }
            }
        } catch (_) { /* history index is optional */ }
        return null;
    };
'''

NEW_SELECTOR = '''    const easternDate = () => {
        const parts = new Intl.DateTimeFormat("en-US", {
            timeZone: "America/New_York",
            year: "numeric",
            month: "2-digit",
            day: "2-digit",
        }).formatToParts(new Date());
        const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
        return `${values.year}-${values.month}-${values.day}`;
    };

    const findPreviousBoard = async (current) => {
        const runDate = String(current?.run_date || "").trim();
        const today = easternDate();

        // Same-day comparison only applies when the live artifact is actually
        // today's slate. A stale live file is itself historical evidence and
        // must not cause its date to be skipped.
        if (runDate && runDate === today) {
            try {
                const sameDay = await fetchJson(`data/history/${runDate}.json`);
                if (hasDistinctPicks(sameDay, current)) {
                    return { payload: sameDay, sameDay: true };
                }
            } catch (_) { /* same-day archive may not exist yet */ }
        }

        try {
            const index = await fetchJson("data/history/index.json");
            const dates = Array.isArray(index?.dates)
                ? index.dates.map(String).filter((date) => /^\\d{4}-\\d{2}-\\d{2}$/.test(date)).sort().reverse()
                : [];
            for (const date of dates) {
                // Previous is chronological relative to today, not relative
                // to a potentially stale daily_predictions.json run_date.
                if (!date || date >= today) continue;
                try {
                    const archived = await fetchJson(`data/history/${date}.json`);
                    if (Array.isArray(archived?.plays) && archived.plays.length) {
                        return { payload: archived, sameDay: false };
                    }
                } catch (_) { /* try the next preserved date */ }
            }
        } catch (_) { /* history index is optional */ }

        // If the index itself is stale, prefer the older live artifact over
        // jumping back another calendar day.
        if (runDate && runDate < today && Array.isArray(current?.plays) && current.plays.length) {
            return { payload: current, sameDay: false };
        }
        return null;
    };
'''

INDEX_HELPER = '''def refresh_history_index(history_dir: Path) -> Path:
    """Derive user-facing history navigation from the dated archive files."""
    history_dir.mkdir(parents=True, exist_ok=True)
    dates: list[str] = []
    for path in history_dir.glob("????-??-??.json"):
        try:
            dates.append(_valid_date(path.stem))
        except ValueError:
            continue
    dates = sorted(set(dates), reverse=True)
    target = history_dir / "index.json"
    payload = {
        "dates": dates,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
    return target


'''

INDEX_TEST = '''

def test_preserve_all_refreshes_history_index(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    history = data / "history"
    history.mkdir()
    (history / "2026-09-04.json").write_text(json.dumps({"run_date": "2026-09-04", "plays": [{"player": "Old"}]}))
    base = {"run_date": "2026-09-05", "generated_at_utc": "2026-09-05T18:00:00Z"}
    (data / "daily_predictions.json").write_text(json.dumps({**base, "plays": [{"player": "Yesterday"}]}))
    for filename in preserve.PRODUCT_FILENAMES:
        (data / filename).write_text(json.dumps(base))

    preserve.preserve_all(data / "daily_predictions.json", history)

    index = json.loads((history / "index.json").read_text())
    assert index["dates"][:2] == ["2026-09-05", "2026-09-04"]
'''


def patch_roi() -> None:
    text = ROI_SOURCE.read_text(encoding="utf-8")
    if NEW_SELECTOR in text:
        return
    if OLD_SELECTOR not in text:
        raise RuntimeError("previous-board selector block not found")
    ROI_SOURCE.write_text(text.replace(OLD_SELECTOR, NEW_SELECTOR, 1), encoding="utf-8")


def patch_preserver() -> None:
    text = PRESERVE_SCRIPT.read_text(encoding="utf-8")
    if "def refresh_history_index(" not in text:
        anchor = "def preserve_all(board_path: Path, history_dir: Path) -> list[Path]:\n"
        if anchor not in text:
            raise RuntimeError("preserve_all anchor not found")
        text = text.replace(anchor, INDEX_HELPER + anchor, 1)

    old_return = '''    for filename in PRODUCT_FILENAMES:
        target = preserve(board_path.parent / filename, history_dir, product=True)
        if target:
            preserved.append(target)
    return preserved
'''
    new_return = '''    for filename in PRODUCT_FILENAMES:
        target = preserve(board_path.parent / filename, history_dir, product=True)
        if target:
            preserved.append(target)
    refresh_history_index(history_dir)
    return preserved
'''
    if old_return in text:
        text = text.replace(old_return, new_return, 1)
    elif "    refresh_history_index(history_dir)\n    return preserved\n" not in text:
        raise RuntimeError("preserve_all return block not found")
    PRESERVE_SCRIPT.write_text(text, encoding="utf-8")


def patch_test() -> None:
    text = PRESERVE_TEST.read_text(encoding="utf-8")
    if "test_preserve_all_refreshes_history_index" not in text:
        PRESERVE_TEST.write_text(text + INDEX_TEST, encoding="utf-8")


def bust_cache() -> None:
    for path in HTML_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("predictions-roi-overrides.js?v=4", "predictions-roi-overrides.js?v=5"), encoding="utf-8")


def sync_deployed_files() -> None:
    roi_bytes = ROI_SOURCE.read_bytes()
    for path in ROI_DEPLOY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(roi_bytes)

    source_index = REPO_ROOT / "sports/mlb/web/data/history/index.json"
    index_bytes = source_index.read_bytes()
    for path in INDEX_DEPLOY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(index_bytes)


def verify() -> None:
    source_index = json.loads((REPO_ROOT / "sports/mlb/web/data/history/index.json").read_text())
    if not source_index.get("dates") or source_index["dates"][0] != "2026-09-05":
        raise RuntimeError(f"September 5 is not newest archived date: {source_index.get('dates', [])[:3]}")
    for root in (
        REPO_ROOT / "sports/mlb/web/data/history",
        REPO_ROOT / "dist/mlb/data/history",
        REPO_ROOT / "paywall/private-content/app/mlb/data/history",
    ):
        payload = json.loads((root / "2026-09-05.json").read_text())
        plays = payload.get("plays") or []
        if payload.get("run_date") != "2026-09-05" or not plays:
            raise RuntimeError(f"invalid September 5 archive: {root}")
        unresolved = [p for p in plays if p.get("settlement_status") not in {"won", "lost", "push"}]
        if unresolved:
            raise RuntimeError(f"unresolved September 5 plays in {root}: {len(unresolved)}")


def main() -> int:
    patch_roi()
    patch_preserver()
    patch_test()
    bust_cache()

    # Import after patching so preserve_all includes refresh_history_index.
    import importlib.util

    spec = importlib.util.spec_from_file_location("preserve_published_predictions", PRESERVE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load preservation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.preserve_all(module.DEFAULT_BOARD, module.DEFAULT_HISTORY)

    sync_deployed_files()
    verify()
    print("September 5 is now the newest Previous Published Picks slate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
