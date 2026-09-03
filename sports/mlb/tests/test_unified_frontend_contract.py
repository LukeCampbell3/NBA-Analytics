from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_unified_frontend_contract_is_kept_off_the_actionable_board():
    html = (ROOT / "mlb/web/predictions.html").read_text()
    js = (ROOT / "mlb/web/predictions.js").read_text()
    assert 'data/unified_predictions.json' in js
    assert 'data/mlb_engine_manifest.json' in js
    assert 'Prediction request timed out' in js
    assert 'Predictions not yet generated for today' in js
    assert 'Unified schema mismatch' in js
    assert 'id="unifiedEngineContent"' not in html
    assert 'id="v21ShadowContent"' not in html
    assert 'unified-contract.js' in html
    assert "renderUnifiedTicket" in js
    assert "result.singles" not in js
    # Unified singles render only payload.singles; ticket legs render only in
    # the grouped ticket component.
    assert "const singles = Array.isArray(payload?.singles)" in js
    assert "const legs = Array.isArray(ticket.legs)" in js


def test_static_builder_preserves_unified_runtime_contract():
    builder = (ROOT / "site/pipeline/build_static_site.py").read_text()
    assert '"unified-contract.js"' in builder
    assert '"unified_predictions.json"' in builder
    assert '"mlb_engine_manifest.json"' in builder


def test_public_mlb_board_separates_current_picks_from_explicit_history_navigation():
    html = (ROOT / "mlb/web/predictions.html").read_text()
    js = (ROOT / "mlb/web/predictions.js").read_text()

    assert "await this.loadDateIndex()" in js
    assert "this.renderDateNav();" in js
    assert "this.availableDates[0]" not in js
    assert "Older picks are never substituted for today's board." in js
    assert "this.assertCurrentArtifact(payload, \"MLB board\")" in js
    assert "MLB board publication is withheld or under review" in js
    assert "is more than 8 hours old" in js
    assert 'cache: "no-store"' in js
    assert "no picks are displayed or linked." in html
    assert "yesterday's picks are never substituted" in html
    assert 'predictions.js?v=40' in html
    assert 'id="sameGameParlayContent"' in html
    assert 'id="pitcherParlayContent"' in html
    assert 'id="highHitParlayContent"' in html
    assert 'id="exoticMarketsContent"' in html
    assert 'data/history/products/${date}/${filename}' in js
    assert "settlementRow: candidate" in js
    assert 'this.renderRunMeta();' in js


def test_mlb_workflow_has_morning_final_and_next_slate_publications():
    workflow = (ROOT.parent / ".github/workflows/mlb-predictions.yml").read_text()
    assert 'cron: "17 8 * 3-10 *"' in workflow
    assert 'cron: "30 18 * 3-10 *"' in workflow
    assert 'cron: "30 23 * 3-10 *"' in workflow
    assert 'date -d tomorrow +%F' in workflow
