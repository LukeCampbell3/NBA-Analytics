from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_unified_frontend_has_one_grouped_source_and_no_leg_to_single_merge():
    html = (ROOT / "mlb/web/predictions.html").read_text()
    js = (ROOT / "mlb/web/predictions.js").read_text()
    assert 'data/unified_predictions.json' in js
    assert 'data/mlb_engine_manifest.json' in js
    assert 'Prediction request timed out' in js
    assert 'Predictions not yet generated for today' in js
    assert 'Unified schema mismatch' in js
    assert 'id="unifiedEngineContent"' in html
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
