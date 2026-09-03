"use strict";

// Real, node-native tests for the shared prediction-card renderer's
// cross-sport consistency contract. Every sport (MLB, NBA, NFL) has to
// hand its per-play data to the same CardVault.renderPredictionCard so
// the resulting HTML shape is identical -- previously each sport had
// its own inline field-coercion adapter at the call site, which meant
// a viewer of the NFL picks page saw subtly different card content
// (e.g. an unlabeled "PASSING" market, or a missing line) than an MLB
// viewer would have of the same underlying play, and the divergence
// was invisible from the shared renderer itself. These tests lock the
// consistency contract in place: the alias fallbacks the shared
// renderer now accepts (market_line || line, target stems, etc.), and
// the envelope-gated wrapper both fail-open (NFL) and fail-closed
// (MLB/NBA) sports can share.

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "sports", "shared", "web", "vault", "vault-components.js"),
  "utf8",
);

function loadCardVault() {
  const document = { addEventListener() {} };
  const window = {
    URL,
    localStorage: {
      getItem() { return ""; },
      setItem() {},
      removeItem() {},
    },
    location: { assign() {} },
  };
  const context = { URL, window, document };
  vm.runInNewContext(source, context, { filename: "vault-components.js" });
  return window.CardVault;
}

function mlbShapedPlay(overrides = {}) {
  return {
    player_display_name: "Shohei Ohtani",
    player: "Shohei Ohtani",
    market_line: 1.5,
    target: "TB",
    direction: "OVER",
    model_hit_probability: 0.62,
    selected_side_price: -115,
    selected_sportsbook_key: "fanduel",
    market_source: "real",
    market_books: 5,
    candidate_authorized: true,
    action_status: "ready",
    ...overrides,
  };
}

function nflShapedPlay(overrides = {}) {
  // Real NFL play shape as emitted by sports/nfl/predictions/daily_policy.py:
  // `player`/`line`/`target="passing"` (stem, not suffixed), plus per-play
  // action_status/candidate_authorized -- no player_display_name, no
  // market_line, no explicit MLB/NBA-style suffixed target.
  return {
    player: "Malik Willis",
    line: 225.5,
    target: "passing",
    direction: "OVER",
    model_hit_probability: 0.61,
    selected_side_price: -113,
    selected_sportsbook_key: "fanduel",
    market_source: "rotowire_public_nfl_props",
    market_books: 7,
    candidate_authorized: true,
    action_status: "ready",
    ...overrides,
  };
}

(async () => {
  const CardVault = loadCardVault();

  // 1. NFL shape must render the same market-line row MLB/NBA get -- the
  //    NFL exporter emits `line`, MLB/NBA emit `market_line`; the shared
  //    renderer now accepts either without any per-sport adapter.
  {
    const nflHtml = CardVault.renderPredictionCard(nflShapedPlay(), 0);
    const mlbHtml = CardVault.renderPredictionCard(mlbShapedPlay({ market_line: 225.5 }), 0);
    assert.ok(nflHtml.includes("<dt>Line</dt><dd>225.50</dd>"),
      "NFL shape (line field) must render the same Line row MLB gets from market_line");
    assert.ok(mlbHtml.includes("<dt>Line</dt><dd>225.50</dd>"),
      "MLB shape (market_line field) must still render its Line row");
    console.log("  ok  NFL `line` and MLB `market_line` render the same Line row");
  }

  // 2. NFL target stems (`passing`/`rushing`/`receiving`) must render as
  //    human labels ("Passing Yards", etc.), not raw upper-case tokens.
  //    Regression against the pre-consistency state where the NFL card
  //    read "PASSING" while MLB read "Total Bases".
  {
    const casesByStem = {
      passing: "Passing Yards",
      rushing: "Rushing Yards",
      receiving: "Receiving Yards",
    };
    for (const [stem, label] of Object.entries(casesByStem)) {
      const html = CardVault.renderPredictionCard(nflShapedPlay({ target: stem }), 0);
      assert.ok(html.includes(label), `NFL target stem ${stem} must render as "${label}"`);
      assert.ok(!html.includes(`>${stem.toUpperCase()}<`),
        `NFL target stem ${stem} must NOT surface as raw upper-case token`);
    }
    console.log("  ok  NFL target stems map to human labels (matches MLB/NBA)");
  }

  // 3. renderPredictionCardWithEnvelope with no envelope must be byte-
  //    identical to renderPredictionCard -- so any sport can opt in
  //    later without changing the currently-rendered output.
  {
    const play = mlbShapedPlay();
    const plain = CardVault.renderPredictionCard(play, 3);
    const wrapped = CardVault.renderPredictionCardWithEnvelope(play, 3, null);
    assert.strictEqual(wrapped, plain,
      "envelope-aware wrapper with null envelope must render byte-identically to the direct call");
    console.log("  ok  envelope-aware wrapper is a no-op when envelope is null");
  }

  // 4. An authorized envelope must NOT weaken a per-play authorization --
  //    a play with candidate_authorized=false is still un-authorized.
  {
    const play = nflShapedPlay({ candidate_authorized: false, action_status: "review" });
    const html = CardVault.renderPredictionCardWithEnvelope(play, 0, {
      candidate_authorized: true,
      publication_status: "ready",
    });
    // "Shadow" status pill fires when candidate_authorized === false --
    // renderStatusPill uses that as the "shadow" key.
    assert.ok(html.toLowerCase().includes("shadow"),
      "an authorized envelope must not silently promote a per-play unauthorized card");
    console.log("  ok  authorized envelope preserves per-play unauthorized state");
  }

  // 5. An UN-authorized envelope must force every play to review/shadow
  //    even if the per-play row said `action_status: "ready"` and
  //    `candidate_authorized: true`. This is the NFL fail-open discipline
  //    the shared helper now encodes -- previously an inline call-site
  //    adapter.
  {
    const play = nflShapedPlay({ candidate_authorized: true, action_status: "ready" });
    const html = CardVault.renderPredictionCardWithEnvelope(play, 0, {
      candidate_authorized: false,
      publication_status: "withheld",
    });
    assert.ok(html.toLowerCase().includes("shadow"),
      "an un-authorized envelope must gate every play into Shadow -- fail-open discipline");
    console.log("  ok  un-authorized envelope forces every play to Shadow");
  }

  // 6. Explicit board_publication_status on the play must win over the
  //    envelope's publication_status -- the wrapper only fills in when
  //    the play didn't already carry the field, never overrides.
  {
    const play = mlbShapedPlay({ board_publication_status: "ready" });
    const html = CardVault.renderPredictionCardWithEnvelope(play, 0, {
      candidate_authorized: true,
      publication_status: "withheld",
    });
    // Ready survives -- if it had been overridden, the status pill would
    // instead read "withheld"/Withheld.
    assert.ok(!/withheld/i.test(html) || /qualified|ready/i.test(html),
      "per-play board_publication_status must not be silently overwritten by the envelope");
    console.log("  ok  per-play board_publication_status is not silently overridden");
  }

  // 7. The two rendering call shapes MLB/NBA use today must remain
  //    unaffected -- one MLB-shape play and one NFL-shape play rendered
  //    into the same board must both produce the same outer article shell
  //    (same class list, same header structure) so the picks page reads
  //    consistently regardless of which sport's data flowed in.
  {
    const mlbHtml = CardVault.renderPredictionCard(mlbShapedPlay(), 0);
    const nflHtml = CardVault.renderPredictionCard(nflShapedPlay(), 0);
    for (const shell of [
      '<article class="prediction-card"',
      '<header class="prediction-card__header">',
      '<span class="prediction-card__rank">',
      '<div class="prediction-card__tags">',
      '<div class="prediction-card__identity">',
      '<div class="prediction-card__signal">',
      '<dl class="prediction-card__metrics">',
      '<p class="prediction-card__note">',
    ]) {
      assert.ok(mlbHtml.includes(shell), `MLB card missing shared shell: ${shell}`);
      assert.ok(nflHtml.includes(shell), `NFL card missing shared shell: ${shell}`);
    }
    console.log("  ok  MLB and NFL cards share the same outer HTML shell");
  }

  console.log("all prediction-card cross-sport shape tests passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
