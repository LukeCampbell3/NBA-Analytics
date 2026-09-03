"use strict";

// Real regression test for the "Loading board details..." placeholder
// getting stuck when today's board isn't published yet. Two real MLB
// and NBA production paths for this: MLB's assertCurrentArtifact
// throwing when today's slate hasn't been published (the actual state
// the user reported on 2026-09-03, when the latest committed board was
// still 2026-09-02), and NBA's fetch failing outright when
// data/daily_predictions.json is missing for today's build. In both
// paths, the earlier code updated the freshness alert and the board
// area but left the header's own #predictionRunMeta element still
// showing its static "Loading board details..." placeholder text --
// so the top of the page contradicted everything below it. This test
// locks the fix: on the failure branch, the header element MUST be
// replaced with a real "not yet published" / "board unavailable" pill
// instead of leaving the loading text in place.

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.join(__dirname, "..");

function readSource(relPath) {
  return fs.readFileSync(path.join(REPO_ROOT, relPath), "utf8");
}

// 1. MLB: on the load-and-render error branch, we must actually clear
//    the "Loading board details..." placeholder by updating
//    this.elements.runMeta -- NOT just #predictionFreshnessAlert or
//    the cards region. If a future edit removes that update, the
//    header will silently go back to "Loading..." forever whenever
//    today's board isn't verified.
{
  const src = readSource("sports/mlb/web/predictions.js");
  const errorBranch = src.split("async loadAndRender(")[1];
  assert.ok(errorBranch, "loadAndRender not found in MLB predictions.js");
  const catchBlock = errorBranch.split("} catch (error) {")[1];
  assert.ok(catchBlock, "loadAndRender catch block not found in MLB predictions.js");
  const untilNextFn = catchBlock.split("\n    async ")[0];
  assert.ok(
    /this\.elements\.runMeta/.test(untilNextFn),
    "MLB loadAndRender catch block must update this.elements.runMeta so the 'Loading board details...' placeholder is replaced when today's board hasn't been verified",
  );
  assert.ok(
    /Not yet published/i.test(untilNextFn) || /withheld/i.test(untilNextFn),
    "MLB loadAndRender catch block must surface a real 'not yet published'/withheld state on the header, not silently leave 'Loading board details...' in place",
  );
  console.log("  ok  MLB catch branch replaces the runMeta placeholder");
}

// 2. NBA: same discipline -- the NBA page also has a static
//    "Loading board details..." placeholder that must be replaced
//    on the failure branch, not left showing next to a "Board
//    unavailable" empty-state card below it.
{
  const src = readSource("sports/nba/web/predictions.js");
  const errorBranch = src.split("async loadAndRender(")[1];
  assert.ok(errorBranch, "loadAndRender not found in NBA predictions.js");
  const catchBlock = errorBranch.split("} catch (error) {")[1];
  assert.ok(catchBlock, "loadAndRender catch block not found in NBA predictions.js");
  const untilNextFn = catchBlock.split("\n    async ")[0];
  assert.ok(
    /this\.elements\.runMeta/.test(untilNextFn),
    "NBA loadAndRender catch block must update this.elements.runMeta so the 'Loading board details...' placeholder is replaced when the daily board fails to load",
  );
  assert.ok(
    /Board unavailable/i.test(untilNextFn) || /withheld/i.test(untilNextFn),
    "NBA loadAndRender catch block must surface a real 'board unavailable' state on the header",
  );
  console.log("  ok  NBA catch branch replaces the runMeta placeholder");
}

// 3. Both pages must still declare the placeholder in their HTML --
//    the fix is to REPLACE it dynamically, not to remove the initial
//    "Loading board details..." text (which is the honest, correct
//    state before either the load succeeds or fails).
{
  const mlbHtml = readSource("sports/mlb/web/predictions.html");
  const nbaHtml = readSource("sports/nba/web/predictions.html");
  assert.ok(
    /Loading board details/i.test(mlbHtml),
    "MLB predictions.html must still carry a 'Loading board details...' placeholder as the initial header state",
  );
  assert.ok(
    /Loading current board details/i.test(nbaHtml),
    "NBA predictions.html must still carry a 'Loading current board details...' placeholder as the initial header state",
  );
  console.log("  ok  static 'Loading board details...' placeholder preserved on both pages");
}

console.log("all prediction-page run-meta failure-recovery tests passed");
