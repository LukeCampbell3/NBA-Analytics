"use strict";

// Real regression: the drive-pass / post-pass "Shot Destination" court
// used to render a permanently-zero "Corner 3: 0 (0.0%)" legend row
// and a corresponding "0" bubble at the corner-3 court position for
// every player, because the frontend's AR_ZONE_COLORS / AR_ZONE_LABELS
// listed a CORNER_3 zone that the real classifier
// (sports/nba/analytics/advantage_routing/routing/states.py --
// classify_shot_zone_from_text) never emits.
//
// Basketball-Reference play-by-play text does not report corner-vs-
// above-break for a 3PA, so every real 3PA is placed in ABOVE_BREAK_3
// by convention -- with an explicit caveat that the corner was not
// ruled out. The old graphic misread as "this player never gets
// assisted for a corner 3", which is not what the data says.
//
// This test locks two invariants:
//   1. The frontend's zone tables carry only zones the real classifier
//      can actually emit (RIM, SHORT_PAINT, MIDRANGE, ABOVE_BREAK_3),
//      and NEVER a standalone CORNER_3 entry.
//   2. Empirically, every real player JSON's recipient breakdown has a
//      zero CORNER_3 count (or no key at all) -- so any future re-
//      introduction of CORNER_3 in the frontend really would render
//      the same misleading permanent-zero row for every player.

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const REPO_ROOT = path.join(__dirname, "..");
const AR_PAGE_SRC = fs.readFileSync(
  path.join(REPO_ROOT, "sports/nba/web/advantage-analysis-page.js"),
  "utf8",
);

// 1. AR_ZONE_COLORS / AR_ZONE_LABELS objects must NOT declare CORNER_3.
{
  const colorsBlock = AR_PAGE_SRC.match(/const AR_ZONE_COLORS\s*=\s*\{([\s\S]*?)\};/);
  const labelsBlock = AR_PAGE_SRC.match(/const AR_ZONE_LABELS\s*=\s*\{([\s\S]*?)\};/);
  assert.ok(colorsBlock, "AR_ZONE_COLORS block not found");
  assert.ok(labelsBlock, "AR_ZONE_LABELS block not found");
  assert.ok(
    !/\bCORNER_3\s*:/.test(colorsBlock[1]),
    "AR_ZONE_COLORS must not declare a CORNER_3 zone -- the real classifier never emits it, so it would render as a permanent-zero row for every player",
  );
  assert.ok(
    !/\bCORNER_3\s*:/.test(labelsBlock[1]),
    "AR_ZONE_LABELS must not declare a CORNER_3 zone -- the legend would show a permanently-zero 'Corner 3' entry for every player",
  );
  console.log("  ok  AR_ZONE_COLORS and AR_ZONE_LABELS drop CORNER_3");
}

// 2. The court's zoneRegions map (which the SVG loops over to render
//    bubbles) also must not carry a CORNER_3 entry -- otherwise the
//    court itself shows a permanent-zero corner bubble even if the
//    legend was fixed.
{
  const zoneRegionsBlock = AR_PAGE_SRC.match(/const zoneRegions\s*=\s*\{([\s\S]*?)\};/);
  assert.ok(zoneRegionsBlock, "zoneRegions block not found inside renderShotDestination");
  assert.ok(
    !/\bCORNER_3\s*:/.test(zoneRegionsBlock[1]),
    "renderShotDestination.zoneRegions must not include CORNER_3 -- the court bubble would be permanently zero for every player",
  );
  console.log("  ok  renderShotDestination.zoneRegions drops CORNER_3");
}

// 3. Empirical: every real per-player JSON's recipient zone_breakdown
//    either lacks CORNER_3 or has it as zero. If a future data-
//    producer edit introduces real CORNER_3 counts, THIS test loudly
//    tells you to bring the frontend CORNER_3 rendering back rather
//    than silently dropping real data on the floor.
{
  const dataDir = path.join(REPO_ROOT, "sports/nba/web/data/advantage-routing");
  const files = fs.readdirSync(dataDir).filter((f) => f.endsWith(".json"));
  assert.ok(files.length > 0, "expected per-player advantage-routing JSONs to exist");
  let checked = 0;
  let cornerCount = 0;
  for (const name of files) {
    const payload = JSON.parse(fs.readFileSync(path.join(dataDir, name), "utf8"));
    const recipients = payload?.recipients?.recipients || [];
    for (const r of recipients) {
      const zb = r?.zone_breakdown || {};
      const value = Number(zb.CORNER_3 || 0);
      cornerCount += value;
    }
    checked += 1;
  }
  assert.strictEqual(
    cornerCount,
    0,
    `expected no real CORNER_3 counts in ${checked} per-player payloads, but the sum was ${cornerCount} -- if this is now real data, restore the CORNER_3 legend/court bubble in advantage-analysis-page.js`,
  );
  console.log(`  ok  ${checked} real per-player JSONs carry zero CORNER_3 counts (aggregate)`);
}

console.log("all shot-destination honesty tests passed");
