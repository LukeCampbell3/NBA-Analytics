"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(
  path.join(__dirname, "..", "sports", "shared", "web", "vault", "vault-components.js"),
  "utf8",
);

function loadCardVault({ storedRegion = "", storageThrows = false } = {}) {
  let clickHandler = null;
  const navigations = [];
  const document = {
    addEventListener(type, handler) {
      if (type === "click") clickHandler = handler;
    },
  };
  const localStorage = {
    getItem() {
      if (storageThrows) throw new Error("storage disabled");
      return storedRegion;
    },
    setItem(_key, value) {
      if (storageThrows) throw new Error("storage disabled");
      storedRegion = value;
    },
    removeItem() {
      if (storageThrows) throw new Error("storage disabled");
      storedRegion = "";
    },
  };
  const window = {
    URL,
    localStorage,
    location: { assign: (href) => navigations.push(href) },
  };
  const context = { URL, window, document };
  vm.runInNewContext(source, context, { filename: "vault-components.js" });
  return { CardVault: window.CardVault, getClickHandler: () => clickHandler, navigations };
}

function makeLink(regionLinks, fallback) {
  return {
    dataset: {
      deeplinksByRegion: JSON.stringify(regionLinks),
      fallbackUrl: fallback,
    },
    href: fallback,
    getAttribute(name) { return name === "href" ? this.href : ""; },
  };
}

(async () => {
  const pa = "https://sportsbook.fanduel.com/addToBetslip?marketId=742.1&selectionId=11";
  const tn = "https://sportsbook.fanduel.com/addToBetslip?marketId=747.2&selectionId=22";
  const nj = "https://sportsbook.fanduel.com/addToBetslip?marketId=734.3&selectionId=33";

  // Known state: rewrite href during the trusted click and allow native
  // target=_blank navigation; do not invoke the async/popup-blocked path.
  const known = loadCardVault({ storedRegion: "PA" });
  known.CardVault.initFanduelBetslipLinks();
  const knownLink = makeLink({ PA: pa }, nj);
  let knownPrevented = false;
  await known.getClickHandler()({
    target: { closest: () => knownLink },
    preventDefault: () => { knownPrevented = true; },
  });
  assert.strictEqual(knownPrevented, false);
  assert.strictEqual(knownLink.href, pa);
  assert.deepStrictEqual(known.navigations, []);

  // Blocked localStorage: retain the dialog choice in memory and use a
  // same-tab navigation after the async dialog so Safari cannot block it.
  const blocked = loadCardVault({ storageThrows: true });
  blocked.CardVault.promptFanduelRegion = async () => "TN";
  blocked.CardVault.initFanduelBetslipLinks();
  const blockedLink = makeLink({ TN: tn }, nj);
  let blockedPrevented = false;
  await blocked.getClickHandler()({
    target: { closest: () => blockedLink },
    preventDefault: () => { blockedPrevented = true; },
  });
  assert.strictEqual(blockedPrevented, true);
  assert.strictEqual(blocked.CardVault.getFanduelRegion(), "TN");
  assert.deepStrictEqual(blocked.navigations, [tn]);

  console.log("FanDuel frontend deep-link tests passed");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
