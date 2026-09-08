/**
 * In The Cards Analytics -- application shell (global nav).
 * One implementation, reused by every page. Sport JS may provide page-specific
 * context, but the shared shell guarantees that an empty navigation root never
 * leaves a user stranded.
 */
(function initPredictionDeskShell(global) {
  const CardVaultShell = {};

  const DEFAULT_SPORTS = [
    { slug: "nba", label: "NBA", href: "/nba/predictions/" },
    { slug: "mlb", label: "MLB", href: "/mlb/predictions/" },
    { slug: "nfl", label: "NFL", href: "/nfl/projections/" },
    { slug: "f1", label: "F1", href: "/f1/predictions/" },
  ];

  const SPORT_CONTEXT_LINKS = {
    nba: [
      { label: "Board", href: "/nba/predictions/" },
      { label: "Stats", href: "/nba/stats/" },
      { label: "Drive-Pass", href: "/nba/drive-pass/" },
      { label: "Post-Pass", href: "/nba/post-pass/" },
      { label: "Advantage Routing", href: "/nba/advantage-routing/" },
      { label: "Method", href: "/nba/prediction-about/" },
      { label: "NBA-CV", href: "/nba/predictions/nba-cv/" },
    ],
  };

  const SPORT_ACCENTS = { nba: "#ff3b57", mlb: "#1f6f43", nfl: "#244a82", f1: "#b72b2b" };

  CardVaultShell.escapeHtml = function escapeHtml(value) {
    return global.CardVault ? global.CardVault.escapeHtml(value) : String(value ?? "");
  };

  CardVaultShell.normalizePath = function normalizePath(value) {
    const base = global.location?.href || "http://localhost/";
    const url = new URL(value || "/", base);
    let path = url.pathname.toLowerCase().replace(/\/index\.html$/, "/");
    path = path.replace(/\.html$/, "/").replace(/\/+$/, "/");
    return path || "/";
  };

  CardVaultShell.inferSportSlug = function inferSportSlug(value) {
    const path = CardVaultShell.normalizePath(value || global.location?.pathname || "/");
    return path.split("/").filter(Boolean)[0] || "";
  };

  function renderNav(root, config, sports) {
    const { brandTitle = "In The Cards Analytics", brandHref = "/", sportSlug = "", sportAccent = "#9a681f", navLinks = [], showDisclaimer = true } = config;
    document.body.classList.add("vault-theme");
    document.documentElement.style.setProperty("--vault-sport-accent", sportAccent);
    const currentPath = CardVaultShell.normalizePath(global.location?.pathname || "/");
    const primaryLinks = [{ label: "Overview", href: "/" }, ...sports.map((sport) => ({ label: sport.label, href: sport.href, slug: sport.slug }))];
    const primaryHtml = primaryLinks.map((link) => {
      const path = CardVaultShell.normalizePath(link.href);
      const active = path === "/" ? currentPath === "/" : (link.slug === sportSlug || currentPath.startsWith(path));
      return `<a class="vault-nav-link${active ? " is-active" : ""}" href="${CardVaultShell.escapeHtml(link.href)}">${CardVaultShell.escapeHtml(link.label)}</a>`;
    }).join("");
    const contextHtml = navLinks.map((link) => {
      const normalized = CardVaultShell.normalizePath(link.href || "#");
      const active = Boolean(link.active) || currentPath === normalized;
      return `<a class="vault-context-link${active ? " is-active" : ""}" href="${CardVaultShell.escapeHtml(link.href || "#")}">${CardVaultShell.escapeHtml(link.label)}</a>`;
    }).join("");
    root.innerHTML = `<header class="vault-topbar" role="banner"><div class="vault-topbar__inner"><a class="vault-brand" href="${CardVaultShell.escapeHtml(brandHref)}" aria-label="In The Cards Analytics home"><span class="vault-brand-title">${CardVaultShell.escapeHtml(brandTitle)}</span></a><button type="button" class="vault-menu-btn" id="vaultNavToggle" aria-expanded="false" aria-controls="vaultNavLinks" aria-label="Open navigation"><span aria-hidden="true"></span></button><div class="vault-navigation" id="vaultNavLinks"><nav class="vault-primary-nav" aria-label="Sports">${primaryHtml}</nav>${contextHtml ? `<nav class="vault-context-nav" aria-label="Section pages">${contextHtml}</nav>` : ""}${showDisclaimer ? `<button type="button" class="vault-info-trigger vault-shell-info" aria-label="About this site" data-info="Independent, research-only model predictions. Review model status, data freshness, and evidence before treating a prediction as executable.">i</button>` : ""}</div></div></header>`;
    const toggle = document.getElementById("vaultNavToggle");
    const links = document.getElementById("vaultNavLinks");
    if (toggle && links) toggle.addEventListener("click", () => { const open = links.classList.toggle("is-open"); toggle.setAttribute("aria-expanded", open ? "true" : "false"); toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation"); });
  }

  CardVaultShell.mount = function mount(config = {}) {
    const root = document.getElementById("vaultShellRoot");
    if (!root) return;
    if (root.querySelector("[data-nba-static-nav]")) return;
    const sportSlug = config.sportSlug || CardVaultShell.inferSportSlug();
    const resolvedConfig = { ...config, sportSlug, sportAccent: config.sportAccent || SPORT_ACCENTS[sportSlug] || "#9a681f", navLinks: Array.isArray(config.navLinks) && config.navLinks.length ? config.navLinks : (SPORT_CONTEXT_LINKS[sportSlug] || []) };
    renderNav(root, resolvedConfig, config.sports || DEFAULT_SPORTS);
    if (!config.sports) {
      fetch("/data/sports.json", { cache: "no-store" }).then((response) => (response.ok ? response.json() : null)).then((catalog) => {
        if (!Array.isArray(catalog) || !catalog.length) return;
        const sports = catalog.filter((sport) => sport && sport.slug && (sport.status === "active" || sport.status === "shadow")).map((sport) => ({ slug: sport.slug, label: String(sport.slug).toUpperCase(), href: sport.entry_href || `/${sport.slug}/predictions/` }));
        if (sports.length && !root.querySelector("[data-nba-static-nav]")) renderNav(root, resolvedConfig, sports);
      }).catch(() => {});
    }
  };

  CardVaultShell.navFromPages = function navFromPages(pages, currentPath = "") {
    const path = String(currentPath || global.location?.pathname || "").toLowerCase();
    return (pages || []).map((page) => { const href = page.href || "#"; const slug = href.toLowerCase().replace(/\/$/, "").split("/").filter(Boolean).pop() || ""; return { label: page.label, href, active: Boolean(slug && path.includes(slug)) }; });
  };

  function autoMount() {
    const root = document.getElementById("vaultShellRoot");
    if (!root || root.children.length) return;
    CardVaultShell.mount({ brandTitle: "In The Cards Analytics", brandHref: "/", sportSlug: CardVaultShell.inferSportSlug(), showDisclaimer: true });
  }

  const scheduleAutoMount = () => global.setTimeout(autoMount, 0);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", scheduleAutoMount, { once: true }); else scheduleAutoMount();
  global.CardVaultShell = CardVaultShell;
})(typeof window !== "undefined" ? window : globalThis);
