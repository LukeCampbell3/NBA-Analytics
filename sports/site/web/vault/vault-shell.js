/**
 * In The Cards Analytics -- application shell (global nav).
 * One implementation, reused by every page. Sport JS must not implement
 * its own navigation -- call CardVaultShell.mount() instead.
 */
(function initPredictionDeskShell(global) {
  const CardVaultShell = {};

  // Static fallback so the top nav is correct on first paint (no flash of
  // a missing sport). /data/sports.json (built from each sport's
  // site.json) is the metadata source of truth and is fetched right after
  // to pick up any sport added since this file last shipped.
  const DEFAULT_SPORTS = [
    { slug: "nba", label: "NBA", href: "/nba/predictions/" },
    { slug: "mlb", label: "MLB", href: "/mlb/predictions/" },
    { slug: "nfl", label: "NFL", href: "/nfl/projections/" },
  ];

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

  function renderNav(root, config, sports) {
    const {
      brandTitle = "In The Cards Analytics",
      brandHref = "/",
      sportSlug = "",
      sportAccent = "#9a681f",
      navLinks = [],
      showDisclaimer = true,
    } = config;

    document.body.classList.add("vault-theme");
    document.documentElement.style.setProperty("--vault-sport-accent", sportAccent);

    const currentPath = CardVaultShell.normalizePath(global.location?.pathname || "/");
    const primaryLinks = [
      { label: "Overview", href: "/" },
      ...sports.map((sport) => ({ label: sport.label, href: sport.href, slug: sport.slug })),
    ];

    const primaryHtml = primaryLinks.map((link) => {
      const path = CardVaultShell.normalizePath(link.href);
      const active = path === "/"
        ? currentPath === "/"
        : (link.slug === sportSlug || currentPath.startsWith(path));
      return `<a class="vault-nav-link${active ? " is-active" : ""}" href="${CardVaultShell.escapeHtml(link.href)}">${CardVaultShell.escapeHtml(link.label)}</a>`;
    }).join("");

    const contextHtml = navLinks.map((link) => {
      const normalized = CardVaultShell.normalizePath(link.href || "#");
      const active = Boolean(link.active) || currentPath === normalized;
      return `<a class="vault-context-link${active ? " is-active" : ""}" href="${CardVaultShell.escapeHtml(link.href || "#")}">${CardVaultShell.escapeHtml(link.label)}</a>`;
    }).join("");

    root.innerHTML = `
      <header class="vault-topbar" role="banner">
        <div class="vault-topbar__inner">
          <a class="vault-brand" href="${CardVaultShell.escapeHtml(brandHref)}" aria-label="In The Cards Analytics home">
            <span class="vault-brand-title">${CardVaultShell.escapeHtml(brandTitle)}</span>
          </a>
          <button type="button" class="vault-menu-btn" id="vaultNavToggle" aria-expanded="false" aria-controls="vaultNavLinks" aria-label="Open navigation">
            <span aria-hidden="true"></span>
          </button>
          <div class="vault-navigation" id="vaultNavLinks">
            <nav class="vault-primary-nav" aria-label="Sports">${primaryHtml}</nav>
            ${contextHtml ? `<nav class="vault-context-nav" aria-label="Section pages">${contextHtml}</nav>` : ""}
            ${showDisclaimer ? `<button type="button" class="vault-info-trigger vault-shell-info" aria-label="About this site" data-info="Independent, research-only model predictions. Review model status, data freshness, and evidence before treating a prediction as executable.">i</button>` : ""}
          </div>
        </div>
      </header>
    `;

    const toggle = document.getElementById("vaultNavToggle");
    const links = document.getElementById("vaultNavLinks");
    if (toggle && links) {
      toggle.addEventListener("click", () => {
        const open = links.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      });
    }
  }

  CardVaultShell.mount = function mount(config = {}) {
    const root = document.getElementById("vaultShellRoot");
    if (!root) return;

    renderNav(root, config, config.sports || DEFAULT_SPORTS);

    // Metadata-driven refresh: pick up the live sport catalog (built from
    // each sport's site.json) so a newly published sport appears in the
    // global nav everywhere without editing this file. Falls back silently
    // to the static list above if the catalog can't be fetched.
    if (!config.sports) {
      fetch("/data/sports.json", { cache: "no-store" })
        .then((response) => (response.ok ? response.json() : null))
        .then((catalog) => {
          if (!Array.isArray(catalog) || !catalog.length) return;
          const sports = catalog
            .filter((sport) => sport && sport.slug && (sport.status === "active" || sport.status === "shadow"))
            .map((sport) => ({ slug: sport.slug, label: String(sport.slug).toUpperCase(), href: sport.entry_href || `/${sport.slug}/predictions/` }));
          if (sports.length) renderNav(root, config, sports);
        })
        .catch(() => {});
    }
  };

  CardVaultShell.navFromPages = function navFromPages(pages, currentPath = "") {
    const path = String(currentPath || global.location?.pathname || "").toLowerCase();
    return (pages || []).map((page) => {
      const href = page.href || "#";
      const slug = href.toLowerCase().replace(/\/$/, "").split("/").filter(Boolean).pop() || "";
      return { label: page.label, href, active: Boolean(slug && path.includes(slug)) };
    });
  };

  global.CardVaultShell = CardVaultShell;
})(typeof window !== "undefined" ? window : globalThis);
