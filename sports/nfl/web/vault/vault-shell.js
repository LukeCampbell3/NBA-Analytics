/**
 * Prediction Desk application shell.
 */
(function initPredictionDeskShell(global) {
  const CardVaultShell = {};

  const DEFAULT_SPORTS = [
    { slug: "nba", label: "NBA", href: "/nba/predictions/" },
    { slug: "mlb", label: "MLB", href: "/mlb/predictions/" },
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

  CardVaultShell.mount = function mount(config = {}) {
    const root = document.getElementById("vaultShellRoot");
    if (!root) return;

    const {
      brandTitle = "Prediction Desk",
      brandHref = "/",
      sportSlug = "",
      sportAccent = "#2563eb",
      navLinks = [],
      sports = DEFAULT_SPORTS,
      showDisclaimer = true,
    } = config;

    document.body.classList.add("vault-theme");
    document.documentElement.style.setProperty("--vault-sport-accent", sportAccent);

    const currentPath = CardVaultShell.normalizePath(global.location?.pathname || "/");
    const primaryLinks = [
      { label: "Overview", href: "/" },
      ...sports.map((sport) => ({
        label: sport.label,
        href: sport.href,
        slug: sport.slug,
      })),
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
          <a class="vault-brand" href="${CardVaultShell.escapeHtml(brandHref)}" aria-label="Prediction Desk overview">
            <span class="vault-brand-mark" aria-hidden="true">P</span>
            <span>
              <span class="vault-brand-kicker">Analytics</span>
              <span class="vault-brand-title">${CardVaultShell.escapeHtml(brandTitle)}</span>
            </span>
          </a>
          <button type="button" class="vault-menu-btn" id="vaultNavToggle" aria-expanded="false" aria-controls="vaultNavLinks" aria-label="Open navigation">
            <span aria-hidden="true"></span>
          </button>
          <div class="vault-navigation" id="vaultNavLinks">
            <nav class="vault-primary-nav" aria-label="Prediction workspaces">${primaryHtml}</nav>
            ${contextHtml ? `<nav class="vault-context-nav" aria-label="Workspace pages">${contextHtml}</nav>` : ""}
            ${showDisclaimer ? `<button type="button" class="vault-info-trigger vault-shell-info" aria-label="About prediction signals" data-info="Research signals only. Review model status, data freshness, and supporting context before using a board.">i</button>` : ""}
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
