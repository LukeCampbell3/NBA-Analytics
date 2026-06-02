/**
 * Card Vault App Shell — top bar, sport switcher, workspace nav
 */
(function initCardVaultShell(global) {
  const CardVaultShell = {};

  const DEFAULT_SPORTS = [
    { slug: "nba", label: "NBA", href: "/nba/" },
    { slug: "mlb", label: "MLB", href: "/mlb/" },
    { slug: "nfl", label: "NFL", href: "/nfl/" },
  ];

  CardVaultShell.escapeHtml = function escapeHtml(value) {
    return global.CardVault ? global.CardVault.escapeHtml(value) : String(value ?? "");
  };

  CardVaultShell.mount = function mount(config = {}) {
    const root = document.getElementById("vaultShellRoot");
    if (!root) return;

    const {
      brandTitle = "In The Cards Analytics",
      brandHref = "/",
      workspaceLabel = "Workspace Hub",
      workspaceHref = "/",
      sportSlug = "",
      sportAccent = "#f59e0b",
      breadcrumbs = [],
      navLinks = [],
      sports = DEFAULT_SPORTS,
      showDisclaimer = true,
    } = config;

    document.body.classList.add("vault-theme");
    if (sportAccent) {
      document.documentElement.style.setProperty("--vault-sport-accent", sportAccent);
    }

    const sportPills = sports.map((s) => {
      const active = sportSlug && s.slug === sportSlug ? " is-active" : "";
      return `<a class="vault-sport-pill${active}" href="${CardVaultShell.escapeHtml(s.href)}">${CardVaultShell.escapeHtml(s.label)}</a>`;
    }).join("");

    const crumbItems = [
      { label: "Vault Hub", href: brandHref },
      ...(workspaceLabel ? [{ label: workspaceLabel, href: workspaceHref }] : []),
      ...breadcrumbs,
    ];
    const crumbHtml = crumbItems.map((item, i) => {
      const isLast = i === crumbItems.length - 1 && (!breadcrumbs.length || i === crumbItems.length - 1);
      if (isLast && i === crumbItems.length - 1 && breadcrumbs.length) {
        return `<span aria-current="page">${CardVaultShell.escapeHtml(item.label)}</span>`;
      }
      if (i === crumbItems.length - 1 && !breadcrumbs.length) {
        return `<span aria-current="page">${CardVaultShell.escapeHtml(item.label)}</span>`;
      }
      return `<a href="${CardVaultShell.escapeHtml(item.href)}">${CardVaultShell.escapeHtml(item.label)}</a>`;
    }).join('<span class="vault-breadcrumb-sep" aria-hidden="true">/</span>');

    const navHtml = navLinks.map((link) => `
      <a class="vault-sport-pill${link.active ? " is-active" : ""}" href="${CardVaultShell.escapeHtml(link.href)}">${CardVaultShell.escapeHtml(link.label)}</a>
    `).join("");

    root.innerHTML = `
      <header class="vault-topbar" role="banner">
        <a class="vault-brand" href="${CardVaultShell.escapeHtml(brandHref)}">
          <span class="vault-brand-kicker">In The Cards</span>
          <span class="vault-brand-title">${CardVaultShell.escapeHtml(brandTitle)}</span>
        </a>
        <nav class="vault-breadcrumb" aria-label="Breadcrumb">${crumbHtml}</nav>
        <div class="vault-sport-switcher" aria-label="Sport workspaces">${sportPills}</div>
        <button type="button" class="vault-menu-btn" id="vaultNavToggle" aria-expanded="false" aria-controls="vaultNavLinks">Menu</button>
        ${navLinks.length ? `<nav id="vaultNavLinks" class="vault-nav-drawer-links vault-sport-switcher" aria-label="Workspace pages">${navHtml}</nav>` : ""}
      </header>
      ${showDisclaimer ? `<p class="vault-disclaimer">Analysis support only — model leans, confidence bands, and validation status are research signals, not automatic decisions.</p>` : ""}
    `;

    const toggle = document.getElementById("vaultNavToggle");
    const links = document.getElementById("vaultNavLinks");
    if (toggle && links) {
      toggle.addEventListener("click", () => {
        const open = links.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }
  };

  /** Build nav from sport manifest page list */
  CardVaultShell.navFromPages = function navFromPages(pages, currentPath = "") {
    const path = String(currentPath || window.location.pathname).toLowerCase();
    return (pages || []).map((page) => {
      const href = page.href || "#";
      const normalized = href.toLowerCase().replace(/\/$/, "");
      const active = path.includes(normalized.split("/").filter(Boolean).pop() || "___");
      return { label: page.label, href, active };
    });
  };

  global.CardVaultShell = CardVaultShell;
})(typeof window !== "undefined" ? window : globalThis);
