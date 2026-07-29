/**
 * Card Vault App Shell — top bar, sport switcher, workspace nav
 */
(function initCardVaultShell(global) {
  const CardVaultShell = {};

  const DEFAULT_SPORTS = [
    { slug: "nba", label: "NBA", href: "/nba/predictions.html" },
    { slug: "mlb", label: "MLB", href: "/mlb/predictions.html" },
  ];

  const DEFAULT_SITE_NAV_LINKS = [
    { label: "Prediction Desk", href: "/" },
    { label: "NBA Predictions", href: "/nba/predictions.html" },
    { label: "NBA Method", href: "/nba/prediction-about.html" },
    { label: "MLB Predictions", href: "/mlb/predictions.html" },
    { label: "MLB Method", href: "/mlb/prediction-about.html" },
  ];

  CardVaultShell.escapeHtml = function escapeHtml(value) {
    return global.CardVault ? global.CardVault.escapeHtml(value) : String(value ?? "");
  };

  CardVaultShell.normalizePath = function normalizePath(value) {
    const base = global.location?.href || "http://localhost/";
    const url = new URL(value || "/", base);
    let path = url.pathname.toLowerCase().replace(/\/index\.html$/, "/");
    path = path.replace(/\/+$/, "/");
    return path || "/";
  };

  CardVaultShell.mount = function mount(config = {}) {
    const root = document.getElementById("vaultShellRoot");
    if (!root) return;

    const {
      brandTitle = "Prediction Analytics",
      brandHref = "/",
      workspaceLabel = "Predictions",
      workspaceHref = "/",
      sportSlug = "",
      sportAccent = "#2563eb",
      breadcrumbs = [],
      navLinks = [],
      siteNavLinks = DEFAULT_SITE_NAV_LINKS,
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
      { label: "Desk", href: brandHref },
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

    const currentPath = CardVaultShell.normalizePath(global.location?.pathname || "/");
    const menuLinksByPath = new Map();
    [...siteNavLinks, ...navLinks].forEach((link) => {
      const href = link?.href || "#";
      const normalized = CardVaultShell.normalizePath(href);
      if (!menuLinksByPath.has(normalized)) {
        menuLinksByPath.set(normalized, { ...link, href, normalized });
      }
    });

    let bestActivePath = "";
    menuLinksByPath.forEach((link, path) => {
      const pathMatches = path === "/"
        ? currentPath === "/"
        : currentPath === path || currentPath.startsWith(path);
      if ((link.active || pathMatches) && path.length > bestActivePath.length) {
        bestActivePath = path;
      }
    });

    const menuLinks = [...menuLinksByPath.values()];
    const navHtml = menuLinks.map((link) => `
      <a class="vault-sport-pill${link.normalized === bestActivePath ? " is-active" : ""}" href="${CardVaultShell.escapeHtml(link.href)}">${CardVaultShell.escapeHtml(link.label)}</a>
    `).join("");

    root.innerHTML = `
      <header class="vault-topbar" role="banner">
        <a class="vault-brand" href="${CardVaultShell.escapeHtml(brandHref)}">
          <span class="vault-brand-kicker">Prediction Desk</span>
          <span class="vault-brand-title">${CardVaultShell.escapeHtml(brandTitle)}</span>
        </a>
        <nav class="vault-breadcrumb" aria-label="Breadcrumb">${crumbHtml}</nav>
        <div class="vault-sport-switcher" aria-label="Sport workspaces">${sportPills}</div>
        ${showDisclaimer ? `<button type="button" class="vault-info-trigger" aria-label="About research signals" data-info="Analysis support only. Model leans, confidence bands, and validation status are research signals, not automatic decisions.">i</button>` : ""}
        <button type="button" class="vault-menu-btn" id="vaultNavToggle" aria-expanded="false" aria-controls="vaultNavLinks">Menu</button>
        ${menuLinks.length ? `<nav id="vaultNavLinks" class="vault-nav-drawer-links vault-sport-switcher" aria-label="Site navigation">${navHtml}</nav>` : ""}
      </header>
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
