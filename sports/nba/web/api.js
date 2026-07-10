function mountShell() {
  if (!window.CardVaultShell) return;
  window.CardVaultShell.mount({
    brandTitle: "NBA Analytics",
    brandHref: "/",
    workspaceLabel: "NBA",
    workspaceHref: "/nba/",
    sportSlug: "nba",
    sportAccent: "#f59e0b",
    breadcrumbs: [{ label: "API Docs", href: "api.html" }],
    navLinks: [
      { label: "Dashboard", href: "index.html" },
      { label: "PAR Records", href: "par.html" },
      { label: "Board", href: "predictions.html" },
      { label: "Safe-State", href: "safe-state.html" },
      { label: "Pricing", href: "pricing.html" },
      { label: "Account", href: "account.html" },
      { label: "API", href: "api.html", active: true },
    ],
    showDisclaimer: false,
  });
}

document.addEventListener("DOMContentLoaded", mountShell);
