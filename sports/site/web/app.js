async function loadSportsManifest() {
    const response = await fetch(`data/sports.json?v=${Date.now()}`);
    if (!response.ok) {
        throw new Error(`Failed to load sports manifest (HTTP ${response.status})`);
    }
    return response.json();
}

function renderSummary(sports) {
    const sportCount = document.getElementById("sportCount");
    const activeCount = document.getElementById("activeCount");
    const pageCount = document.getElementById("pageCount");

    const active = sports.filter((sport) => sport.status === "active").length;
    const pages = sports.reduce((sum, sport) => sum + (Array.isArray(sport.pages) ? sport.pages.length : 0), 0);

    if (sportCount) sportCount.textContent = String(sports.length);
    if (activeCount) activeCount.textContent = String(active);
    if (pageCount) pageCount.textContent = String(pages);
}

function renderSportsGrid(sports) {
    const grid = document.getElementById("sportsGrid");
    if (!grid) return;

    const cv = window.CardVault;
    if (!cv) {
        console.error("CardVault not loaded");
        return;
    }

    if (!Array.isArray(sports) || sports.length === 0) {
        grid.innerHTML = cv.renderEmptyState(
            "No vault doors found",
            "No sport workspaces were discovered in the current build.",
            "Add sports/<slug>/web with an index.html to register a new workspace."
        );
        return;
    }

    grid.innerHTML = sports.map((sport) => cv.renderSportWorkspaceCard(sport)).join("");
}

function mountHubShell() {
    if (!window.CardVaultShell) return;

    window.CardVaultShell.mount({
        brandTitle: "Analytics Vault Hub",
        brandHref: "/",
        workspaceLabel: "",
        sportSlug: "",
        sportAccent: "#38bdf8",
        breadcrumbs: [{ label: "Vault Hub", href: "/" }],
        navLinks: [],
        showDisclaimer: true,
    });
}

function showLoadingGrid() {
    const grid = document.getElementById("sportsGrid");
    if (grid && window.CardVault) {
        grid.innerHTML = window.CardVault.renderSkeletonCard(3);
    }
}

async function init() {
    mountHubShell();
    showLoadingGrid();

    try {
        const sports = await loadSportsManifest();
        renderSummary(sports);
        renderSportsGrid(sports);
    } catch (error) {
        console.error(error);
        const grid = document.getElementById("sportsGrid");
        if (grid && window.CardVault) {
            grid.innerHTML = window.CardVault.renderEmptyState(
                "Unable to load vault manifest",
                error.message,
                "Rebuild the static site to regenerate data/sports.json."
            );
        }
    }
}

document.addEventListener("DOMContentLoaded", init);
