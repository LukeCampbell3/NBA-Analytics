async function loadSportsManifest() {
    const response = await fetch(`data/sports.json?v=${Date.now()}`);
    if (!response.ok) {
        throw new Error(`Failed to load prediction manifest (HTTP ${response.status})`);
    }
    return response.json();
}

function predictionPages(sport) {
    return (sport.pages || []).filter((page) => page.slug === "predictions" || page.slug === "prediction-about");
}

function renderSummary(sports) {
    const sportCount = document.getElementById("sportCount");
    const boardCount = document.getElementById("boardCount");
    const methodCount = document.getElementById("methodCount");
    const surfaceCount = document.getElementById("predictionSurfaceCount");

    const pages = sports.flatMap(predictionPages);
    const boards = pages.filter((page) => page.slug === "predictions").length;
    const methods = pages.filter((page) => page.slug === "prediction-about").length;

    if (sportCount) sportCount.textContent = String(sports.length);
    if (boardCount) boardCount.textContent = String(boards);
    if (methodCount) methodCount.textContent = String(methods);
    if (surfaceCount) surfaceCount.textContent = String(pages.length);
}

function renderSportsGrid(sports) {
    const grid = document.getElementById("sportsGrid");
    if (!grid) return;

    const cv = window.CardVault;
    if (!cv) {
        console.error("CardVault not loaded");
        return;
    }

    const predictionSports = (sports || [])
        .map((sport) => ({ ...sport, pages: predictionPages(sport) }))
        .filter((sport) => sport.pages.some((page) => page.slug === "predictions"));

    if (!predictionSports.length) {
        grid.innerHTML = cv.renderEmptyState(
            "No prediction pages found",
            "No sport prediction boards were discovered in the current build.",
            "Add a predictions.html page under sports/<slug>/web to publish a board."
        );
        return;
    }

    grid.innerHTML = predictionSports.map((sport) => cv.renderSportWorkspaceCard({
        ...sport,
        entry_href: sport.pages.find((page) => page.slug === "predictions")?.href || sport.entry_href,
    })).join("");
}

function mountHubShell() {
    if (!window.CardVaultShell) return;

    window.CardVaultShell.mount({
        brandTitle: "Prediction Analytics",
        brandHref: "/",
        workspaceLabel: "",
        sportSlug: "",
        sportAccent: "#2563eb",
        breadcrumbs: [{ label: "Prediction Desk", href: "/" }],
        navLinks: [],
        showDisclaimer: true,
    });
}

function showLoadingGrid() {
    const grid = document.getElementById("sportsGrid");
    if (grid && window.CardVault) {
        grid.innerHTML = window.CardVault.renderSkeletonCard(2);
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
                "Unable to load prediction manifest",
                error.message,
                "Rebuild the static site to regenerate data/sports.json."
            );
        }
    }
}

document.addEventListener("DOMContentLoaded", init);
