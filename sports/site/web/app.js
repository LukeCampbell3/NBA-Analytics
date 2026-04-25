async function loadSportsManifest() {
    const response = await fetch(`data/sports.json?v=${Date.now()}`);
    if (!response.ok) {
        throw new Error(`Failed to load sports manifest (HTTP ${response.status})`);
    }
    return response.json();
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
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

    if (!Array.isArray(sports) || sports.length === 0) {
        grid.innerHTML = `
            <article class="sport-card sport-card-empty">
                <p>No sport workspaces were discovered in the current build.</p>
            </article>
        `;
        return;
    }

    grid.innerHTML = sports.map((sport) => {
        const pages = Array.isArray(sport.pages) ? sport.pages : [];
        const pageLinks = pages.slice(0, 5).map((page) => `
            <a class="page-chip" href="${escapeHtml(page.href)}">${escapeHtml(page.label)}</a>
        `).join("");

        return `
            <article class="sport-card" style="--sport-accent:${escapeHtml(sport.accent)}; --sport-surface:${escapeHtml(sport.surface)};">
                <div class="sport-card-top">
                    <span class="sport-status">${escapeHtml(sport.status_label)}</span>
                    <span class="sport-slug">/${escapeHtml(sport.slug)}</span>
                </div>
                <h3>${escapeHtml(sport.title)}</h3>
                <p class="sport-tagline">${escapeHtml(sport.tagline)}</p>
                <p class="sport-summary">${escapeHtml(sport.summary)}</p>
                <div class="page-chip-row">
                    ${pageLinks || '<span class="page-chip page-chip-muted">No published pages yet</span>'}
                </div>
                <a class="sport-cta" href="${escapeHtml(sport.entry_href)}">Open ${escapeHtml(sport.title)}</a>
            </article>
        `;
    }).join("");
}

async function init() {
    try {
        const sports = await loadSportsManifest();
        renderSummary(sports);
        renderSportsGrid(sports);
    } catch (error) {
        console.error(error);
        const grid = document.getElementById("sportsGrid");
        if (grid) {
            grid.innerHTML = `
                <article class="sport-card sport-card-empty">
                    <p>${escapeHtml(error.message)}</p>
                </article>
            `;
        }
    }
}

document.addEventListener("DOMContentLoaded", init);
