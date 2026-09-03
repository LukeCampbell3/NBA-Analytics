async function fetchSports() {
    const response = await fetch("data/sports.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

function mountShell() {
    if (!window.CardVaultShell) return;
    window.CardVaultShell.mount({
        brandHref: "/",
        sportSlug: "",
        navLinks: [],
        showDisclaimer: true,
    });
}

function renderSportCard(sport) {
    const cv = window.CardVault;
    const href = sport.entry_href || `/${sport.slug}/predictions/`;
    return `
        <article class="sport-card" style="--card-accent:${cv.escapeAttr(sport.accent)};">
            <h3>${cv.escapeHtml(sport.title)}</h3>
            <p>${cv.escapeHtml(sport.tagline)}</p>
            <a class="sport-card__link" href="${cv.escapeAttr(href)}">View ${cv.escapeHtml(sport.slug.toUpperCase())} &rarr;</a>
        </article>`;
}

async function init() {
    mountShell();
    const grid = document.getElementById("sportsGrid");
    const summary = document.getElementById("deskSummary");
    try {
        const sports = await fetchSports();
        grid.innerHTML = sports.map(renderSportCard).join("");
        summary.textContent = `${sports.length} sport${sports.length === 1 ? "" : "s"} publishing predictions`;
    } catch (error) {
        grid.innerHTML = '<div class="site-error">Sport catalog is temporarily unavailable.</div>';
        summary.textContent = "Catalog unavailable";
    }
}

document.addEventListener("DOMContentLoaded", init);
