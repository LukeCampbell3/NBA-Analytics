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

// Golf's model pipeline exists but has not published a live route yet
// (no predictions.html, no generated data) -- shown honestly as
// unavailable rather than either hidden or linked to a page that 404s.
function renderComingSoonCard(label) {
    const cv = window.CardVault;
    return `
        <article class="sport-card sport-card--unavailable">
            <h3>${cv.escapeHtml(label)}</h3>
            <p>Model in development. Not yet publishing predictions.</p>
            <span class="sport-card__link">Coming soon</span>
        </article>`;
}

async function init() {
    mountShell();
    const grid = document.getElementById("sportsGrid");
    const summary = document.getElementById("deskSummary");
    try {
        const sports = await fetchSports();
        const known = new Set(sports.map((s) => s.slug));
        let html = sports.map(renderSportCard).join("");
        if (!known.has("golf")) html += renderComingSoonCard("Golf");
        grid.innerHTML = html;
        summary.textContent = `${sports.length} sport${sports.length === 1 ? "" : "s"} publishing predictions`;
    } catch (error) {
        grid.innerHTML = '<div class="site-error">Sport catalog is temporarily unavailable.</div>';
        summary.textContent = "Catalog unavailable";
    }
}

document.addEventListener("DOMContentLoaded", init);
