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
        brandTitle: "Prediction Bounties",
        sportAccent: "#8a5820",
        navLinks: [
            { href: "/pricing/", label: "Pricing" },
            { href: "/login/", label: "Sign in" },
        ],
        showDisclaimer: true,
    });
}

function renderSport(sport) {
    const cv = window.CardVault;
    return `
        <article class="desk-board-card" style="--sport-accent:${cv.escapeAttr(sport.accent)};">
            <div class="desk-board-card__topline">
                ${cv.renderStatusPill("active", cv.escapeHtml(sport.status_label || "Available"))}
                <span class="desk-board-card__run">Members only</span>
            </div>
            <div class="desk-board-card__feature">
                <div>
                    <p class="desk-board-card__eyebrow">${cv.escapeHtml(sport.slug)} model desk</p>
                    <h3>${cv.escapeHtml(sport.title)}</h3>
                    <p class="desk-board-card__tagline">${cv.escapeHtml(sport.tagline)}</p>
                </div>
            </div>
            <p>${cv.escapeHtml(sport.summary)}</p>
            <div class="desk-board-card__actions">
                <a class="desk-board-card__primary" href="/login/">Sign in to view</a>
                <a class="desk-board-card__secondary" href="/pricing/">View membership</a>
            </div>
        </article>`;
}

async function init() {
    mountShell();
    const grid = document.getElementById("sportsGrid");
    const summary = document.getElementById("deskSummary");
    try {
        const sports = await fetchSports();
        grid.innerHTML = sports.map(renderSport).join("");
        summary.textContent = `${sports.length} protected model desks`;
    } catch (error) {
        grid.innerHTML = '<div class="desk-board-error">Membership catalog is temporarily unavailable.</div>';
        summary.textContent = "Catalog unavailable";
    }
}

document.addEventListener("DOMContentLoaded", init);
