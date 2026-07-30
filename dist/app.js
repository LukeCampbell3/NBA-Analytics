async function fetchJson(path) {
    const response = await fetch(`${path}?v=${Date.now()}`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}

function predictionPages(sport) {
    return (sport.pages || []).filter((page) => page.slug === "predictions" || page.slug === "prediction-about");
}

function pageHref(sport, slug) {
    return predictionPages(sport).find((page) => page.slug === slug)?.href || `/${sport.slug}/${slug}/`;
}

function publicationState(data) {
    const raw = String(data?.publication_status || "ready").toLowerCase();
    if (raw === "ready" || raw === "published") {
        return { tone: "active", label: "Published" };
    }
    if (raw === "review") {
        return { tone: "stale", label: "Review" };
    }
    return { tone: "withheld", label: "Withheld" };
}

function topSignal(data) {
    const plays = Array.isArray(data?.plays) ? data.plays : [];
    if (!plays.length) return "n/a";
    const sorted = plays.slice().sort((a, b) => {
        const ev = (Number(b.ev) || 0) - (Number(a.ev) || 0);
        if (Math.abs(ev) > 1e-9) return ev;
        return (Number(b.abs_edge) || Number(b.edge) || 0) - (Number(a.abs_edge) || Number(a.edge) || 0);
    });
    const play = sorted[0];
    const value = play.ev != null ? Number(play.ev) * 100 : Number(play.abs_edge ?? play.edge);
    if (!Number.isFinite(value)) return "n/a";
    return play.ev != null ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}%` : `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function boardSize(data) {
    const plays = Array.isArray(data?.plays) ? data.plays.length : 0;
    const summaryCount = Number(data?.summary?.play_count);
    return String(plays || (Number.isFinite(summaryCount) ? summaryCount : 0));
}

function renderBoardCard(sport, data) {
    const cv = window.CardVault;
    const state = publicationState(data);
    const throughDate = data?.through_date || "n/a";
    const runDate = data?.run_date || "n/a";
    const boardHref = pageHref(sport, "predictions");
    const methodHref = pageHref(sport, "prediction-about");
    const actionLabel = state.label === "Published" ? "Open board" : "Review board";

    return `
        <article class="desk-board-card" style="--sport-accent:${cv.escapeAttr(sport.accent)};">
            <div class="desk-board-card__topline">
                ${cv.renderStatusPill(state.tone, state.label)}
                <span class="desk-board-card__run">Run ${cv.escapeHtml(runDate)}</span>
            </div>
            <h3>${cv.escapeHtml(sport.title)}</h3>
            <p class="desk-board-card__tagline">${cv.escapeHtml(sport.tagline)}</p>
            <div class="desk-board-card__metrics">
                <div class="desk-board-card__metric">
                    <span>Board size</span>
                    <strong>${cv.escapeHtml(boardSize(data))}</strong>
                </div>
                <div class="desk-board-card__metric">
                    <span>Data through</span>
                    <strong>${cv.escapeHtml(throughDate)}</strong>
                </div>
                <div class="desk-board-card__metric">
                    <span>Top signal</span>
                    <strong>${cv.escapeHtml(topSignal(data))}</strong>
                </div>
            </div>
            <div class="desk-board-card__actions">
                <a class="desk-board-card__primary" href="${cv.escapeAttr(boardHref)}">${cv.escapeHtml(actionLabel)}</a>
                <a class="desk-board-card__secondary" href="${cv.escapeAttr(methodHref)}">Methodology</a>
            </div>
        </article>
    `;
}

function mountShell() {
    if (!window.CardVaultShell) return;
    window.CardVaultShell.mount({
        brandTitle: "Prediction Desk",
        brandHref: "/",
        sportSlug: "",
        sportAccent: "#2563eb",
        navLinks: [],
        showDisclaimer: true,
    });
}

async function init() {
    mountShell();
    const grid = document.getElementById("sportsGrid");
    const summary = document.getElementById("deskSummary");

    try {
        const manifest = await fetchJson("data/sports.json");
        const sports = (manifest || [])
            .map((sport) => ({ ...sport, pages: predictionPages(sport) }))
            .filter((sport) => sport.pages.some((page) => page.slug === "predictions"));
        const results = await Promise.all(sports.map(async (sport) => {
            try {
                return { sport, data: await fetchJson(`${sport.slug}/data/daily_predictions.json`) };
            } catch (error) {
                return { sport, data: null, error };
            }
        }));

        grid.innerHTML = results.map(({ sport, data, error }) => {
            if (error || !data) {
                return `<article class="desk-board-error">${window.CardVault.escapeHtml(sport.title)} board data is unavailable.</article>`;
            }
            return renderBoardCard(sport, data);
        }).join("");

        const publishedCount = results.filter(({ data }) => publicationState(data).label === "Published").length;
        summary.textContent = `${publishedCount} published / ${sports.length} total boards`;
    } catch (error) {
        console.error(error);
        grid.innerHTML = `<div class="desk-board-error">Unable to load prediction workspaces: ${window.CardVault.escapeHtml(error.message)}</div>`;
        summary.textContent = "Board status unavailable";
    }
}

document.addEventListener("DOMContentLoaded", init);
