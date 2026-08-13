class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.availableDates = [];
        this.activeDate = null;
        this.currentDate = null;
        this.activeParlayLegCount = null;
        this.elements = {
            cards: document.getElementById("predictionCards"),
            empty: document.getElementById("predictionEmpty"),
            runMeta: document.getElementById("predictionRunMeta"),
            parlaySection: document.getElementById("dailyParlaySection"),
            parlayContent: document.getElementById("dailyParlayContent"),
            poolTitle: document.getElementById("predictionPoolTitle"),
            dateNav: document.getElementById("predictionDateNav"),
        };
        this.init();
    }

    init() {
        this.mountShell();
        if (window.CardVault && this.elements.cards) {
            this.elements.cards.innerHTML = window.CardVault.renderSkeletonCard(6);
        }
        this.loadDatesAndRender();
    }

    mountShell() {
        if (!window.CardVaultShell) return;

        window.CardVaultShell.mount({
            brandTitle: "Prediction Bounties",
            brandHref: "/",
            sportSlug: "mlb",
            sportAccent: "#087f5b",
            navLinks: [
                { label: "Board", href: "/mlb/predictions/", active: true },
                { label: "Method", href: "/mlb/prediction-about/", active: false },
            ],
            showDisclaimer: true,
        });
    }

    async loadDatesAndRender() {
        await this.loadDateIndex();
        const currentLoaded = await this.loadAndRender(null);
        if (!currentLoaded && this.availableDates.length) {
            await this.loadAndRender(this.availableDates[0]);
        }
        this.renderDateNav();
    }

    async loadDateIndex() {
        try {
            const response = await fetch(`data/history/index.json?v=${Date.now()}`);
            if (!response.ok) return;
            const index = await response.json();
            this.availableDates = Array.isArray(index.dates)
                ? index.dates
                    .map((date) => String(date))
                    .filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date))
                    .sort()
                    .reverse()
                : [];
        } catch (_) { /* history is optional */ }
    }

    async loadAndRender(date) {
        try {
            await this.load(date);
            this.renderDailyParlay();
            this.renderCards();
            return true;
        } catch (error) {
            console.error(error);
            if (window.CardVault && this.elements.cards) {
                this.elements.cards.innerHTML = window.CardVault.renderEmptyState(
                    "Board unavailable",
                    `Unable to load MLB predictions: ${error.message}`,
                    "Check that data/daily_predictions.json exists for this build."
                );
            }
            return false;
        }
    }

    async load(date) {
        const url = date
            ? `data/history/${date}.json?v=${Date.now()}`
            : `data/daily_predictions.json?v=${Date.now()}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.data = await response.json();
        this.activeDate = this.data?.run_date || date || null;
        if (!date) this.currentDate = this.activeDate;
        const publicationStatus = String(this.data?.publication_status || "ready").toLowerCase();
        this.plays = Array.isArray(this.data.plays)
            ? this.data.plays.map((play) => ({ ...play, board_publication_status: publicationStatus }))
            : [];
        this.plays.sort((a, b) => {
            const parlayDiff = Number(Boolean(b.parlay_candidate)) - Number(Boolean(a.parlay_candidate));
            if (parlayDiff !== 0) return parlayDiff;
            const evDiff = (Number(b.ev) || 0) - (Number(a.ev) || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (Number(b.abs_edge) || Number(b.edge) || 0) - (Number(a.abs_edge) || Number(a.edge) || 0);
        });
        this.renderRunMeta();
        const authorizationEnabled = Boolean(this.data?.policy_governance?.candidate_authorization_enabled);
        if (this.elements.poolTitle) {
            this.elements.poolTitle.textContent = authorizationEnabled ? "Authorized Pool" : "Shadow Candidate Pool";
        }
    }

    renderDateNav() {
        const nav = this.elements.dateNav;
        if (!nav) return;

        const dates = [this.currentDate, ...this.availableDates]
            .filter((date, index, values) => date && values.indexOf(date) === index);
        if (dates.length < 2) {
            nav.innerHTML = "";
            return;
        }

        const buttons = dates.map((date) => {
            const isActive = date === this.activeDate;
            return `<button type="button" class="date-nav__btn${isActive ? " is-active" : ""}" data-date="${this.escapeHtml(date)}" aria-pressed="${isActive}">${this.escapeHtml(this.formatDateLabel(date))}</button>`;
        }).join("");
        nav.innerHTML = `<div class="date-nav__scroll">${buttons}</div>`;

        nav.querySelectorAll(".date-nav__btn").forEach((button) => {
            button.addEventListener("click", async () => {
                const date = button.dataset.date;
                if (date === this.activeDate) return;
                if (this.elements.cards) {
                    this.elements.cards.innerHTML = window.CardVault
                        ? window.CardVault.renderSkeletonCard(4)
                        : "";
                }
                await this.loadAndRender(date === this.currentDate ? null : date);
                this.renderDateNav();
            });
        });
    }

    formatDateLabel(dateValue) {
        try {
            const displayDate = new Date(`${dateValue}T12:00:00`);
            const today = new Date().toISOString().slice(0, 10);
            if (dateValue === this.currentDate) return dateValue === today ? "Today" : "Current";
            return displayDate.toLocaleDateString("en-US", { month: "short", day: "numeric" });
        } catch (_) {
            return dateValue;
        }
    }

    renderRunMeta() {
        const runDate = this.data?.run_date || "n/a";
        const throughDate = this.data?.through_date || "n/a";
        const policy = this.data?.policy_profile || "n/a";
        const publicationStatus = String(this.data?.publication_status || "ready").toLowerCase();
        const authorizationEnabled = Boolean(this.data?.policy_governance?.candidate_authorization_enabled);
        const publicationLabel = !authorizationEnabled ? "Shadow only" : publicationStatus === "ready" ? "Published" : publicationStatus === "review" ? "Review" : "Withheld";
        const publicationTone = !authorizationEnabled ? "stale" : publicationStatus === "ready" ? "active" : publicationStatus === "review" ? "stale" : "withheld";
        const stale = publicationStatus !== "ready";
        const quality = this.data?.data_quality || {};
        const lagText = Number.isFinite(Number(quality.lag_days)) ? `${Number(quality.lag_days)}d` : "n/a";

        if (this.elements.runMeta && window.CardVault) {
            this.elements.runMeta.innerHTML = `
                ${window.CardVault.renderStatusPill(publicationTone, publicationLabel)}
                <span class="prediction-run-meta__item">Run <strong>${this.escapeHtml(runDate)}</strong></span>
                <span class="prediction-run-meta__item">Data through <strong>${this.escapeHtml(throughDate)}</strong></span>
                <span class="prediction-run-meta__item">Lag <strong>${this.escapeHtml(lagText)}</strong></span>
                <span class="prediction-run-meta__item">Signals <strong>${this.plays.length}</strong></span>
                <span class="prediction-run-meta__item">Policy <strong>${this.escapeHtml(policy)}</strong></span>
            `;
        } else if (this.elements.runMeta) {
            this.elements.runMeta.textContent = `Run ${runDate} | Data through ${throughDate} | Policy ${policy} | ${publicationLabel}`;
        }
    }

    renderCards() {
        const cv = window.CardVault;
        if (!cv) {
            console.error("CardVault not loaded");
            return;
        }

        if (!this.plays.length) {
            const message = String(this.data?.publication_message || "No analytical signals are available for this run.").trim();
            const emptyEl = this.elements.empty;
            if (emptyEl) {
                emptyEl.style.display = "block";
                const msgP = emptyEl.querySelector("p");
                if (msgP) msgP.textContent = message || "No analytical signals are available for this run.";
            }
            this.elements.cards.innerHTML = "";
            return;
        }

        if (this.elements.empty) {
            this.elements.empty.style.display = "none";
        }

        this.elements.cards.innerHTML = this.plays
            .map((play, index) => cv.renderPredictionCard(play, index))
            .join("");
    }

    renderDailyParlay() {
        const section = this.elements.parlaySection;
        const content = this.elements.parlayContent;
        if (!section || !content) return;

        const parlay = this.data?.daily_parlay || {};
        const selectedTicket = parlay?.selected_ticket || null;
        const ladder = Array.isArray(parlay?.ticket_ladder) && parlay.ticket_ladder.length
            ? parlay.ticket_ladder.filter((item) => item && Array.isArray(item.legs) && item.legs.length)
            : selectedTicket ? [selectedTicket] : [];
        const defaultLegCount = Number(selectedTicket?.leg_count || ladder[0]?.leg_count || 0);
        if (!ladder.some((item) => Number(item.leg_count) === Number(this.activeParlayLegCount))) {
            this.activeParlayLegCount = defaultLegCount;
        }
        const ticket = ladder.find((item) => Number(item.leg_count) === Number(this.activeParlayLegCount)) || selectedTicket;
        const status = String(ticket?.status || parlay?.status || "withheld").toLowerCase();
        const candidateAuthorized = Boolean(ticket?.candidate_authorized);
        const statusLabel = !candidateAuthorized && ticket ? "Shadow only" : status === "ready" ? "Ready" : status === "review" ? "Lineup review" : "Withheld";
        const statusTone = !candidateAuthorized && ticket ? "stale" : status === "ready" ? "active" : status === "review" ? "stale" : "withheld";

        if (!ticket || !Array.isArray(ticket.legs) || !ticket.legs.length) {
            content.innerHTML = `
                <div class="daily-parlay__header">
                    <div>
                        <p class="vault-page-kicker">Parlay ladder</p>
                        <h2 id="dailyParlayTitle">MLB Parlays</h2>
                    </div>
                    ${window.CardVault ? window.CardVault.renderStatusPill(statusTone, statusLabel) : ""}
                </div>
                <p class="daily-parlay__empty">${this.escapeHtml(parlay?.reason || "No ticket cleared today's consistency gates.")}</p>
            `;
            return;
        }

        const legs = ticket.legs.map((leg, index) => {
            const player = leg.player_display_name || leg.player || "Unknown Player";
            const target = window.CardVault ? window.CardVault.formatTargetLabel(leg.target) : String(leg.target || "");
            const line = this.formatNumber(leg.market_line, 1);
            const price = this.formatAmerican(leg.selected_side_price);
            const probability = this.formatPct(leg.estimated_graded_hit_rate);
            const lineup = String(leg.lineup_status || "unconfirmed");
            const lineupLabel = lineup === "confirmed" ? "Confirmed" : lineup === "not_in_posted_lineup" ? "Out" : "Pending";
            return `
                <div class="daily-parlay__leg">
                    <span class="daily-parlay__leg-number">${String(index + 1).padStart(2, "0")}</span>
                    <div class="daily-parlay__leg-copy">
                        <strong>${this.escapeHtml(player)}</strong>
                        <span>${this.escapeHtml(`OVER ${line} ${target}`)}</span>
                    </div>
                    <div class="daily-parlay__leg-market">
                        <strong>${this.escapeHtml(price)}</strong>
                        <span>${this.escapeHtml(probability)} / ${this.escapeHtml(lineupLabel)}</span>
                    </div>
                </div>
            `;
        }).join("");

        const ticketTabs = ladder.length > 1 ? `
            <div class="daily-parlay__tabs" role="tablist" aria-label="Parlay length">
                ${ladder.map((item) => {
                    const legCount = Number(item.leg_count || 0);
                    const active = legCount === Number(this.activeParlayLegCount);
                    return `<button type="button" class="daily-parlay__tab${active ? " is-active" : ""}" data-parlay-legs="${legCount}" role="tab" aria-selected="${active}">${legCount} legs</button>`;
                }).join("")}
            </div>
        ` : "";
        const ticketTier = String(ticket.ticket_tier || "consistency");
        content.innerHTML = `
            <div class="daily-parlay__header">
                <div>
                    <p class="vault-page-kicker">${this.escapeHtml(ticketTier)} / ${this.escapeHtml(ticket.leg_count)} legs</p>
                    <h2 id="dailyParlayTitle">MLB Parlays</h2>
                </div>
                ${window.CardVault ? window.CardVault.renderStatusPill(statusTone, statusLabel) : ""}
            </div>
            ${ticketTabs}
            <div class="daily-parlay__metrics">
                <div><span>Projected hit</span><strong>${this.escapeHtml(this.formatPct(ticket.projected_probability))}</strong></div>
                <div><span>Combined price</span><strong>${this.escapeHtml(this.formatAmerican(ticket.combined_american_price))}</strong></div>
                <div><span>Expected return</span><strong>${this.escapeHtml(this.formatSignedPct(ticket.expected_return_per_unit))}</strong></div>
                <div><span>Sportsbook</span><strong>${this.escapeHtml(ticket.sportsbook || "n/a")}</strong></div>
            </div>
            <div class="daily-parlay__legs">${legs}</div>
            <p class="daily-parlay__state">${this.escapeHtml(parlay.reason || "")}</p>
        `;
        content.querySelectorAll("[data-parlay-legs]").forEach((button) => {
            button.addEventListener("click", () => {
                const legCount = Number(button.dataset.parlayLegs);
                if (!Number.isFinite(legCount) || legCount === Number(this.activeParlayLegCount)) return;
                this.activeParlayLegCount = legCount;
                this.renderDailyParlay();
            });
        });
    }

    formatNumber(value, digits = 2) {
        const number = Number(value);
        return Number.isFinite(number) ? number.toFixed(digits) : "n/a";
    }

    formatPct(value) {
        const number = Number(value);
        return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "n/a";
    }

    formatSignedPct(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return "n/a";
        return `${number >= 0 ? "+" : ""}${(number * 100).toFixed(1)}%`;
    }

    formatAmerican(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return "n/a";
        const rounded = Math.round(number);
        return `${rounded > 0 ? "+" : ""}${rounded}`;
    }

    escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#39;");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    new DailyPredictionsPage();
});
