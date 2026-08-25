class F1PredictionBoard {
    constructor() {
        this.data = null;
        this.el = {
            eventTitle: document.getElementById("eventTitle"),
            eventCountdown: document.getElementById("eventCountdown"),
            eventMeta: document.getElementById("eventMeta"),
            runFacts: document.getElementById("runFacts"),
            status: document.getElementById("status"),
            metrics: document.getElementById("metrics"),
            table: document.getElementById("projectionTable"),
            plays: document.getElementById("plays"),
            picksSummary: document.getElementById("picksSummary"),
        };
        this.init();
    }

    async init() {
        this.mountShell();
        try {
            const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`, { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
            this.render();
        } catch (error) {
            this.el.runFacts.textContent = `Unable to load the Formula 1 board: ${error.message}`;
            this.el.status.innerHTML = "<p>The current board is unavailable.</p>";
        }
    }

    mountShell() {
        window.CardVaultShell?.mount({
            brandTitle: "In The Cards Analytics", brandHref: "/", sportSlug: "f1", sportAccent: "#d00000",
            navLinks: [
                { label: "Predictions", href: "/f1/predictions/", active: true },
                { label: "Method", href: "/f1/prediction-about/", active: false },
            ],
            showDisclaimer: true,
        });
    }

    render() {
        const event = this.data.event;
        const quality = this.data.data_quality || {};
        const model = this.data.model || {};
        const market = this.data.market || {};
        const projections = Array.isArray(this.data.projections) ? this.data.projections : [];
        const plays = Array.isArray(this.data.plays) ? this.data.plays : [];
        this.el.eventTitle.textContent = event?.race_name || "No Upcoming Race";
        this.renderEvent(event, projections, market);
        this.el.runFacts.innerHTML = [
            `Board ${this.formatDate(this.data.run_date, { month: "short", day: "numeric" })}`,
            `${quality.market_observations || 0} live winner prices`,
            quality.starting_grid_positions ? `${quality.starting_grid_positions} grid positions known` : "Grid pending",
        ].map((item) => `<span>${this.escape(item)}</span>`).join("");
        this.el.status.innerHTML = `<p><strong>${quality.status === "shadow" ? "Next-event board ready." : "Market signals withheld."}</strong> ${this.escape(quality.reason || "No status detail supplied.")}</p>`;
        this.el.metrics.innerHTML = this.cards([
            ["Model races", model.training_races], ["Training rows", model.training_rows],
            ["Market offers", quality.market_observations], ["Edge candidates", plays.length],
        ]);
        this.renderPlays(plays);
        this.renderTable(projections);
    }

    renderEvent(event, projections, market) {
        if (!event) {
            this.el.eventCountdown.textContent = "Schedule unavailable";
            this.el.eventMeta.innerHTML = "";
            return;
        }
        const raceDate = new Date(`${event.date}T${event.time_utc || "00:00:00Z"}`);
        const days = Math.ceil((raceDate.getTime() - Date.now()) / 86400000);
        this.el.eventCountdown.textContent = days > 1 ? `Race in ${days} days` : days === 1 ? "Race tomorrow" : days === 0 ? "Race day" : "Race completed";
        const timeLabel = Number.isNaN(raceDate.getTime())
            ? event.time_utc || "Time pending"
            : new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(raceDate);
        const source = market.provider === "free_exchange_consensus" ? "Free exchanges" : market.provider || "No market";
        this.el.eventMeta.innerHTML = [
            ["Round", `${event.round} of ${event.season}`],
            ["Race start", timeLabel],
            ["Circuit", event.circuit || "TBD"],
            ["Field / source", `${projections.length} drivers \u00b7 ${source}`],
        ].map(([label, value]) => `<div><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></div>`).join("");
    }

    renderTable(rows) {
        if (!rows.length) {
            this.el.table.innerHTML = '<div class="prediction-about-empty">No upcoming driver field is available.</div>';
            return;
        }
        const body = rows.map((row) => `<tr>
            <td><span class="f1-rank">${this.escape(row.model_rank)}</span><strong>${this.escape(row.driver)}</strong><span class="f1-constructor">${this.escape(row.constructor)}</span></td>
            <td>${row.grid_position ?? "\u2014"}</td>
            <td><strong>${this.pct(row.win_probability)}</strong></td>
            <td>${this.pct(row.podium_probability)}</td>
            <td>${this.pct(row.top6_probability)}</td>
            <td>${this.pct(row.market_probability)}</td>
            <td class="${Number(row.edge) > 0 ? "f1-positive" : ""}">${this.signedPct(row.edge)}</td>
            <td>${row.best_price == null ? "\u2014" : `${this.formatPrice(row.best_price)} \u00b7 ${this.escape(row.best_book)}`}</td>
        </tr>`).join("");
        this.el.table.innerHTML = `<table class="prediction-about-table"><thead><tr><th>Driver</th><th>Grid</th><th>Win</th><th>Podium</th><th>Top 6</th><th>Market</th><th>Edge</th><th>Best price</th></tr></thead><tbody>${body}</tbody></table>`;
    }

    renderPlays(plays) {
        if (!plays.length) {
            this.el.picksSummary.textContent = "No qualifying edge at current prices";
            this.el.plays.innerHTML = '<div class="prediction-about-empty">No race-winner price currently clears the shadow edge gate.</div>';
            return;
        }
        const ranked = [...plays].sort((a, b) => Number(b.edge) - Number(a.edge));
        this.el.picksSummary.textContent = `${ranked.length} shadow pick${ranked.length === 1 ? "" : "s"} clear the 3% edge gate`;
        this.el.plays.innerHTML = ranked.map((play, index) => {
            const modelWidth = Math.min(100, Number(play.win_probability) * 100);
            const marketWidth = Math.min(100, Number(play.market_probability) * 100);
            return `<article class="f1-pick">
                <div class="f1-pick-topline"><span class="f1-pick-number">${index + 1}</span><span class="f1-edge-badge">${this.signedPct(play.edge)} edge</span></div>
                <h3>${this.escape(play.driver)}</h3>
                <p class="f1-pick-team">${this.escape(play.constructor)} \u00b7 Model rank ${this.escape(play.model_rank)}</p>
                <div class="f1-probability-row"><span>Model win</span><strong>${this.pct(play.win_probability)}</strong></div>
                <div class="f1-probability-track"><span style="width:${modelWidth}%"></span></div>
                <div class="f1-probability-row"><span>Market implied</span><strong>${this.pct(play.market_probability)}</strong></div>
                <div class="f1-probability-track is-market"><span style="width:${marketWidth}%"></span></div>
                <div class="f1-price-row"><span>Best YES ask</span><strong>${this.formatPrice(play.best_price)}</strong><small>${this.escape(play.best_book)}</small></div>
            </article>`;
        }).join("");
    }

    cards(items) {
        return items.map(([label, value]) => `<article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value ?? "n/a")}</strong></article>`).join("");
    }
    pct(value) { return value !== null && value !== "" && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "\u2014"; }
    signedPct(value) { return value !== null && value !== "" && Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "\u2014"; }
    num(value) { return value !== null && value !== "" && Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "n/a"; }
    formatPrice(value) { return value !== null && value !== "" && Number.isFinite(Number(value)) ? `${Number(value) > 0 ? "+" : ""}${Math.round(Number(value))}` : "\u2014"; }
    formatDate(value, options = {}) {
        const parsed = new Date(`${value}T12:00:00Z`);
        return Number.isNaN(parsed.getTime()) ? value || "n/a" : new Intl.DateTimeFormat(undefined, { timeZone: "UTC", ...options }).format(parsed);
    }
    escape(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;"); }
}
document.addEventListener("DOMContentLoaded", () => new F1PredictionBoard());
