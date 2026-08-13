class NflPredictionBoard {
    constructor() {
        this.data = null;
        this.marketEvidence = null;
        this.elements = {
            runFacts: document.getElementById("runFacts"),
            gate: document.getElementById("gateSummary"),
            overall: document.getElementById("overallMetrics"),
            board: document.getElementById("currentBoard"),
            parlay: document.getElementById("dailyParlay"),
            marketStatus: document.getElementById("marketReplayStatus"),
            marketMetrics: document.getElementById("marketReplayMetrics"),
            marketBaselines: document.getElementById("marketBaselines"),
            marketWeekly: document.getElementById("marketWeekly"),
        };
        this.init();
    }

    async init() {
        this.mountShell();
        try {
            const [dailyResponse, marketResponse] = await Promise.all([
                fetch(`data/daily_predictions.json?v=${Date.now()}`),
                fetch(`data/market_validation_summary.json?v=${Date.now()}`),
            ]);
            if (!dailyResponse.ok) throw new Error(`HTTP ${dailyResponse.status}`);
            this.data = await dailyResponse.json();
            this.marketEvidence = marketResponse.ok ? await marketResponse.json() : null;
            this.render();
        } catch (error) {
            console.error(error);
            this.elements.runFacts.textContent = `Unable to load NFL picks: ${error.message}`;
        }
    }

    mountShell() {
        if (!window.CardVaultShell) return;
        window.CardVaultShell.mount({
            brandTitle: "Prediction Bounties",
            brandHref: "/",
            sportSlug: "nfl",
            sportAccent: "#b42318",
            navLinks: [
                { label: "Picks", href: "/nfl/predictions/", active: true },
                { label: "Fantasy Draft", href: "/nfl/fantasy/", active: false },
                { label: "Method", href: "/nfl/prediction-about/", active: false },
            ],
            showDisclaimer: true,
        });
    }

    render() {
        const plays = Array.isArray(this.data.plays) ? this.data.plays : [];
        const quality = this.data.data_quality || {};
        const selection = this.data.selection || {};
        const shadow = this.data.mode === "live_shadow";
        this.elements.runFacts.innerHTML = [
            `Slate ${this.escape(this.data.run_date || "n/a")}`,
            `Generated ${this.escape(this.formatTime(this.data.generated_at_utc))}`,
            `${plays.length} candidate${plays.length === 1 ? "" : "s"}`,
            `${this.escape(this.formatInt(quality.complete_market_observations))} market observations`,
            shadow ? "Shadow mode" : "Historical report",
        ].map((item) => `<span>${item}</span>`).join("");

        const withheld = this.data.publication_status !== "shadow_current_pool";
        this.elements.gate.innerHTML = `<p><strong>${withheld ? "No current pick published." : "Current candidates found."}</strong> ${this.escape(quality.reason || "These candidates passed the frozen model and execution gates but are not authorized for staking while prospective certification is inactive.")}</p>`;
        const evidence = this.data.historical_evidence || this.marketEvidence?.final_test || {};
        const cards = [
            ["Candidates", this.formatInt(plays.length)],
            ["Books Required", this.formatInt(selection.minimum_books)],
            ["Price Range", this.formatPriceRange(selection.american_price_range)],
            ["Locked Record", evidence.wins != null ? `${this.formatInt(evidence.wins)}-${this.formatInt(evidence.losses)}` : "n/a"],
            ["Locked Hit Rate", this.formatPct(evidence.hit_rate)],
            ["Locked ROI", this.formatSignedPct(evidence.roi)],
        ];
        this.elements.overall.innerHTML = cards.map(([label, value]) => `
            <article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></article>
        `).join("");
        this.renderBoard(plays);
        this.renderParlay();
        this.renderMarketReplay();
    }

    renderBoard(plays) {
        if (!plays.length) {
            this.elements.board.innerHTML = "<p>No playable passing-yard candidate survived this slate.</p>";
            return;
        }
        const rows = plays.map((play) => `<tr>
            <td>${this.escape(play.player)}</td>
            <td>${this.escape(`${play.team || "?"} vs ${play.opponent || "?"}`)}</td>
            <td><strong>${this.escape(`${play.direction} ${this.formatNum(play.line, 1)}`)}</strong></td>
            <td>${this.escape(this.formatAmerican(play.selected_side_price))}</td>
            <td>${this.escape(play.selected_sportsbook_key || "n/a")}</td>
            <td>${this.escape(this.formatPct(play.model_hit_probability))}</td>
            <td>${this.escape(this.formatInt(play.market_books))}</td>
        </tr>`).join("");
        this.elements.board.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Player</th><th>Matchup</th><th>Play</th><th>Odds</th><th>Book</th><th>Model</th><th>Books</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
    }

    renderParlay() {
        const parlay = this.data.daily_parlay || {};
        const ticket = parlay.selected_ticket;
        if (!ticket) {
            this.elements.parlay.innerHTML = `<p><strong>Withheld.</strong> ${this.escape(parlay.reason || "No distinct-game ticket was available.")}</p>`;
            return;
        }
        const legs = (ticket.legs || []).map((leg) => `${leg.player} ${leg.direction} ${this.formatNum(leg.line, 1)}`).join(" + ");
        this.elements.parlay.innerHTML = `<p><strong>Shadow ticket only:</strong> ${this.escape(legs)} at ${this.escape(ticket.sportsbook_key)}. ${this.escape(parlay.reason || "The parlay policy is not authorized.")}</p>`;
    }

    renderMarketReplay() {
        const evidence = this.marketEvidence;
        if (!evidence) {
            this.elements.marketStatus.innerHTML = "<p>Locked market replay evidence is unavailable.</p>";
            this.elements.marketMetrics.innerHTML = "";
            this.elements.marketBaselines.innerHTML = "";
            this.elements.marketWeekly.innerHTML = "";
            return;
        }
        const final = evidence.final_test || {};
        const policy = evidence.locked_policy || {};
        const deployment = evidence.gates?.deployment || {};
        const stats = evidence.statistical_evidence || {};
        this.elements.marketStatus.innerHTML = `<p><strong>Singles passed the historical holdout; live authorization remains ${this.escape(deployment.status || "blocked")}.</strong> ${this.escape(deployment.reason || "Prospective evidence is required.")}</p>`;
        const cards = [
            ["Validated Market", (evidence.validated_targets || []).join(", ") || "n/a"],
            ["Weekly Cap", this.formatInt(policy.weekly_top_n)],
            ["Record", `${this.formatInt(final.wins)}-${this.formatInt(final.losses)}`],
            ["Hit Rate", this.formatPct(final.hit_rate)],
            ["ROI", this.formatSignedPct(final.roi)],
            ["Profit", `${this.formatSignedNum(final.profit_units, 2)}u`],
            ["Clustered Hit 95%", this.formatRange(stats.week_cluster_hit_rate_95, false)],
            ["Clustered ROI 95%", this.formatRange(stats.week_cluster_roi_95, true)],
        ];
        this.elements.marketMetrics.innerHTML = cards.map(([label, value]) => `
            <article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></article>
        `).join("");

        const baselines = evidence.baselines || {};
        const baselineRows = [
            ["Production selector", final],
            ["Always under", baselines.always_under || {}],
            ["Point projection side", baselines.point_projection_side || {}],
        ].map(([label, row]) => `<tr>
            <td>${this.escape(label)}</td><td>${this.escape(this.formatInt(row.graded_decisions))}</td>
            <td>${this.escape(`${this.formatInt(row.wins)}-${this.formatInt(row.losses)}`)}</td>
            <td>${this.escape(this.formatPct(row.hit_rate))}</td><td>${this.escape(this.formatSignedPct(row.roi))}</td>
        </tr>`).join("");
        this.elements.marketBaselines.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Policy</th><th>N</th><th>Record</th><th>Hit rate</th><th>ROI</th></tr></thead>
            <tbody>${baselineRows}</tbody>
        </table>`;

        const weeklyRows = (evidence.weekly || []).map((row) => `<tr>
            <td>W${this.escape(this.formatInt(row.week))}</td><td>${this.escape(this.formatInt(row.picks))}</td>
            <td>${this.escape(`${this.formatInt(row.wins)}-${this.formatInt(row.losses)}`)}</td>
            <td>${this.escape(this.formatPct(row.hit_rate))}</td><td>${this.escape(this.formatSignedPct(row.roi))}</td>
            <td>${this.escape(`${this.formatSignedNum(row.profit_units, 2)}u`)}</td>
        </tr>`).join("");
        this.elements.marketWeekly.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Week</th><th>Picks</th><th>Record</th><th>Hit rate</th><th>ROI</th><th>Units</th></tr></thead>
            <tbody>${weeklyRows}</tbody>
        </table>`;
    }

    formatTime(value) {
        if (!value) return "n/a";
        const parsed = new Date(value);
        return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    }
    formatPct(value) { return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a"; }
    formatSignedPct(value) { return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "n/a"; }
    formatSignedNum(value, places = 2) { return Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(places)}` : "n/a"; }
    formatRange(values, signed = false) {
        if (!Array.isArray(values) || values.length !== 2) return "n/a";
        const formatter = signed ? this.formatSignedPct.bind(this) : this.formatPct.bind(this);
        return `${formatter(values[0])}-${formatter(values[1])}`;
    }
    formatNum(value, places = 2) { return Number.isFinite(Number(value)) ? Number(value).toFixed(places) : "n/a"; }
    formatInt(value) { return Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "n/a"; }
    formatAmerican(value) { return Number.isFinite(Number(value)) ? `${Number(value) > 0 ? "+" : ""}${Math.round(Number(value))}` : "n/a"; }
    formatPriceRange(values) { return Array.isArray(values) && values.length === 2 ? `${this.formatAmerican(values[0])} to ${this.formatAmerican(values[1])}` : "n/a"; }
    escape(value) {
        return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }
}

document.addEventListener("DOMContentLoaded", () => new NflPredictionBoard());
