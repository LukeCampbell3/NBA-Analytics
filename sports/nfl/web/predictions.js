class NflPredictionBoard {
    constructor() {
        this.data = null;
        this.marketEvidence = null;
        this.weekPool = null;
        this.weekMarketBoard = null;
        this.position = "ALL";
        this.elements = {
            runFacts: document.getElementById("runFacts"),
            gate: document.getElementById("gateSummary"),
            overall: document.getElementById("overallMetrics"),
            board: document.getElementById("currentBoard"),
            parlay: document.getElementById("dailyParlay"),
            weekPoolStatus: document.getElementById("weekPoolStatus"),
            weekPoolMetrics: document.getElementById("weekPoolMetrics"),
            weekProjectionPool: document.getElementById("weekProjectionPool"),
            weekPositionFilters: document.getElementById("weekPositionFilters"),
            parlayWatchlistStatus: document.getElementById("parlayWatchlistStatus"),
            parlayWatchlists: document.getElementById("parlayWatchlists"),
            parlayV2Section: document.getElementById("parlayV2Section"),
            parlayV2Content: document.getElementById("parlayV2Content"),
        };
        this.bindControls();
        this.init();
    }

    bindControls() {
        this.elements.weekPositionFilters?.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-position]");
            if (!button) return;
            this.position = button.dataset.position || "ALL";
            this.elements.weekPositionFilters.querySelectorAll("button").forEach((item) => {
                item.classList.toggle("is-active", item === button);
            });
            this.renderWeekPool();
        });
    }

    async init() {
        this.mountShell();
        try {
            const [dailyResponse, marketResponse, weekResponse, weekMarketResponse] = await Promise.all([
                fetch(`data/daily_predictions.json?v=${Date.now()}`),
                fetch(`data/market_validation_summary.json?v=${Date.now()}`),
                fetch(`data/week_1_pool.json?v=${Date.now()}`),
                fetch(`data/week_1_market_board.json?v=${Date.now()}`),
            ]);
            if (!dailyResponse.ok) throw new Error(`HTTP ${dailyResponse.status}`);
            if (!weekResponse.ok) throw new Error(`Week pool HTTP ${weekResponse.status}`);
            this.data = await dailyResponse.json();
            this.marketEvidence = marketResponse.ok ? await marketResponse.json() : null;
            this.weekPool = await weekResponse.json();
            this.weekMarketBoard = weekMarketResponse.ok ? await weekMarketResponse.json() : null;
            this.render();
        } catch (error) {
            console.error(error);
            this.elements.runFacts.textContent = `Unable to load NFL picks: ${error.message}`;
        }
    }

    mountShell() {
        if (!window.CardVaultShell) return;
        window.CardVaultShell.mount({
            brandTitle: "In The Cards Analytics",
            brandHref: "/",
            sportSlug: "nfl",
            sportAccent: "#b42318",
            navLinks: [
                { label: "Predictions", href: "/nfl/predictions/", active: true },
                { label: "Fantasy", href: "/nfl/fantasy/", active: false },
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
        const week = this.weekPool || {};
        this.elements.runFacts.innerHTML = [
            `${this.escape(week.season || "NFL")} Week ${this.escape(week.week || "n/a")}`,
            `Generated ${this.escape(this.formatTime(week.generated_at_utc))}`,
            `${this.escape(this.formatInt(week.games))} games`,
            `${this.escape(this.formatInt(week.players))} player projections`,
            `${plays.length} candidate${plays.length === 1 ? "" : "s"}`,
            `${this.escape(this.formatInt(quality.complete_market_observations))} market observations`,
            shadow ? "Shadow mode" : "Historical report",
        ].map((item) => `<span>${item}</span>`).join("");

        this.renderWeekPool();
        this.renderParlayWatchlists();

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
        this.renderParlayV2();
    }

    renderWeekPool() {
        const data = this.weekPool || {};
        const pool = Array.isArray(data.pool) ? data.pool : [];
        const visiblePool = this.position === "ALL"
            ? pool
            : pool.filter((row) => row.position === this.position);
        const validation = data.validation || {};
        const targetValidation = validation.targets || {};
        const counts = data.position_counts || {};
        const awaiting = data.market_status !== "lines_available";
        this.elements.weekPoolStatus.innerHTML = `<p><strong>${awaiting ? "Projection pool ready; market lines pending." : "Projection and market pools available."}</strong> ${this.escape(data.scope || "These are performance projections, not sportsbook picks.")}</p>`;
        const cards = [
            ["Games", this.formatInt(data.games)],
            ["All Players", this.formatInt(data.players)],
            ["QBs", this.formatInt(counts.QB)],
            ["RBs", this.formatInt(counts.RB)],
            ["WRs", this.formatInt(counts.WR)],
            ["TEs", this.formatInt(counts.TE)],
            ["Pass MAE", `${this.formatNum(targetValidation.passing?.mae, 1)} yd`],
            ["Rush MAE", `${this.formatNum(targetValidation.rushing?.mae, 1)} yd`],
            ["Receive MAE", `${this.formatNum(targetValidation.receiving?.mae, 1)} yd`],
        ];
        this.elements.weekPoolMetrics.innerHTML = cards.map(([label, value]) => `<article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></article>`).join("");
        if (!visiblePool.length) {
            this.elements.weekProjectionPool.innerHTML = "<p>No Week 1 projections are available.</p>";
            return;
        }
        const rows = visiblePool.map((row) => `<tr>
            <td>${this.escape(this.formatInt(row.projection_rank))}</td>
            <td><strong>${this.escape(row.player)}</strong><br><small>${this.escape(`${row.depth_role || row.position} · ${row.team} ${row.venue === "home" ? "vs" : "at"} ${row.opponent}`)}</small></td>
            <td>${this.escape(this.formatKickoff(row.kickoff_utc))}</td>
            <td>${this.escape(row.target_label || row.target)}</td>
            <td><strong>${this.escape(this.formatNum(row.projection, 1))}</strong></td>
            <td>${this.escape(`${this.formatNum(row.p10, 0)}–${this.formatNum(row.p90, 0)}`)}</td>
            <td>${this.escape(row.market_line == null ? "Awaiting line" : this.formatNum(row.market_line, 1))}</td>
        </tr>`).join("");
        this.elements.weekProjectionPool.innerHTML = `<table class="prediction-about-table"><thead><tr><th>RK</th><th>Player / Matchup</th><th>Kickoff</th><th>Target</th><th>Projection</th><th>P10–P90</th><th>Market</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    renderParlayWatchlists() {
        const data = this.weekPool || {};
        const policy = data.parlay_policy || {};
        const livePools = this.weekMarketBoard?.pools || {};
        const marketTickets = Object.entries(livePools).map(([name, legs]) => ({
            name: name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()),
            note: "RotoWire Week 1 prices joined to the frozen projection pool.",
            legs: Array.isArray(legs) ? legs : [],
            marketBacked: true,
        })).filter((ticket) => ticket.legs.length);
        const watchlists = marketTickets.length ? marketTickets : (Array.isArray(data.parlay_watchlists) ? data.parlay_watchlists : []);
        this.elements.parlayWatchlistStatus.innerHTML = marketTickets.length
            ? `<p><strong>Week 1 market-backed shadow pools.</strong> Real two-sided RotoWire odds are attached, but joint probabilities and execution remain withheld until dependency calibration and an executable combined quote exist.</p>`
            : `<p><strong>Projection templates only — not bets.</strong> ${this.escape(policy.reason || "Authentic two-sided lines are required before any leg can be evaluated.")} No line, direction, odds, or staking authorization has been assigned.</p>`;
        if (!watchlists.length) {
            this.elements.parlayWatchlists.innerHTML = "<p>No Week 1 parlay watchlists are available.</p>";
            return;
        }
        this.elements.parlayWatchlists.innerHTML = watchlists.map((ticket) => {
            const legs = (ticket.legs || []).map((leg) => `<div class="week-parlay-leg">
                <span class="week-parlay-position">${this.escape(leg.position)}</span>
                <span><strong>${this.escape(leg.player)}</strong><small>${this.escape(`${leg.team} vs ${leg.opponent} · ${String(leg.target || "").replaceAll("_", " ")}`)}</small></span>
                <span class="week-parlay-projection"><strong>${this.escape(leg.side ? `${leg.side} ${this.formatNum(leg.line, 1)}` : this.formatNum(leg.projection, 1))}</strong><small>${this.escape(leg.side ? `${leg.bookmaker} ${this.formatOdds(leg.price)}` : "projected")}</small></span>
            </div>`).join("");
            return `<article class="week-parlay-card">
                <header><div><h3>${this.escape(ticket.name)}</h3><p>${this.escape(ticket.note)}</p></div><span class="week-parlay-status">Awaiting lines</span></header>
                <div class="week-parlay-legs">${legs}</div>
            </article>`;
        }).join("");
    }

    renderBoard(plays) {
        if (!plays.length) {
            this.elements.board.innerHTML = "<p>No playable passing-yard candidate survived this slate.</p>";
            return;
        }
        const cv = window.CardVault;
        if (!cv) {
            this.elements.board.innerHTML = "<p>The bounty notices could not be loaded.</p>";
            return;
        }
        this.elements.board.innerHTML = plays.map((play, index) => cv.renderPredictionCard({
            ...play,
            rank: play.rank || index + 1,
            player_display_name: play.player,
            target: play.target || "passing_yards",
            market_line: play.line,
            model_hit_probability: play.model_hit_probability,
            candidate_authorized: this.data?.candidate_authorized === true && play.candidate_authorized !== false,
            action_status: this.data?.candidate_authorized === true ? play.action_status : "review",
            board_publication_status: this.data?.publication_status,
        }, index)).join("");
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

    /**
     * PARLAY_POLICY_V2 -- the new, theory-grounded 2-leg parlay path
     * (sports/nfl/parlay_v2/, ported from MLB's PARLAY_CERTIFICATION_V2).
     * Reads ONLY this.data.parlays -- entirely separate from the old
     * "Parlay Policy" section above (this.data.daily_parlay), which stays
     * untouched as diagnostic-only CONTROL. Status language is restricted
     * to the same allowed vocabulary MLB's board uses -- never
     * "guaranteed" / "safe bet" / "proven winner" / "lock".
     */
    renderParlayV2() {
        const section = this.elements.parlayV2Section;
        const content = this.elements.parlayV2Content;
        if (!section || !content) return;

        const parlay = this.data?.parlays || {};
        const statusLabel = this.formatParlayV2StatusLabel(parlay.policy_status);
        const statusTone = this.formatParlayV2StatusTone(parlay.policy_status);

        const executionLabel = parlay.shadow_execution_status === "EXECUTED_SHADOW" ? "Shadow only (selected)" : "Not executed";
        const worldGateLabels = { REQUIRED: "Required", BOUNDED_RISK: "Bounded risk", OBSERVE_ONLY: "Observe-only" };
        const worldGateLabel = worldGateLabels[parlay.world_gate_mode] || "n/a";
        const statusFooter = `Policy: ${this.escape(parlay.policy_version || "n/a")} / Research status: ${this.escape(statusLabel)} / World gate: ${this.escape(worldGateLabel)} / Execution: ${this.escape(executionLabel)}`;

        if (parlay.action !== "ACT" || !parlay.selected_parlay) {
            const reason = String(parlay.abstain_reason || "").trim();
            const shadow = parlay.shadow_candidate;
            const shadowBlock = shadow ? `
                <p class="daily-parlay__empty">This week's V2 shadow candidate -- not certified, no stake authorized</p>
                ${this.renderParlayV2Legs(shadow)}
            ` : `<p class="daily-parlay__empty">${this.escape(this.formatParlayV2AbstainReason(reason, parlay))}</p>`;
            content.innerHTML = `
                <div class="daily-parlay__header">
                    <div>
                        <p class="vault-page-kicker">Theory-grounded 2-leg parlay</p>
                        <h2 id="parlayV2Title">This Week's V2 Shadow Candidate</h2>
                    </div>
                    ${window.CardVault ? window.CardVault.renderStatusPill(statusTone, "Abstain") : ""}
                </div>
                ${shadowBlock}
                <p class="daily-parlay__state">${this.escape(this.formatParlayV2AbstainReason(reason, parlay))} ${statusFooter}</p>
            `;
            return;
        }

        content.innerHTML = `
            <div class="daily-parlay__header">
                <div>
                    <p class="vault-page-kicker">Theory-grounded 2-leg parlay</p>
                    <h2 id="parlayV2Title">This Week's V2 Shadow Candidate</h2>
                </div>
                ${window.CardVault ? window.CardVault.renderStatusPill(statusTone, "Selected -- shadow only") : ""}
            </div>
            ${this.renderParlayV2Legs(parlay.selected_parlay)}
            <p class="daily-parlay__state">${statusFooter}</p>
        `;
    }

    /**
     * Deliberately does NOT show a probability/score next to either leg
     * (for a certified pick or a shadow candidate alike) -- matches MLB's
     * board: this program's own research found that ranking/displaying by
     * raw model probability concentrates the frozen marginal model's
     * worst overconfidence, so surfacing that number here would
     * misleadingly suggest a reliability this system has not established.
     */
    renderParlayV2Legs(pair) {
        if (!window.CardVault) return "";
        const legs = [pair.leg_1, pair.leg_2].filter(Boolean);
        const cards = legs.map((leg, index) => {
            const displayName = String(leg.player || "").trim() || "Unknown player";
            const nameParts = displayName.split(/\s+/).filter(Boolean);
            const monogram = nameParts.length >= 2
                ? `${nameParts[0][0]}${nameParts[nameParts.length - 1][0]}`.toUpperCase()
                : (nameParts[0] || "NA").slice(0, 2).toUpperCase();
            const lineText = this.formatNum(leg.line, 1);
            return window.CardVault.renderLegCard({
                rank: index + 1,
                monogram,
                photoUrl: String(leg.player_headshot_url || "").trim(),
                photoFallbackUrl: String(leg.player_headshot_fallback_url || "").trim(),
                name: displayName,
                market: `${leg.side || ""} ${String(leg.target || "").replaceAll("_", " ")}`.trim(),
                context: lineText !== "n/a" ? `Line ${lineText}` : "",
            });
        }).join("");
        return `<div class="vault-board vault-board--legs">${cards}</div>`;
    }

    formatParlayV2StatusLabel(policyStatus) {
        const status = String(policyStatus || "").toUpperCase();
        const labels = {
            DEVELOPMENT: "Shadow",
            FROZEN_PROSPECTIVE_INCONCLUSIVE: "Prospective inconclusive",
            FROZEN_POLICY_PROSPECTIVELY_SUPPORTED: "Supported current",
            SUPPORTED_CURRENT: "Supported current",
            PRODUCTION_DEMOTED: "Production demoted",
        };
        return labels[status] || "Shadow";
    }

    formatParlayV2StatusTone(policyStatus) {
        const status = String(policyStatus || "").toUpperCase();
        if (status === "SUPPORTED_CURRENT" || status === "FROZEN_POLICY_PROSPECTIVELY_SUPPORTED") return "active";
        if (status === "PRODUCTION_DEMOTED") return "withheld";
        return "stale"; // DEVELOPMENT / FROZEN_PROSPECTIVE_INCONCLUSIVE / unknown
    }

    formatParlayV2AbstainReason(reason, parlay) {
        const messages = {
            NO_REAL_QUOTE: "No real market quote available for this week's slate.",
            NO_CANDIDATES: "No cross-event, cross-player candidate pairs exist for this week's slate.",
            NO_STATE_SUPPORT: "Not enough independent prior weeks have accumulated yet.",
            NO_LEG_MARKET_SUPPORT: "Not enough prior settled observations for this market type yet.",
            NO_LEG_LINE_SUPPORT: "Not enough prior settled observations for this exact line yet.",
            NO_PAIR_IN_SUPPORT: "No pair currently meets the frozen support requirements.",
            PRICE_OUT_OF_RANGE: "The best available price fell outside the frozen accepted range.",
            NO_PAIR_PASSES_FROZEN_POLICY: "No pair cleared the frozen certification requirements this week.",
            OPERATIONALLY_INELIGIBLE: "This week is not operationally eligible for a parlay decision.",
            POLICY_NOT_FROZEN: "The V2 policy has not yet been frozen for prospective use.",
            CERTIFICATION_STREAM_NOT_READY: "Not enough real prospective history has accumulated yet.",
            PARLAY_V2_ARTIFACT_UNAVAILABLE: "V2 parlay data is not available for this run.",
        };
        let message = messages[reason] || "No qualifying parlay was selected for this week.";
        // Real, honest progress numbers -- never fabricated -- straight
        // from the same ledger the policy itself reads.
        if (reason === "NO_STATE_SUPPORT" && parlay && Number.isFinite(parlay.independent_slate_count) && Number.isFinite(parlay.independent_slate_count_required)) {
            message += ` (${parlay.independent_slate_count} of ${parlay.independent_slate_count_required} independent prior weeks so far.)`;
        }
        return message;
    }

    formatTime(value) {
        if (!value) return "n/a";
        const parsed = new Date(value);
        return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    }
    formatKickoff(value) { const parsed = new Date(value); return Number.isNaN(parsed.valueOf()) ? "n/a" : parsed.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }); }
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
