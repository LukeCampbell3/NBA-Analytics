class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.availableDates = [];
        this.activeDate = null;
        this.currentDate = null;
        this.elements = {
            cards: document.getElementById("predictionCards"),
            empty: document.getElementById("predictionEmpty"),
            runMeta: document.getElementById("predictionRunMeta"),
            parlayV2Content: document.getElementById("parlayV2Content"),
            sameGameParlayContent: document.getElementById("sameGameParlayContent"),
            pitcherParlayContent: document.getElementById("pitcherParlayContent"),
            highHitParlayContent: document.getElementById("highHitParlayContent"),
            dateNav: document.getElementById("predictionDateNav"),
        };
        this.init();
    }

    init() {
        this.mountShell();
        // Every real "Add to FanDuel Betslip" link on the page resolves
        // itself at click time (viewer's state already known -> opens
        // immediately; not known yet -> a one-time real prompt, then
        // opens) -- no page-level control to mount. See
        // CardVault.initFanduelBetslipLinks.
        window.CardVault?.initFanduelBetslipLinks();
        if (window.CardVault && this.elements.cards) {
            this.elements.cards.innerHTML = window.CardVault.renderSkeletonCard(6);
        }
        this.loadDatesAndRender();
        this.loadSameGameParlay();
        this.loadPitcherParlay();
        this.loadHighHitParlay();
    }

    mountShell() {
        if (!window.CardVaultShell) return;

        window.CardVaultShell.mount({
            brandTitle: "In The Cards Analytics",
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
            this.renderParlayV2();
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
        const basePlays = Array.isArray(this.data.plays)
            ? this.data.plays.map((play) => ({ ...play, board_publication_status: publicationStatus }))
            : [];
        this.plays = this.mergeLegacySoloBets(basePlays);
        this.plays.sort((a, b) => {
            const parlayDiff = Number(Boolean(b.parlay_candidate)) - Number(Boolean(a.parlay_candidate));
            if (parlayDiff !== 0) return parlayDiff;
            const evDiff = (Number(b.ev) || 0) - (Number(a.ev) || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (Number(b.abs_edge) || Number(b.edge) || 0) - (Number(a.abs_edge) || Number(a.edge) || 0);
        });
        this.renderRunMeta();
    }

    /**
     * The legacy parlay ticket system (daily_parlay) is CONTROL/diagnostic
     * only -- it has never been V2-certified, so its legs are never shown
     * bundled as a "parlay" claim. Instead each leg is folded into Solo
     * Bets as an individual, unauthorized candidate (candidate_authorized
     * always follows the ticket's own value, so an unauthorized ticket's
     * legs correctly render with the same "Shadow only" treatment as any
     * other uncertified single play). Skips a leg already present in the
     * base singles pool (matched by play_key, falling back to
     * player+target+market_line) so nothing is shown twice.
     */
    mergeLegacySoloBets(basePlays) {
        const ticket = this.data?.daily_parlay?.selected_ticket;
        if (!ticket || !Array.isArray(ticket.legs) || !ticket.legs.length) return basePlays;
        // Defense in depth: profit_boost is a deliberately permanent-
        // shadow, alternate-line longshot product (real leg probability
        // floor 0.18, real ticket floor 0.10 -- see select_daily_parlay.py)
        // that must never be presented as a regular single pick, however
        // it reaches this payload. The real backend fix keeps it out of
        // selected_ticket already (select_daily_parlay.py's build_payload);
        // this check is a second, independent guard against the same real
        // failure mode (a sub-50%-probability leg shown with the same
        // card format and prominence as a real hit-rate-gated pick).
        if (String(ticket.ticket_tier || "").toLowerCase() === "profit_boost") return basePlays;

        const legKey = (item) => String(
            item?.play_key || `${item?.player || item?.player_display_name || ""}|${item?.target || ""}|${item?.market_line || ""}`
        );
        const existingKeys = new Set(basePlays.map(legKey));
        const candidateAuthorized = Boolean(ticket.candidate_authorized);

        const legacyPlays = ticket.legs
            .filter((leg) => !existingKeys.has(legKey(leg)))
            .map((leg) => ({
                ...leg,
                candidate_authorized: candidateAuthorized,
                action_status: leg.action_status || ticket.status,
                ev: leg.ev != null ? leg.ev : leg.expected_value_per_unit,
                edge: leg.edge != null ? leg.edge : (Number.isFinite(Number(leg.prediction)) && Number.isFinite(Number(leg.market_line))
                    ? Number(leg.prediction) - Number(leg.market_line)
                    : undefined),
                parlay_candidate: false, // deliberately not tagged "parlay" -- see method docstring
            }));
        return [...basePlays, ...legacyPlays];
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

    /**
     * PARLAY_POLICY_V2 -- the only "parlay" product path (the legacy
     * ticket system's legs are folded into Solo Bets, see
     * mergeLegacySoloBets). Reads ONLY this.data.parlays. Status
     * language is restricted to the allowed vocabulary (mission section
     * 10) -- never "guaranteed" / "safe bet" / "proven winner" / "lock".
     */
    renderParlayV2() {
        const content = this.elements.parlayV2Content;
        if (!content) return;

        const parlay = this.data?.parlays || {};
        const statusLabel = this.formatParlayV2StatusLabel(parlay.policy_status);
        const statusTone = this.formatParlayV2StatusTone(parlay.policy_status);

        // Status footer: Policy (which frozen action rule), Research
        // status (policy_status -- certification progress, never a
        // profitability claim), World gate (world_gate_mode -- whether
        // world/counterexample diagnostics could have blocked this
        // decision; OBSERVE_ONLY means they never can, so a selection
        // here is NEVER a claim that the world certificate "passed" --
        // see decision_record.world_certificate_diagnostics.certified,
        // which stays honestly false/irrelevant under OBSERVE_ONLY),
        // Execution (shadow_execution_status -- whether TODAY's decision
        // actually selected a real frozen wager).
        const executionLabel = parlay.shadow_execution_status === "EXECUTED_SHADOW" ? "Shadow only (selected)" : "Not executed";
        const worldGateLabels = { REQUIRED: "Required", BOUNDED_RISK: "Bounded risk", OBSERVE_ONLY: "Observe-only" };
        const worldGateLabel = worldGateLabels[parlay.world_gate_mode] || "n/a";
        const statusFooter = `Policy: ${this.escapeHtml(parlay.policy_version || "n/a")} / Research status: ${this.escapeHtml(statusLabel)} / World gate: ${this.escapeHtml(worldGateLabel)} / Execution: ${this.escapeHtml(executionLabel)}`;

        if (parlay.action !== "ACT" || !parlay.selected_parlay) {
            const reason = String(parlay.abstain_reason || "").trim();
            const shadow = parlay.shadow_candidate;
            const shadowBlock = shadow ? `
                <p class="daily-parlay__empty">Today's V2 shadow candidate -- not certified, no stake authorized</p>
                ${this.renderParlayV2Legs(shadow)}
            ` : `<p class="daily-parlay__empty">${this.escapeHtml(this.formatParlayV2AbstainReason(reason, parlay))}</p>`;
            content.innerHTML = `
                <div class="daily-parlay__header daily-parlay__header--status-only">
                    ${window.CardVault ? window.CardVault.renderStatusPill(statusTone, "Abstain") : ""}
                    ${shadow && window.CardVault ? window.CardVault.renderParlaySettlementBadge([shadow.leg_1, shadow.leg_2]) : ""}
                </div>
                ${shadowBlock}
                <p class="daily-parlay__state">${this.escapeHtml(this.formatParlayV2AbstainReason(reason, parlay))} ${statusFooter}</p>
            `;
            return;
        }

        content.innerHTML = `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill(statusTone, "Selected -- shadow only") : ""}
                ${window.CardVault ? window.CardVault.renderParlaySettlementBadge([parlay.selected_parlay.leg_1, parlay.selected_parlay.leg_2]) : ""}
            </div>
            ${this.renderParlayV2Legs(parlay.selected_parlay)}
            <p class="daily-parlay__state">${statusFooter}</p>
        `;
    }

    /**
     * Deliberately does NOT show a probability/score next to either leg
     * (for a certified pick or a shadow candidate alike) -- see
     * run_parlay_v2._best_shadow_candidate's docstring: this program's
     * own research found that ranking/displaying by raw model
     * probability concentrates the frozen marginal model's worst
     * overconfidence, so surfacing that number here would misleadingly
     * suggest a reliability this system has not established.
     */
    renderParlayV2Legs(pair) {
        if (!window.CardVault) return "";
        const legs = [pair.leg_1, pair.leg_2].filter(Boolean);
        const cards = legs.map((leg, index) => {
            const direction = String(leg.side || "").toUpperCase() === "UNDER" ? "UNDER" : "OVER";
            const target = window.CardVault.formatTargetLabel(leg.target);
            const displayName = String(leg.player || "").replaceAll("_", " ").trim() || "Unknown player";
            const nameParts = displayName.split(/\s+/).filter(Boolean);
            const monogram = nameParts.length >= 2
                ? `${nameParts[0][0]}${nameParts[nameParts.length - 1][0]}`.toUpperCase()
                : (nameParts[0] || "NA").slice(0, 2).toUpperCase();
            const lineText = this.formatNumber(leg.line, 1);
            return window.CardVault.renderLegCard({
                rank: index + 1,
                monogram,
                photoUrl: String(leg.player_headshot_url || "").trim(),
                photoFallbackUrl: String(leg.player_headshot_fallback_url || "").trim(),
                name: displayName,
                market: `${direction} ${target}`,
                context: lineText !== "n/a" ? `Line ${lineText}` : "",
                betslipUrl: leg.sportsbook_deeplink || "",
                deeplinksByRegion: leg.deeplinks_by_region || null,
                settlementRow: leg,
            });
        }).join("");
        return `<div class="vault-board vault-board--legs">${cards}</div>`;
    }

    /**
     * Same-Game Parlay -- real cross-market (moneyline + full total + F5
     * total) combos, priced with a joint Monte Carlo simulation so the
     * legs' real correlation is reflected rather than assumed away. This
     * is a SEPARATE, brand-new policy from PARLAY_POLICY_V2 above: its
     * own data/same_game_predictions.json, its own empty calibration
     * ledger. Loaded independently of the main board (never blocks or
     * fails the rest of the page if this file is missing/stale) --
     * mirrors loadDateIndex()'s "optional" fetch pattern.
     */
    async loadSameGameParlay() {
        const content = this.elements.sameGameParlayContent;
        if (!content) return;
        try {
            const response = await fetch(`data/same_game_predictions.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.sameGameData = await response.json();
        } catch (_error) {
            this.sameGameData = null;
        }
        this.renderSameGameParlay();
    }

    renderSameGameParlay() {
        const content = this.elements.sameGameParlayContent;
        if (!content) return;
        const data = this.sameGameData;

        if (!data || data.status !== "ok" || !Array.isArray(data.games) || !data.games.length) {
            content.innerHTML = this.sameGameParlayHeader() + `
                <p class="daily-parlay__empty">No MLB games scheduled today.</p>
            `;
            return;
        }

        const games = data.games;
        const authorizedCount = Number(data.candidate_authorized_count) || 0;
        const pricedCount = games.filter((game) => game.status === "ok").length;
        const statusFooter = `Policy: shadow_only_v1 / Games scheduled: ${games.length} / Priced: ${pricedCount} / Authorized: ${authorizedCount}`;

        // Real candidates flattened across the whole slate -- only the single
        // best (highest real model EV) combo is shown, matching the V2
        // section's singular "Today's Shadow Candidate" framing above.
        const allCombos = [];
        for (const game of games) {
            if (Array.isArray(game.combo_candidates)) {
                for (const combo of game.combo_candidates) allCombos.push({ game, combo });
            }
        }

        if (!allCombos.length) {
            const reason = data.odds_status && data.odds_status !== "success"
                ? "Live market odds not yet available for today's slate."
                : "No real cross-market combo cleared pricing for today's slate.";
            content.innerHTML = this.sameGameParlayHeader() + `
                <p class="daily-parlay__empty">${this.escapeHtml(reason)} A same-game combo will appear here once real moneyline/total/F5 lines are posted and priced.</p>
                <p class="daily-parlay__state">${this.escapeHtml(statusFooter)}</p>
            `;
            return;
        }

        allCombos.sort((a, b) => (Number(b.combo.expected_value_per_unit) ?? -Infinity) - (Number(a.combo.expected_value_per_unit) ?? -Infinity));
        const best = allCombos[0];
        const extraCount = allCombos.length - 1;

        content.innerHTML = this.sameGameParlayHeader() + `
            <div class="same-game-parlay__grid">${this.renderSameGameCombo(best.game, best.combo)}</div>
            ${extraCount > 0 ? `<p class="daily-parlay__state">+${extraCount} more real combo${extraCount === 1 ? "" : "s"} priced across today's slate</p>` : ""}
            <p class="daily-parlay__state">${this.escapeHtml(statusFooter)}</p>
        `;
    }

    sameGameParlayHeader() {
        return `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill("stale", "Shadow only") : ""}
            </div>
        `;
    }

    renderSameGameCombo(game, combo) {
        const matchup = `${this.escapeHtml(game.away_team || "")} @ ${this.escapeHtml(game.home_team || "")}`;
        const starters = `${this.escapeHtml(game.away_starter_name || "TBD")} vs ${this.escapeHtml(game.home_starter_name || "TBD")}`;
        const authorized = Boolean(combo.candidate_authorized);
        const pillTone = authorized ? "active" : "stale";
        const pillLabel = authorized ? "Selected -- shadow only" : "Shadow only";

        const joint = this.formatPct(combo.real_joint_model_probability);
        const rawMarketJoint = this.formatPct(combo.naive_market_joint_raw_probability);
        const noVigMarketJoint = this.formatPct(combo.naive_no_vig_combo_probability);
        const edge = this.formatSignedPp(combo.probability_edge);
        const ev = this.formatSignedPct(combo.expected_value_per_unit);

        return `
            <article class="same-game-parlay__card">
                <div class="daily-parlay__header">
                    <div>
                        <strong>${matchup}</strong>
                        <span>${starters}</span>
                    </div>
                    ${window.CardVault ? window.CardVault.renderStatusPill(pillTone, pillLabel) : ""}
                    ${window.CardVault ? window.CardVault.renderParlaySettlementBadge([combo.leg_a, combo.leg_b]) : ""}
                </div>
                <div class="vault-board vault-board--legs">
                    ${this.renderSameGameLeg(combo.leg_a, game, 1)}
                    ${this.renderSameGameLeg(combo.leg_b, game, 2)}
                </div>
                <div class="same-game-parlay__metrics">
                    <span>Model joint probability <strong>${joint}</strong></span>
                    <span>Naive market joint (raw) <strong>${rawMarketJoint}</strong></span>
                    <span>Naive market joint (no-vig) <strong>${noVigMarketJoint}</strong></span>
                    <span>Joint edge vs. no-vig market <strong>${edge}</strong></span>
                    <span>Synthetic-price EV <strong>${ev}</strong></span>
                </div>
                <!-- "Synthetic-price EV" -- this game's combo_decimal_price is
                     the two real legs' own decimal prices multiplied together,
                     never an actual FanDuel same-game-parlay quote (which
                     would price the legs' real correlation into one number).
                     Never call this figure "sportsbook executable EV" until a
                     real SGP quote is captured and used instead. -->
            </article>
        `;
    }

    renderSameGameLeg(leg, game, index) {
        if (!leg || !window.CardVault) return "";
        const name = this.formatSameGameLegLabel(leg, game);
        const monogram = name.replace(/[^A-Za-z]/g, "").slice(0, 2).toUpperCase() || "NA";
        return window.CardVault.renderLegCard({
            rank: index,
            monogram,
            name,
            market: this.formatSameGameMarketLabel(leg.market),
            metrics: [
                ["Odds", this.formatAmerican(leg.price_american)],
                ["Book", leg.sportsbook || ""],
            ],
            betslipUrl: leg.sportsbook_deeplink || "",
            deeplinksByRegion: leg.deeplinks_by_region || null,
            settlementRow: leg,
        });
    }

    formatSameGameLegLabel(leg, game) {
        if (leg.market === "moneyline") {
            const team = leg.side === "home" ? game.home_team : game.away_team;
            return `${team || "?"} ML`;
        }
        const side = leg.side === "over" ? "Over" : "Under";
        const line = this.formatNumber(leg.line, 1);
        return `${side} ${line}`;
    }

    formatSameGameMarketLabel(market) {
        const labels = { moneyline: "Moneyline", game_total: "Game Total", first_5_innings_total: "F5 Total" };
        return labels[market] || String(market || "");
    }

    /**
     * Pitcher Parlay -- real cross-game, pitcher-strikeouts-only 2-leg
     * parlay (run_mlb_pitcher_parlay_daily.py / select_mlb_pitcher_
     * parlay.py). Two different real starting pitchers in two different
     * real games have no real shared game state, so unlike the same-game
     * combo above, the real joint probability here really is the naive
     * independence product of each leg's own real model probability --
     * see that module's docstring. A brand-new, additive, own-payload
     * product (pitcher_parlay_predictions.json), loaded independently of
     * the rest of the page the same way the same-game combo is.
     */
    async loadPitcherParlay() {
        const content = this.elements.pitcherParlayContent;
        if (!content) return;
        try {
            const response = await fetch(`data/pitcher_parlay_predictions.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.pitcherParlayData = await response.json();
        } catch (_error) {
            this.pitcherParlayData = null;
        }
        this.renderPitcherParlay();
    }

    renderPitcherParlay() {
        const content = this.elements.pitcherParlayContent;
        if (!content) return;
        const data = this.pitcherParlayData;

        if (!data || data.status !== "ok") {
            content.innerHTML = this.pitcherParlayHeader() + `
                <p class="daily-parlay__empty">${this.escapeHtml(this.formatPitcherParlayStatusReason(data))}</p>
            `;
            return;
        }

        const parlay = data.parlay;
        if (!parlay) {
            content.innerHTML = this.pitcherParlayHeader() + `
                <p class="daily-parlay__empty">No real cross-game pitcher-strikeouts pair cleared pricing today.</p>
                <p class="daily-parlay__state">${this.escapeHtml(`Real probable starters: ${data.real_starters_posted ?? 0} / Real priced legs: ${data.real_priced_legs ?? 0}`)}</p>
            `;
            return;
        }

        const authorized = Boolean(parlay.candidate_authorized);
        const pillTone = authorized ? "active" : "stale";
        const pillLabel = authorized ? "Selected -- shadow only" : "Shadow only";
        const joint = this.formatPct(parlay.real_joint_model_probability);
        const rawMarketJoint = this.formatPct(parlay.naive_market_joint_raw_probability);
        const noVigMarketJoint = this.formatPct(parlay.naive_no_vig_combo_probability);
        const edge = parlay.probability_edge != null ? this.formatSignedPp(parlay.probability_edge) : "n/a";
        const ev = parlay.expected_value_per_unit != null ? this.formatSignedPct(parlay.expected_value_per_unit) : "n/a";

        content.innerHTML = this.pitcherParlayHeader(pillTone, pillLabel, [parlay.leg_a, parlay.leg_b]) + `
            <div class="vault-board vault-board--legs">
                ${this.renderPitcherKLeg(parlay.leg_a, 1)}
                ${this.renderPitcherKLeg(parlay.leg_b, 2)}
            </div>
            <div class="same-game-parlay__metrics">
                <span>Model joint probability <strong>${joint}</strong></span>
                <span>Naive market joint (raw) <strong>${rawMarketJoint}</strong></span>
                <span>Naive market joint (no-vig) <strong>${noVigMarketJoint}</strong></span>
                <span>Joint edge vs. no-vig market <strong>${edge}</strong></span>
                <span>Model EV <strong>${ev}</strong></span>
            </div>
        `;
    }

    pitcherParlayHeader(pillTone = "stale", pillLabel = "Shadow only", legs = []) {
        return `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill(pillTone, pillLabel) : ""}
                ${window.CardVault ? window.CardVault.renderParlaySettlementBadge(legs) : ""}
            </div>
        `;
    }

    renderPitcherKLeg(leg, index) {
        if (!leg || !window.CardVault) return "";
        const name = String(leg.pitcher_name || "").trim() || "Unknown pitcher";
        const nameParts = name.split(/\s+/).filter(Boolean);
        const monogram = nameParts.length >= 2
            ? `${nameParts[0][0]}${nameParts[nameParts.length - 1][0]}`.toUpperCase()
            : (nameParts[0] || "NA").slice(0, 2).toUpperCase();
        const side = leg.side === "under" ? "Under" : "Over";
        const matchup = [leg.team, leg.opponent].filter(Boolean).join(" vs. ");
        return window.CardVault.renderLegCard({
            rank: index,
            monogram,
            name,
            market: `${side} ${this.formatNumber(leg.line, 1)} Strikeouts`,
            context: matchup,
            metrics: [
                ["Odds", this.formatAmerican(leg.price_american)],
                ["Book", leg.sportsbook || ""],
            ],
            betslipUrl: leg.sportsbook_deeplink || "",
            deeplinksByRegion: leg.deeplinks_by_region || null,
            settlementRow: leg,
        });
    }

    formatPitcherParlayStatusReason(data) {
        const reasons = {
            no_real_games_scheduled_today: "No MLB games scheduled today.",
            no_real_probable_starters_posted_yet: "No real probable starters posted yet for today's slate.",
        };
        return reasons[data?.status] || "Pitcher parlay data is not available for this run.";
    }

    /**
     * High-Hit Parlay -- v12 Phase 3's HIGH_HIT_PARLAY_V1
     * (select_high_hit_parlay.py): real cross-game combos built directly
     * from today's own v11-eligible single-prop pool, joint-probability-
     * safe (every leg independently clears v11's own probability floor;
     * the combo itself must clear a real joint-probability floor). This
     * is the one parlay group that IS a high-hit-probability claim -- see
     * its own construction.leg_probability_floor / joint_probability_floor
     * in the payload. A brand-new, additive, own-payload product
     * (high_hit_parlay_predictions.json), loaded independently of the
     * rest of the page the same way the other parlay products are.
     */
    async loadHighHitParlay() {
        const content = this.elements.highHitParlayContent;
        if (!content) return;
        try {
            const response = await fetch(`data/high_hit_parlay_predictions.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.highHitParlayData = await response.json();
        } catch (_error) {
            this.highHitParlayData = null;
        }
        this.renderHighHitParlay();
    }

    renderHighHitParlay() {
        const content = this.elements.highHitParlayContent;
        if (!content) return;
        const data = this.highHitParlayData;
        const construction = data?.construction || {};
        const legFloor = this.formatPct(construction.leg_probability_floor);
        const jointFloor = this.formatPct(construction.joint_probability_floor);
        const footer = `Leg floor: ${legFloor} / Combined floor: ${jointFloor} / Eligible legs today: ${data?.legs_eligible ?? 0}`;

        if (!data || !Array.isArray(data.parlays) || !data.parlays.length) {
            content.innerHTML = this.highHitParlayHeader() + `
                <p class="daily-parlay__empty">No real combination of today's eligible legs cleared the combined-probability floor.</p>
                <p class="daily-parlay__state">${this.escapeHtml(footer)}</p>
            `;
            return;
        }

        const cards = data.parlays.map((parlay) => this.renderHighHitCombo(parlay)).join("");
        content.innerHTML = this.highHitParlayHeader() + `
            <div class="same-game-parlay__grid">${cards}</div>
            <p class="daily-parlay__state">${this.escapeHtml(footer)}</p>
        `;
    }

    highHitParlayHeader() {
        return `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill("stale", "Shadow only") : ""}
            </div>
        `;
    }

    renderHighHitCombo(parlay) {
        const joint = this.formatPct(parlay.joint_probability);
        const price = this.formatNumber(parlay.decimal_price, 2);
        const ev = this.formatSignedPct(parlay.expected_value_per_unit);
        const legs = Array.isArray(parlay.legs) ? parlay.legs : [];

        return `
            <article class="same-game-parlay__card">
                <div class="daily-parlay__header">
                    <div><strong>${legs.length}-leg high-hit combo</strong></div>
                    ${window.CardVault ? window.CardVault.renderParlaySettlementBadge(legs) : ""}
                </div>
                <div class="vault-board vault-board--legs">
                    ${legs.map((leg, index) => this.renderHighHitLeg(leg, index + 1)).join("")}
                </div>
                <div class="same-game-parlay__metrics">
                    <span>Joint probability <strong>${joint}</strong></span>
                    <span>Combined decimal price <strong>${price}</strong></span>
                    <span>Model EV <strong>${ev}</strong></span>
                </div>
            </article>
        `;
    }

    renderHighHitLeg(leg, index) {
        if (!leg || !window.CardVault) return "";
        const name = String(leg.player || "").trim() || "Unknown player";
        const nameParts = name.split(/\s+/).filter(Boolean);
        const monogram = nameParts.length >= 2
            ? `${nameParts[0][0]}${nameParts[nameParts.length - 1][0]}`.toUpperCase()
            : (nameParts[0] || "NA").slice(0, 2).toUpperCase();
        const side = String(leg.direction || "").toUpperCase() === "UNDER" ? "Under" : "Over";
        return window.CardVault.renderLegCard({
            rank: index,
            monogram,
            name,
            market: `${side} ${this.formatNumber(leg.market_line, 1)} ${leg.target || ""}`.trim(),
            context: leg.team || "",
            metrics: [
                ["Probability", this.formatPct(leg.probability)],
                ["Odds", this.formatAmerican(leg.american_price)],
                ["Book", leg.sportsbook || ""],
            ],
            settlementRow: leg,
        });
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
        return labels[status] || "Prospective inconclusive";
    }

    formatParlayV2StatusTone(policyStatus) {
        const status = String(policyStatus || "").toUpperCase();
        if (status === "SUPPORTED_CURRENT" || status === "FROZEN_POLICY_PROSPECTIVELY_SUPPORTED") return "active";
        if (status === "PRODUCTION_DEMOTED") return "withheld";
        return "stale"; // DEVELOPMENT / FROZEN_PROSPECTIVE_INCONCLUSIVE / unknown
    }

    formatParlayV2AbstainReason(reason, parlay) {
        const messages = {
            NO_REAL_QUOTE: "No real market quote available for today's slate.",
            NO_CANDIDATES: "No cross-game candidate pairs exist for today's slate.",
            NO_STATE_SUPPORT: "Not enough independent prior slates have accumulated yet.",
            NO_LEG_MARKET_SUPPORT: "Not enough prior settled observations for this market type yet.",
            NO_LEG_LINE_SUPPORT: "Not enough prior settled observations for this exact line yet.",
            NO_PAIR_IN_SUPPORT: "No pair currently meets the frozen support requirements.",
            PRICE_OUT_OF_RANGE: "The best available price fell outside the frozen accepted range.",
            NO_PAIR_PASSES_FROZEN_POLICY: "No pair cleared the frozen certification requirements today.",
            OPERATIONALLY_INELIGIBLE: "Today's slate is not operationally eligible for a parlay decision.",
            POLICY_NOT_FROZEN: "The V2 policy has not yet been frozen for prospective use.",
            CERTIFICATION_STREAM_NOT_READY: "Not enough real prospective history has accumulated yet.",
            PARLAY_V2_ARTIFACT_UNAVAILABLE: "V2 parlay data is not available for this run.",
        };
        let message = messages[reason] || "No qualifying parlay was selected for this slate.";
        // Real, honest progress numbers -- never fabricated -- straight
        // from the same ledger the policy itself reads.
        if (reason === "NO_STATE_SUPPORT" && parlay && Number.isFinite(parlay.independent_slate_count) && Number.isFinite(parlay.independent_slate_count_required)) {
            message += ` (${parlay.independent_slate_count} of ${parlay.independent_slate_count_required} independent prior slates so far.)`;
        }
        return message;
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

    // Percentage-POINT formatter for a difference of two probabilities --
    // deliberately distinct from formatSignedPct (a signed % return like
    // EV), see vault-components.js's own CardVault.formatSignedPp.
    formatSignedPp(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return "n/a";
        return `${number >= 0 ? "+" : ""}${(number * 100).toFixed(1)} pp`;
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
