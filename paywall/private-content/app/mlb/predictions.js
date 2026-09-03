class DailyPredictionsPage {
    constructor() {
        this.data = null;
        this.plays = [];
        this.availableDates = [];
        this.activeDate = null;
        this.currentDate = null;
        this.maxCurrentArtifactAgeMs = 8 * 60 * 60 * 1000;
        this.elements = {
            cards: document.getElementById("predictionCards"),
            empty: document.getElementById("predictionEmpty"),
            runMeta: document.getElementById("predictionRunMeta"),
            parlayV2Content: document.getElementById("parlayV2Content"),
            sameGameParlayContent: document.getElementById("sameGameParlayContent"),
            pitcherParlayContent: document.getElementById("pitcherParlayContent"),
            highHitParlayContent: document.getElementById("highHitParlayContent"),
            v4SinglesContent: document.getElementById("v4SinglesContent"),
            exoticMarketsContent: document.getElementById("exoticMarketsContent"),
            unifiedEngineContent: document.getElementById("unifiedEngineContent"),
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
    }

    async loadUnifiedEngine() {
        const target = this.elements.unifiedEngineContent;
        if (!target) return;
        try {
            const manifest = await this.fetchUnifiedJson(`data/mlb_engine_manifest.json?v=${Date.now()}`);
            const payload = await this.fetchUnifiedJson(`data/unified_predictions.json?v=${Date.now()}`);
            this.validateUnifiedPayload(payload, manifest);
            this.renderUnifiedEngine(payload, manifest);
        } catch (error) {
            target.innerHTML = this.renderUnifiedNotice("Unified data unavailable", `${error.message || "Load failed"}. The legacy production board remains active.`);
        }
    }

    async fetchUnifiedJson(url, timeoutMs = 8000) {
        if (window.MlbUnifiedContract) {
            return window.MlbUnifiedContract.fetchJson(url, { timeoutMs });
        }
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, { cache: "no-store", credentials: "same-origin", signal: controller.signal });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const payload = await response.json();
            if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("Malformed or empty JSON");
            return payload;
        } catch (error) {
            if (error?.name === "AbortError") throw new Error("Prediction request timed out");
            throw error;
        } finally {
            clearTimeout(timeout);
        }
    }

    easternDate() {
        const parts = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(new Date());
        const value = Object.fromEntries(parts.map((part) => [part.type, part.value]));
        return `${value.year}-${value.month}-${value.day}`;
    }

    validateUnifiedPayload(payload, manifest) {
        if (window.MlbUnifiedContract) {
            return window.MlbUnifiedContract.validate(payload, manifest, { today: this.easternDate() });
        }
        if (payload.schema_version !== "unified_mlb_v1") throw new Error("Unified schema mismatch");
        if (!payload.generation_id || !payload.generated_at_utc || !payload.policy_hash || !payload.run_date) throw new Error("Unified artifact is incomplete");
        if (payload.policy_hash !== manifest.policy_hash) throw new Error("Unified policy hash mismatch");
        if (payload.run_date !== this.easternDate()) throw new Error("Predictions not yet generated for today's slate");
        const generated = Date.parse(payload.generated_at_utc);
        if (!Number.isFinite(generated) || Date.now() - generated > 30 * 60 * 60 * 1000) throw new Error("Unified artifact is stale");
        if (!Array.isArray(payload.singles) || !payload.parlays || !payload.evidence) throw new Error("Unified artifact contract is incomplete");
    }

    renderUnifiedNotice(title, detail) {
        return `<article class="parlay-ticket"><div class="parlay-ticket__header"><strong>${this.escapeHtml(title)}</strong><span class="vault-status vault-status--stale">Shadow</span></div><p>${this.escapeHtml(detail)}</p></article>`;
    }

    renderUnifiedEngine(payload, manifest) {
        const target = this.elements.unifiedEngineContent;
        if (!target) return;
        const evidence = payload?.evidence || {};
        const unifiedActive = manifest?.active_engine === "unified";
        const singles = Array.isArray(payload?.singles) ? payload.singles : [];
        const classes = [
            ["2-Leg", payload?.parlays?.two_leg],
            ["3-Leg", payload?.parlays?.three_leg],
            ["4-Leg", payload?.parlays?.four_leg],
        ];
        const singleHtml = singles.length
            ? `<div class="unified-ticket-grid">${singles.map((candidate) => this.renderUnifiedSingle(candidate)).join("")}</div>`
            : this.renderUnifiedNotice("Singles abstain", "No candidate cleared probability, uncertainty, support, identity, price, and conservative-EV gates.");
        const parlayHtml = classes.map(([label, tickets]) => {
            const list = Array.isArray(tickets) ? tickets : [];
            return `<div class="parlay-group"><p class="parlay-group__label">Best Qualified ${label}</p>${list.length ? list.map((ticket) => this.renderUnifiedTicket(ticket)).join("") : this.renderUnifiedNotice(`${label} abstain`, "The independently evaluated safe set produced no qualified ticket.")}</div>`;
        }).join("");
        target.innerHTML = `<p class="parlay-group__note">Engine: ${unifiedActive ? "Unified active" : "Legacy active / unified shadow"} · Evidence: ${this.escapeHtml(evidence.state || "DEVELOPMENT")} · Execution: ${evidence.publication_authority && unifiedActive ? "authorized" : "not authorized"}</p><div class="parlay-group"><p class="parlay-group__label">Singles</p>${singleHtml}</div>${parlayHtml}`;
    }

    renderUnifiedSingle(candidate) {
        return `<article class="parlay-ticket"><div class="parlay-ticket__header"><strong>${this.escapeHtml(candidate.subject_id || candidate.team || "Candidate")}</strong><span class="vault-status vault-status--stale">Shadow</span></div><p>${this.escapeHtml(String(candidate.side || "").toUpperCase())} ${this.escapeHtml(candidate.market_type)} ${this.escapeHtml(candidate.line ?? "")}</p><dl class="parlay-ticket__metrics"><div><dt>Usable probability</dt><dd>${this.formatPct(candidate.usable_probability)}</dd></div><div><dt>Price</dt><dd>${this.formatAmerican(candidate.american_price)}</dd></div><div><dt>Break-even</dt><dd>${this.formatPct(candidate.market_break_even_probability)}</dd></div><div><dt>Edge</dt><dd>${this.formatSignedPp(candidate.probability_edge)}</dd></div><div><dt>Conservative EV</dt><dd>${this.formatSignedPct(candidate.conservative_expected_value)}</dd></div></dl></article>`;
    }

    renderUnifiedTicket(ticket) {
        const legs = Array.isArray(ticket.legs) ? ticket.legs : [];
        const legHtml = legs.map((leg, index) => `<li><strong>${index + 1}. ${this.escapeHtml(leg.subject_id || leg.team || "Leg")}</strong><span>${this.escapeHtml(String(leg.side || "").toUpperCase())} ${this.escapeHtml(leg.market_type)} ${this.escapeHtml(leg.line ?? "")} · ${this.formatPct(leg.usable_probability)} · ${this.formatAmerican(leg.american_price)}</span></li>`).join("");
        return `<article class="parlay-ticket"><div class="parlay-ticket__header"><strong>${ticket.leg_count}-Leg ${ticket.ticket_type === "same_game" ? "Same-Game" : "Parlay"}</strong><span class="vault-status vault-status--stale">Shadow</span></div><ol class="parlay-ticket__legs">${legHtml}</ol><dl class="parlay-ticket__metrics"><div><dt>Joint probability</dt><dd>${this.formatPct(ticket.joint_probability)}</dd></div><div><dt>Break-even</dt><dd>${this.formatPct(ticket.break_even_probability)}</dd></div><div><dt>Joint edge</dt><dd>${this.formatSignedPp(ticket.probability_edge)}</dd></div><div><dt>Conservative EV</dt><dd>${this.formatSignedPct(ticket.conservative_expected_value)}</dd></div><div><dt>Dependency delta</dt><dd>${this.formatSignedPp(ticket.dependency_delta)}</dd></div></dl></article>`;
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
        if (currentLoaded) {
            await this.loadPickProducts(null);
        } else {
            this.renderDependentProductsUnavailable();
        }
        this.renderDateNav();
    }

    productUrl(filename, date = null) {
        return date
            ? `data/history/products/${date}/${filename}?v=${Date.now()}`
            : `data/${filename}?v=${Date.now()}`;
    }

    async loadPickProducts(date = null) {
        await Promise.all([
            this.loadSameGameParlay(date),
            this.loadPitcherParlay(date),
            this.loadHighHitParlay(date),
            this.loadExoticMarkets(date),
        ]);
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
            this.renderV4Singles();
            this.renderCards();
            return true;
        } catch (error) {
            console.error(error);
            this.plays = [];
            if (this.data) {
                this.renderRunMeta();
            } else if (this.elements.runMeta) {
                this.elements.runMeta.textContent = "MLB board unavailable";
            }
            this.renderFreshnessAlert("Current MLB picks unavailable", error.message || "Freshness verification failed.");
            if (window.CardVault && this.elements.cards) {
                this.elements.cards.innerHTML = window.CardVault.renderEmptyState(
                    "No current picks are displayed",
                    `The latest MLB board could not be verified for today: ${error.message}`,
                    "Refresh later. Older picks are never substituted for today's board."
                );
            }
            if (this.elements.empty) this.elements.empty.style.display = "none";
            return false;
        }
    }

    async load(date) {
        const url = date
            ? `data/history/${date}.json?v=${Date.now()}`
            : `data/daily_predictions.json?v=${Date.now()}`;
        const response = await fetch(url, {
            cache: "no-store",
            credentials: "same-origin",
            headers: { "Cache-Control": "no-cache" },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        this.data = payload;
        this.activeDate = payload?.run_date || date || null;
        if (!date) this.currentDate = this.activeDate;
        if (!date) {
            this.assertCurrentArtifact(payload, "MLB board");
            this.renderRunMeta();
            if (String(payload.publication_status || "").toLowerCase() !== "ready") {
                throw new Error("MLB board publication is withheld or under review");
            }
            if (!Array.isArray(payload.plays)) throw new Error("MLB board has no valid plays collection");
        }
        const publicationStatus = String(this.data?.publication_status || "ready").toLowerCase();
        const basePlays = Array.isArray(this.data.plays)
            ? this.data.plays.map((play) => ({ ...play, board_publication_status: publicationStatus }))
            : [];
        this.plays = basePlays;
        this.plays.sort((a, b) => {
            const parlayDiff = Number(Boolean(b.parlay_candidate)) - Number(Boolean(a.parlay_candidate));
            if (parlayDiff !== 0) return parlayDiff;
            const evDiff = (Number(b.ev) || 0) - (Number(a.ev) || 0);
            if (Math.abs(evDiff) > 1e-9) return evDiff;
            return (Number(b.abs_edge) || Number(b.edge) || 0) - (Number(a.abs_edge) || Number(a.edge) || 0);
        });
        this.renderRunMeta();
        if (!date) this.renderFreshnessAlert("", "");
    }

    assertCurrentArtifact(payload, label = "MLB artifact") {
        if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
            throw new Error(`${label} is malformed`);
        }
        const today = this.easternDate();
        const runDate = String(payload.run_date || "");
        if (runDate !== today) {
            throw new Error(`${label} is for ${runDate || "an unknown date"}, not today's ${today} slate`);
        }
        const generatedAt = Date.parse(payload.generated_at_utc || "");
        if (!Number.isFinite(generatedAt)) throw new Error(`${label} has no valid generation timestamp`);
        const ageMs = Date.now() - generatedAt;
        if (ageMs < -15 * 60 * 1000) throw new Error(`${label} has an invalid future timestamp`);
        if (ageMs > this.maxCurrentArtifactAgeMs) throw new Error(`${label} is more than 8 hours old`);
        return payload;
    }

    renderFreshnessAlert(title, detail) {
        const target = document.getElementById("predictionFreshnessAlert");
        if (!target) return;
        target.innerHTML = title ? this.renderUnifiedNotice(title, detail) : "";
    }

    renderDependentProductsUnavailable() {
        const message = this.renderUnifiedNotice(
            "Unavailable until today's board is verified",
            "No picks or betslip actions are shown because the current MLB slate failed freshness validation."
        );
        [
            this.elements.parlayV2Content,
            this.elements.sameGameParlayContent,
            this.elements.pitcherParlayContent,
            this.elements.highHitParlayContent,
            this.elements.v4SinglesContent,
            this.elements.exoticMarketsContent,
            this.elements.unifiedEngineContent,
            document.getElementById("v21ShadowContent"),
        ].filter(Boolean).forEach((target) => { target.innerHTML = message; });
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
                const archiveDate = date === this.currentDate ? null : date;
                const loaded = await this.loadAndRender(archiveDate);
                if (loaded) await this.loadPickProducts(archiveDate);
                this.renderDateNav();
            });
        });
    }

    formatDateLabel(dateValue) {
        try {
            const displayDate = new Date(`${dateValue}T12:00:00`);
            const today = this.easternDate();
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
        const viewingArchive = Boolean(this.activeDate && this.currentDate && this.activeDate !== this.currentDate);
        const publicationLabel = viewingArchive ? "Archived" : !authorizationEnabled ? "Shadow only" : publicationStatus === "ready" ? "Published" : publicationStatus === "review" ? "Review" : "Withheld";
        const publicationTone = viewingArchive ? "stale" : !authorizationEnabled ? "stale" : publicationStatus === "ready" ? "active" : publicationStatus === "review" ? "stale" : "withheld";
        const stale = publicationStatus !== "ready";
        const quality = this.data?.data_quality || {};
        const lagText = Number.isFinite(Number(quality.lag_days)) ? `${Number(quality.lag_days)}d` : "n/a";
        const officialCount = this.plays.filter((play) => play.issuance_board === "OFFICIAL" || play.issuance_board === "LEGACY_IMPORT").length;
        const lateAddCount = this.plays.filter((play) => play.issuance_board === "LATE_ADD").length;
        const issuanceText = this.data?.publication_protocol
            ? `<span class="prediction-run-meta__item">Official <strong>${officialCount}</strong></span><span class="prediction-run-meta__item">Late adds <strong>${lateAddCount}</strong></span>`
            : "";

        if (this.elements.runMeta && window.CardVault) {
            this.elements.runMeta.innerHTML = `
                ${window.CardVault.renderStatusPill(publicationTone, publicationLabel)}
                <span class="prediction-run-meta__item">Run <strong>${this.escapeHtml(runDate)}</strong></span>
                <span class="prediction-run-meta__item">Data through <strong>${this.escapeHtml(throughDate)}</strong></span>
                <span class="prediction-run-meta__item">Lag <strong>${this.escapeHtml(lagText)}</strong></span>
                <span class="prediction-run-meta__item">Signals <strong>${this.plays.length}</strong></span>
                ${issuanceText}
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

        const official = this.plays.filter((play) => play.issuance_board !== "LATE_ADD");
        const lateAdds = this.plays.filter((play) => play.issuance_board === "LATE_ADD");
        const renderGroup = (label, plays, offset = 0) => plays.length
            ? `<section class="prediction-issuance-group"><h3>${this.escapeHtml(label)}</h3>${plays.map((play, index) => cv.renderPredictionCard(play, offset + index)).join("")}</section>`
            : "";
        this.elements.cards.innerHTML = this.data?.publication_protocol
            ? `${renderGroup("11:30 Official Board", official)}${renderGroup("5:30 Late Adds", lateAdds, official.length)}`
            : this.plays.map((play, index) => cv.renderPredictionCard(play, index)).join("");
    }

    renderV4Singles() {
        const content = this.elements.v4SinglesContent;
        if (!content) return;
        const shadow = this.data?.v4_singles_shadow || {};
        const plays = Array.isArray(shadow.plays) ? shadow.plays : [];
        const priorSlates = Number(shadow.strictly_prior_settled_slates) || 0;
        const footer = `Policy: ${shadow.version || "V4 unavailable"} / Evidence: ${shadow.evidence_status || "uncertified"} / Prior prospective slates: ${priorSlates} / Execution: not authorized`;
        if (!plays.length) {
            const reason = String(shadow.status || "UNAVAILABLE").toUpperCase() === "UNAVAILABLE"
                ? "No current V4 report is available."
                : "No single bet cleared the V4 confidence, exact-price edge, and positive-EV gates.";
            content.innerHTML = `
                <div class="daily-parlay__header daily-parlay__header--status-only">
                    ${window.CardVault ? window.CardVault.renderStatusPill("stale", "Shadow — abstain") : ""}
                </div>
                <p class="daily-parlay__empty">${this.escapeHtml(reason)}</p>
                <p class="daily-parlay__state">${this.escapeHtml(footer)}</p>
            `;
            return;
        }
        content.innerHTML = `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill("stale", `${plays.length} shadow candidate${plays.length === 1 ? "" : "s"}`) : ""}
            </div>
            <div class="vault-board vault-board--legs">
                ${plays.map((play, index) => this.renderV4Single(play, index + 1)).join("")}
            </div>
            <p class="daily-parlay__state">${this.escapeHtml(footer)}</p>
        `;
    }

    renderV4Single(play, index) {
        if (!window.CardVault) return "";
        const name = String(play.player || "").trim() || "Unknown player";
        const parts = name.split(/\s+/).filter(Boolean);
        const monogram = parts.length >= 2
            ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase()
            : (parts[0] || "NA").slice(0, 2).toUpperCase();
        return window.CardVault.renderLegCard({
            rank: index,
            monogram,
            name,
            market: `${String(play.direction || "OVER").toUpperCase()} ${this.formatNumber(play.line, 1)} ${play.target || "H"}`,
            context: "V4 singles shadow — no stake authorized",
            metrics: [
                ["P balanced", this.formatPct(play.balanced_probability)],
                ["Price", this.formatAmerican(play.american_price)],
                ["Edge", this.formatSignedPp(play.probability_edge)],
                ["EV", this.formatSignedPct(play.decision_ev)],
            ],
            betslipUrl: play.sportsbook_deeplink || "",
            deeplinksByRegion: play.deeplinks_by_region || null,
            settlementRow: play,
        });
    }

    /**
     * PARLAY_POLICY_V2 -- the only "parlay" product path. The legacy
     * ticket system (daily_parlay.selected_ticket) is CONTROL/diagnostic
     * only and is no longer surfaced on this page at all (removed
     * 2026-08-29: its 62% leg-probability floor is looser than the real
     * singles publication policy's 65% floor, so a merged leg could
     * appear as a Solo Bet the singles policy itself would have
     * rejected -- a real parlay research leg is not an independently
     * selected straight bet). Reads ONLY this.data.parlays. Status
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
    async loadSameGameParlay(date = null) {
        const content = this.elements.sameGameParlayContent;
        if (!content) return;
        try {
            const response = await fetch(this.productUrl("same_game_predictions.json", date));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.sameGameData = await response.json();
            if (!date) this.assertCurrentArtifact(this.sameGameData, "Same-game parlay board");
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
            const strict = Array.isArray(game.combo_candidates) ? game.combo_candidates : [];
            const fallback = strict.length ? [] : (Array.isArray(game.exploratory_ev_candidates) ? game.exploratory_ev_candidates : []);
            for (const combo of [...strict, ...fallback]) allCombos.push({ game, combo, withheld: !strict.includes(combo) });
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
        const withheld = best.withheld ? "Best real priced fallback — withheld from the tight-quality set." : "";

        content.innerHTML = this.sameGameParlayHeader() + `
            <div class="same-game-parlay__grid">${this.renderSameGameCombo(best.game, best.combo, best.withheld)}</div>
            ${withheld ? `<p class="daily-parlay__state">${this.escapeHtml(withheld)}</p>` : ""}
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

    renderSameGameCombo(game, combo, isWithheldFallback = false) {
        const matchup = `${this.escapeHtml(game.away_team || "")} @ ${this.escapeHtml(game.home_team || "")}`;
        const starters = `${this.escapeHtml(game.away_starter_name || "TBD")} vs ${this.escapeHtml(game.home_starter_name || "TBD")}`;
        const authorized = Boolean(combo.candidate_authorized);
        const pillTone = authorized ? "active" : "stale";
        const pillLabel = isWithheldFallback ? "Withheld fallback — shadow only" : (authorized ? "Selected -- shadow only" : "Shadow only");

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
    async loadPitcherParlay(date = null) {
        const content = this.elements.pitcherParlayContent;
        if (!content) return;
        try {
            const response = await fetch(this.productUrl("pitcher_parlay_predictions.json", date));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.pitcherParlayData = await response.json();
            if (!date) this.assertCurrentArtifact(this.pitcherParlayData, "Pitcher parlay board");
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

        const parlay = data.parlay || data.max_hit_control;
        const isWithheldFallback = !data.parlay && Boolean(data.max_hit_control);
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
    async loadHighHitParlay(date = null) {
        const content = this.elements.highHitParlayContent;
        if (!content) return;
        try {
            const response = await fetch(this.productUrl("high_hit_parlay_predictions.json", date));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.highHitParlayData = await response.json();
            if (!date) this.assertCurrentArtifact(this.highHitParlayData, "High-hit parlay board");
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

        const published = Array.isArray(data?.parlays) ? data.parlays : [];
        const fallback = data?.shadow_fallback;
        if (!data || (!published.length && !fallback)) {
            content.innerHTML = this.highHitParlayHeader() + `
                <p class="daily-parlay__empty">No real combination of today's eligible legs cleared the combined-probability floor.</p>
                <p class="daily-parlay__state">${this.escapeHtml(footer)}</p>
            `;
            return;
        }

        const cards = (published.length ? published : [fallback]).map((parlay) => this.renderHighHitCombo(parlay)).join("");
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
                    <div><strong>${legs.length}-leg high-hit combo</strong>${parlay.selection_status === "WITHHELD_PRODUCT_GATES" ? " — withheld fallback" : ""}</div>
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

    async loadExoticMarkets(date = null) {
        const content = this.elements.exoticMarketsContent;
        if (!content) return;
        try {
            const response = await fetch(this.productUrl("exotic_market_predictions.json", date));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.exoticMarketsData = await response.json();
            if (!date) this.assertCurrentArtifact(this.exoticMarketsData, "Exotic-market board");
        } catch (_error) {
            this.exoticMarketsData = null;
        }
        this.renderExoticMarkets();
    }

    renderExoticMarkets() {
        const content = this.elements.exoticMarketsContent;
        if (!content) return;
        const data = this.exoticMarketsData;
        if (!data || data.status !== "ok") {
            content.innerHTML = `<p class="daily-parlay__empty">No current exotic-market report is available.</p>`;
            return;
        }
        const candidates = Array.isArray(data.candidates) ? data.candidates.slice(0, 6) : [];
        const registry = Array.isArray(data.market_registry) ? data.market_registry : [];
        const cards = candidates.map((candidate, index) => this.renderExoticCandidate(candidate, index + 1)).join("");
        const blocked = registry
            .filter((market) => !String(market.readiness || "").startsWith("SCORABLE"))
            .map((market) => `${market.market}: ${market.readiness}`)
            .join(" / ");
        content.innerHTML = `
            <div class="daily-parlay__header daily-parlay__header--status-only">
                ${window.CardVault ? window.CardVault.renderStatusPill("stale", "Shadow only") : ""}
            </div>
            ${cards ? `<div class="vault-board vault-board--legs">${cards}</div>` : `<p class="daily-parlay__empty">No matching real total lines are priced yet.</p>`}
            <p class="daily-parlay__state">Policy: ${this.escapeHtml(data.policy || "EXOTIC_MARKETS_V1_SHADOW")} / Scored: ${candidates.length} / Execution: not authorized</p>
            <p class="daily-parlay__state">Model gates: ${this.escapeHtml(blocked || "none")}</p>
        `;
    }

    renderExoticCandidate(candidate, index) {
        if (!window.CardVault) return "";
        const matchup = `${candidate.away_team || "?"} @ ${candidate.home_team || "?"}`;
        const side = String(candidate.side || "").toLowerCase() === "under" ? "Under" : "Over";
        return window.CardVault.renderLegCard({
            rank: index,
            monogram: "XM",
            name: `${side} ${this.formatNumber(candidate.line, 1)}`,
            market: this.formatSameGameMarketLabel(candidate.market),
            context: matchup,
            metrics: [
                ["Probability", this.formatPct(candidate.model_probability)],
                ["Diagnostic EV", this.formatSignedPct(candidate.expected_value_per_unit)],
                ["Odds", this.formatAmerican(candidate.price_american)],
            ],
            betslipUrl: candidate.sportsbook_deeplink || "",
            deeplinksByRegion: candidate.deeplinks_by_region || null,
            settlementRow: candidate,
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
