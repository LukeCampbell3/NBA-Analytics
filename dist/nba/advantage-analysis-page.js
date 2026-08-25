// Shared engine for the split advantage-routing pages (Stats, Drive-Pass,
// Post-Pass). Each page is a genuinely separate, bookmarkable route with
// its own HTML -- not a query-string mode switch on one shared page -- but
// they read the same per-player JSON and share the same real rendering
// logic, so that logic lives here once instead of being copy-pasted three
// times and drifting.
//
// Every render* method guards on the DOM elements it needs: a page that
// doesn't include a given section's markup simply skips that method, so
// the same engine safely powers a baseline-only page (Stats) and a
// full mode-specific page (Drive-Pass / Post-Pass) with no page-specific
// branching required beyond the fixed `mode` in AR_PAGE_CONFIG.
//
// The original bundled advantage-routing.html/.js page is left untouched
// for backward compatibility with any existing links -- this is a
// parallel implementation, not a replacement for it.

const AR_DATA_ROOT = "data/advantage-routing";

const AR_ZONE_COLORS = {
    RIM: "#c02c3a",
    SHORT_PAINT: "#e0793f",
    MIDRANGE: "#c9a227",
    CORNER_3: "#2f9e6b",
    ABOVE_BREAK_3: "#3a7bd5",
};

const AR_ZONE_LABELS = {
    RIM: "Rim",
    SHORT_PAINT: "Short paint",
    MIDRANGE: "Midrange",
    CORNER_3: "Corner 3",
    ABOVE_BREAK_3: "Three (corner + above-break combined)",
};

class AdvantageAnalysisPage {
    constructor(config) {
        this.config = Object.assign({ mode: null, pageLabel: "", navHref: "" }, config || {});
        this.elements = this.collectElements();
        this.players = [];
        this.playerCache = new Map();
        this.currentPlayer = null;
        this.bindControls();
        this.init();
    }

    collectElements() {
        const ids = [
            "arRunFacts", "arPlayerSelect", "arDataUnavailableNotice",
            "arBaselineGrid",
            "arGravityChart",
            "arRecipientNote", "arRecipientTableBody", "arRecipientTable",
            "arHalfCourt", "arZoneLegend",
            "arResearchSummary",
            "arUsageSlider", "arUsageOut", "arPassTendencySlider", "arPassTendencyOut",
            "arAdvancedToggle", "arAdvancedGroup",
            "arElasticitySlider", "arElasticityOut", "arRetentionSlider", "arRetentionOut",
            "arTurnoverSlider", "arTurnoverOut", "arSaturationSlider", "arSaturationOut",
            "arResetButton", "arCausalityExplanation",
            "arBaselineColumn", "arSimColumn", "arScenarioSelect", "arMonteCarlo",
            "arCompareSelectors", "arCompareTable",
            "arProvenance",
        ];
        const map = {};
        ids.forEach((id) => { map[id] = document.getElementById(id); });
        return map;
    }

    navLinks() {
        return [
            { label: "Board", href: "/nba/predictions/", active: this.config.pageLabel === "Board" },
            { label: "Stats", href: "/nba/stats/", active: this.config.pageLabel === "Stats" },
            { label: "Drive-Pass", href: "/nba/drive-pass/", active: this.config.pageLabel === "Drive-Pass" },
            { label: "Post-Pass", href: "/nba/post-pass/", active: this.config.pageLabel === "Post-Pass" },
            { label: "Method", href: "/nba/prediction-about/", active: this.config.pageLabel === "Method" },
        ];
    }

    mountShell() {
        if (!window.CardVaultShell) return;
        window.CardVaultShell.mount({
            brandTitle: "In The Cards Analytics",
            brandHref: "/",
            sportSlug: "nba",
            sportAccent: "#c02c3a",
            navLinks: this.navLinks(),
            showDisclaimer: true,
        });
    }

    async init() {
        this.mountShell();
        try {
            const response = await fetch(`${AR_DATA_ROOT}/players.json?v=${Date.now()}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const index = await response.json();
            this.players = index.players || [];
            this.populatePlayerSelect();
            this.populateCompareSelectors();
            if (this.players.length) {
                await this.selectPlayer(this.players[0].slug);
            } else if (this.elements.arRunFacts) {
                this.elements.arRunFacts.textContent = "No advantage-routing player artifacts are available yet.";
            }
        } catch (error) {
            console.error(error);
            if (this.elements.arRunFacts) {
                this.elements.arRunFacts.textContent = `Unable to load advantage-routing data: ${error.message}`;
            }
        }
    }

    bindControls() {
        if (this.elements.arPlayerSelect) {
            this.elements.arPlayerSelect.addEventListener("change", (event) => this.selectPlayer(event.target.value));
        }

        if (this.elements.arAdvancedToggle && this.elements.arAdvancedGroup) {
            this.elements.arAdvancedToggle.addEventListener("click", () => {
                const expanded = this.elements.arAdvancedToggle.getAttribute("aria-expanded") === "true";
                this.elements.arAdvancedToggle.setAttribute("aria-expanded", String(!expanded));
                this.elements.arAdvancedGroup.hidden = expanded;
                this.elements.arAdvancedToggle.textContent = expanded ? "Advanced controls" : "Hide advanced controls";
            });
        }

        const sliders = [
            this.elements.arUsageSlider, this.elements.arPassTendencySlider, this.elements.arElasticitySlider,
            this.elements.arRetentionSlider, this.elements.arTurnoverSlider, this.elements.arSaturationSlider,
        ].filter(Boolean);
        sliders.forEach((slider) => slider.addEventListener("input", () => this.renderSimulation()));

        if (this.elements.arScenarioSelect) {
            this.elements.arScenarioSelect.addEventListener("change", () => this.renderSimulation());
        }
        if (this.elements.arResetButton) {
            this.elements.arResetButton.addEventListener("click", () => this.resetToBaseline());
        }
        if (this.elements.arRecipientTable) {
            this.elements.arRecipientTable.addEventListener("click", (event) => {
                const th = event.target.closest("th[data-sort]");
                if (!th) return;
                this.sortRecipients(th.dataset.sort);
            });
        }
    }

    populatePlayerSelect() {
        if (!this.elements.arPlayerSelect) return;
        this.elements.arPlayerSelect.innerHTML = this.players
            .map((p) => `<option value="${this.escape(p.slug)}">${this.escape(p.name)}</option>`)
            .join("");
    }

    populateCompareSelectors() {
        if (!this.elements.arCompareSelectors) return;
        const options = this.players.map((p) => `<option value="${this.escape(p.slug)}">${this.escape(p.name)}</option>`).join("");
        this.elements.arCompareSelectors.innerHTML = [0, 1, 2, 3]
            .map((i) => `<select data-compare-index="${i}"><option value="">-- none --</option>${options}</select>`)
            .join("");
        this.elements.arCompareSelectors.querySelectorAll("select").forEach((select, i) => {
            if (this.players[i]) select.value = this.players[i].slug;
            select.addEventListener("change", () => this.renderComparison());
        });
        this.renderComparison();
    }

    async loadPlayer(slug) {
        if (this.playerCache.has(slug)) return this.playerCache.get(slug);
        const response = await fetch(`${AR_DATA_ROOT}/${slug}.json?v=${Date.now()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        this.playerCache.set(slug, data);
        return data;
    }

    async selectPlayer(slug) {
        try {
            const data = await this.loadPlayer(slug);
            this.currentPlayer = data;
            if (this.elements.arPlayerSelect) this.elements.arPlayerSelect.value = slug;
            this.render();
        } catch (error) {
            console.error(error);
            if (this.elements.arRunFacts) {
                this.elements.arRunFacts.textContent = `Unable to load ${slug}: ${error.message}`;
            }
        }
    }

    render() {
        const d = this.currentPlayer;
        if (!d) return;
        if (this.elements.arRunFacts) {
            const gp = this.metricValue(d.baseline.games_played);
            const usg = this.metricValue(d.baseline.usage_pct);
            this.elements.arRunFacts.innerHTML = [
                `${this.escape(d.player.name)}`,
                `${this.escape(d.player.season)}`,
                `${this.formatNum(gp, 0)} real games`,
                `${this.formatNum(usg, 1)}% real USG`,
                `${d.recipients.sample_size} real sampled assists`,
            ].map((item) => `<span>${item}</span>`).join("");
        }

        this.renderBaseline();
        this.renderGravity();
        this.renderRecipients();
        this.renderShotDestination();
        this.renderResearchSummary();
        this.renderModeSpecific();
        this.renderProvenance();
        this.resetToBaseline();
        this.renderComparison();
    }

    renderModeSpecific() {
        if (!this.config.mode || !this.elements.arDataUnavailableNotice) return;
        const d = this.currentPlayer;
        if (!d) return;
        const node = this.config.mode === "drive" ? d.drive : (this.config.mode === "post" ? d.post : d.interior_hub);
        if (!node) return;
        const routingVector = node.routing_vector;
        if (routingVector && routingVector.status === "UNAVAILABLE") {
            this.elements.arDataUnavailableNotice.hidden = false;
            this.elements.arDataUnavailableNotice.innerHTML =
                `<strong>Routing-state vector unavailable for this mode.</strong> ${this.escape(routingVector.reason || "")}`;
        } else {
            this.elements.arDataUnavailableNotice.hidden = true;
        }
    }

    // ---------------- baseline ----------------

    renderBaseline() {
        if (!this.elements.arBaselineGrid) return;
        const b = this.currentPlayer.baseline;
        const cards = [
            ["USG%", b.usage_pct, (v) => this.formatNum(v, 1) + "%"],
            ["Minutes/G", b.minutes_per_game, (v) => this.formatNum(v, 1)],
            ["Decision touches/G", b.decision_touches_per_game, (v) => this.formatNum(v, 1)],
            ["AST/G", b.ast_per_game, (v) => this.formatNum(v, 2)],
            ["TOV/G", b.tov_per_game, (v) => this.formatNum(v, 2)],
            ["AST / decision touch", b.ast_per_decision_touch, (v) => this.formatNum(v, 3)],
            ["TOV / decision touch", b.tov_per_decision_touch, (v) => this.formatNum(v, 3)],
            ["Advantage-pass %", b.advantage_pass_pct, (v) => this.formatNum(v, 1) + "%"],
        ];
        this.elements.arBaselineGrid.innerHTML = cards.map(([label, metric, fmt]) => this.metricCard(label, metric, fmt)).join("");
    }

    metricCard(label, metric, formatter) {
        const badge = this.provenanceBadge(metric.status);
        const value = metric.value === null || metric.value === undefined ? "n/a" : formatter(metric.value);
        const title = metric.status === "UNAVAILABLE" ? (metric.reason || "") : (metric.method || metric.source || "");
        return `<article class="prediction-about-metric-card" title="${this.escape(title)}"><span>${this.escape(label)}${badge}</span><strong>${this.escape(value)}</strong></article>`;
    }

    provenanceBadge(status) {
        const cls = {
            OBSERVED: "ar-badge--observed", DERIVED: "ar-badge--derived", RECONSTRUCTED: "ar-badge--reconstructed",
            SIMULATED: "ar-badge--simulated", UNAVAILABLE: "ar-badge--unavailable",
        }[status] || "ar-badge--unavailable";
        return `<span class="ar-badge ${cls}">${this.escape(status || "n/a")}</span>`;
    }

    // ---------------- gravity ----------------

    renderGravity() {
        if (!this.elements.arGravityChart) return;
        const gravity = this.currentPlayer.gravity;
        const mechanisms = ["PAINT_FACEUP_GRAVITY", "VERTICAL_GRAVITY", "SHORT_ROLL_GRAVITY", "POP_GRAVITY", "PERIMETER_GRAVITY", "POST_SCORING_GRAVITY"];
        this.elements.arGravityChart.innerHTML = mechanisms.map((mech) => {
            const metrics = gravity.components[mech] || {};
            const values = Object.values(metrics).filter((m) => m.value !== null && m.value !== undefined && typeof m.value === "number");
            const anyReal = values.length > 0;
            const primary = values.find((m) => m.status !== "UNAVAILABLE");
            const barWidth = primary ? Math.min(100, Math.max(4, Math.abs(primary.value) * (primary.value <= 1 ? 100 : 3))) : 0;
            const componentChips = Object.entries(metrics).map(([name, m]) => {
                const val = m.value === null || m.value === undefined ? "n/a" : (typeof m.value === "number" ? m.value.toFixed(3) : m.value);
                return `<span class="ar-component-chip" title="${this.escape(m.method || m.source || m.reason || "")}">${this.escape(name)}: ${this.escape(val)} ${this.provenanceBadge(m.status)}</span>`;
            }).join("");
            return `
                <div class="ar-gravity-row">
                    <span class="ar-gravity-name">${this.escape(mech.replace("_GRAVITY", "").replace("_", " "))}</span>
                    <div class="ar-gravity-bar-track"><div class="ar-gravity-bar-fill" style="width:${anyReal ? barWidth : 0}%; opacity:${anyReal ? 1 : 0.25}"></div></div>
                    <span>${anyReal ? this.provenanceBadge(primary ? primary.status : "UNAVAILABLE") : this.provenanceBadge("UNAVAILABLE")}</span>
                    <div class="ar-gravity-components">${componentChips}</div>
                </div>`;
        }).join("");
    }

    // ---------------- recipients ----------------

    renderRecipients() {
        if (!this.elements.arRecipientTableBody) return;
        const network = this.currentPlayer.recipients;
        if (this.elements.arRecipientNote) this.elements.arRecipientNote.textContent = network.sample_description || "";
        this._recipients = (network.recipients || []).slice();
        this._recipientSort = { key: "assists", dir: -1 };
        this.paintRecipientTable();
    }

    sortRecipients(key) {
        if (this._recipientSort.key === key) {
            this._recipientSort.dir *= -1;
        } else {
            this._recipientSort = { key, dir: -1 };
        }
        this.paintRecipientTable();
    }

    paintRecipientTable() {
        if (!this.elements.arRecipientTableBody) return;
        const { key, dir } = this._recipientSort;
        const rows = this._recipients.slice().sort((a, b) => {
            const av = a[key] ? a[key].value : 0;
            const bv = b[key] ? b[key].value : 0;
            return dir * ((av || 0) - (bv || 0));
        });
        this.elements.arRecipientTableBody.innerHTML = rows.map((r) => `
            <tr>
                <td>${this.formatNum(r.assists.value, 0)}</td>
                <td>${this.escape(r.recipient_label)}</td>
                <td>${this.formatNum((r.assist_share.value || 0) * 100, 1)}%</td>
                <td>${r.high_value_share_index.value === null ? "n/a" : this.formatNum(r.high_value_share_index.value, 2)}</td>
                <td>${this.escape(r.most_common_resulting_shot ? (AR_ZONE_LABELS[r.most_common_resulting_shot] || r.most_common_resulting_shot) : "n/a")}</td>
            </tr>
        `).join("");
        if (this.elements.arRecipientTable) {
            this.elements.arRecipientTable.querySelectorAll("th[data-sort]").forEach((th) => {
                th.classList.toggle("is-sorted", th.dataset.sort === key);
            });
        }
    }

    // ---------------- shot destination (half court) ----------------

    renderShotDestination() {
        if (!this.elements.arHalfCourt) return;
        const network = this.currentPlayer.recipients;
        const zoneTotals = {};
        let total = 0;
        (network.recipients || []).forEach((r) => {
            Object.entries(r.zone_breakdown || {}).forEach(([zone, count]) => {
                zoneTotals[zone] = (zoneTotals[zone] || 0) + count;
                total += count;
            });
        });

        const svg = this.elements.arHalfCourt;
        const zoneRegions = {
            RIM: { cx: 250, cy: 60, r: 45 },
            SHORT_PAINT: { cx: 250, cy: 130, r: 55 },
            MIDRANGE: { cx: 250, cy: 230, r: 90 },
            CORNER_3: { cx: 60, cy: 380, r: 45 },
            ABOVE_BREAK_3: { cx: 250, cy: 380, r: 90 },
        };
        let svgContent = `
            <rect x="10" y="10" width="480" height="450" fill="none" stroke="currentColor" stroke-opacity="0.25"/>
            <rect x="170" y="10" width="160" height="190" fill="none" stroke="currentColor" stroke-opacity="0.35"/>
            <circle cx="250" cy="60" r="60" fill="none" stroke="currentColor" stroke-opacity="0.35"/>
            <path d="M 30 10 A 220 220 0 0 0 30 450" fill="none" stroke="currentColor" stroke-opacity="0.35"/>
            <path d="M 470 10 A 220 220 0 0 1 470 450" fill="none" stroke="currentColor" stroke-opacity="0.35"/>
        `;
        Object.entries(zoneRegions).forEach(([zone, pos]) => {
            const count = zoneTotals[zone] || 0;
            const share = total ? count / total : 0;
            const color = AR_ZONE_COLORS[zone] || "#888";
            const opacity = total ? 0.15 + share * 0.75 : 0.08;
            svgContent += `<circle cx="${pos.cx}" cy="${pos.cy}" r="${pos.r}" fill="${color}" fill-opacity="${opacity.toFixed(2)}" stroke="${color}" stroke-opacity="0.6"/>`;
            svgContent += `<text x="${pos.cx}" y="${pos.cy}" text-anchor="middle" dominant-baseline="middle" font-size="14" fill="currentColor">${count}</text>`;
        });
        svg.innerHTML = svgContent;

        if (this.elements.arZoneLegend) {
            this.elements.arZoneLegend.innerHTML = Object.entries(AR_ZONE_LABELS).map(([zone, label]) => {
                const count = zoneTotals[zone] || 0;
                const share = total ? ((count / total) * 100).toFixed(1) : "0.0";
                return `<span class="ar-zone-legend-item"><span class="ar-zone-swatch" style="background:${AR_ZONE_COLORS[zone]}"></span>${this.escape(label)}: ${count} (${share}%)</span>`;
            }).join("");
        }
    }

    // ---------------- research summary ----------------

    renderResearchSummary() {
        if (!this.elements.arResearchSummary) return;
        const summary = this.currentPlayer.research_summary;
        const chips = (summary.archetype || []).map((a) => `<span class="ar-archetype-chip">${this.escape(a.replace(/_/g, " "))}</span>`).join("");
        const recipients = (summary.best_recipients || []).map((r) => `${this.escape(r.label)} (${this.formatNum((r.assist_share || 0) * 100, 1)}% of sampled assists)`).join(", ");
        const caveats = (summary.caveats || []).map((c) => `<p class="ar-caveat">${this.escape(c)}</p>`).join("");
        this.elements.arResearchSummary.innerHTML = `
            <div>${chips}<span class="ar-badge ar-badge--derived">confidence ${(summary.confidence * 100).toFixed(0)}%</span></div>
            <p>${this.escape(summary.simulation_finding)}</p>
            <p><strong>Primary gravity:</strong> ${this.escape((summary.primary_gravity || []).join(", ") || "n/a")}</p>
            <p><strong>Highest-leverage recipients (sampled):</strong> ${recipients || "n/a"}</p>
            <p><strong>Role constraint:</strong> ${this.escape(summary.role_constraint)}</p>
            ${caveats}
        `;
    }

    // ---------------- simulation ----------------

    resetToBaseline() {
        if (!this.elements.arUsageSlider) return;
        const b = this.currentPlayer.baseline;
        const usg = this.metricValue(b.usage_pct) || 15;
        this.elements.arUsageSlider.min = Math.max(1, usg * 0.5).toFixed(1);
        this.elements.arUsageSlider.max = Math.min(40, usg * 3).toFixed(1);
        this.elements.arUsageSlider.value = usg;
        if (this.elements.arPassTendencySlider) this.elements.arPassTendencySlider.value = 0;
        if (this.elements.arElasticitySlider) this.elements.arElasticitySlider.value = 0.6;
        if (this.elements.arRetentionSlider) this.elements.arRetentionSlider.value = 100;
        if (this.elements.arTurnoverSlider) this.elements.arTurnoverSlider.value = 0;
        if (this.elements.arSaturationSlider) this.elements.arSaturationSlider.value = 55;
        if (this.elements.arScenarioSelect) this.elements.arScenarioSelect.value = "NEUTRAL";
        this.renderSimulation();
    }

    saturationRetention(h, k) {
        return Math.exp(-k * Math.max(0, h - 1));
    }

    simulateLive(baseline, params) {
        const currentUsage = baseline.usage_pct || 1;
        const h = currentUsage > 0 ? 1 + params.elasticity * (params.targetUsage / currentUsage - 1) : 1;
        const saturation = this.saturationRetention(h, params.saturationK);
        const retention = params.retentionOverride !== null ? params.retentionOverride : saturation;
        const turnoverGrowth = params.turnoverOverride !== null ? params.turnoverOverride : (saturation > 0 ? 1 / saturation - 1 : 0);

        const simDecisionTouches = baseline.decisionTouches * h;
        const simPasses = simDecisionTouches * (1 + params.passTendencyChange);
        const simAssists = simPasses * baseline.astPerTouch * retention;
        const simMakes = simPasses * baseline.makesPerTouch * retention;
        const simTurnovers = simPasses * baseline.tovPerTouch * (1 + turnoverGrowth);

        return { h, saturation, retention, turnoverGrowth, simDecisionTouches, simPasses, simAssists, simMakes, simTurnovers };
    }

    renderSimulation() {
        if (!this.elements.arUsageSlider) return;
        const d = this.currentPlayer;
        if (!d) return;
        const b = d.baseline;

        const targetUsage = parseFloat(this.elements.arUsageSlider.value);
        const passTendencyChange = parseFloat(this.elements.arPassTendencySlider.value) / 100;
        const elasticity = parseFloat(this.elements.arElasticitySlider.value);
        const retentionSliderValue = parseFloat(this.elements.arRetentionSlider.value) / 100;
        const turnoverSliderValue = parseFloat(this.elements.arTurnoverSlider.value) / 100;
        const saturationK = parseFloat(this.elements.arSaturationSlider.value) / 100;

        this.elements.arUsageOut.textContent = `${targetUsage.toFixed(1)}%`;
        this.elements.arPassTendencyOut.textContent = `${passTendencyChange >= 0 ? "+" : ""}${(passTendencyChange * 100).toFixed(0)}%`;
        this.elements.arElasticityOut.textContent = elasticity.toFixed(2);
        this.elements.arRetentionOut.textContent = `${(retentionSliderValue * 100).toFixed(0)}%`;
        this.elements.arTurnoverOut.textContent = `${turnoverSliderValue >= 0 ? "+" : ""}${(turnoverSliderValue * 100).toFixed(0)}%`;
        this.elements.arSaturationOut.textContent = saturationK.toFixed(2);

        const baseline = {
            usage_pct: this.metricValue(b.usage_pct),
            decisionTouches: this.metricValue(b.decision_touches_per_game),
            astPerTouch: this.metricValue(b.ast_per_decision_touch),
            tovPerTouch: this.metricValue(b.tov_per_decision_touch),
            makesPerTouch: this.metricValue(b.makes_per_decision_touch),
        };

        const advancedOpen = this.elements.arAdvancedToggle && this.elements.arAdvancedToggle.getAttribute("aria-expanded") === "true";
        const params = {
            targetUsage, elasticity, passTendencyChange, saturationK,
            retentionOverride: advancedOpen ? retentionSliderValue : null,
            turnoverOverride: advancedOpen ? turnoverSliderValue : null,
        };
        const sim = this.simulateLive(baseline, params);

        const explanation = `Target usage ${targetUsage.toFixed(1)}% vs. current ${this.formatNum(baseline.usage_pct, 1)}%. `
            + `With decision-touch elasticity ${elasticity.toFixed(2)}, ${(elasticity * 100).toFixed(0)}% of the proportional role change is assumed to become additional decision touches (H=${sim.h.toFixed(2)}). `
            + `Pass tendency is changed by ${passTendencyChange >= 0 ? "+" : ""}${(passTendencyChange * 100).toFixed(0)}%. `
            + `Role saturation retains ${(sim.retention * 100).toFixed(0)}% of baseline efficiency, and turnover risk changes by ${sim.turnoverGrowth >= 0 ? "+" : ""}${(sim.turnoverGrowth * 100).toFixed(0)}%. `
            + `This is a SIMULATED, conditional projection under these explicit assumptions -- never a forecast.`;
        if (this.elements.arCausalityExplanation) this.elements.arCausalityExplanation.textContent = explanation;

        if (this.elements.arBaselineColumn) {
            this.elements.arBaselineColumn.innerHTML = [
                ["Decision touches/G", baseline.decisionTouches, 1],
                ["Assists/G", this.metricValue(b.ast_per_game), 2],
                ["Turnovers/G", this.metricValue(b.tov_per_game), 2],
            ].map(([label, value]) => this.bvsRow(label, value, null, 1)).join("");
        }

        if (this.elements.arSimColumn) {
            this.elements.arSimColumn.innerHTML = [
                ["Decision touches/G", sim.simDecisionTouches, this.metricValue(b.decision_touches_per_game)],
                ["Assists/G", sim.simAssists, this.metricValue(b.ast_per_game)],
                ["Receiver makes/G (assisted only)", sim.simMakes, null],
                ["Turnovers/G", sim.simTurnovers, this.metricValue(b.tov_per_game)],
            ].map(([label, value, baselineValue]) => this.bvsRow(label, value, baselineValue, 2)).join("");
        }

        this.renderMonteCarlo();
    }

    bvsRow(label, value, baselineValue, digits) {
        let deltaHtml = "";
        if (baselineValue !== null && baselineValue !== undefined && Number.isFinite(value)) {
            const delta = value - baselineValue;
            const cls = delta > 0 ? "ar-delta-up" : (delta < 0 ? "ar-delta-down" : "");
            deltaHtml = ` <span class="${cls}">(${delta >= 0 ? "+" : ""}${delta.toFixed(digits)})</span>`;
        }
        return `<div class="ar-bvs-metric-row"><span>${this.escape(label)}</span><span>${this.formatNum(value, digits)}${deltaHtml}</span></div>`;
    }

    renderMonteCarlo() {
        if (!this.elements.arMonteCarlo) return;
        const mc = this.currentPlayer.simulation_parameters && this.currentPlayer.simulation_parameters.monte_carlo;
        if (!mc || !mc.assists) {
            this.elements.arMonteCarlo.innerHTML = `<p>${this.provenanceBadge("UNAVAILABLE")} No Monte Carlo result was computed for this player (insufficient real baseline data).</p>`;
            return;
        }
        const cards = [
            ["Assists/G", mc.assists],
            ["Turnovers/G", mc.turnovers],
            ["Receiver makes/G", mc.receiver_makes],
        ];
        this.elements.arMonteCarlo.innerHTML = cards.map(([label, stats]) => {
            const range = stats.p90 - stats.p10 || 1;
            const fillPct = 100;
            return `
                <div class="ar-mc-card">
                    <span>${this.escape(label)}</span>
                    <strong>${this.formatNum(stats.median, 2)}</strong>
                    <span>P10 ${this.formatNum(stats.p10, 2)} &ndash; P90 ${this.formatNum(stats.p90, 2)}</span>
                    <div class="ar-mc-range-bar"><div class="ar-mc-range-bar-fill" style="left:0%; width:${fillPct}%"></div></div>
                </div>`;
        }).join("") + `<p class="ar-caveat">${mc.n_draws} draws, seed ${mc.seed} (reproducible).</p>`;
    }

    // ---------------- comparison ----------------

    async renderComparison() {
        if (!this.elements.arCompareSelectors || !this.elements.arCompareTable) return;
        const selects = Array.from(this.elements.arCompareSelectors.querySelectorAll("select"));
        const slugs = selects.map((s) => s.value).filter(Boolean);
        if (!slugs.length) {
            this.elements.arCompareTable.innerHTML = "";
            return;
        }
        const players = await Promise.all(slugs.map((slug) => this.loadPlayer(slug).catch(() => null)));
        const rows = [
            ["Usage %", (p) => this.formatNum(this.metricValue(p.baseline.usage_pct), 1)],
            ["Decision touches/G", (p) => this.formatNum(this.metricValue(p.baseline.decision_touches_per_game), 1)],
            ["AST/decision touch", (p) => this.formatNum(this.metricValue(p.baseline.ast_per_decision_touch), 3)],
            ["TOV/decision touch", (p) => this.formatNum(this.metricValue(p.baseline.tov_per_decision_touch), 3)],
            ["Sampled assists", (p) => p.recipients.sample_size],
            ["Archetype", (p) => (p.research_summary.archetype || []).join(", ")],
            ["Primary gravity", (p) => (p.research_summary.primary_gravity || []).join(", ")],
            ["Role constraint", (p) => p.research_summary.role_constraint],
        ];
        const header = `<thead><tr><th>Metric</th>${players.map((p) => `<th>${p ? this.escape(p.player.name) : "n/a"}</th>`).join("")}</tr></thead>`;
        const body = `<tbody>${rows.map(([label, fn]) => `<tr><td>${this.escape(label)}</td>${players.map((p) => `<td>${p ? this.escape(String(fn(p))) : "n/a"}</td>`).join("")}</tr>`).join("")}</tbody>`;
        this.elements.arCompareTable.innerHTML = header + body;
    }

    // ---------------- provenance ----------------

    renderProvenance() {
        if (!this.elements.arProvenance) return;
        const prov = this.currentPlayer.provenance;
        this.elements.arProvenance.innerHTML = `
            <p><strong>Season:</strong> ${this.escape(prov.season)} &middot; <strong>Generated:</strong> ${this.escape(this.formatTime(prov.generated_at_utc))}</p>
            <p><strong>Box scores:</strong> <code>${this.escape(prov.box_score_source || "n/a")}</code></p>
            <p><strong>Basketball-Reference sample:</strong> ${prov.bball_ref_games_sampled ? prov.bball_ref_games_sampled.length : 0} of ${prov.bball_ref_games_available_total || 0} real games -- ${this.escape(prov.bball_ref_sampling_method || "")}</p>
            <p><strong>stats.nba.com reachable:</strong> ${prov.stats_nba_com_reachable ? "yes" : "no"} -- ${this.escape(prov.stats_nba_com_note || "")}</p>
        `;
    }

    // ---------------- utils ----------------

    metricValue(metric) {
        if (!metric) return null;
        return metric.value === undefined ? null : metric.value;
    }

    formatTime(value) {
        if (!value) return "n/a";
        const parsed = new Date(value);
        return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    }

    formatNum(value, digits = 2) {
        const number = Number(value);
        return Number.isFinite(number) ? number.toFixed(digits) : "n/a";
    }

    escape(value) {
        return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }
}

document.addEventListener("DOMContentLoaded", () => new AdvantageAnalysisPage(window.AR_PAGE_CONFIG));
