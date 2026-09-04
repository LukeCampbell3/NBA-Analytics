/**
 * Drive-Pass -- modernized frontend controller.
 *
 * Standalone page controller (Drive-Pass only). The shared
 * `advantage-analysis-page.js` engine that also powers the Post-Pass page
 * is intentionally NOT modified; Post-Pass keeps using it verbatim while
 * Drive-Pass gets this dedicated modern implementation.
 *
 * Reads the same per-player artifacts under `data/advantage-routing/` --
 * no server or data changes -- and renders them into a mobile-first,
 * savant / craftednba-inspired layout with a hero card, section tabs,
 * a real half-court zone chart, and a live simulator.
 */
(function () {
    "use strict";

    const DATA_ROOT = "data/advantage-routing";
    const MODE = "drive";

    const ZONE_LABELS = {
        RIM: "Rim",
        SHORT_PAINT: "Short paint",
        MIDRANGE: "Midrange",
        CORNER_3: "Corner 3",
        ABOVE_BREAK_3: "Above-break 3",
    };

    const ZONE_VAR = {
        RIM: "--dp-zone-rim",
        SHORT_PAINT: "--dp-zone-short",
        MIDRANGE: "--dp-zone-mid",
        CORNER_3: "--dp-zone-c3",
        ABOVE_BREAK_3: "--dp-zone-ab3",
    };

    const ZONE_ORDER = ["RIM", "SHORT_PAINT", "MIDRANGE", "CORNER_3", "ABOVE_BREAK_3"];

    const GRAVITY_MECHANISMS = [
        { key: "PAINT_FACEUP_GRAVITY", label: "Paint face-up" },
        { key: "VERTICAL_GRAVITY", label: "Vertical" },
        { key: "SHORT_ROLL_GRAVITY", label: "Short roll" },
        { key: "POP_GRAVITY", label: "Pop" },
        { key: "PERIMETER_GRAVITY", label: "Perimeter" },
        { key: "POST_SCORING_GRAVITY", label: "Post" },
    ];

    const RECIPIENT_SORTS = [
        { key: "assists", label: "Assists" },
        { key: "assist_share", label: "Share" },
        { key: "high_value_share_index", label: "High-value" },
    ];

    // ---------------- utils ----------------

    const $ = (id) => document.getElementById(id);

    function esc(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }

    function num(value, digits = 2) {
        const n = Number(value);
        return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
    }

    function metricValue(metric) {
        if (!metric) return null;
        return metric.value === undefined ? null : metric.value;
    }

    function badgeClass(status) {
        return {
            OBSERVED: "dp-badge--observed",
            DERIVED: "dp-badge--derived",
            RECONSTRUCTED: "dp-badge--reconstructed",
            SIMULATED: "dp-badge--simulated",
            UNAVAILABLE: "dp-badge--unavailable",
        }[status] || "dp-badge--unavailable";
    }

    function badge(status, opts = {}) {
        const cls = badgeClass(status);
        const label = opts.short ? status.slice(0, 3) : status;
        return `<span class="dp-badge ${cls}" title="${esc(status || "n/a")}">${esc(label || "n/a")}</span>`;
    }

    function initialsFromName(name) {
        return String(name || "")
            .split(/\s+/).filter(Boolean).slice(0, 2)
            .map((w) => w[0]?.toUpperCase() || "")
            .join("");
    }

    function formatTime(value) {
        if (!value) return "n/a";
        const parsed = new Date(value);
        return Number.isNaN(parsed.valueOf())
            ? String(value)
            : parsed.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    }

    // ---------------- half court SVG ----------------

    // Real NBA half-court geometry in feet, scaled to the SVG viewBox.
    // Coordinate system: x=[0,50] (sideline to sideline), y=[0,47] (baseline
    // out to half court). The rim center is at (25, 4.75). Zones are drawn
    // as translucent shaded regions; opacity is driven by share of sampled
    // assists into that zone.
    function drawCourt(svgEl, zoneTotals, totalAssists) {
        const W = 500, H = 470;
        const scale = (feet, axis) => (axis === "x" ? feet * (W / 50) : feet * (H / 47));
        const px = (fx, fy) => `${scale(fx, "x").toFixed(2)},${scale(fy, "y").toFixed(2)}`;

        // Zone shade colors via CSS var lookup.
        const style = getComputedStyle(svgEl);
        const zoneColor = (zone) => style.getPropertyValue(ZONE_VAR[zone]).trim() || "#888";

        const share = (zone) => {
            const c = zoneTotals[zone] || 0;
            if (!totalAssists) return 0;
            return c / totalAssists;
        };
        const opacity = (zone) => (totalAssists ? 0.20 + share(zone) * 0.70 : 0.09);

        // Rim center in feet
        const rimX = 25, rimY = 4.75;

        // Compose SVG content
        const parts = [];

        // Zone: RIM (restricted arc, 4 ft radius)
        parts.push(`<circle cx="${scale(rimX, "x")}" cy="${scale(rimY, "y")}" r="${scale(4, "x")}"
            fill="${zoneColor("RIM")}" fill-opacity="${opacity("RIM").toFixed(3)}"
            stroke="${zoneColor("RIM")}" stroke-opacity="0.6"/>`);

        // Zone: SHORT_PAINT (paint minus rim). Paint = 16ft wide (x 17..33) by 19ft deep (y 0..19).
        // We shade as an annulus around rim inside paint bounds.
        parts.push(`<path d="
            M ${px(17, 0)} L ${px(33, 0)} L ${px(33, 19)} L ${px(17, 19)} Z"
            fill="${zoneColor("SHORT_PAINT")}" fill-opacity="${opacity("SHORT_PAINT").toFixed(3)}"/>`);

        // Zone: MIDRANGE - from paint boundary to arc (23.75 ft) ring segment
        // Draw as sweeping arc region between paint and 3pt arc, capped at corner y=14.
        const midOuterR = 23.75;
        parts.push(`<path d="
            M ${px(3, 14)}
            L ${px(3, 0)}
            L ${px(17, 0)}
            L ${px(17, 19)}
            L ${px(33, 19)}
            L ${px(33, 0)}
            L ${px(47, 0)}
            L ${px(47, 14)}
            A ${scale(midOuterR, "x")} ${scale(midOuterR, "y")} 0 0 1 ${px(3, 14)}
            Z"
            fill="${zoneColor("MIDRANGE")}" fill-opacity="${opacity("MIDRANGE").toFixed(3)}"/>`);

        // Zone: CORNER_3 - two corner strips (x < 3 or x > 47), y up to 14 ft
        const c3Op = opacity("CORNER_3").toFixed(3);
        parts.push(`<path d="M ${px(0, 0)} L ${px(3, 0)} L ${px(3, 14)} L ${px(0, 14)} Z"
            fill="${zoneColor("CORNER_3")}" fill-opacity="${c3Op}"/>`);
        parts.push(`<path d="M ${px(47, 0)} L ${px(50, 0)} L ${px(50, 14)} L ${px(47, 14)} Z"
            fill="${zoneColor("CORNER_3")}" fill-opacity="${c3Op}"/>`);

        // Zone: ABOVE_BREAK_3 - outside the arc from corners up to half-court
        parts.push(`<path d="
            M ${px(0, 14)}
            L ${px(3, 14)}
            A ${scale(midOuterR, "x")} ${scale(midOuterR, "y")} 0 0 0 ${px(47, 14)}
            L ${px(50, 14)}
            L ${px(50, 47)}
            L ${px(0, 47)}
            Z"
            fill="${zoneColor("ABOVE_BREAK_3")}" fill-opacity="${opacity("ABOVE_BREAK_3").toFixed(3)}"/>`);

        // --- court lines on top ---
        const lineOpts = `stroke="currentColor" stroke-opacity="0.5" fill="none"`;

        // Baseline & sidelines & half-court line
        parts.push(`<path ${lineOpts} d="M ${px(0, 0)} L ${px(50, 0)} L ${px(50, 47)} L ${px(0, 47)} Z"/>`);
        // Paint rectangle
        parts.push(`<rect ${lineOpts} x="${scale(17, "x")}" y="${scale(0, "y")}"
            width="${scale(16, "x")}" height="${scale(19, "y")}"/>`);
        // Free-throw circle
        parts.push(`<circle ${lineOpts} cx="${scale(25, "x")}" cy="${scale(19, "y")}" r="${scale(6, "x")}"/>`);
        // Rim
        parts.push(`<circle ${lineOpts} cx="${scale(rimX, "x")}" cy="${scale(rimY, "y")}" r="${scale(0.75, "x")}"
            stroke-opacity="0.75"/>`);
        // Backboard
        parts.push(`<line ${lineOpts} x1="${scale(22, "x")}" y1="${scale(4, "y")}"
            x2="${scale(28, "x")}" y2="${scale(4, "y")}" stroke-width="2" stroke-opacity="0.7"/>`);
        // 3-point line (corners + arc)
        parts.push(`<line ${lineOpts} x1="${scale(3, "x")}" y1="${scale(0, "y")}"
            x2="${scale(3, "x")}" y2="${scale(14, "y")}"/>`);
        parts.push(`<line ${lineOpts} x1="${scale(47, "x")}" y1="${scale(0, "y")}"
            x2="${scale(47, "x")}" y2="${scale(14, "y")}"/>`);
        parts.push(`<path ${lineOpts} d="M ${px(3, 14)}
            A ${scale(midOuterR, "x")} ${scale(midOuterR, "y")} 0 0 0 ${px(47, 14)}"/>`);
        // Center circle at half court
        parts.push(`<circle ${lineOpts} cx="${scale(25, "x")}" cy="${scale(47, "y")}" r="${scale(6, "x")}"
            stroke-dasharray="4 4"/>`);

        // Zone labels with count (centered inside zone)
        const zoneLabelPos = {
            RIM: { fx: 25, fy: 4.75 },
            SHORT_PAINT: { fx: 25, fy: 13 },
            MIDRANGE: { fx: 25, fy: 22 },
            CORNER_3: { fx: 6.5, fy: 7 },
            ABOVE_BREAK_3: { fx: 25, fy: 33 },
        };
        Object.entries(zoneLabelPos).forEach(([zone, pos]) => {
            const c = zoneTotals[zone] || 0;
            if (!c && !totalAssists) return;
            parts.push(`<text x="${scale(pos.fx, "x")}" y="${scale(pos.fy, "y")}"
                text-anchor="middle" dominant-baseline="middle"
                font-size="18" font-weight="700"
                fill="currentColor" style="paint-order: stroke; stroke: var(--dp-bg-soft); stroke-width: 3px; stroke-linejoin: round;">${c}</text>`);
        });

        svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svgEl.innerHTML = parts.join("\n");
    }

    // ---------------- controller ----------------

    class DrivePassPage {
        constructor() {
            this.el = this.collect();
            this.players = [];
            this.playerCache = new Map();
            this.currentPlayer = null;
            this.compareSlugs = [];
            this.recipientSort = { key: "assists", dir: "desc" };
            this.activeSection = "overview";

            this.bind();
            this.mountShell();
            this.init();
        }

        collect() {
            const ids = [
                "dpSearch", "dpPlayerSelect", "dpHeroAvatar", "dpHeroName", "dpHeroMeta",
                "dpStatStrip", "dpUnavailable", "dpTabs",
                "dpOverviewGrid",
                "dpGravityChart",
                "dpRecipientSort", "dpRecipientsList",
                "dpCourt", "dpCourtLegend",
                "dpSummary",
                "dpUsage", "dpUsageOut", "dpPass", "dpPassOut",
                "dpElast", "dpElastOut", "dpRet", "dpRetOut",
                "dpTov", "dpTovOut", "dpSat", "dpSatOut",
                "dpAdvancedBtn", "dpAdvancedGroup", "dpResetBtn",
                "dpCausality", "dpBvsBaseline", "dpBvsSim", "dpScenario", "dpMonteCarlo",
                "dpCompareChips", "dpCompareTableWrap", "dpCompareAdd",
                "dpProvenance",
            ];
            const map = {};
            ids.forEach((id) => { map[id] = document.getElementById(id); });
            return map;
        }

        mountShell() {
            if (!window.CardVaultShell) return;
            window.CardVaultShell.mount({
                brandTitle: "In The Cards Analytics",
                brandHref: "/",
                sportSlug: "nba",
                sportAccent: "#c02c3a",
                navLinks: [
                    { label: "Board", href: "/nba/predictions/" },
                    { label: "Stats", href: "/nba/stats/" },
                    { label: "Drive-Pass", href: "/nba/drive-pass/", active: true },
                    { label: "Post-Pass", href: "/nba/post-pass/" },
                    { label: "Method", href: "/nba/prediction-about/" },
                ],
                showDisclaimer: true,
            });
        }

        bind() {
            if (this.el.dpPlayerSelect) {
                this.el.dpPlayerSelect.addEventListener("change", (e) => this.selectPlayer(e.target.value));
            }
            if (this.el.dpSearch) {
                this.el.dpSearch.addEventListener("input", () => this.filterPlayers());
            }
            if (this.el.dpTabs) {
                this.el.dpTabs.addEventListener("click", (e) => {
                    const btn = e.target.closest(".dp-tab");
                    if (!btn) return;
                    this.setSection(btn.dataset.section);
                });
            }
            if (this.el.dpRecipientSort) {
                this.el.dpRecipientSort.addEventListener("click", (e) => {
                    const btn = e.target.closest(".dp-sort-chip");
                    if (!btn) return;
                    const key = btn.dataset.sort;
                    if (this.recipientSort.key === key) {
                        this.recipientSort.dir = this.recipientSort.dir === "desc" ? "asc" : "desc";
                    } else {
                        this.recipientSort = { key, dir: "desc" };
                    }
                    this.renderRecipients();
                });
            }
            if (this.el.dpAdvancedBtn && this.el.dpAdvancedGroup) {
                this.el.dpAdvancedBtn.addEventListener("click", () => {
                    const open = this.el.dpAdvancedBtn.getAttribute("aria-expanded") === "true";
                    this.el.dpAdvancedBtn.setAttribute("aria-expanded", String(!open));
                    this.el.dpAdvancedGroup.hidden = open;
                    this.el.dpAdvancedBtn.textContent = open ? "Show advanced controls" : "Hide advanced controls";
                });
            }
            [this.el.dpUsage, this.el.dpPass, this.el.dpElast, this.el.dpRet,
             this.el.dpTov, this.el.dpSat, this.el.dpScenario].filter(Boolean).forEach((c) =>
                c.addEventListener("input", () => this.renderSimulation())
            );
            if (this.el.dpScenario) {
                this.el.dpScenario.addEventListener("change", () => this.renderSimulation());
            }
            if (this.el.dpResetBtn) {
                this.el.dpResetBtn.addEventListener("click", () => this.resetSimToBaseline());
            }
            if (this.el.dpCompareAdd) {
                this.el.dpCompareAdd.addEventListener("click", () => this.addCompareSlot());
            }
            if (this.el.dpCompareChips) {
                this.el.dpCompareChips.addEventListener("click", (e) => {
                    const btn = e.target.closest("[data-remove-slug]");
                    if (!btn) return;
                    this.removeCompareSlug(btn.dataset.removeSlug);
                });
                this.el.dpCompareChips.addEventListener("change", (e) => {
                    const sel = e.target.closest("select[data-compare-select]");
                    if (!sel) return;
                    const slug = sel.value;
                    if (slug && !this.compareSlugs.includes(slug)) {
                        this.compareSlugs.push(slug);
                    }
                    this.renderCompare();
                });
            }
        }

        async init() {
            try {
                const r = await fetch(`${DATA_ROOT}/players.json?v=${Date.now()}`);
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const idx = await r.json();
                this.players = idx.players || [];
                this.populateSelect();
                if (this.players.length) {
                    // Preselect first player
                    this.compareSlugs = this.players.slice(0, 2).map((p) => p.slug);
                    await this.selectPlayer(this.players[0].slug);
                } else if (this.el.dpHeroMeta) {
                    this.el.dpHeroMeta.textContent = "No advantage-routing artifacts are available yet.";
                }
            } catch (e) {
                console.error(e);
                if (this.el.dpHeroMeta) {
                    this.el.dpHeroMeta.textContent = `Unable to load advantage-routing data: ${e.message}`;
                }
            }
        }

        populateSelect() {
            if (!this.el.dpPlayerSelect) return;
            this.el.dpPlayerSelect.innerHTML = this.players
                .map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join("");
        }

        filterPlayers() {
            if (!this.el.dpPlayerSelect || !this.el.dpSearch) return;
            const q = this.el.dpSearch.value.toLowerCase().trim();
            const matches = this.players.filter((p) => !q || p.name.toLowerCase().includes(q));
            const preserveValue = this.el.dpPlayerSelect.value;
            this.el.dpPlayerSelect.innerHTML = matches
                .map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join("");
            // Prefer to keep the current player if still in the filtered set,
            // otherwise auto-select the first match.
            if (matches.some((p) => p.slug === preserveValue)) {
                this.el.dpPlayerSelect.value = preserveValue;
            } else if (matches.length) {
                this.el.dpPlayerSelect.value = matches[0].slug;
                this.selectPlayer(matches[0].slug);
            }
        }

        async loadPlayer(slug) {
            if (this.playerCache.has(slug)) return this.playerCache.get(slug);
            const r = await fetch(`${DATA_ROOT}/${slug}.json?v=${Date.now()}`);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const d = await r.json();
            this.playerCache.set(slug, d);
            return d;
        }

        async selectPlayer(slug) {
            try {
                const data = await this.loadPlayer(slug);
                this.currentPlayer = data;
                if (this.el.dpPlayerSelect && this.el.dpPlayerSelect.value !== slug) {
                    this.el.dpPlayerSelect.value = slug;
                }
                this.renderAll();
            } catch (e) {
                console.error(e);
                if (this.el.dpHeroMeta) {
                    this.el.dpHeroMeta.textContent = `Unable to load ${slug}: ${e.message}`;
                }
            }
        }

        setSection(id) {
            if (!id) return;
            this.activeSection = id;
            document.querySelectorAll(".dp-tab").forEach((t) => {
                t.classList.toggle("is-active", t.dataset.section === id);
                t.setAttribute("aria-selected", String(t.dataset.section === id));
            });
            document.querySelectorAll("[data-panel]").forEach((p) => {
                p.classList.toggle("dp-panel-hidden", p.dataset.panel !== id);
            });
        }

        // ---------------- render ----------------

        renderAll() {
            const d = this.currentPlayer;
            if (!d) return;
            this.renderHero();
            this.renderModeNotice();
            this.renderOverview();
            this.renderGravity();
            this.renderRecipients();
            this.renderShotDest();
            this.renderSummary();
            this.resetSimToBaseline();
            this.renderProvenance();
            this.renderCompare();
        }

        renderHero() {
            const d = this.currentPlayer;
            const b = d.baseline;
            if (this.el.dpHeroName) this.el.dpHeroName.textContent = d.player.name;
            if (this.el.dpHeroMeta) {
                const gp = metricValue(b.games_played);
                const mpg = metricValue(b.minutes_per_game);
                this.el.dpHeroMeta.innerHTML = [
                    `<strong>${esc(d.player.season)}</strong> season`,
                    `<strong>${num(gp, 0)}</strong> real games`,
                    `<strong>${num(mpg, 1)}</strong> MPG`,
                    `Sampled <strong>${d.recipients.sample_size}</strong> assists`,
                ].map((s) => `<span>${s}</span>`).join("");
            }
            if (this.el.dpHeroAvatar) {
                this.el.dpHeroAvatar.textContent = initialsFromName(d.player.name);
            }
            if (this.el.dpStatStrip) {
                const tiles = [
                    { label: "USG%", metric: b.usage_pct, format: (v) => `${num(v, 1)}%` },
                    { label: "Decision Touches/G", metric: b.decision_touches_per_game, format: (v) => num(v, 1) },
                    { label: "AST/G", metric: b.ast_per_game, format: (v) => num(v, 1) },
                    { label: "TOV/G", metric: b.tov_per_game, format: (v) => num(v, 1) },
                    { label: "AST/Touch", metric: b.ast_per_decision_touch, format: (v) => num(v, 3) },
                ];
                this.el.dpStatStrip.innerHTML = tiles.map((t) => {
                    const v = metricValue(t.metric);
                    const display = v === null ? "n/a" : t.format(v);
                    return `<div class="dp-stat-tile" title="${esc(t.metric?.method || t.metric?.source || "")}">
                        <span class="dp-stat-tile__label">${esc(t.label)}</span>
                        <span class="dp-stat-tile__value">${esc(display)}</span>
                    </div>`;
                }).join("");
            }
        }

        renderModeNotice() {
            if (!this.el.dpUnavailable) return;
            const node = this.currentPlayer.drive;
            const rv = node && node.routing_vector;
            if (rv && rv.status === "UNAVAILABLE") {
                this.el.dpUnavailable.hidden = false;
                this.el.dpUnavailable.innerHTML =
                    `<strong>Drive routing-state vector unavailable.</strong> ${esc(rv.reason || "")}`;
            } else {
                this.el.dpUnavailable.hidden = true;
            }
        }

        renderOverview() {
            if (!this.el.dpOverviewGrid) return;
            const b = this.currentPlayer.baseline;
            const cards = [
                ["USG%", b.usage_pct, (v) => `${num(v, 1)}%`],
                ["Decision touches/G", b.decision_touches_per_game, (v) => num(v, 1)],
                ["AST/G", b.ast_per_game, (v) => num(v, 2)],
                ["TOV/G", b.tov_per_game, (v) => num(v, 2)],
                ["AST/decision touch", b.ast_per_decision_touch, (v) => num(v, 3)],
                ["TOV/decision touch", b.tov_per_decision_touch, (v) => num(v, 3)],
                ["Advantage-pass %", b.advantage_pass_pct, (v) => `${num(v, 1)}%`],
                ["Minutes/G", b.minutes_per_game, (v) => num(v, 1)],
            ];
            this.el.dpOverviewGrid.innerHTML = cards.map(([label, metric, fmt]) => {
                const v = metricValue(metric);
                const value = v === null ? "n/a" : fmt(v);
                const title = metric?.status === "UNAVAILABLE"
                    ? (metric.reason || "")
                    : (metric?.method || metric?.source || "");
                return `<article class="dp-metric-card" title="${esc(title)}">
                    <div class="dp-metric-card__label">${esc(label)} ${badge(metric?.status, { short: true })}</div>
                    <div class="dp-metric-card__value">${esc(value)}</div>
                </article>`;
            }).join("");
        }

        renderGravity() {
            if (!this.el.dpGravityChart) return;
            const gravity = this.currentPlayer.gravity;
            const rows = GRAVITY_MECHANISMS.map((mech) => {
                const metrics = gravity.components[mech.key] || {};
                const values = Object.values(metrics).filter(
                    (m) => m.value !== null && m.value !== undefined && typeof m.value === "number"
                );
                const anyReal = values.length > 0;
                const primary = values.find((m) => m.status !== "UNAVAILABLE");
                const rawBar = primary ? Math.abs(primary.value) * (primary.value <= 1 ? 100 : 3) : 0;
                const barWidth = anyReal ? Math.min(100, Math.max(4, rawBar)) : 0;
                const chips = Object.entries(metrics).map(([name, m]) => {
                    const val = m.value === null || m.value === undefined
                        ? "n/a"
                        : (typeof m.value === "number" ? m.value.toFixed(3) : String(m.value));
                    return `<span class="dp-chip" title="${esc(m.method || m.source || m.reason || "")}">
                        ${esc(name.toLowerCase().replaceAll("_", " "))}: <strong>${esc(val)}</strong>
                        ${badge(m.status, { short: true })}
                    </span>`;
                }).join("");
                const badgeHtml = badge(primary ? primary.status : "UNAVAILABLE", { short: true });
                return `<div class="dp-gravity-row">
                    <div class="dp-gravity-name">${esc(mech.label)} ${badgeHtml}</div>
                    <div class="dp-gravity-bar">
                        <div class="dp-gravity-bar__fill${anyReal ? "" : " dp-gravity-bar__fill--muted"}"
                             style="width:${anyReal ? barWidth.toFixed(1) : 100}%"></div>
                    </div>
                    <div class="dp-gravity-components">${chips || `<span class="dp-chip">no components published</span>`}</div>
                </div>`;
            }).join("");
            this.el.dpGravityChart.innerHTML = rows;
        }

        renderRecipients() {
            if (!this.el.dpRecipientsList) return;
            const network = this.currentPlayer.recipients;
            const recipients = (network.recipients || []).slice();
            const { key, dir } = this.recipientSort;

            recipients.sort((a, b) => {
                const av = a[key] ? (a[key].value ?? 0) : 0;
                const bv = b[key] ? (b[key].value ?? 0) : 0;
                return (dir === "desc" ? -1 : 1) * (av - bv);
            });

            const maxAssists = recipients.reduce(
                (m, r) => Math.max(m, r.assists?.value || 0), 0
            ) || 1;

            if (this.el.dpRecipientSort) {
                this.el.dpRecipientSort.innerHTML = RECIPIENT_SORTS.map((s) => {
                    const active = s.key === key;
                    return `<button class="dp-sort-chip${active ? " is-active" : ""}"
                        data-sort="${esc(s.key)}" data-dir="${esc(dir)}" type="button">${esc(s.label)}</button>`;
                }).join("");
            }

            if (!recipients.length) {
                this.el.dpRecipientsList.innerHTML =
                    `<p class="dp-caveat">No sampled recipients in this artifact.</p>`;
                return;
            }

            this.el.dpRecipientsList.innerHTML = recipients.map((r) => {
                const assists = r.assists?.value || 0;
                const share = (r.assist_share?.value || 0) * 100;
                const hv = r.high_value_share_index?.value;
                const zoneLabel = r.most_common_resulting_shot
                    ? (ZONE_LABELS[r.most_common_resulting_shot] || r.most_common_resulting_shot)
                    : "n/a";
                const pct = (assists / maxAssists) * 100;
                return `<div class="dp-recipient-row">
                    <div class="dp-recipient-row__name">${esc(r.recipient_label)}</div>
                    <div class="dp-recipient-row__num">${num(assists, 0)} ast</div>
                    <div class="dp-recipient-row__bar">
                        <div class="dp-recipient-row__bar-fill" style="width:${pct.toFixed(1)}%"></div>
                    </div>
                    <div class="dp-recipient-row__meta">
                        <span>Share <strong>${num(share, 1)}%</strong></span>
                        <span>High-value idx <strong>${hv === null || hv === undefined ? "n/a" : num(hv, 2)}</strong></span>
                        <span>Common shot <strong>${esc(zoneLabel)}</strong></span>
                    </div>
                </div>`;
            }).join("");
        }

        renderShotDest() {
            if (!this.el.dpCourt) return;
            const network = this.currentPlayer.recipients;
            const zoneTotals = {};
            let total = 0;
            (network.recipients || []).forEach((r) => {
                Object.entries(r.zone_breakdown || {}).forEach(([zone, count]) => {
                    zoneTotals[zone] = (zoneTotals[zone] || 0) + count;
                    total += count;
                });
            });

            drawCourt(this.el.dpCourt, zoneTotals, total);

            if (this.el.dpCourtLegend) {
                this.el.dpCourtLegend.innerHTML = ZONE_ORDER.map((zone) => {
                    const c = zoneTotals[zone] || 0;
                    const share = total ? ((c / total) * 100).toFixed(1) : "0.0";
                    return `<div class="dp-legend-item">
                        <span class="dp-legend-swatch" style="background: var(${ZONE_VAR[zone]});"></span>
                        <span>${esc(ZONE_LABELS[zone])}</span>
                        <strong>${c} &middot; ${share}%</strong>
                    </div>`;
                }).join("");
            }
        }

        renderSummary() {
            if (!this.el.dpSummary) return;
            const s = this.currentPlayer.research_summary || {};
            const chips = (s.archetype || []).map(
                (a) => `<span class="dp-archetype">${esc(a.replaceAll("_", " "))}</span>`
            ).join("");
            const conf = Number.isFinite(s.confidence) ? `${(s.confidence * 100).toFixed(0)}%` : "n/a";
            const receivers = (s.best_recipients || []).map(
                (r) => `${esc(r.label)} <strong>(${num((r.assist_share || 0) * 100, 1)}%)</strong>`
            ).join(", ");
            const caveats = (s.caveats || []).map((c) => `<p class="dp-caveat">${esc(c)}</p>`).join("");
            this.el.dpSummary.innerHTML = `
                <div>${chips || ""}<span class="dp-badge dp-badge--derived">Confidence ${conf}</span></div>
                <p>${esc(s.simulation_finding || "")}</p>
                <p><strong>Primary gravity:</strong> ${esc((s.primary_gravity || []).join(", ") || "n/a")}</p>
                <p><strong>Highest-leverage recipients (sampled):</strong> ${receivers || "n/a"}</p>
                <p><strong>Role constraint:</strong> ${esc(s.role_constraint || "n/a")}</p>
                ${caveats}
            `;
        }

        // ---------------- simulator ----------------

        resetSimToBaseline() {
            if (!this.el.dpUsage) return;
            const b = this.currentPlayer.baseline;
            const usg = metricValue(b.usage_pct) || 15;
            this.el.dpUsage.min = Math.max(1, usg * 0.5).toFixed(1);
            this.el.dpUsage.max = Math.min(40, usg * 3).toFixed(1);
            this.el.dpUsage.value = usg;
            if (this.el.dpPass) this.el.dpPass.value = 0;
            if (this.el.dpElast) this.el.dpElast.value = 0.6;
            if (this.el.dpRet) this.el.dpRet.value = 100;
            if (this.el.dpTov) this.el.dpTov.value = 0;
            if (this.el.dpSat) this.el.dpSat.value = 55;
            if (this.el.dpScenario) this.el.dpScenario.value = "NEUTRAL";
            this.renderSimulation();
        }

        saturationRetention(h, k) {
            return Math.exp(-k * Math.max(0, h - 1));
        }

        simulateLive(baseline, params) {
            const currentUsage = baseline.usage_pct || 1;
            const h = currentUsage > 0
                ? 1 + params.elasticity * (params.targetUsage / currentUsage - 1)
                : 1;
            const saturation = this.saturationRetention(h, params.saturationK);
            const retention = params.retentionOverride !== null
                ? params.retentionOverride : saturation;
            const turnoverGrowth = params.turnoverOverride !== null
                ? params.turnoverOverride : (saturation > 0 ? 1 / saturation - 1 : 0);

            const simDecisionTouches = baseline.decisionTouches * h;
            const simPasses = simDecisionTouches * (1 + params.passTendencyChange);
            const simAssists = simPasses * baseline.astPerTouch * retention;
            const simMakes = simPasses * baseline.makesPerTouch * retention;
            const simTurnovers = simPasses * baseline.tovPerTouch * (1 + turnoverGrowth);

            return { h, saturation, retention, turnoverGrowth,
                     simDecisionTouches, simPasses, simAssists, simMakes, simTurnovers };
        }

        bvsRow(label, value, baselineValue, digits) {
            let delta = "";
            if (baselineValue !== null && baselineValue !== undefined && Number.isFinite(value)) {
                const d = value - baselineValue;
                const cls = d > 0 ? "dp-delta-up" : (d < 0 ? "dp-delta-down" : "");
                delta = ` <span class="${cls}">(${d >= 0 ? "+" : ""}${d.toFixed(digits)})</span>`;
            }
            return `<div class="dp-bvs-row"><span>${esc(label)}</span><span class="dp-num">${num(value, digits)}${delta}</span></div>`;
        }

        renderSimulation() {
            if (!this.el.dpUsage) return;
            const d = this.currentPlayer;
            if (!d) return;
            const b = d.baseline;

            const targetUsage = parseFloat(this.el.dpUsage.value);
            const passTendencyChange = parseFloat(this.el.dpPass.value) / 100;
            const elasticity = parseFloat(this.el.dpElast.value);
            const retentionSlider = parseFloat(this.el.dpRet.value) / 100;
            const turnoverSlider = parseFloat(this.el.dpTov.value) / 100;
            const saturationK = parseFloat(this.el.dpSat.value) / 100;

            if (this.el.dpUsageOut) this.el.dpUsageOut.textContent = `${targetUsage.toFixed(1)}%`;
            if (this.el.dpPassOut) this.el.dpPassOut.textContent =
                `${passTendencyChange >= 0 ? "+" : ""}${(passTendencyChange * 100).toFixed(0)}%`;
            if (this.el.dpElastOut) this.el.dpElastOut.textContent = elasticity.toFixed(2);
            if (this.el.dpRetOut) this.el.dpRetOut.textContent = `${(retentionSlider * 100).toFixed(0)}%`;
            if (this.el.dpTovOut) this.el.dpTovOut.textContent =
                `${turnoverSlider >= 0 ? "+" : ""}${(turnoverSlider * 100).toFixed(0)}%`;
            if (this.el.dpSatOut) this.el.dpSatOut.textContent = saturationK.toFixed(2);

            const baseline = {
                usage_pct: metricValue(b.usage_pct),
                decisionTouches: metricValue(b.decision_touches_per_game),
                astPerTouch: metricValue(b.ast_per_decision_touch),
                tovPerTouch: metricValue(b.tov_per_decision_touch),
                makesPerTouch: metricValue(b.makes_per_decision_touch),
            };

            const advancedOpen = this.el.dpAdvancedBtn &&
                this.el.dpAdvancedBtn.getAttribute("aria-expanded") === "true";
            const params = {
                targetUsage, elasticity, passTendencyChange, saturationK,
                retentionOverride: advancedOpen ? retentionSlider : null,
                turnoverOverride: advancedOpen ? turnoverSlider : null,
            };
            const sim = this.simulateLive(baseline, params);

            if (this.el.dpCausality) {
                this.el.dpCausality.innerHTML =
                    `Target usage <strong>${targetUsage.toFixed(1)}%</strong> vs. current
                    <strong>${num(baseline.usage_pct, 1)}%</strong>. With decision-touch elasticity
                    <strong>${elasticity.toFixed(2)}</strong>, ${(elasticity * 100).toFixed(0)}% of the
                    proportional role change is assumed to become additional decision touches
                    (H=${sim.h.toFixed(2)}). Pass tendency is changed by
                    <strong>${passTendencyChange >= 0 ? "+" : ""}${(passTendencyChange * 100).toFixed(0)}%</strong>.
                    Role saturation retains <strong>${(sim.retention * 100).toFixed(0)}%</strong> of baseline
                    efficiency, and turnover risk changes by
                    <strong>${sim.turnoverGrowth >= 0 ? "+" : ""}${(sim.turnoverGrowth * 100).toFixed(0)}%</strong>.
                    This is a <em>simulated</em>, conditional projection under these explicit assumptions --
                    never a forecast.`;
            }

            if (this.el.dpBvsBaseline) {
                this.el.dpBvsBaseline.innerHTML = [
                    ["Decision touches/G", baseline.decisionTouches, null, 1],
                    ["Assists/G", metricValue(b.ast_per_game), null, 2],
                    ["Turnovers/G", metricValue(b.tov_per_game), null, 2],
                ].map(([l, v, bv, dg]) => this.bvsRow(l, v, bv, dg)).join("");
            }
            if (this.el.dpBvsSim) {
                this.el.dpBvsSim.innerHTML = [
                    ["Decision touches/G", sim.simDecisionTouches, metricValue(b.decision_touches_per_game), 1],
                    ["Assists/G", sim.simAssists, metricValue(b.ast_per_game), 2],
                    ["Receiver makes/G", sim.simMakes, null, 2],
                    ["Turnovers/G", sim.simTurnovers, metricValue(b.tov_per_game), 2],
                ].map(([l, v, bv, dg]) => this.bvsRow(l, v, bv, dg)).join("");
            }

            this.renderMonteCarlo();
        }

        renderMonteCarlo() {
            if (!this.el.dpMonteCarlo) return;
            const mc = this.currentPlayer.simulation_parameters &&
                       this.currentPlayer.simulation_parameters.monte_carlo;
            if (!mc || !mc.assists) {
                this.el.dpMonteCarlo.innerHTML =
                    `<p class="dp-caveat">${badge("UNAVAILABLE")} No Monte Carlo result available for this player.</p>`;
                return;
            }
            const cards = [
                ["Assists/G", mc.assists],
                ["Turnovers/G", mc.turnovers],
                ["Receiver makes/G", mc.receiver_makes],
            ];
            const cardsHtml = cards.map(([label, stats]) => `
                <div class="dp-mc-card">
                    <span class="dp-mc-card__label">${esc(label)}</span>
                    <span class="dp-mc-card__value">${num(stats.median, 2)}</span>
                    <span class="dp-mc-card__range">P10 ${num(stats.p10, 2)} &ndash; P90 ${num(stats.p90, 2)}</span>
                    <div class="dp-mc-card__bar"><div class="dp-mc-card__bar-fill" style="width: 100%"></div></div>
                </div>`).join("");
            this.el.dpMonteCarlo.innerHTML = cardsHtml
                + `<p class="dp-caveat" style="grid-column: 1 / -1">${mc.n_draws} draws, seed ${mc.seed} (reproducible).</p>`;
        }

        // ---------------- compare ----------------

        addCompareSlot() {
            if (!this.el.dpCompareChips) return;
            // Only allow adding when we have players remaining
            const notPicked = this.players.find((p) => !this.compareSlugs.includes(p.slug));
            if (!notPicked) return;
            this.compareSlugs.push(notPicked.slug);
            this.renderCompare();
        }

        removeCompareSlug(slug) {
            this.compareSlugs = this.compareSlugs.filter((s) => s !== slug);
            this.renderCompare();
        }

        async renderCompare() {
            if (!this.el.dpCompareChips || !this.el.dpCompareTableWrap) return;
            const bySlug = Object.fromEntries(this.players.map((p) => [p.slug, p]));
            const chipsHtml = this.compareSlugs.map((slug) => {
                const p = bySlug[slug];
                return `<span class="dp-compare-chip">
                    ${esc(p ? p.name : slug)}
                    <button type="button" aria-label="Remove ${esc(p ? p.name : slug)}"
                            data-remove-slug="${esc(slug)}">×</button>
                </span>`;
            }).join("");
            const remaining = this.players.filter((p) => !this.compareSlugs.includes(p.slug));
            const addSelect = remaining.length
                ? `<label class="dp-compare-chip" style="padding: 0;">
                    <select data-compare-select class="dp-select"
                            style="border: 0; background: transparent; min-height: 34px;
                                   padding: 0 26px 0 12px;">
                        <option value="">+ add player</option>
                        ${remaining.map((p) => `<option value="${esc(p.slug)}">${esc(p.name)}</option>`).join("")}
                    </select>
                </label>`
                : `<span class="dp-compare-add" aria-disabled="true">All players added</span>`;
            this.el.dpCompareChips.innerHTML = chipsHtml + addSelect;

            if (!this.compareSlugs.length) {
                this.el.dpCompareTableWrap.innerHTML =
                    `<div class="dp-compare-empty">Add players above to build a side-by-side comparison.</div>`;
                return;
            }
            const loaded = await Promise.all(
                this.compareSlugs.map((slug) => this.loadPlayer(slug).catch(() => null))
            );
            const rows = [
                ["USG %",              (p) => num(metricValue(p.baseline.usage_pct), 1)],
                ["Decision touches/G", (p) => num(metricValue(p.baseline.decision_touches_per_game), 1)],
                ["AST/G",              (p) => num(metricValue(p.baseline.ast_per_game), 2)],
                ["TOV/G",              (p) => num(metricValue(p.baseline.tov_per_game), 2)],
                ["AST/decision touch", (p) => num(metricValue(p.baseline.ast_per_decision_touch), 3)],
                ["TOV/decision touch", (p) => num(metricValue(p.baseline.tov_per_decision_touch), 3)],
                ["Sampled assists",    (p) => p.recipients.sample_size],
                ["Archetype",          (p) => (p.research_summary.archetype || []).map((a) => a.replaceAll("_", " ")).join(", ")],
                ["Primary gravity",    (p) => (p.research_summary.primary_gravity || []).join(", ")],
                ["Role constraint",    (p) => p.research_summary.role_constraint],
            ];
            const header = `<thead><tr><th>Metric</th>${
                loaded.map((p) => `<th>${p ? esc(p.player.name) : "n/a"}</th>`).join("")
            }</tr></thead>`;
            const body = `<tbody>${
                rows.map(([label, fn]) => `<tr>
                    <td>${esc(label)}</td>
                    ${loaded.map((p) => `<td>${p ? esc(String(fn(p))) : "n/a"}</td>`).join("")}
                </tr>`).join("")
            }</tbody>`;
            this.el.dpCompareTableWrap.innerHTML =
                `<div class="dp-compare-scroll"><table class="dp-compare-table">${header}${body}</table></div>`;
        }

        renderProvenance() {
            if (!this.el.dpProvenance) return;
            const p = this.currentPlayer.provenance || {};
            const sampled = p.bball_ref_games_sampled ? p.bball_ref_games_sampled.length : 0;
            this.el.dpProvenance.innerHTML = `
                <p><strong>Season:</strong> ${esc(p.season)} &middot; <strong>Generated:</strong> ${esc(formatTime(p.generated_at_utc))}</p>
                <p><strong>Box scores:</strong> <code>${esc(p.box_score_source || "n/a")}</code></p>
                <p><strong>Basketball-Reference sample:</strong> ${sampled} of ${p.bball_ref_games_available_total || 0} real games &mdash; ${esc(p.bball_ref_sampling_method || "")}</p>
                <p><strong>stats.nba.com reachable:</strong> ${p.stats_nba_com_reachable ? "yes" : "no"} &mdash; ${esc(p.stats_nba_com_note || "")}</p>
            `;
        }
    }

    document.addEventListener("DOMContentLoaded", () => new DrivePassPage());
})();
