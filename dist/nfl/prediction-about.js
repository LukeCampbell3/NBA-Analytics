document.addEventListener("DOMContentLoaded", async () => {
    if (window.CardVaultShell) {
        window.CardVaultShell.mount({
            brandTitle: "In The Cards Analytics", brandHref: "/", sportSlug: "nfl", sportAccent: "#b42318",
            navLinks: [
                { label: "Projections", href: "/nfl/projections/", active: false },
                { label: "Picks", href: "/nfl/picks/", active: false },
                { label: "Fantasy", href: "/nfl/fantasy/", active: false },
                { label: "Method", href: "/nfl/prediction-about/", active: true },
            ],
            showDisclaimer: true,
        });
    }
    const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    const formatInt = (value) => Number.isFinite(Number(value)) ? String(Math.round(Number(value))) : "n/a";
    const formatPct = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a";
    const formatSignedPct = (value) => Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "n/a";
    const formatSignedNum = (value, places = 2) => Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(places)}` : "n/a";
    const formatRange = (values, signed = false) => {
        if (!Array.isArray(values) || values.length !== 2) return "n/a";
        const formatter = signed ? formatSignedPct : formatPct;
        return `${formatter(values[0])}-${formatter(values[1])}`;
    };

    /**
     * Locked singles backtest, baseline comparison, and weekly replay --
     * moved here from predictions.html (spec section 17: prediction pages
     * show current signals, deep validation evidence belongs on the
     * methodology page). predictions.html keeps only the concise
     * Locked Record/Hit Rate/ROI summary cards.
     */
    function renderMarketReplay(evidence) {
        const statusEl = document.getElementById("marketReplayStatus");
        const metricsEl = document.getElementById("marketReplayMetrics");
        const baselinesEl = document.getElementById("marketBaselines");
        const weeklyEl = document.getElementById("marketWeekly");
        if (!statusEl || !metricsEl || !baselinesEl || !weeklyEl) return;
        if (!evidence) {
            statusEl.innerHTML = "<p>Locked market replay evidence is unavailable.</p>";
            metricsEl.innerHTML = "";
            baselinesEl.innerHTML = "";
            weeklyEl.innerHTML = "";
            return;
        }
        const final = evidence.final_test || {};
        const policy = evidence.locked_policy || {};
        const deployment = evidence.gates?.deployment || {};
        const stats = evidence.statistical_evidence || {};
        statusEl.innerHTML = `<p><strong>Singles passed the historical holdout; live authorization remains ${escape(deployment.status || "blocked")}.</strong> ${escape(deployment.reason || "Prospective evidence is required.")}</p>`;
        const cards = [
            ["Validated Market", (evidence.validated_targets || []).join(", ") || "n/a"],
            ["Weekly Cap", formatInt(policy.weekly_top_n)],
            ["Record", `${formatInt(final.wins)}-${formatInt(final.losses)}`],
            ["Hit Rate", formatPct(final.hit_rate)],
            ["ROI", formatSignedPct(final.roi)],
            ["Profit", `${formatSignedNum(final.profit_units, 2)}u`],
            ["Clustered Hit 95%", formatRange(stats.week_cluster_hit_rate_95, false)],
            ["Clustered ROI 95%", formatRange(stats.week_cluster_roi_95, true)],
        ];
        metricsEl.innerHTML = cards.map(([label, value]) => `
            <article class="prediction-about-metric-card"><span>${escape(label)}</span><strong>${escape(value)}</strong></article>
        `).join("");

        const baselines = evidence.baselines || {};
        const baselineRows = [
            ["Production selector", final],
            ["Always under", baselines.always_under || {}],
            ["Point projection side", baselines.point_projection_side || {}],
        ].map(([label, row]) => `<tr>
            <td>${escape(label)}</td><td>${escape(formatInt(row.graded_decisions))}</td>
            <td>${escape(`${formatInt(row.wins)}-${formatInt(row.losses)}`)}</td>
            <td>${escape(formatPct(row.hit_rate))}</td><td>${escape(formatSignedPct(row.roi))}</td>
        </tr>`).join("");
        baselinesEl.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Policy</th><th>N</th><th>Record</th><th>Hit rate</th><th>ROI</th></tr></thead>
            <tbody>${baselineRows}</tbody>
        </table>`;

        const weeklyRows = (evidence.weekly || []).map((row) => `<tr>
            <td>W${escape(formatInt(row.week))}</td><td>${escape(formatInt(row.picks))}</td>
            <td>${escape(`${formatInt(row.wins)}-${formatInt(row.losses)}`)}</td>
            <td>${escape(formatPct(row.hit_rate))}</td><td>${escape(formatSignedPct(row.roi))}</td>
            <td>${escape(`${formatSignedNum(row.profit_units, 2)}u`)}</td>
        </tr>`).join("");
        weeklyEl.innerHTML = `<table class="prediction-about-table">
            <thead><tr><th>Week</th><th>Picks</th><th>Record</th><th>Hit rate</th><th>ROI</th><th>Units</th></tr></thead>
            <tbody>${weeklyRows}</tbody>
        </table>`;
    }

    try {
        const [response, marketResponse] = await Promise.all([
            fetch(`data/daily_predictions.json?v=${Date.now()}`),
            fetch(`data/market_validation_summary.json?v=${Date.now()}`),
        ]);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const market = marketResponse.ok ? await marketResponse.json() : null;
        const method = data.methodology || {};
        const sourceSeasons = method.source_seasons || [];
        const sourceRange = sourceSeasons.length ? `${sourceSeasons[0]}–${sourceSeasons[sourceSeasons.length - 1]}` : "n/a";
        document.getElementById("methodFacts").innerHTML = `
            <span>Sources ${escape(sourceRange)}</span>
            <span>Meta folds ${escape((method.meta_seasons || []).join(", "))}</span>
            <span>Holdout ${escape(method.holdout_season || "n/a")}</span>`;
        document.getElementById("leakageControls").innerHTML = (method.leakage_controls || [])
            .map((item) => `<li>${escape(item)}</li>`).join("");
        document.getElementById("scopeNote").innerHTML = `<p><strong>Scope:</strong> ${escape(method.scope_note || "n/a")}</p>`;
        if (market) {
            const marketMethod = market.methodology || {};
            const final = market.final_test || {};
            const deployment = market.gates?.deployment || {};
            document.getElementById("marketMethodFacts").innerHTML = `<p><strong>Locked policy:</strong> ${escape(marketMethod.selected_architecture || "n/a")}, top ${escape(marketMethod.weekly_top_n || "n/a")} per week, ${escape(((marketMethod.minimum_side_probability || 0) * 100).toFixed(0))}% probability floor. Final replay: ${escape(final.wins || 0)}–${escape(final.losses || 0)} (${escape(((final.hit_rate || 0) * 100).toFixed(2))}%), ${escape(((final.roi || 0) * 100).toFixed(2))}% ROI.</p>`;
            document.getElementById("marketGateFacts").innerHTML = `<p><strong>Replay status:</strong> ${escape(market.status || "n/a")}. <strong>Deployment ${escape(deployment.status || "blocked")}:</strong> ${escape(deployment.reason || "Source provenance is unresolved.")}</p>`;
            document.getElementById("marketLimitations").innerHTML = (market.limitations || [])
                .map((item) => `<li>${escape(item)}</li>`).join("");
            renderMarketReplay(market);
        } else {
            document.getElementById("marketMethodFacts").textContent = "Market methodology payload unavailable.";
            document.getElementById("marketGateFacts").textContent = "Market replay payload unavailable.";
            renderMarketReplay(null);
        }
    } catch (error) {
        document.getElementById("methodFacts").textContent = `Methodology unavailable: ${error.message}`;
    }
});
