document.addEventListener("DOMContentLoaded", async () => {
    if (window.CardVaultShell) {
        window.CardVaultShell.mount({
            brandTitle: "Prediction Bounties", brandHref: "/", sportSlug: "nfl", sportAccent: "#7c3aed",
            navLinks: [
                { label: "Model Report", href: "/nfl/predictions/", active: false },
                { label: "Method", href: "/nfl/prediction-about/", active: true },
            ],
            showDisclaimer: true,
        });
    }
    const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
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
        } else {
            document.getElementById("marketMethodFacts").textContent = "Market methodology payload unavailable.";
            document.getElementById("marketGateFacts").textContent = "Market replay payload unavailable.";
        }
    } catch (error) {
        document.getElementById("methodFacts").textContent = `Methodology unavailable: ${error.message}`;
    }
});
