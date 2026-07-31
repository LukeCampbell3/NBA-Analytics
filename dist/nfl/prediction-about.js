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
        const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
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
    } catch (error) {
        document.getElementById("methodFacts").textContent = `Methodology unavailable: ${error.message}`;
    }
});
