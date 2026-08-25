document.addEventListener("DOMContentLoaded", async () => {
    window.CardVaultShell?.mount({
        brandTitle: "In The Cards Analytics", brandHref: "/", sportSlug: "f1", sportAccent: "#d00000",
        navLinks: [
            { label: "Race Board", href: "/f1/predictions/", active: false },
            { label: "Method", href: "/f1/prediction-about/", active: true },
        ],
        showDisclaimer: true,
    });
    const facts = document.getElementById("aboutRunFacts");
    try {
        const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const model = data.model || {};
        const event = data.event || {};
        facts.innerHTML = [
            `Model ${model.name || "n/a"}`, `Trained through ${model.trained_through || "n/a"}`,
            `${model.training_races || 0} races`, event.race_name ? `Next: ${event.race_name}` : "No upcoming race",
        ].map((value) => `<span>${String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</span>`).join("");

        // Backtest and prospective-tracker evidence -- moved here from
        // predictions.html (spec section 17/20: deep validation belongs
        // on the methodology page, not the current-signals page).
        const esc = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
        const pct = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "n/a";
        const num = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "n/a";
        const cards = (items) => items.map(([label, value]) => `<article class="prediction-about-metric-card"><span>${esc(label)}</span><strong>${esc(value ?? "n/a")}</strong></article>`).join("");
        const backtest = model.backtest || {};
        const backtestEl = document.getElementById("backtest");
        if (backtestEl) {
            backtestEl.innerHTML = cards([
                ["Holdout races", backtest.holdout_races],
                ["Top-pick winners", pct(backtest.winner_top_pick_accuracy)],
                ["Winner Brier", num(backtest.winner_brier)],
                ["Winner log loss", num(backtest.winner_log_loss)],
                ["Podium Brier", num(backtest.podium_brier)],
                ["Top-six Brier", num(backtest.top6_brier)],
            ]);
        }
        const prospective = data.prospective_evaluation || {};
        const prospectiveEl = document.getElementById("prospective");
        if (prospectiveEl) {
            prospectiveEl.innerHTML = cards([
                ["Settled snapshots", prospective.settled_snapshots],
                ["Distinct races", prospective.settled_races],
                ["Top-pick winners", pct(prospective.top_pick_accuracy)],
                ["Winner Brier", num(prospective.winner_brier)],
                ["Winner log loss", num(prospective.winner_log_loss)],
                ["Shadow play hit rate", pct(prospective.play_hit_rate)],
            ]);
        }
    } catch (error) {
        facts.textContent = `Current model metadata unavailable: ${error.message}`;
    }
});
