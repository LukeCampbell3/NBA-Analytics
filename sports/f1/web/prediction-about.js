document.addEventListener("DOMContentLoaded", async () => {
    window.CardVaultShell?.mount({
        brandTitle: "Prediction Bounties", brandHref: "/", sportSlug: "f1", sportAccent: "#d00000",
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
    } catch (error) {
        facts.textContent = `Current model metadata unavailable: ${error.message}`;
    }
});
