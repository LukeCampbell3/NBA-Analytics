const GOLF_MARKET_LABELS = { WINNER: "Winner", TOP_5: "Top 5", TOP_10: "Top 10", TOP_20: "Top 20", MAKE_CUT: "Make Cut" };

class GolfPredictionBoard {
    constructor() {
        this.data = null;
        this.el = {
            eventTitle: document.getElementById("eventTitle"),
            eventCountdown: document.getElementById("eventCountdown"),
            eventMeta: document.getElementById("eventMeta"),
            runFacts: document.getElementById("runFacts"),
            status: document.getElementById("status"),
            metrics: document.getElementById("metrics"),
            top10Table: document.getElementById("top10Table"),
            top10Summary: document.getElementById("top10Summary"),
            plays: document.getElementById("plays"),
            picksSummary: document.getElementById("picksSummary"),
        };
        this.init();
    }

    async init() {
        this.mountShell();
        try {
            const response = await fetch(`data/daily_predictions.json?v=${Date.now()}`, { cache: "no-store" });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.data = await response.json();
            this.render();
        } catch (error) {
            this.el.runFacts.textContent = `Unable to load the PGA board: ${error.message}`;
            this.el.status.innerHTML = "<p>The current board is unavailable.</p>";
        }
    }

    mountShell() {
        window.CardVaultShell?.mount({
            brandTitle: "In The Cards Analytics", brandHref: "/", sportSlug: "golf", sportAccent: "#1a6b3c",
            navLinks: [
                { label: "Predictions", href: "/golf/predictions/", active: true },
            ],
            showDisclaimer: true,
        });
    }

    render() {
        const status = this.data.status || "unknown";
        const top10 = Array.isArray(this.data.top_10) ? this.data.top_10 : [];
        const candidates = Array.isArray(this.data.candidates) ? this.data.candidates : [];

        this.el.eventTitle.textContent = this.data.event_name || "No Upcoming PGA Tour Event";
        this.renderEvent(status);
        this.el.runFacts.innerHTML = [
            `Board generated ${this.formatDateTime(this.data.generated_at_utc)}`,
            `Field: ${this.data.field_size ?? 0} players`,
            `Odds: ${this.formatOddsStatus(this.data.odds_status)}`,
        ].map((item) => `<span>${this.escape(item)}</span>`).join("");

        this.el.status.innerHTML = `<p><strong>${this.escape(this.statusHeadline(status))}</strong> ${this.escape(this.statusDetail(status))}</p>`;
        this.el.metrics.innerHTML = this.cards([
            ["Field size", this.data.field_size],
            ["Real market candidates", candidates.length],
            ["Authorized for staking", this.data.candidate_authorized_count ?? 0],
            ["Odds status", this.formatOddsStatus(this.data.odds_status)],
        ]);

        this.renderTop10(top10);
        this.renderPlays(candidates);
    }

    renderEvent(status) {
        const start = this.data.event_start_utc;
        if (!this.data.event_id || !start) {
            this.el.eventCountdown.textContent = "Schedule unavailable";
            this.el.eventMeta.innerHTML = "";
            return;
        }
        const startDate = new Date(start);
        const days = Math.ceil((startDate.getTime() - Date.now()) / 86400000);
        this.el.eventCountdown.textContent = days > 1 ? `Starts in ${days} days` : days === 1 ? "Starts tomorrow" : days === 0 ? "Tournament underway" : "Tournament completed";
        const timeLabel = Number.isNaN(startDate.getTime())
            ? start
            : new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" }).format(startDate);
        this.el.eventMeta.innerHTML = [
            ["Tournament", this.data.event_name || "TBD"],
            ["Starts", timeLabel],
            ["Field status", this.formatFieldStatus(status)],
            ["Field size", `${this.data.field_size ?? 0} players`],
        ].map(([label, value]) => `<div><span>${this.escape(label)}</span><strong>${this.escape(value)}</strong></div>`).join("");
    }

    renderTop10(rows) {
        if (!rows.length) {
            this.el.top10Summary.textContent = "No projected field available yet";
            this.el.top10Table.innerHTML = '<div class="prediction-about-empty">No real field is posted for the next event yet.</div>';
            return;
        }
        this.el.top10Summary.textContent = `Ranked by projected tournament total score`;
        const body = rows.map((row, index) => `<tr>
            <td><span class="golf-rank">${index + 1}</span><strong>${this.escape(row.player_name)}</strong></td>
            <td>${this.num(row.projected_round_score)}</td>
            <td>${this.signedScore(row.projected_total_score)}</td>
            <td>${this.pct(row.win_probability)}</td>
            <td>${this.pct(row.top10_probability)}</td>
            <td>${row.form_rounds_observed ?? 0}</td>
        </tr>`).join("");
        this.el.top10Table.innerHTML = `<table class="prediction-about-table"><thead><tr><th>Player</th><th>Proj. round</th><th>Proj. total</th><th>Win %</th><th>Top-10 %</th><th>Rounds of form</th></tr></thead><tbody>${body}</tbody></table>`;
    }

    renderPlays(candidates) {
        const priced = candidates.filter((c) => c.selected_side_price !== null && c.selected_side_price !== undefined);
        if (!priced.length) {
            this.el.picksSummary.textContent = "No real market price currently clears this board's gates";
            this.el.plays.innerHTML = '<div class="prediction-about-empty">No real, priced candidate is available right now -- golf market coverage is thin outside outright-winner markets, and a real cause is always shown here rather than a guessed pick.</div>';
            return;
        }
        const ranked = [...priced].sort((a, b) => Number(b.expected_value_per_unit ?? -1) - Number(a.expected_value_per_unit ?? -1));
        const authorizedCount = ranked.filter((c) => c.candidate_authorized).length;
        this.el.picksSummary.textContent = `${ranked.length} real, priced candidate${ranked.length === 1 ? "" : "s"} · ${authorizedCount} authorized for staking`;
        this.el.plays.innerHTML = ranked.map((play, index) => {
            const modelWidth = Math.min(100, Number(play.model_probability) * 100);
            const marketWidth = play.no_vig_market_probability == null ? 0 : Math.min(100, Number(play.no_vig_market_probability) * 100);
            const badge = play.candidate_authorized
                ? '<span class="golf-authorized-badge">Authorized</span>'
                : '<span class="golf-shadow-badge">Shadow only</span>';
            return `<article class="golf-pick">
                <div class="golf-pick-topline"><span class="golf-pick-number">${index + 1}</span>${badge}</div>
                <h3>${this.escape(play.player_name)}</h3>
                <p class="golf-pick-market">${this.escape(GOLF_MARKET_LABELS[play.market] || play.market)}</p>
                <div class="golf-probability-row"><span>Model probability</span><strong>${this.pct(play.model_probability)}</strong></div>
                <div class="golf-probability-track"><span style="width:${modelWidth}%"></span></div>
                <div class="golf-probability-row"><span>No-vig market</span><strong>${this.pct(play.no_vig_market_probability)}</strong></div>
                <div class="golf-probability-track is-market"><span style="width:${marketWidth}%"></span></div>
                <div class="golf-price-row"><span>Best price</span><strong>${this.formatPrice(play.selected_side_price)}</strong><small>${this.escape(play.selected_sportsbook_key || "—")} · EV ${this.signedPct(play.expected_value_per_unit)}</small></div>
                ${play.support_blocking_dimensions && play.support_blocking_dimensions.length ? `<p class="golf-blocking-note">Awaiting real evidence: ${this.escape(play.support_blocking_dimensions.join(", "))}</p>` : ""}
            </article>`;
        }).join("");
    }

    statusHeadline(status) {
        if (status === "ok") return "Board ready.";
        if (status === "field_not_posted") return "Field not posted yet.";
        if (status === "no_event_in_calendar") return "No upcoming event found.";
        return "Board status unknown.";
    }
    statusDetail(status) {
        if (status === "ok") return "Real field, real recent-form projections, and real market candidates are shown below.";
        if (status === "field_not_posted") return "ESPN has not posted the real tournament field yet -- this typically appears 1-2 days before the first tee time.";
        if (status === "no_event_in_calendar") return "No real PGA Tour event is currently listed in the season calendar (a real, expected off-season state).";
        return "See board readiness for detail.";
    }
    formatFieldStatus(status) { return status === "ok" ? "Posted" : status === "field_not_posted" ? "Not yet posted" : "Unknown"; }
    formatOddsStatus(status) {
        if (status === "success") return "Live";
        if (status === "no_active_golf_market") return "No real market active";
        if (status === "missing_credentials") return "Unavailable";
        if (status === "no_props") return "No priced markets";
        return status || "Unknown";
    }

    cards(items) { return items.map(([label, value]) => `<article class="prediction-about-metric-card"><span>${this.escape(label)}</span><strong>${this.escape(value ?? "n/a")}</strong></article>`).join(""); }
    pct(value) { return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—"; }
    signedPct(value) { return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(1)}%` : "—"; }
    num(value) { return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "—"; }
    signedScore(value) { return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(1) : "—"; }
    formatPrice(value) { return value !== null && value !== undefined && Number.isFinite(Number(value)) ? `${Number(value) > 0 ? "+" : ""}${Math.round(Number(value))}` : "—"; }
    formatDateTime(value) {
        const parsed = new Date(value);
        return Number.isNaN(parsed.getTime()) ? value || "n/a" : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(parsed);
    }
    escape(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;"); }
}
document.addEventListener("DOMContentLoaded", () => new GolfPredictionBoard());
