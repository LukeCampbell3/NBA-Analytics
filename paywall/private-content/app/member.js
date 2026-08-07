(function memberApp() {
    const title = document.getElementById("memberTitle");
    const status = document.getElementById("memberStatus");
    const desks = document.getElementById("memberDesks");
    const grid = document.getElementById("sportsGrid");
    const checkout = document.getElementById("checkoutButton");
    const billing = document.getElementById("billingButton");
    const logout = document.getElementById("logoutButton");
    const gatewayBase = window.PaywallConfig.gatewayBase;

    function csrfToken() {
        const prefix = "__Host-csrf=";
        const value = document.cookie.split("; ").find((item) => item.startsWith(prefix));
        return value ? decodeURIComponent(value.slice(prefix.length)) : "";
    }

    async function api(path, method = "GET") {
        const headers = { "Accept": "application/json" };
        if (method !== "GET") {
            headers["Content-Type"] = "application/json";
            headers["X-CSRF-Token"] = csrfToken();
        }
        const response = await fetch(gatewayBase + path, { method, headers, credentials: "same-origin", cache: "no-store", body: method === "GET" ? undefined : "{}" });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
        return body;
    }

    function sportCard(sport) {
        const cv = window.CardVault;
        return `<article class="desk-board-card" style="--sport-accent:${cv.escapeAttr(sport.accent)}"><p class="desk-board-card__eyebrow">${cv.escapeHtml(sport.slug)} desk</p><h3>${cv.escapeHtml(sport.title)}</h3><p>${cv.escapeHtml(sport.summary)}</p><div class="desk-board-card__actions"><a class="desk-board-card__primary" href="${cv.escapeAttr(gatewayBase + sport.entry_href)}">Open board</a></div></article>`;
    }

    async function loadDesks() {
        const response = await fetch("/data/sports.json", { credentials: "same-origin", cache: "no-store" });
        if (!response.ok) throw new Error("Unable to load member desks");
        const sports = await response.json();
        grid.innerHTML = sports.map(sportCard).join("");
        desks.hidden = false;
    }

    checkout.addEventListener("click", async () => {
        checkout.disabled = true;
        try { window.location.assign((await api("/api/checkout", "POST")).url); }
        catch (_) { status.textContent = "Checkout is temporarily unavailable. Please try again."; checkout.disabled = false; }
    });
    billing.addEventListener("click", async () => {
        billing.disabled = true;
        try { window.location.assign((await api("/api/billing-portal", "POST")).url); }
        catch (_) { status.textContent = "Billing controls are temporarily unavailable."; billing.disabled = false; }
    });
    logout.addEventListener("click", async () => {
        try { await fetch(gatewayBase + "/auth/logout", { method: "POST", credentials: "same-origin", redirect: "follow", headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken() }, body: "{}" }); }
        finally { window.location.assign("/"); }
    });

    api("/api/account/status").then(async (account) => {
        title.textContent = `Welcome, ${account.display_name || "member"}`;
        if (account.has_access) {
            status.textContent = `Your ${account.plan} membership is active.`;
            billing.hidden = false;
            await loadDesks();
        } else {
            status.textContent = "Your account is signed in but does not currently have content access.";
            checkout.hidden = false;
            if (account.status !== "pending") billing.hidden = false;
        }
    }).catch(() => window.location.replace("/login/"));
})();
