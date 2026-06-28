function mountShell() {
  if (!window.CardVaultShell) return;
  window.CardVaultShell.mount({
    brandTitle: "NBA Analytics",
    brandHref: "/",
    workspaceLabel: "NBA",
    workspaceHref: "/nba/",
    sportSlug: "nba",
    sportAccent: "#f59e0b",
    breadcrumbs: [{ label: "Account", href: "account.html" }],
    navLinks: [
      { label: "Dashboard", href: "index.html" },
      { label: "Board", href: "predictions.html" },
      { label: "Safe-State", href: "safe-state.html" },
      { label: "Pricing", href: "pricing.html" },
      { label: "Account", href: "account.html", active: true },
      { label: "API", href: "api.html" },
    ],
    showDisclaimer: false,
  });
}

function renderSummary(me, entitlements) {
  const el = document.getElementById("accountSummary");
  if (!me) {
    el.innerHTML = `<p class="vault-page-lead">Sign in to view your plan and entitlements.</p>`;
    document.getElementById("apiKeySection").hidden = true;
    return;
  }
  const plan = entitlements?.capabilities?.plan_id || me.entitlements?.plan_id || "free";
  el.innerHTML = `
    <p><strong>Email:</strong> ${me.user?.email || "unknown"}</p>
    <p><strong>Plan:</strong> ${plan}</p>
    <p><strong>Subscription status:</strong> ${me.subscription?.status || "none"}</p>
    <p><strong>Capabilities:</strong></p>
    <ul class="hero-points">
      <li>Full safe-state board: ${entitlements?.capabilities?.can_view_full_safe_state ? "yes" : "locked"}</li>
      <li>Candidate pool: ${entitlements?.capabilities?.can_view_candidate_pool ? "yes" : "locked"}</li>
      <li>CSV export: ${entitlements?.capabilities?.can_export_csv ? "yes" : "locked"}</li>
      <li>API access: ${entitlements?.capabilities?.can_use_api ? "yes" : "locked"}</li>
      <li>API usage today: ${entitlements?.usage_today ?? 0} / ${entitlements?.api_limit ?? 0}</li>
    </ul>
    <button type="button" id="manageBillingBtn" class="vault-door-cta">Manage billing</button>
  `;
  document.getElementById("manageBillingBtn")?.addEventListener("click", async () => {
    try {
      const portal = await window.NbaAuthClient.openBillingPortal();
      window.location.href = portal.portal_url;
    } catch (error) {
      alert(error.message);
    }
  });
  document.getElementById("apiKeySection").hidden = !entitlements?.capabilities?.can_use_api;
}

async function refreshAccount() {
  const me = await window.NbaAuthClient.fetchMe();
  const entitlements = await window.NbaAuthClient.fetchEntitlements();
  renderSummary(me, entitlements);
  if (entitlements?.capabilities?.can_use_api) {
    await renderApiKeys();
  }
}

async function renderApiKeys() {
  const list = document.getElementById("apiKeyList");
  const payload = await window.NbaAuthClient.listApiKeys();
  list.innerHTML = (payload.keys || []).map((key) => `
    <div class="vault-metric-chip" style="margin:8px 0;display:flex;gap:8px;align-items:center;">
      <span>${key.name || "Key"} · ${key.key_prefix}… · ${key.status}</span>
      <button type="button" data-id="${key.id}" class="revoke-key-btn vault-sport-pill">Revoke</button>
    </div>
  `).join("") || `<p class="vault-page-lead">No API keys yet.</p>`;
  list.querySelectorAll(".revoke-key-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await window.NbaAuthClient.revokeApiKey(btn.getAttribute("data-id"));
      await renderApiKeys();
    });
  });
}

async function init() {
  mountShell();
  document.getElementById("accountLoginBtn").addEventListener("click", async () => {
    const email = document.getElementById("accountEmail").value.trim();
    if (!email) return alert("Enter an email");
    await window.NbaAuthClient.devLogin(email);
    await refreshAccount();
  });
  document.getElementById("accountLogoutBtn").addEventListener("click", async () => {
    window.NbaAuthClient.setToken("");
    await refreshAccount();
  });
  document.getElementById("createApiKeyBtn")?.addEventListener("click", async () => {
    const name = document.getElementById("apiKeyName").value.trim() || "Default";
    const created = await window.NbaAuthClient.createApiKey(name);
    const once = document.getElementById("apiKeyOnce");
    once.hidden = false;
    once.textContent = `Copy now — shown once only:\n${created.api_key}`;
    await renderApiKeys();
  });
  await refreshAccount();
}

document.addEventListener("DOMContentLoaded", init);
