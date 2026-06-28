/**
 * Shared auth + billing client for NBA Analytics static frontend.
 * Never stores Stripe secrets — only session tokens in localStorage.
 */
(function initNbaAuthClient(global) {
  const STORAGE_KEY = "nba_analytics_access_token";

  function apiBase() {
    const meta = document.querySelector('meta[name="nba-api-base"]');
    return (meta && meta.content) || global.NBA_API_BASE || "";
  }

  function getToken() {
    return localStorage.getItem(STORAGE_KEY) || "";
  }

  function setToken(token) {
    if (token) localStorage.setItem(STORAGE_KEY, token);
    else localStorage.removeItem(STORAGE_KEY);
  }

  async function apiFetch(path, options = {}) {
    const headers = Object.assign({ "Content-Type": "application/json" }, options.headers || {});
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    const response = await fetch(`${apiBase()}${path}`, { ...options, headers });
    const text = await response.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (_e) { body = { raw: text }; }
    if (!response.ok) {
      const message = (body && (body.detail || body.message)) || `HTTP ${response.status}`;
      throw new Error(typeof message === "string" ? message : JSON.stringify(message));
    }
    return body;
  }

  async function devLogin(email) {
    const result = await apiFetch("/api/auth/dev-session", {
      method: "POST",
      body: JSON.stringify({ email }),
    });
    setToken(result.access_token);
    return result;
  }

  async function fetchMe() {
    if (!getToken()) return null;
    try {
      return await apiFetch("/api/me");
    } catch (_e) {
      setToken("");
      return null;
    }
  }

  async function fetchEntitlements() {
    return apiFetch("/api/entitlements");
  }

  async function startCheckout(planId) {
    const origin = window.location.origin;
    const path = window.location.pathname.replace(/[^/]+$/, "");
    return apiFetch("/api/billing/create-checkout-session", {
      method: "POST",
      body: JSON.stringify({
        plan_id: planId,
        success_url: `${origin}${path}account.html?checkout=success`,
        cancel_url: `${origin}${path}pricing.html?checkout=canceled`,
      }),
    });
  }

  async function openBillingPortal() {
    const origin = window.location.origin;
    const path = window.location.pathname.replace(/[^/]+$/, "");
    return apiFetch("/api/billing/create-portal-session", {
      method: "POST",
      body: JSON.stringify({ return_url: `${origin}${path}account.html` }),
    });
  }

  async function listApiKeys() {
    return apiFetch("/api/api-keys");
  }

  async function createApiKey(name) {
    return apiFetch("/api/api-keys", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
  }

  async function revokeApiKey(id) {
    return apiFetch(`/api/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
  }

  global.NbaAuthClient = {
    apiBase,
    getToken,
    setToken,
    devLogin,
    fetchMe,
    fetchEntitlements,
    startCheckout,
    openBillingPortal,
    listApiKeys,
    createApiKey,
    revokeApiKey,
    apiFetch,
  };
})(window);
