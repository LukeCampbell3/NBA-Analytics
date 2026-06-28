const PLAN_COPY = {
  free: {
    name: "Free Research",
    price: "$0",
    bullets: ["3 safe-state preview cards/day", "Simulation previews", "24h settlement delay", "Shadow/research labels"],
    cta: "Current preview",
    planId: null,
  },
  plus: {
    name: "Plus Analytics",
    price: "$19/mo",
    bullets: ["Full safe-state board", "Simulation cards", "Settlement tracking", "No settlement delay"],
    cta: "Upgrade to Plus",
    planId: "plus",
  },
  pro: {
    name: "Pro Research",
    price: "$49/mo",
    bullets: ["Candidate pool visibility", "Advanced simulation filters", "CSV export", "Live/pre-lock context"],
    cta: "Upgrade to Pro",
    planId: "pro",
  },
  api: {
    name: "API Access",
    price: "$99/mo",
    bullets: ["Token-based JSON API", "Daily request limits", "Historical export access", "Usage dashboard"],
    cta: "Get API Access",
    planId: "api",
  },
};

function mountShell() {
  if (!window.CardVaultShell) return;
  window.CardVaultShell.mount({
    brandTitle: "NBA Analytics",
    brandHref: "/",
    workspaceLabel: "NBA",
    workspaceHref: "/nba/",
    sportSlug: "nba",
    sportAccent: "#f59e0b",
    breadcrumbs: [{ label: "Pricing", href: "pricing.html" }],
    navLinks: [
      { label: "Dashboard", href: "index.html" },
      { label: "Board", href: "predictions.html" },
      { label: "Safe-State", href: "safe-state.html" },
      { label: "Pricing", href: "pricing.html", active: true },
      { label: "Account", href: "account.html" },
      { label: "API", href: "api.html" },
    ],
    showDisclaimer: false,
  });
}

function renderPlanCard(key, plan) {
  const copy = PLAN_COPY[key] || { name: plan.name, price: `$${(plan.monthly_price_cents / 100).toFixed(0)}/mo`, bullets: [], cta: "Select", planId: key };
  return `
    <article class="vault-door" style="--sport-accent:#f59e0b">
      <div class="vault-door-top">${key === "free" ? '<span class="vault-status-pill vault-status-pill--active">Preview</span>' : `<span class="vault-status-pill">${copy.name}</span>`}</div>
      <h3>${copy.name}</h3>
      <p class="vault-door-tagline">${copy.price}</p>
      <ul class="hero-points">${copy.bullets.map((b) => `<li>${b}</li>`).join("")}</ul>
      ${copy.planId ? `<button type="button" class="vault-door-cta checkout-btn" data-plan="${copy.planId}">${copy.cta}</button>` : `<span class="vault-metric-chip">Included with site preview</span>`}
    </article>
  `;
}

async function init() {
  mountShell();
  const grid = document.getElementById("pricingGrid");
  let plans = Object.keys(PLAN_COPY);
  try {
    const payload = await fetch(`${window.NbaAuthClient?.apiBase?.() || ""}/api/plans`).then((r) => r.json());
    if (Array.isArray(payload.plans) && payload.plans.length) {
      plans = payload.plans.map((p) => p.id);
    }
  } catch (_e) {
    /* static fallback */
  }
  grid.innerHTML = plans.map((id) => renderPlanCard(id, PLAN_COPY[id] || {})).join("");
  grid.querySelectorAll(".checkout-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        const planId = btn.getAttribute("data-plan");
        const session = await window.NbaAuthClient.startCheckout(planId);
        window.location.href = session.checkout_url;
      } catch (error) {
        alert(error.message || "Checkout unavailable. Sign in first or configure Stripe server-side.");
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", init);
