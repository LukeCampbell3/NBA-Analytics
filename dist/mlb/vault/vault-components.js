/**
 * The Card Vault System — shared render helpers (vanilla JS)
 * Use across hub, sport workspaces, prediction boards, validation views.
 */
(function initCardVault(global) {
  const CardVault = {};

  CardVault.escapeHtml = function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  };

  CardVault.escapeAttr = function escapeAttr(value) {
    return CardVault.escapeHtml(value).replaceAll("`", "&#96;");
  };

  CardVault.formatNumber = function formatNumber(value, digits = 2) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
  };

  CardVault.formatPct = function formatPct(value, digits = 1) {
    const n = Number(value);
    return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : "n/a";
  };

  CardVault.formatSignedPct = function formatSignedPct(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    const pct = (n * 100).toFixed(1);
    return `${n >= 0 ? "+" : ""}${pct}%`;
  };

  /**
   * Map analytical context to card classification (not gambling tiers).
   * Future sports can pass explicit variant from payload.
   */
  CardVault.resolveVariant = function resolveVariant(ctx = {}) {
    if (ctx.variant) return String(ctx.variant).toLowerCase().replace(/_/g, "-");
    const confidence = Number(ctx.confidence);
    const volatility = String(ctx.volatility || "").toLowerCase();
    const sample = Number(ctx.sampleSize);
    const signal = String(ctx.signalStrength || "").toLowerCase();
    const isRookie = Boolean(ctx.rookie || ctx.lowSample);

    if (ctx.premium || signal === "premium") return "patch";
    if (ctx.star || signal === "star") return "manga";
    if (isRookie || (Number.isFinite(sample) && sample < 15)) return "rookie";
    if (volatility === "high") return "tiger-ice";
    if (volatility === "unusual") return "zebra-ice";
    if (Number.isFinite(confidence) && confidence >= 0.75) return "blue-ice";
    if (Number.isFinite(confidence) && confidence >= 0.6) return "refractor";
    if (ctx.historical) return "retro";
    if (signal === "high-value") return "auto";
    return "chrome";
  };

  CardVault.variantLabel = function variantLabel(variant) {
    const labels = {
      chrome: "Chrome",
      refractor: "Refractor",
      retro: "Retro",
      "blue-ice": "Blue Ice",
      "zebra-ice": "Zebra Ice",
      "tiger-ice": "Tiger Ice",
      rookie: "Rookie",
      manga: "Manga",
      auto: "Auto",
      patch: "Patch",
      "auto-patch": "Auto Patch",
    };
    return labels[variant] || "Chrome";
  };

  CardVault.confidenceBand = function confidenceBand(score) {
    const n = Number(score);
    if (!Number.isFinite(n)) return "c";
    if (n >= 0.8) return "a";
    if (n >= 0.65) return "b";
    if (n >= 0.45) return "c";
    return "d";
  };

  CardVault.renderMetricChip = function renderMetricChip(label, value, tone = "") {
    const toneClass = tone ? ` vault-metric-chip--${CardVault.escapeAttr(tone)}` : "";
    return `
      <span class="vault-metric-chip${toneClass}">
        <span class="vault-metric-chip__label">${CardVault.escapeHtml(label)}</span>
        <span class="vault-metric-chip__value">${CardVault.escapeHtml(value)}</span>
      </span>
    `;
  };

  CardVault.renderConfidenceBand = function renderConfidenceBand(score, label = "Confidence") {
    const band = CardVault.confidenceBand(score);
    const pct = Number.isFinite(Number(score)) ? Math.round(Number(score) * 100) : null;
    return `
      <span class="vault-confidence vault-confidence--${band}" title="${CardVault.escapeAttr(label)}">
        <span class="vault-confidence__text">${CardVault.escapeHtml(label)} ${pct !== null ? `${pct}%` : "n/a"}</span>
        <span class="vault-confidence__bar" aria-hidden="true"><span class="vault-confidence__fill"></span></span>
      </span>
    `;
  };

  CardVault.renderStatusPill = function renderStatusPill(status, label) {
    const key = String(status || "active").toLowerCase();
    const text = label || key;
    return `<span class="vault-status-pill vault-status-pill--${CardVault.escapeAttr(key)}">${CardVault.escapeHtml(text)}</span>`;
  };

  CardVault.renderEvidenceBadge = function renderEvidenceBadge(count, label = "Evidence") {
    return `<span class="vault-evidence"><strong>${CardVault.escapeHtml(String(count ?? 0))}</strong> ${CardVault.escapeHtml(label)}</span>`;
  };

  CardVault.renderDataFreshness = function renderDataFreshness(updatedAt, stale = false) {
    const cls = stale ? "vault-freshness vault-freshness--stale" : "vault-freshness";
    return `<span class="${cls}">Updated ${CardVault.escapeHtml(updatedAt || "unknown")}</span>`;
  };

  CardVault.renderEmptyState = function renderEmptyState(title, message, hint = "") {
    return `
      <div class="vault-empty" role="status">
        <h3>${CardVault.escapeHtml(title)}</h3>
        <p>${CardVault.escapeHtml(message)}</p>
        ${hint ? `<p class="vault-page-lead">${CardVault.escapeHtml(hint)}</p>` : ""}
      </div>
    `;
  };

  CardVault.renderSkeletonCard = function renderSkeletonCard(count = 1) {
    return Array.from({ length: count }, () => `
      <article class="player-card player-card--skeleton" aria-hidden="true">
        <div class="player-card__body">
          <div class="player-card__shimmer" style="width:60%"></div>
          <div class="player-card__shimmer" style="width:40%;margin-top:8px"></div>
          <div class="player-card__shimmer" style="width:80%;margin-top:16px"></div>
        </div>
      </article>
    `).join("");
  };

  CardVault.renderSportWorkspaceCard = function renderSportWorkspaceCard(sport) {
    const pages = Array.isArray(sport.pages) ? sport.pages : [];
    const chips = pages.slice(0, 4).map((p) => `
      <a class="vault-metric-chip" href="${CardVault.escapeAttr(p.href)}">${CardVault.escapeHtml(p.label)}</a>
    `).join("");
    return `
      <article class="vault-door" style="--sport-accent:${CardVault.escapeAttr(sport.accent)};">
        <div class="vault-door-top">
          ${CardVault.renderStatusPill(sport.status, sport.status_label)}
          <span class="vault-metric-chip"><span class="vault-metric-chip__label">NS</span>/${CardVault.escapeHtml(sport.slug)}</span>
        </div>
        <h3>${CardVault.escapeHtml(sport.title)}</h3>
        <p class="vault-door-tagline">${CardVault.escapeHtml(sport.tagline)}</p>
        <p class="vault-page-lead">${CardVault.escapeHtml(sport.summary)}</p>
        <div class="player-card__metrics" style="margin: var(--vault-space-4) 0;">
          ${chips || CardVault.renderMetricChip("Pages", "0 published")}
        </div>
        <a class="vault-door-cta" href="${CardVault.escapeAttr(sport.entry_href)}">Open ${CardVault.escapeHtml(sport.title)} predictions</a>
      </article>
    `;
  };

  /**
   * Living player card — core component
   * @param {object} opts
   */
  CardVault.renderPlayerCard = function renderPlayerCard(opts) {
    const {
      variant: variantIn,
      density = "default",
      context = "board",
      interactive = false,
      href = "",
      locked = false,
      playerName = "Unknown",
      team = "",
      position = "",
      opponent = "",
      headshotUrl = "",
      monogram = "",
      signal = "",
      why = "",
      metrics = [],
      confidence,
      confidenceLabel = "Confidence band",
      footer = "",
      analyticalTag = "",
      statusPill = null,
    } = opts;

    const variant = CardVault.resolveVariant({ ...opts, variant: variantIn });
    const classification = CardVault.variantLabel(variant);
    const premium = ["refractor", "blue-ice", "manga", "auto", "patch", "auto-patch"].includes(variant);
    const densityClass = density === "compact" ? " player-card--compact" : density === "dossier" ? " player-card--dossier" : "";
    const contextClass = context ? ` player-card--${CardVault.escapeAttr(context)}` : "";
    const tag = analyticalTag || classification;

    const avatar = headshotUrl
      ? `<img class="player-card__avatar" src="${CardVault.escapeAttr(headshotUrl)}" alt="" loading="lazy" onerror="this.replaceWith(this.nextElementSibling)" /><div class="player-card__avatar player-card__avatar-fallback" hidden>${CardVault.escapeHtml(monogram || "?")}</div>`
      : `<div class="player-card__avatar player-card__avatar-fallback">${CardVault.escapeHtml(monogram || "?")}</div>`;

    const metricHtml = (metrics || []).map((m) => CardVault.renderMetricChip(m.label, m.value, m.tone || "")).join("");
    const confHtml = confidence != null ? CardVault.renderConfidenceBand(confidence, confidenceLabel) : "";
    const statusHtml = statusPill ? CardVault.renderStatusPill(statusPill.status, statusPill.label) : "";

    const inner = `
      <div class="player-card__edge" aria-hidden="true"></div>
      <div class="player-card__foil" aria-hidden="true"></div>
      ${locked ? '<div class="player-card__lock">Research preview</div>' : ""}
      <div class="player-card__strip">
        <span class="player-card__classification">${CardVault.escapeHtml(classification)}</span>
        <span>${CardVault.escapeHtml(tag)}</span>
      </div>
      <div class="player-card__body">
        <div class="player-card__identity">
          ${avatar}
          <div>
            <h2 class="player-card__name">${CardVault.escapeHtml(playerName)}</h2>
            <p class="player-card__meta">${CardVault.escapeHtml([team, position, opponent].filter(Boolean).join(" · "))}</p>
          </div>
        </div>
        ${signal ? `<p class="player-card__signal">${CardVault.escapeHtml(signal)}</p>` : ""}
        ${why ? `<p class="player-card__why">${CardVault.escapeHtml(why)}</p>` : ""}
        <div class="player-card__metrics">${metricHtml}</div>
      </div>
      <div class="player-card__footer">
        <span>${confHtml} ${statusHtml}</span>
        <span>${CardVault.escapeHtml(footer)}</span>
      </div>
    `;

    const classes = [
      "player-card",
      `player-card--${variant}`,
      premium ? "player-card--premium" : "",
      locked ? "player-card--locked" : "",
      interactive ? "player-card--interactive" : "",
      densityClass,
      contextClass,
    ].filter(Boolean).join(" ");

    const aria = `Analytical card for ${playerName}. ${classification}. ${signal || ""}`;

    if (interactive && href) {
      return `<a class="${classes}" href="${CardVault.escapeAttr(href)}" aria-label="${CardVault.escapeAttr(aria)}">${inner}</a>`;
    }
    if (interactive) {
      return `<button type="button" class="${classes}" aria-label="${CardVault.escapeAttr(aria)}">${inner}</button>`;
    }
    return `<article class="${classes}" aria-label="${CardVault.escapeAttr(aria)}">${inner}</article>`;
  };

  CardVault.formatTargetLabel = function formatTargetLabel(target) {
    const lookup = {
      H: "Hits",
      TB: "Total Bases",
      R: "Runs",
      K: "Pitcher K",
      HR: "Home Runs",
      RBI: "RBIs",
      ER: "Earned Runs",
    };
    const key = String(target || "").toUpperCase();
    return lookup[key] || key || "Market";
  };

  /** Prediction board card — uses player card language (NBA + MLB payloads) */
  CardVault.renderPredictionCard = function renderPredictionCard(play, index = 0) {
    const tierRaw = String(play.confidence_tier || play.recommendation || "consider").toLowerCase();
    const directionRaw = String(play.direction || "").toUpperCase();
    const direction = directionRaw === "UNDER" ? "UNDER" : "OVER";
    const displayName = String(play.player_display_name || play.player || "").replaceAll("_", " ").trim() || "Unknown Player";
    const headshotUrl = String(play.player_headshot_url || "").trim();
    const id = Number(play.player_id);
    const resolvedHeadshot = headshotUrl || (Number.isFinite(id) && id > 0
      ? `https://cdn.nba.com/headshots/nba/latest/1040x760/${id}.png`
      : "");
    const parts = displayName.split(/\s+/).filter(Boolean);
    const monogram = parts.length >= 2 ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase() : (parts[0] || "NA").slice(0, 2).toUpperCase();

    const riskFlags = Array.isArray(play.risk_flags) ? play.risk_flags.map((flag) => String(flag || "").trim()).filter(Boolean) : [];
    const actionStatus = String(play.action_status || play.publication_status || "").toLowerCase();
    const needsReview = actionStatus === "review" || riskFlags.length > 0 || play.model_estimate_status === "review";
    const tier = needsReview
      ? "review"
      : (play.parlay_candidate ? "parlay" : (["elite", "strong", "consider", "pass"].includes(tierRaw) ? tierRaw : "consider"));

    const lineText = CardVault.formatNumber(play.market_line);
    const predText = CardVault.formatNumber(play.prediction);
    const targetLabel = CardVault.formatTargetLabel(play.target);
    const edgeValue = play.abs_edge != null ? play.abs_edge : play.edge;
    const edgeText = CardVault.formatNumber(edgeValue);
    const evText = play.ev != null ? CardVault.formatSignedPct(play.ev) : CardVault.formatSignedNumber(edgeValue);

    const gameText = [play.market_away_team, play.market_home_team].filter(Boolean).join(" @ ");
    const sourceText = String(play.market_source || "").toLowerCase() === "real" ? "Book line" : play.market_source ? "Benchmark line" : "";
    const footerParts = [play.market_date, gameText, sourceText].filter(Boolean);
    const footer = footerParts.join(" - ");

    const riskLabels = {
      stale_history: "Stale data",
      lineup_unconfirmed: "Lineup",
      roster_unverified: "Roster",
      team_mismatch: "Team check",
      game_date_mismatch: "Date check",
      push_exposure: "Push risk",
      multi_game_slate_review: "Slate check",
    };

    const directionDelta = Number.isFinite(Number(edgeValue))
      ? `${CardVault.formatNumber(Math.abs(Number(edgeValue)))} ${direction === "UNDER" ? "below" : "above"} the market line`
      : `aligned to the ${direction} side of the market`;
    const why = needsReview
      ? "Review before action: stale data, lineup status, push exposure, or slate context may affect settlement."
      : play.parlay_candidate
      ? `Model lean pairs with ${String(play.parlay_partner_name || "another tagged leg").trim()} (${CardVault.formatPct(play.parlay_projected_hit_rate)} projected alignment).`
      : `The model projection is ${directionDelta}.`;

    const photoHtml = resolvedHeadshot
      ? `<img class="prediction-card__photo-img" src="${CardVault.escapeAttr(resolvedHeadshot)}" alt="" loading="lazy" onerror="this.replaceWith(this.nextElementSibling)" /><span class="prediction-card__fallback">${CardVault.escapeHtml(monogram)}</span>`
      : `<span class="prediction-card__fallback">${CardVault.escapeHtml(monogram)}</span>`;
    const parlayTag = play.parlay_candidate && !needsReview ? '<span class="prediction-card__tag prediction-card__tag--parlay">Parlay</span>' : "";
    const tierTag = needsReview ? "" : `<span class="prediction-card__tag">${CardVault.escapeHtml(tier)}</span>`;
    const publicationStatus = String(play.board_publication_status || play.publication_status || "ready").toLowerCase();
    const statusTag = needsReview
      ? '<span class="prediction-card__tag prediction-card__tag--risk">Review</span>'
      : publicationStatus !== "ready" && publicationStatus !== "published"
        ? '<span class="prediction-card__tag prediction-card__tag--risk">Withheld</span>'
        : '<span class="prediction-card__tag">Published</span>';
    const riskTags = riskFlags
      .slice(0, 3)
      .map((flag) => `<span class="prediction-card__tag prediction-card__tag--risk">${CardVault.escapeHtml(riskLabels[flag] || flag.replaceAll("_", " "))}</span>`)
      .join("");
    const hitRate = play.estimated_graded_hit_rate != null ? CardVault.formatPct(play.estimated_graded_hit_rate) : "n/a";
    const pushText = play.estimated_push_probability != null ? CardVault.formatPct(play.estimated_push_probability) : "n/a";
    const valueScore = play.value_score != null ? CardVault.formatNumber(play.value_score) : "n/a";
    const metrics = [
      ["Line", lineText],
      ["Projection", predText],
      ["Edge", edgeText],
      ...(hitRate !== "n/a" ? [["Model", hitRate]] : []),
      ...(pushText !== "n/a" ? [["Push", pushText]] : []),
      ...(valueScore !== "n/a" ? [["Value", valueScore]] : []),
    ];
    const metricHtml = metrics
      .map(([label, value]) => `<div><dt>${CardVault.escapeHtml(label)}</dt><dd>${CardVault.escapeHtml(value)}</dd></div>`)
      .join("");
    const signalLabel = play.ev != null ? "Expected value" : "Edge";

    const rank = String(play.rank || index + 1).padStart(2, "0");

    return `
      <article class="prediction-card prediction-card--${CardVault.escapeAttr(tier)}" data-direction="${CardVault.escapeAttr(direction)}" aria-label="Prediction card for ${CardVault.escapeAttr(displayName)} ${CardVault.escapeAttr(direction)} ${CardVault.escapeAttr(targetLabel)}">
        <header class="prediction-card__header">
          <div class="prediction-card__bounty-heading">
            <span class="prediction-card__rank">Bounty ${CardVault.escapeHtml(rank)}</span>
            <span class="prediction-card__wanted">Wanted</span>
          </div>
          <div class="prediction-card__tags">${parlayTag}${tierTag}${statusTag}${riskTags}</div>
        </header>
        <div class="prediction-card__identity">
          <div class="prediction-card__photo">${photoHtml}</div>
          <div class="prediction-card__identity-copy">
            <h3 class="prediction-card__name">${CardVault.escapeHtml(displayName)}</h3>
            <p class="prediction-card__market">${CardVault.escapeHtml(direction)} ${CardVault.escapeHtml(targetLabel)}</p>
            ${footer ? `<p class="prediction-card__context">${CardVault.escapeHtml(footer)}</p>` : ""}
          </div>
        </div>
        <div class="prediction-card__signal">
          <span class="prediction-card__signal-label">Reward signal · ${CardVault.escapeHtml(signalLabel)}</span>
          <strong>${CardVault.escapeHtml(evText)}</strong>
        </div>
        <dl class="prediction-card__metrics">${metricHtml}</dl>
        <p class="prediction-card__note">${CardVault.escapeHtml(why)}</p>
      </article>
    `;
  };

  CardVault.formatSignedNumber = function formatSignedNumber(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
  };

  global.CardVault = CardVault;
})(typeof window !== "undefined" ? window : globalThis);
