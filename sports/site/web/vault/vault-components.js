/**
 * In The Cards Analytics -- shared render helpers (vanilla JS)
 * Use across the landing page, prediction boards, and methodology views.
 *
 * renderPlayerCard / resolveVariant / variantLabel (a trading-card rarity
 * system) and renderSportWorkspaceCard (workspace "door" cards) were
 * removed here as dead code -- neither is referenced by any publicly
 * shipped page; see the route audit.
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

  CardVault.formatSignedNumber = function formatSignedNumber(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
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

  CardVault.confidenceBand = function confidenceBand(score) {
    const n = Number(score);
    if (!Number.isFinite(n)) return "c";
    if (n >= 0.8) return "a";
    if (n >= 0.65) return "b";
    if (n >= 0.45) return "c";
    return "d";
  };

  /**
   * One canonical status vocabulary, used everywhere a prediction's
   * publication/authorization state is shown. Do not invent per-sport
   * synonyms -- extend this map instead.
   */
  CardVault.STATUS_LABELS = {
    qualified: "Qualified",
    published: "Qualified",
    ready: "Qualified",
    shadow: "Shadow",
    review: "Pending",
    pending: "Pending",
    withheld: "Withheld",
    unavailable: "Unavailable",
    "research-only": "Research Only",
    research_only: "Research Only",
    active: "Qualified",
    stale: "Pending",
    error: "Unavailable",
  };

  CardVault.renderStatusPill = function renderStatusPill(status, label) {
    const key = String(status || "pending").toLowerCase().replaceAll("_", "-");
    const text = label || CardVault.STATUS_LABELS[key] || key;
    return `<span class="status-pill status-pill--${CardVault.escapeAttr(key)}">${CardVault.escapeHtml(text)}</span>`;
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
      <article class="prediction-card" aria-hidden="true">
        <div class="prediction-card__note">Loading...</div>
      </article>
    `).join("");
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
      PASSING_YARDS: "Passing Yards",
      RUSHING_YARDS: "Rushing Yards",
      RECEIVING_YARDS: "Receiving Yards",
    };
    const key = String(target || "").toUpperCase();
    return lookup[key] || key || "Market";
  };

  /**
   * Prediction card -- the primary decision unit on every sport's board.
   * Leads with player / market / projection / model probability / edge /
   * odds / status (see spec section 14). Everything lower-priority
   * (support depth, policy identifiers, simulation metadata, push
   * exposure, source detail) lives behind a <details> "Details" disclosure
   * rather than crowding the card.
   */
  CardVault.renderPredictionCard = function renderPredictionCard(play, index = 0) {
    const directionRaw = String(play.direction || "").toUpperCase();
    const direction = directionRaw === "UNDER" ? "UNDER" : "OVER";
    const displayName = String(play.player_display_name || play.player || "").replaceAll("_", " ").trim() || "Unknown player";
    const headshotUrl = String(play.player_headshot_url || "").trim();
    const id = Number(play.player_id);
    const resolvedHeadshot = headshotUrl || (Number.isFinite(id) && id > 0
      ? `https://cdn.nba.com/headshots/nba/latest/1040x760/${id}.png`
      : "");
    const parts = displayName.split(/\s+/).filter(Boolean);
    const monogram = parts.length >= 2 ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase() : (parts[0] || "NA").slice(0, 2).toUpperCase();

    const riskFlags = Array.isArray(play.risk_flags) ? play.risk_flags.map((flag) => String(flag || "").trim()).filter(Boolean) : [];
    if (play.candidate_authorized === false && !riskFlags.includes("policy_uncertified")) riskFlags.push("policy_uncertified");
    const actionStatus = String(play.action_status || play.publication_status || "").toLowerCase();
    const needsReview = actionStatus === "review" || riskFlags.length > 0 || play.model_estimate_status === "review";

    const lineText = CardVault.formatNumber(play.market_line);
    const predText = CardVault.formatNumber(play.prediction);
    const targetLabel = CardVault.formatTargetLabel(play.target);
    const edgeValue = play.abs_edge != null ? play.abs_edge : play.edge;
    const edgeText = CardVault.formatNumber(edgeValue);
    const modelProbability = play.estimated_graded_hit_rate != null
      ? play.estimated_graded_hit_rate
      : play.model_hit_probability;
    const signalText = play.ev != null
      ? CardVault.formatSignedPct(play.ev)
      : edgeValue != null
        ? CardVault.formatSignedNumber(edgeValue)
        : CardVault.formatPct(modelProbability);

    const gameText = [play.market_away_team, play.market_home_team].filter(Boolean).join(" @ ")
      || [play.team, play.opponent].filter(Boolean).join(" vs ");
    const bookName = String(play.selected_sportsbook_key || "")
      .split("_")
      .filter(Boolean)
      .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
      .join(" ");
    const sourceText = bookName
      ? bookName
      : String(play.market_source || "").toLowerCase() === "real"
        ? "Book line"
        : play.market_source ? "Benchmark line" : "";
    const footerParts = [play.market_date, gameText, sourceText].filter(Boolean);
    const footer = footerParts.join(" · ");

    const riskLabels = {
      stale_history: "Stale data",
      lineup_unconfirmed: "Lineup unconfirmed",
      roster_unverified: "Roster unverified",
      team_mismatch: "Team check",
      game_date_mismatch: "Date check",
      push_exposure: "Push risk",
      policy_uncertified: "Shadow",
      multi_game_slate_review: "Slate check",
    };

    const statusKey = riskFlags.includes("policy_uncertified")
      ? "shadow"
      : needsReview
        ? "pending"
        : (String(play.board_publication_status || play.publication_status || "ready").toLowerCase() === "ready"
          || String(play.board_publication_status || play.publication_status || "").toLowerCase() === "published")
          ? "qualified"
          : "withheld";

    const why = riskFlags.includes("policy_uncertified")
      ? "Shadow: this candidate is not yet authorized for staking, pending certification evidence."
      : needsReview
      ? "Pending review: stale data, lineup status, push exposure, or slate context may affect settlement."
      : play.parlay_candidate
      ? `Pairs with ${String(play.parlay_partner_name || "another tagged leg").trim()} (${CardVault.formatPct(play.parlay_projected_hit_rate)} projected alignment).`
      : Number.isFinite(Number(edgeValue))
        ? `Model projection is ${CardVault.formatNumber(Math.abs(Number(edgeValue)))} ${direction === "UNDER" ? "below" : "above"} the market line.`
        : `Model projection is aligned to the ${direction} side of the market.`;

    const photoHtml = resolvedHeadshot
      ? `<img class="prediction-card__photo-img" src="${CardVault.escapeAttr(resolvedHeadshot)}" alt="" loading="lazy" onerror="this.replaceWith(this.nextElementSibling)" /><span class="prediction-card__fallback">${CardVault.escapeHtml(monogram)}</span>`
      : `<span class="prediction-card__fallback">${CardVault.escapeHtml(monogram)}</span>`;
    const parlayTag = play.parlay_candidate && !needsReview ? '<span class="prediction-card__tag prediction-card__tag--parlay">Parlay</span>' : "";
    const riskTags = riskFlags
      .filter((flag) => flag !== "policy_uncertified")
      .slice(0, 2)
      .map((flag) => `<span class="prediction-card__tag prediction-card__tag--risk">${CardVault.escapeHtml(riskLabels[flag] || flag.replaceAll("_", " "))}</span>`)
      .join("");

    const hitRate = modelProbability != null ? CardVault.formatPct(modelProbability) : "n/a";
    const oddsText = play.selected_side_price != null ? CardVault.formatSignedNumber(play.selected_side_price).replace(/\.00$/, "") : "n/a";
    const pushText = play.estimated_push_probability != null ? CardVault.formatPct(play.estimated_push_probability) : null;
    const valueScore = play.value_score != null ? CardVault.formatNumber(play.value_score) : null;

    // Primary metrics: what the user needs first. Line/Projection/Model/Odds.
    const primaryMetrics = [
      ...(lineText !== "n/a" ? [["Line", lineText]] : []),
      ...(predText !== "n/a" ? [["Projection", predText]] : []),
      ...(hitRate !== "n/a" ? [["Model probability", hitRate]] : []),
      ["Odds", oddsText],
    ];
    const primaryMetricHtml = primaryMetrics
      .map(([label, value]) => `<div><dt>${CardVault.escapeHtml(label)}</dt><dd>${CardVault.escapeHtml(value)}</dd></div>`)
      .join("");

    // Secondary/audit detail: support depth, policy state, source, value
    // score, push exposure -- behind progressive disclosure, not deleted.
    const detailRows = [
      ...(edgeText !== "n/a" ? [["Edge", edgeText]] : []),
      ...(pushText ? [["Push probability", pushText]] : []),
      ...(valueScore ? [["Value score", valueScore]] : []),
      ...(play.market_books != null ? [["Books compared", CardVault.formatNumber(play.market_books, 0)]] : []),
      ...(sourceText ? [["Source", sourceText]] : []),
      ...(play.rank != null ? [["Rank", String(play.rank)]] : []),
    ];
    const detailHtml = detailRows.length
      ? `<dl class="prediction-card__metrics" style="grid-template-columns:1fr;margin-top:8px;">${detailRows.map(([label, value]) => `<div><dt>${CardVault.escapeHtml(label)}</dt><dd>${CardVault.escapeHtml(value)}</dd></div>`).join("")}</dl>`
      : "";

    return `
      <article class="prediction-card" data-direction="${CardVault.escapeAttr(direction)}" aria-label="Prediction for ${CardVault.escapeAttr(displayName)}, ${CardVault.escapeAttr(direction)} ${CardVault.escapeAttr(targetLabel)}">
        <header class="prediction-card__header">
          <span class="prediction-card__rank">${String(play.rank || index + 1)}</span>
          <div class="prediction-card__tags">${parlayTag}${CardVault.renderStatusPill(statusKey)}${riskTags}</div>
        </header>
        <div class="prediction-card__identity">
          <div class="prediction-card__photo">${photoHtml}</div>
          <div>
            <h3 class="prediction-card__name">${CardVault.escapeHtml(displayName)}</h3>
            <p class="prediction-card__market">${CardVault.escapeHtml(direction)} ${CardVault.escapeHtml(targetLabel)}</p>
            ${footer ? `<p class="prediction-card__context">${CardVault.escapeHtml(footer)}</p>` : ""}
          </div>
        </div>
        <div class="prediction-card__signal">
          <span class="prediction-card__signal-label">${CardVault.escapeHtml(edgeValue != null ? "Edge" : "Model probability")}</span>
          <strong>${CardVault.escapeHtml(signalText)}</strong>
        </div>
        <dl class="prediction-card__metrics">${primaryMetricHtml}</dl>
        <p class="prediction-card__note">${CardVault.escapeHtml(why)}</p>
        ${detailHtml ? `<details class="disclosure"><summary>Details</summary><div class="disclosure-body">${detailHtml}</div></details>` : ""}
      </article>
    `;
  };

  global.CardVault = CardVault;
})(typeof window !== "undefined" ? window : globalThis);
