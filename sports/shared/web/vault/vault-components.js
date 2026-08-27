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

  /**
   * Validates a real FanDuel "Add to Betslip" deep link before ever
   * rendering it -- host/scheme/path allowlist only, never a generic
   * open-redirect passthrough. Shared by every sport/product that
   * renders one (single-leg plays, parlay pairs, same-game combos):
   * every one of them is built server-side from FanDuel's own real
   * odds feed (see sports/mlb/parlay_v2/fanduel_betslip.py and
   * fanduel_public_mlb_provider.py), but this re-validates on the
   * client too rather than trusting the payload blindly.
   */
  CardVault.safeFanDuelBetslipUrl = function safeFanDuelBetslipUrl(value) {
    try {
      const url = new URL(String(value || ""));
      const allowedHosts = new Set(["account.sportsbook.fanduel.com", "sportsbook.fanduel.com"]);
      if (url.protocol !== "https:" || !allowedHosts.has(url.hostname.toLowerCase())) return "";
      if (!url.pathname.toLowerCase().endsWith("/addtobetslip")) return "";
      return url.toString();
    } catch (_error) {
      return "";
    }
  };

  /**
   * Shared photo markup for both card renderers below. Chains a primary
   * headshot URL to an optional fallback URL (e.g. MLB's secondary CDN
   * mirror) and finally to the monogram -- each <img> falls through to
   * the next element on its own onerror, so a broken or missing image
   * never leaves a blank card. Never guesses a URL from an id: only
   * real URLs supplied by the caller are ever rendered.
   */
  CardVault.renderPhotoHtml = function renderPhotoHtml(primaryUrl, fallbackUrl, monogram) {
    const fallbackSpan = `<span class="prediction-card__fallback">${CardVault.escapeHtml(monogram)}</span>`;
    if (!primaryUrl) return fallbackSpan;
    // Flat chain of siblings: each <img>'s onerror swaps itself out for
    // whatever comes next (the fallback image, or finally the monogram),
    // so a broken/missing photo never leaves a blank card.
    const img = (url) => `<img class="prediction-card__photo-img" src="${CardVault.escapeAttr(url)}" alt="" loading="lazy" onerror="this.replaceWith(this.nextElementSibling)" />`;
    return fallbackUrl
      ? `${img(primaryUrl)}${img(fallbackUrl)}${fallbackSpan}`
      : `${img(primaryUrl)}${fallbackSpan}`;
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

  /**
   * Real breakeven probability implied by a single American price -- pure
   * arithmetic on a number the payload already carries (selected_side_
   * price), never a guess or a second data source. This is the SAME
   * side's own vig-included implied probability (not de-vigged against
   * an opposite-side quote), which is exactly what "probability edge"
   * needs: how much the model's own calibrated probability clears what
   * this exact price requires to break even.
   */
  CardVault.impliedProbabilityFromAmerican = function impliedProbabilityFromAmerican(price) {
    const n = Number(price);
    if (!Number.isFinite(n) || n === 0) return null;
    return n > 0 ? 100 / (n + 100) : -n / (-n + 100);
  };

  /** Signed percentage-POINT formatter -- for a difference of two
   * probabilities (e.g. model probability minus breakeven probability),
   * which is a distinct quantity from a signed percentage return (EV)
   * and should never share formatSignedPct's "%" suffix. */
  CardVault.formatSignedPp = function formatSignedPp(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "n/a";
    const pp = (n * 100).toFixed(1);
    return `${n >= 0 ? "+" : ""}${pp} pp`;
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
    // Headshot URL comes only from the exporter's own real data --
    // never guessed from a player_id, since the CDN path pattern that
    // works for one sport (e.g. cdn.nba.com) is wrong for every other
    // sport and would silently 404 or point at an unrelated image.
    const resolvedHeadshot = String(play.player_headshot_url || "").trim();
    const fallbackHeadshot = String(play.player_headshot_fallback_url || "").trim();
    const parts = displayName.split(/\s+/).filter(Boolean);
    const monogram = parts.length >= 2 ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase() : (parts[0] || "NA").slice(0, 2).toUpperCase();

    // "lineup_unconfirmed" is a real risk at publication time (this
    // board is built once, hours before first pitch), but it goes
    // stale the moment the real game actually starts -- whoever's
    // playing is playing, so a still-showing "unconfirmed" badge past
    // commence_time_utc is misleading rather than informative. Every
    // OTHER risk flag (stale_history, roster_unverified, etc.) is
    // unaffected -- this drops only the one flag that time itself
    // resolves, using a real field (commence_time_utc) the payload
    // already carries, never a guess.
    const gameHasStarted = (() => {
      const commence = Date.parse(String(play.commence_time_utc || ""));
      return Number.isFinite(commence) && Date.now() >= commence;
    })();
    const riskFlags = Array.isArray(play.risk_flags)
      ? play.risk_flags
        .map((flag) => String(flag || "").trim())
        .filter(Boolean)
        .filter((flag) => flag !== "lineup_unconfirmed" || !gameHasStarted)
      : [];
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
    // Real, distinct quantities: EV (expected return per unit staked)
    // and probability edge (calibrated hit probability minus what this
    // exact price requires to break even) are NOT the same number and
    // must never share a label -- a high-EV, low-probability pick (see
    // e.g. the pitcher-strikeouts parlay) is real and correct, but
    // labeling its EV as "Edge" reads as a much stronger probability
    // claim than the model is actually making.
    const signalLabel = play.ev != null ? "Model EV" : edgeValue != null ? "Edge" : "Model probability";
    const signalText = play.ev != null
      ? CardVault.formatSignedPct(play.ev)
      : edgeValue != null
        ? CardVault.formatSignedNumber(edgeValue)
        : CardVault.formatPct(modelProbability);
    const breakevenProbability = CardVault.impliedProbabilityFromAmerican(play.selected_side_price);
    const probabilityEdge = modelProbability != null && breakevenProbability != null
      ? Number(modelProbability) - breakevenProbability
      : null;

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

    const photoHtml = CardVault.renderPhotoHtml(resolvedHeadshot, fallbackHeadshot, monogram);
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

    // Primary metrics: what the user needs first. Line/Projection/Model/
    // Prob. edge/Odds. "Prob. edge" is a real, distinct quantity from
    // the EV shown above (see signalLabel/probabilityEdge) -- shown
    // alongside it, never in place of it, so a viewer never has to
    // infer probability edge from an EV number that isn't one.
    const primaryMetrics = [
      ...(lineText !== "n/a" ? [["Line", lineText]] : []),
      ...(predText !== "n/a" ? [["Projection", predText]] : []),
      ...(hitRate !== "n/a" ? [["Model probability", hitRate]] : []),
      ...(probabilityEdge != null ? [["Prob. edge", CardVault.formatSignedPp(probabilityEdge)]] : []),
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

    // Real single-leg "Add to Betslip" link -- only when this exact play
    // was actually priced at FanDuel (selected_sportsbook_key) AND
    // carries FanDuel's own real deep link for that selection
    // (sportsbook_deeplink). Re-validated against the host/path
    // allowlist before ever rendering; never a guessed or generic link.
    const betslipUrl = String(play.selected_sportsbook_key || "").trim().toLowerCase() === "fanduel"
      ? CardVault.safeFanDuelBetslipUrl(play.sportsbook_deeplink)
      : "";
    const betslipHtml = betslipUrl
      ? `<a class="prediction-card__betslip-link" href="${CardVault.escapeAttr(betslipUrl)}" target="_blank" rel="noopener noreferrer">Add to FanDuel Betslip</a>`
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
          <span class="prediction-card__signal-label">${CardVault.escapeHtml(signalLabel)}</span>
          <strong>${CardVault.escapeHtml(signalText)}</strong>
        </div>
        <dl class="prediction-card__metrics">${primaryMetricHtml}</dl>
        <p class="prediction-card__note">${CardVault.escapeHtml(why)}</p>
        ${betslipHtml}
        ${detailHtml ? `<details class="disclosure"><summary>Details</summary><div class="disclosure-body">${detailHtml}</div></details>` : ""}
      </article>
    `;
  };

  /**
   * Same visual unit as renderPredictionCard (the same .prediction-card
   * markup/CSS), for a pick that isn't a full board play -- a parlay leg,
   * same-game combo leg, or anything else that's still fundamentally one
   * pick and deserves the same card treatment a top-of-board pick gets,
   * not a bare list row. Takes an already-normalized spec rather than a
   * raw play/leg object: callers own the field mapping for their own
   * data shape (a player-prop leg and a team-market leg look nothing
   * alike), this only owns the shared rendering. Never fabricates a
   * field -- pass only what's real; `metrics` and `context` are omitted
   * cleanly when empty, exactly like renderPredictionCard's own optional
   * rows.
   */
  CardVault.renderLegCard = function renderLegCard({
    rank, statusTone = "", statusLabel = "", monogram = "", photoUrl = "", photoFallbackUrl = "",
    name = "", market = "", context = "", metrics = [], note = "", betslipUrl = "",
  } = {}) {
    const photoHtml = CardVault.renderPhotoHtml(photoUrl, photoFallbackUrl, monogram);
    const metricHtml = metrics
      .filter(([, value]) => value != null && value !== "")
      .map(([label, value]) => `<div><dt>${CardVault.escapeHtml(label)}</dt><dd>${CardVault.escapeHtml(value)}</dd></div>`)
      .join("");

    // Real single-leg "Add to Betslip" link for THIS leg only -- same
    // validation, same URL FanDuel's own feed issues, same treatment
    // renderPredictionCard gives a top-of-board pick. Parlay products
    // deliberately do NOT combine legs into one multi-leg deep link
    // here: that combined-URL scheme was never confirmed against real
    // FanDuel behavior and failed a real, logged-in device test, so
    // each leg gets its own real, individually-verified link instead.
    const safeBetslipUrl = CardVault.safeFanDuelBetslipUrl(betslipUrl);
    const betslipHtml = safeBetslipUrl
      ? `<a class="prediction-card__betslip-link" href="${CardVault.escapeAttr(safeBetslipUrl)}" target="_blank" rel="noopener noreferrer">Add to FanDuel Betslip</a>`
      : "";

    return `
      <article class="prediction-card" aria-label="${CardVault.escapeAttr(`${name}, ${market}`)}">
        <header class="prediction-card__header">
          <span class="prediction-card__rank">${CardVault.escapeHtml(String(rank ?? ""))}</span>
          <div class="prediction-card__tags">${statusLabel ? CardVault.renderStatusPill(statusTone, statusLabel) : ""}</div>
        </header>
        <div class="prediction-card__identity">
          <div class="prediction-card__photo">${photoHtml}</div>
          <div>
            <h3 class="prediction-card__name">${CardVault.escapeHtml(name)}</h3>
            <p class="prediction-card__market">${CardVault.escapeHtml(market)}</p>
            ${context ? `<p class="prediction-card__context">${CardVault.escapeHtml(context)}</p>` : ""}
          </div>
        </div>
        ${metricHtml ? `<dl class="prediction-card__metrics">${metricHtml}</dl>` : ""}
        ${note ? `<p class="prediction-card__note">${CardVault.escapeHtml(note)}</p>` : ""}
        ${betslipHtml}
      </article>
    `;
  };

  global.CardVault = CardVault;
})(typeof window !== "undefined" ? window : globalThis);
