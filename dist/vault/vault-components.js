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
   * FanDuel Sportsbook is a real, state-by-state licensed operator: each
   * region is a genuinely separate sportsbook instance with its own real
   * marketId/selectionId for the identical player/market/line (confirmed
   * directly -- fanduel_public_mlb_provider.py's own real fetch returns a
   * different marketId per region for the same real prop). A deep link
   * built under one region only actually adds to a viewer's betslip if
   * their real FanDuel account is in that same region -- this mirrors
   * sports/mlb/predictions/odds/providers/fanduel_regions.py's real,
   * verified list (kept in sync by hand; both are small, rarely-changing
   * lists, not worth a build step to share).
   */
  CardVault.FANDUEL_STATE_NAMES = {
    NY: "New York", PA: "Pennsylvania", OH: "Ohio", IL: "Illinois", MI: "Michigan",
    NC: "North Carolina", VA: "Virginia", AZ: "Arizona", TN: "Tennessee", IN: "Indiana",
    MA: "Massachusetts", MD: "Maryland", MO: "Missouri", CO: "Colorado", WV: "West Virginia",
    LA: "Louisiana", KY: "Kentucky", CT: "Connecticut", IA: "Iowa", KS: "Kansas",
    AR: "Arkansas", WY: "Wyoming", VT: "Vermont", NJ: "New Jersey", DC: "District of Columbia",
  };
  CardVault.FANDUEL_REGION_STORAGE_KEY = "itc_fanduel_region";
  CardVault._fanduelRegionMemory = "";

  /**
   * The viewer's own persisted state choice, per-browser (localStorage --
   * private to this viewer, never sent anywhere, never seen by other
   * viewers). Wrapped in try/catch: a private window, cleared site data,
   * or a storage-blocking browser setting must never break the page --
   * see the artifact/browser-storage discipline this repo already
   * follows elsewhere. Returns "" (no selection) rather than guessing a
   * default state for the viewer.
   */
  CardVault.getFanduelRegion = function getFanduelRegion() {
    try {
      const value = String(global.localStorage?.getItem(CardVault.FANDUEL_REGION_STORAGE_KEY) || "").toUpperCase();
      if (CardVault.FANDUEL_STATE_NAMES[value]) {
        CardVault._fanduelRegionMemory = value;
        return value;
      }
    } catch (_error) {
      // Fall through to the in-memory choice. Storage restrictions must
      // not make the just-selected state disappear before URL resolution.
    }
    return CardVault.FANDUEL_STATE_NAMES[CardVault._fanduelRegionMemory]
      ? CardVault._fanduelRegionMemory
      : "";
  };

  CardVault.setFanduelRegion = function setFanduelRegion(state) {
    const normalized = String(state || "").toUpperCase();
    CardVault._fanduelRegionMemory = CardVault.FANDUEL_STATE_NAMES[normalized] ? normalized : "";
    try {
      if (normalized && CardVault.FANDUEL_STATE_NAMES[normalized]) {
        global.localStorage?.setItem(CardVault.FANDUEL_REGION_STORAGE_KEY, normalized);
      } else {
        global.localStorage?.removeItem(CardVault.FANDUEL_REGION_STORAGE_KEY);
      }
    } catch (_error) { /* storage unavailable -- selection just won't persist this session */ }
  };

  /**
   * A real link for the viewer's own selected state when one is
   * available in `deeplinksByRegion` (the payload's real per-state map,
   * see enrich_parlay_leg_betslip.py), else the original single-region
   * `fallbackUrl` (which only actually resolves for a viewer whose real
   * account happens to be in whichever region the pipeline fetched under
   * -- historically NJ). Both are re-validated through
   * safeFanDuelBetslipUrl before ever being rendered.
   */
  CardVault.resolveBetslipUrl = function resolveBetslipUrl(deeplinksByRegion, fallbackUrl) {
    const region = CardVault.getFanduelRegion();
    if (region && deeplinksByRegion && deeplinksByRegion[region]) {
      const regional = CardVault.safeFanDuelBetslipUrl(deeplinksByRegion[region]);
      if (regional) return regional;
    }
    return CardVault.safeFanDuelBetslipUrl(fallbackUrl);
  };

  /**
   * No persistent page-level control -- every real FanDuel betslip link
   * on the page instead resolves itself the moment it's clicked:
   *   - viewer's state already known (getFanduelRegion()) -> the correct
   *     real regional link opens immediately, no interruption.
   *   - not known yet -> the click is held, a one-time real prompt asks
   *     which state to use, and once answered the same real click's link
   *     opens -- the state is then remembered (localStorage) so every
   *     later click across the whole site just works with no prompt.
   * A single delegated document-level click listener (initFanduelBetslip
   * Links, called once at page load) drives every link tagged
   * data-fanduel-betslip="1" (see renderPredictionCard/renderLegCard);
   * this file only needs to be loaded once for it to cover the whole page.
   */
  CardVault.FANDUEL_BETSLIP_LINK_SELECTOR = '[data-fanduel-betslip="1"]';

  CardVault.resolveFanduelBetslipHref = function resolveFanduelBetslipHref(element) {
    const region = CardVault.getFanduelRegion();
    let deeplinksByRegion = {};
    try {
      deeplinksByRegion = JSON.parse(element.dataset.deeplinksByRegion || "{}");
    } catch (_error) { /* malformed/absent -- falls back below */ }
    if (region && deeplinksByRegion[region]) {
      const regional = CardVault.safeFanDuelBetslipUrl(deeplinksByRegion[region]);
      if (regional) return regional;
    }
    return CardVault.safeFanDuelBetslipUrl(element.dataset.fallbackUrl || element.getAttribute("href"));
  };

  /**
   * Builds (once) and shows the real one-time "which state" prompt as a
   * native <dialog> -- keyboard/ESC/focus-trap handled by the browser,
   * no framework needed. Resolves with the chosen 2-letter state code,
   * or "" if the viewer dismissed it without choosing.
   */
  CardVault.promptFanduelRegion = function promptFanduelRegion() {
    let dialog = document.getElementById("fanduelRegionDialog");
    if (!dialog) {
      const options = Object.entries(CardVault.FANDUEL_STATE_NAMES)
        .sort((a, b) => a[1].localeCompare(b[1]))
        .map(([code, name]) => `<option value="${CardVault.escapeAttr(code)}">${CardVault.escapeHtml(name)}</option>`)
        .join("");
      dialog = document.createElement("dialog");
      dialog.id = "fanduelRegionDialog";
      dialog.className = "fanduel-region-dialog";
      dialog.innerHTML = `
        <form method="dialog" class="fanduel-region-dialog__form">
          <h3 class="fanduel-region-dialog__title">Which state is your FanDuel account in?</h3>
          <p class="fanduel-region-dialog__note">FanDuel prices each state separately, so this makes sure the link actually adds the bet to your slip. Asked once -- your browser remembers it after this.</p>
          <select class="fanduel-region-dialog__select" required>
            <option value="" disabled selected>Select your state</option>
            ${options}
          </select>
          <div class="fanduel-region-dialog__actions">
            <button type="submit" value="cancel" formnovalidate class="fanduel-region-dialog__cancel">Not now</button>
            <button type="submit" value="continue" class="fanduel-region-dialog__continue">Continue to FanDuel</button>
          </div>
        </form>
      `;
      document.body.appendChild(dialog);
    }
    const select = dialog.querySelector("select");
    select.value = "";
    return new Promise((resolve) => {
      const onClose = () => {
        dialog.removeEventListener("close", onClose);
        resolve(dialog.returnValue === "continue" ? select.value : "");
      };
      dialog.addEventListener("close", onClose);
      dialog.showModal();
    });
  };

  /**
   * The shared "Add to FanDuel Betslip" anchor markup -- used by both
   * renderPredictionCard and renderLegCard so the click-resolution data
   * attributes are built in exactly one place. `href` is the best link
   * known at render time (progressive enhancement -- still a real,
   * working link if JS never runs); the data attributes let
   * initFanduelBetslipLinks() re-resolve it fresh at click time against
   * whatever the viewer's region turns out to be by then.
   */
  CardVault.renderFanduelBetslipAnchor = function renderFanduelBetslipAnchor(fallbackUrl, deeplinksByRegion) {
    const href = CardVault.resolveBetslipUrl(deeplinksByRegion, fallbackUrl);
    if (!href) return "";
    const dataAttrs = deeplinksByRegion
      ? ` data-fanduel-betslip="1" data-fallback-url="${CardVault.escapeAttr(CardVault.safeFanDuelBetslipUrl(fallbackUrl))}" data-deeplinks-by-region="${CardVault.escapeAttr(JSON.stringify(deeplinksByRegion))}"`
      : "";
    return `<a class="prediction-card__betslip-link" href="${CardVault.escapeAttr(href)}"${dataAttrs} target="_blank" rel="noopener noreferrer">Add to FanDuel Betslip</a>`;
  };

  /**
   * Wires every real betslip link on the page, exactly once. Delegated
   * on document so it covers links added by a later re-render too --
   * callers never need to re-bind anything.
   */
  CardVault.initFanduelBetslipLinks = function initFanduelBetslipLinks() {
    if (CardVault._fanduelBetslipLinksInitialized) return;
    CardVault._fanduelBetslipLinksInitialized = true;
    document.addEventListener("click", async (event) => {
      const link = event.target.closest?.(CardVault.FANDUEL_BETSLIP_LINK_SELECTOR);
      if (!link) return;

      // When the state is already known, preserve the browser's native
      // anchor navigation. Updating href inside the trusted click keeps
      // target=_blank/app handoff eligible on Safari instead of routing it
      // through a later window.open() call that its popup blocker can reject.
      if (CardVault.getFanduelRegion()) {
        const href = CardVault.resolveFanduelBetslipHref(link);
        if (href) link.href = href;
        return;
      }

      event.preventDefault();
      if (!CardVault.getFanduelRegion()) {
        const chosen = await CardVault.promptFanduelRegion();
        if (!chosen) return; // dismissed -- no link opened, nothing saved
        CardVault.setFanduelRegion(chosen);
      }
      const href = CardVault.resolveFanduelBetslipHref(link);
      // The state dialog makes this asynchronous, so opening a new window
      // here is no longer reliably covered by the original user activation
      // on iOS Safari. Same-tab navigation is not popup-blocked and lets
      // FanDuel perform its normal web/app handoff.
      if (href) global.location.assign(href);
    });
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

  /**
   * Real settled game outcome -- a distinct concept from the
   * publication/authorization vocabulary above (STATUS_LABELS/
   * renderStatusPill's own docstring scopes that map to "a prediction's
   * publication/authorization state"). This reads settlement_status,
   * written only by sports/mlb/scripts/settle_published_predictions.py
   * once the real underlying MLB game is final and the real boxscore
   * stat has been compared to the real market line -- never a guess, and
   * never shown for a still-pending or never-attempted row (both render
   * nothing here, exactly like every other optional badge in this file).
   */
  CardVault.SETTLEMENT_LABELS = { won: "Won", lost: "Lost", push: "Push" };
  CardVault.renderSettlementBadge = function renderSettlementBadge(row) {
    const status = String(row?.settlement_status || "").toLowerCase();
    const label = CardVault.SETTLEMENT_LABELS[status];
    if (!label) return "";
    return `<span class="status-pill status-pill--${CardVault.escapeAttr(status)}">${CardVault.escapeHtml(label)}</span>`;
  };

  /**
   * Whole-parlay outcome from its legs' own settlement_status -- so a
   * viewer can tell at a glance whether the real parlay itself hit, not
   * just each individual leg (real sportsbook parlay logic: one real
   * loss loses the whole parlay regardless of the other legs; a push
   * leg drops out rather than counting as a loss). Returns null (render
   * nothing) until every leg has a real resolved outcome, UNLESS a leg
   * has already lost -- a parlay is real-lost the moment any one leg is,
   * with no need to wait on the rest.
   */
  CardVault.combineLegSettlementStatuses = function combineLegSettlementStatuses(rows) {
    const statuses = (rows || [])
      .filter(Boolean)
      .map((row) => String(row.settlement_status || "").toLowerCase());
    if (!statuses.length) return null;
    if (statuses.some((status) => status === "lost")) return "lost";
    if (statuses.some((status) => status !== "won" && status !== "push")) return null;
    return statuses.every((status) => status === "push") ? "push" : "won";
  };

  CardVault.PARLAY_SETTLEMENT_LABELS = { won: "Parlay Won", lost: "Parlay Lost", push: "Parlay Push" };
  CardVault.renderParlaySettlementBadge = function renderParlaySettlementBadge(rows) {
    const status = CardVault.combineLegSettlementStatuses(rows);
    if (!status) return "";
    return `<span class="status-pill status-pill--${CardVault.escapeAttr(status)}">${CardVault.escapeHtml(CardVault.PARLAY_SETTLEMENT_LABELS[status])}</span>`;
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
    // (sportsbook_deeplink). Resolved fresh at click time against the
    // viewer's own state -- see renderFanduelBetslipAnchor /
    // initFanduelBetslipLinks.
    const betslipHtml = String(play.selected_sportsbook_key || "").trim().toLowerCase() === "fanduel"
      ? CardVault.renderFanduelBetslipAnchor(play.sportsbook_deeplink, play.deeplinks_by_region)
      : "";

    return `
      <article class="prediction-card" data-direction="${CardVault.escapeAttr(direction)}" aria-label="Prediction for ${CardVault.escapeAttr(displayName)}, ${CardVault.escapeAttr(direction)} ${CardVault.escapeAttr(targetLabel)}">
        <header class="prediction-card__header">
          <span class="prediction-card__rank">${String(play.rank || index + 1)}</span>
          <div class="prediction-card__tags">${parlayTag}${CardVault.renderStatusPill(statusKey)}${CardVault.renderSettlementBadge(play)}${riskTags}</div>
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
    name = "", market = "", context = "", metrics = [], note = "", betslipUrl = "", deeplinksByRegion = null, settlementRow = null,
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
    // Resolved fresh at click time against the viewer's own state -- see
    // renderFanduelBetslipAnchor / initFanduelBetslipLinks.
    const betslipHtml = CardVault.renderFanduelBetslipAnchor(betslipUrl, deeplinksByRegion);

    return `
      <article class="prediction-card" aria-label="${CardVault.escapeAttr(`${name}, ${market}`)}">
        <header class="prediction-card__header">
          <span class="prediction-card__rank">${CardVault.escapeHtml(String(rank ?? ""))}</span>
          <div class="prediction-card__tags">${statusLabel ? CardVault.renderStatusPill(statusTone, statusLabel) : ""}${settlementRow ? CardVault.renderSettlementBadge(settlementRow) : ""}</div>
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
