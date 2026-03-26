// NBA Player Cards Web App

// Polyfill for roundRect (for older browsers)
if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, width, height, radius) {
        if (width < 2 * radius) radius = width / 2;
        if (height < 2 * radius) radius = height / 2;
        this.beginPath();
        this.moveTo(x + radius, y);
        this.arcTo(x + width, y, x + width, y + height, radius);
        this.arcTo(x + width, y + height, x, y + height, radius);
        this.arcTo(x, y + height, x, y, radius);
        this.arcTo(x, y, x + width, y, radius);
        this.closePath();
        return this;
    };
}

class PlayerCardsApp {
    constructor() {
        this.players = [];
        this.filteredPlayers = [];
        this.analyses = [];
        this.valuations = [];
        this.currentSort = 'name';
        this.initialDistributionMetricKey = 'team_fit';
        this.courtImage = null;
        this.distributionRaf = null;
        this.distributionPoints = [];
        this.distributionMetricConfig = null;
        this.teamFitScoreCache = null;
        this.teamFitPercentileCache = null;
        
        this.init();
    }

    async init() {
        await this.loadData();
        await this.loadCourtImage();
        this.setupEventListeners();
        this.populateArchetypeFilter();
        this.populateTeamFilter();
        const sortEl = document.getElementById('sortBy');
        if (sortEl) sortEl.value = this.currentSort;
        this.sortPlayers();
        this.renderPlayers();
        this.updateStats();
    }

    async loadCourtImage() {
        return new Promise((resolve) => {
            this.courtImage = new Image();
            this.courtImage.onload = () => resolve();
            this.courtImage.onerror = () => {
                console.warn('Court image not found, will draw without background');
                resolve();
            };
            this.courtImage.src = 'NBA-half-court.png';
        });
    }

    async loadData() {
        try {
            console.log('Starting to load data...');
            
            // Load player cards from data_sample format
            const cacheBust = `v=${Date.now()}`;
            const cardsResponse = await fetch(`data/cards.json?${cacheBust}`);
            console.log('Cards response status:', cardsResponse.status);
            
            if (!cardsResponse.ok) {
                throw new Error(`Failed to load cards: ${cardsResponse.status}`);
            }
            
            const cardsData = await cardsResponse.json();
            console.log('Cards loaded:', cardsData.length, 'players');
            
            // Handle both array and object formats
            this.players = Array.isArray(cardsData) ? cardsData : [cardsData];

            // Load valuations if available
            try {
                const valuationsResponse = await fetch(`data/valuations.json?${cacheBust}`);
                const valuationsData = await valuationsResponse.json();
                this.valuations = Array.isArray(valuationsData) ? valuationsData : [valuationsData];
                console.log('Valuations loaded:', this.valuations.length);
            } catch (e) {
                console.log('Valuations not available:', e);
                this.valuations = [];
            }

            // Merge valuation data into players and compute normalized value scores.
            this.mergePlayerData();
            this.computePlayerValueScores();
            this.teamProfiles = this.buildTeamProfiles();
            this.computeBreakoutScoreIndex();
            this.teamFitScoreCache = null;
            this.teamFitPercentileCache = null;
            this.filteredPlayers = [...this.players];
            
            console.log('Data loading complete. Total players:', this.players.length);
            document.getElementById('loading').style.display = 'none';
        } catch (error) {
            console.error('Error loading data:', error);
            document.getElementById('loading').innerHTML = `<p>Error loading player data: ${error.message}</p><p>Please ensure data files are available.</p>`;
        }
    }

    mergePlayerData() {
        this.players = this.players.map(player => {
            const playerName = player.player.name;
            
            // Find matching valuation
            const valuation = this.valuations.find(v => 
                v.player?.name === playerName
            );

            return {
                ...player,
                valuation: valuation || null
            };
        });
    }

    getPlayerValue(player) {
        return player.value_metrics?.player_value_score ?? 50;
    }

    computeCompDensity(player) {
        const comps = player.comparables?.similar_players || [];
        if (!comps.length) return 0.35;
        const top = comps.slice(0, 5).map(c => Number(c.similarity_score || 0));
        const avg = top.reduce((s, v) => s + v, 0) / Math.max(1, top.length);
        return this.clampNum(avg, 0, 1);
    }

    computeMechanismMetrics(player, evalOut = null) {
        const fitEval = evalOut || this.evaluateBreakoutScenario(player);
        const fit = fitEval.fitScenario || {};
        const sb = fit.score_breakdown || {};
        const shot = player.shot_profile || {};
        const trad = player.performance?.traditional || {};
        const creation = player.creation_profile || {};
        const trust = (player.v1_1_enhancements?.trust_assessment?.score || 60) / 100;
        const compDensity = this.computeCompDensity(player);

        const threeFreq = Number(shot.three_point_frequency || 0);
        const threePct = Number(trad.three_point_pct || 0);
        const assisted = Number(creation.assisted_rate || 0.55);
        const spacingGravitySkill = this.clampNum(
            (0.45 * this.clampNum(threeFreq / 0.55, 0, 1)) +
            (0.40 * this.clampNum((threePct - 0.30) / 0.14, 0, 1)) +
            (0.15 * this.clampNum(assisted / 0.8, 0, 1)),
            0,
            1
        );

        const suppressionRelief = this.clampNum(
            ((sb.scheme_complementarity || 0) * 0.55) +
            ((sb.role_fit || 0) * 0.30) +
            ((sb.usage_headroom || 0) * 0.15),
            0,
            1
        );

        const portabilityScore = this.clampNum(
            (0.45 * (sb.switch_fit || 0)) +
            (0.35 * (sb.defense_fit || 0)) +
            (0.20 * this.matchupVersatility(player)),
            0,
            1
        );

        const maxClamp = Math.max(1e-6, (fitEval.maxUsageIncrease || 0.1) + ((fitEval.maxMinutesIncrease || 8) / 100));
        const realizedClamp = (fitEval.usageDelta || 0) + ((fitEval.minutesDelta || 0) / 100);
        const clampSeverity = this.clampNum(1 - this.safeDiv(realizedClamp, maxClamp, 0), 0, 1);

        const epm = this.getRawEpm(player);
        const lebron = this.getRawLebron(player);
        const impactPriorDisagreement = (epm !== null && lebron !== null)
            ? this.clampNum(Math.abs(epm - lebron) / 3.0, 0, 1)
            : 0.35;

        const confidence = this.clampNum(
            (0.35 * (fit.confidence || 0.6)) +
            (0.30 * trust) +
            (0.20 * compDensity) +
            (0.15 * (1 - impactPriorDisagreement)),
            0.2,
            0.95
        );

        return {
            spacing_gravity_skill: spacingGravitySkill,
            suppression_relief: suppressionRelief,
            portability_score: portabilityScore,
            clamp_severity: clampSeverity,
            comp_density: compDensity,
            impact_prior_disagreement: impactPriorDisagreement,
            confidence
        };
    }

    buildCardScenarioSummary(player, evalOut, mechanism) {
        const fit = evalOut.fitScenario || {};
        const sb = fit.score_breakdown || {};
        const audit = evalOut.impactAudit || { verdict: 'insufficient_data' };
        const evidence = evalOut.evidence || { coverage: 0.7, grade: 'moderate' };
        const team = fit.team && fit.team !== 'No Clear Team Edge' ? fit.team : (player.player?.team || 'Current team');

        const driverPool = [
            { key: 'spacing gravity', v: mechanism.spacing_gravity_skill, text: 'spacing gravity creates weak-side defensive pull' },
            { key: 'suppression relief', v: mechanism.suppression_relief, text: 'context shift unlocks suppressed on-court value' },
            { key: 'portability', v: mechanism.portability_score, text: 'defensive role is portable across matchups' },
            { key: 'creation fit', v: sb.creation_fit || 0, text: 'on-ball creation fit fills team initiator gap' },
            { key: 'off-ball fit', v: sb.offball_fit || 0, text: 'off-ball profile scales next to primary creators' },
            { key: 'role fit', v: sb.role_fit || 0, text: 'role alignment raises lineup stability' },
            { key: 'usage headroom', v: sb.usage_headroom || 0, text: 'available usage headroom supports controlled expansion' }
        ].sort((a, b) => b.v - a.v);

        const topA = driverPool[0];
        const topB = driverPool[1];
        const topDrivers = [topA?.text, topB?.text].filter(Boolean).join('; ');

        const clampLabel = mechanism.clamp_severity <= 0.35
            ? 'light clamp pressure'
            : (mechanism.clamp_severity <= 0.6 ? 'controlled clamp pressure' : 'heavy clamp pressure');
        const priorLabel = mechanism.impact_prior_disagreement <= 0.35
            ? 'low prior disagreement'
            : (mechanism.impact_prior_disagreement <= 0.6 ? 'moderate prior disagreement' : 'high prior disagreement');
        const signalLabel = (fit.signal_strength || 'weak').toUpperCase();
        const auditLabel = String(audit.verdict || 'insufficient_data').replace('_', ' ');
        return `${team} scenario: ${topDrivers}. Signal ${signalLabel}; ${(mechanism.confidence * 100).toFixed(0)}% confidence with ${clampLabel} and ${priorLabel}. Audit ${auditLabel}; evidence ${(100 * (evidence.coverage || 0)).toFixed(0)}% (${String(evidence.grade || 'moderate').toUpperCase()}).`;
    }

    getPlayerKey(player) {
        const id = player.player?.id;
        if (id) return `id:${id}`;
        return `name:${player.player?.name || 'unknown'}|team:${player.player?.team || 'UNK'}`;
    }

    computeBreakoutScoreIndex() {
        const previous = this.breakoutScoreByKey;
        this.breakoutScoreByKey = null;

        const rows = this.players.map(p => {
            const out = this.evaluateBreakoutScenario(p);
            const raw = Number(out.breakoutScoreRaw ?? out.breakoutScore ?? 0);
            return { key: this.getPlayerKey(p), raw };
        });

        const sorted = rows.map(r => r.raw).sort((a, b) => a - b);
        const n = sorted.length || 1;
        const pctMap = {};

        for (const r of rows) {
            let lo = 0;
            let hi = n;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (sorted[mid] <= r.raw) lo = mid + 1;
                else hi = mid;
            }
            const rank = lo;
            pctMap[r.key] = this.clampNum((rank / n) * 100, 0, 100);
        }

        this.breakoutScoreByKey = pctMap;
        this.breakoutScoreRawByKey = Object.fromEntries(rows.map(r => [r.key, r.raw]));
        if (previous && Object.keys(previous).length && Object.keys(pctMap).length !== Object.keys(previous).length) {
            console.warn('Breakout index size changed:', Object.keys(previous).length, '->', Object.keys(pctMap).length);
        }
    }

    toFiniteNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    }

    getRawEpm(player) {
        return (
            this.toFiniteNumber(player.value_metrics?.epm) ??
            this.toFiniteNumber(player.performance?.advanced?.epm) ??
            this.toFiniteNumber(player.impact?.epm)
        );
    }

    getRawLebron(player) {
        return (
            this.toFiniteNumber(player.value_metrics?.lebron) ??
            this.toFiniteNumber(player.performance?.advanced?.lebron) ??
            this.toFiniteNumber(player.impact?.lebron)
        );
    }

    getNested(obj, path, fallback = null) {
        let cur = obj;
        for (const key of path) {
            if (cur == null || cur[key] === undefined || cur[key] === null) return fallback;
            cur = cur[key];
        }
        return cur;
    }

    getArchetypePeakAge(archetype) {
        const map = {
            lead_guard: 26.5,
            versatile_wing: 27.0,
            stretch_big: 28.0,
            rim_pressure_guard: 26.0,
            '3_and_d_wing': 27.5
        };
        return map[String(archetype || '').toLowerCase()] || 27.0;
    }

    getAgingAdjustmentSummary(player, valuation) {
        const age = Number(player.player?.age || 25);
        const archetype = String(player.identity?.primary_archetype || '');
        const peakAge = Number(valuation?.aging?.peak_age ?? this.getArchetypePeakAge(archetype));
        const mult = this.toFiniteNumber(valuation?.aging?.multipliers?.year_0);
        const multiplier = mult !== null ? mult : this.clampNum(1 - ((age - peakAge) * 0.018), 0.82, 1.12);
        const pct = (multiplier - 1) * 100;
        return {
            peak_age: peakAge,
            multiplier,
            pct,
            source: mult !== null ? 'valuation_curve' : 'archetype_proxy'
        };
    }

    getContractSurplusSummary(valuation) {
        if (!valuation) {
            return { available: false };
        }
        const npvSurplus = this.toFiniteNumber(valuation?.contract?.npv_surplus);
        const byYear = valuation?.contract?.surplus_by_year || {};
        const years = Object.keys(byYear).sort();
        const firstYear = years.length ? years[0] : null;
        const firstYearSurplus = firstYear ? this.toFiniteNumber(byYear[firstYear]) : null;
        const salaryByYear = valuation?.contract?.salary_by_year || {};
        const firstYearSalary = firstYear ? this.toFiniteNumber(salaryByYear[firstYear]) : null;
        const marketByYear = valuation?.market_value?.by_year || {};
        const firstYearMarket = firstYear ? this.toFiniteNumber(marketByYear[firstYear]) : null;
        return {
            available: npvSurplus !== null || firstYearSurplus !== null,
            npv_surplus: npvSurplus,
            current_year: firstYear,
            current_year_surplus: firstYearSurplus,
            current_year_salary: firstYearSalary,
            current_year_market: firstYearMarket
        };
    }

    inferDefenseRoleIdentity(player) {
        const position = String(player.player?.position || '').toLowerCase();
        const mp = player.defense_assessment?.matchup_profile || {};
        const guardShare = Number(mp.vs_guards || 0);
        const wingShare = Number(mp.vs_wings || 0);
        const bigShare = Number(mp.vs_bigs || 0);
        const totalShare = guardShare + wingShare + bigShare;
        const guardPct = totalShare > 0 ? guardShare / totalShare : 0.33;
        const wingPct = totalShare > 0 ? wingShare / totalShare : 0.34;
        const bigPct = totalShare > 0 ? bigShare / totalShare : 0.33;

        const steals = Number(player.performance?.traditional?.steals_per_game || 0);
        const blocks = Number(player.performance?.traditional?.blocks_per_game || 0);
        const foulRate = Number(player.defense_assessment?.estimated_metrics?.foul_rate || 0);
        const versatility = this.matchupVersatility(player);

        let primaryRole = 'Wing Containment';
        if (bigPct >= 0.46 || position === 'big') primaryRole = 'Rim/Interior';
        else if (guardPct >= 0.46 || position === 'guard') primaryRole = 'POA/Navigation';
        else if (wingPct >= 0.46 || position === 'wing') primaryRole = 'Wing Containment';

        const navigationRisk = this.clampNum((0.22 - guardPct) * 2.2 + 0.15, 0, 1);
        const sizeMismatchRisk = this.clampNum(Math.abs(guardPct - bigPct) + (1 - versatility) * 0.55, 0, 1);
        const foulRisk = this.clampNum((foulRate - 3.0) / 2.2, 0, 1);
        const disruption = this.clampNum((steals + blocks) / 2.3, 0, 1);
        const targetingRisk = this.clampNum(
            (0.40 * (1 - versatility)) +
            (0.25 * sizeMismatchRisk) +
            (0.20 * foulRisk) +
            (0.15 * navigationRisk) -
            (0.15 * disruption),
            0,
            1
        );

        const warnings = [];
        if (targetingRisk >= 0.62) warnings.push('High playoff targeting risk in switch-heavy possessions.');
        if (foulRisk >= 0.55) warnings.push('Foul pressure can compress high-minute defensive deployment.');
        if (sizeMismatchRisk >= 0.58) warnings.push('Cross-matchup size profile may require protection coverage.');
        if (!warnings.length) warnings.push('No major playoff targeting breakpoint from current profile proxies.');

        return {
            primary_role: primaryRole,
            role_mix: { guards: guardPct, wings: wingPct, bigs: bigPct },
            targeting_risk: targetingRisk,
            warnings
        };
    }

    computeImpactAudit(player, fitScenario, contributionStrength, breakoutSignal) {
        const epm = this.getRawEpm(player);
        const lebron = this.getRawLebron(player);
        const usable = [epm, lebron].filter(v => v !== null);
        if (!usable.length) {
            return {
                verdict: 'insufficient_data',
                consensus: null,
                disagreement: 1.0,
                required_justification: ['Missing EPM/LEBRON prior anchor values; scenario requires manual review.'],
                bonus: 0
            };
        }

        const consensus = usable.reduce((a, b) => a + b, 0) / usable.length;
        const disagreement = (epm !== null && lebron !== null)
            ? this.clampNum(Math.abs(epm - lebron) / 3.0, 0, 1)
            : 0.35;

        const fitStrength = this.clampNum((fitScenario?.fit_score || 0) / 0.10, 0, 1);
        const modelSignal = this.clampNum((0.5 * contributionStrength) + (0.3 * fitStrength) + (0.2 * breakoutSignal), 0, 1);
        const priorSignal = this.clampNum((consensus + 2.0) / 4.0, 0, 1);
        const signalGap = modelSignal - priorSignal;

        let verdict = 'support';
        if (Math.abs(signalGap) <= 0.18) verdict = 'support';
        else if (signalGap > 0.18 && consensus < -0.2) verdict = 'contradict';
        else if (signalGap < -0.18 && consensus > 0.2) verdict = 'contradict';
        else verdict = 'support';

        const requiredJustification = [];
        if (verdict === 'contradict') {
            requiredJustification.push('Model signal contradicts impact priors; verify role change/injury context before promotion.');
            requiredJustification.push('Confirm projected usage/minutes jump is observable in lineup context.');
        }
        if (disagreement >= 0.6) {
            requiredJustification.push('EPM and LEBRON disagreement is high; confidence should be discounted.');
        }

        const bonus = verdict === 'support' ? 2.5 : (verdict === 'contradict' ? -5.0 : -2.0);
        return {
            verdict,
            consensus,
            disagreement,
            required_justification: requiredJustification,
            bonus
        };
    }

    computeEvidenceQuality(player, valuation) {
        const checks = [
            { key: 'minutes_per_game', ok: this.toFiniteNumber(player.performance?.traditional?.minutes_per_game) !== null },
            { key: 'usage_rate', ok: this.toFiniteNumber(player.performance?.advanced?.usage_rate) !== null },
            { key: 'three_point_pct', ok: this.toFiniteNumber(player.performance?.traditional?.three_point_pct) !== null },
            { key: 'rim_frequency', ok: this.toFiniteNumber(player.shot_profile?.rim_frequency) !== null },
            { key: 'three_point_frequency', ok: this.toFiniteNumber(player.shot_profile?.three_point_frequency) !== null },
            { key: 'assisted_rate', ok: this.toFiniteNumber(player.creation_profile?.assisted_rate) !== null },
            { key: 'matchup_profile', ok: player.defense_assessment?.matchup_profile != null },
            { key: 'trust_score', ok: this.toFiniteNumber(player.v1_1_enhancements?.trust_assessment?.score) !== null },
            { key: 'epm_or_lebron', ok: this.getRawEpm(player) !== null || this.getRawLebron(player) !== null },
            { key: 'contract_surplus', ok: this.toFiniteNumber(valuation?.contract?.npv_surplus) !== null }
        ];
        const total = checks.length;
        const present = checks.filter(x => x.ok).length;
        const coverage = this.safeDiv(present, total, 0);
        const missing = checks.filter(x => !x.ok).map(x => x.key);
        const penalty = this.clampNum((0.82 - coverage) * 0.55, 0, 0.22);
        const grade = coverage >= 0.9 ? 'high' : (coverage >= 0.75 ? 'moderate' : 'limited');
        return { coverage, missing, penalty, grade };
    }

    computeIntrinsicContextLedger(player, fitScenario, mechanism, contributionStrength, usageUpside, minutesUpside, fitEdge) {
        const valueNorm = this.clampNum(this.getPlayerValue(player) / 100, 0, 1);
        const intrinsic = this.clampNum(
            (0.32 * valueNorm) +
            (0.18 * mechanism.spacing_gravity_skill) +
            (0.20 * mechanism.portability_score) +
            (0.12 * (1 - mechanism.impact_prior_disagreement)) +
            (0.18 * (1 - mechanism.clamp_severity)),
            0,
            1
        );
        const context = this.clampNum(
            (0.28 * fitEdge) +
            (0.24 * (fitScenario.score_breakdown?.scheme_complementarity || 0)) +
            (0.18 * (fitScenario.score_breakdown?.usage_headroom || 0)) +
            (0.14 * usageUpside) +
            (0.10 * minutesUpside) +
            (0.06 * contributionStrength),
            0,
            1
        );
        const projectedTotal = this.clampNum((0.52 * intrinsic) + (0.48 * context), 0, 1);
        const residual = this.clampNum(1 - projectedTotal, 0, 1);
        const contextShare = this.safeDiv(context, (intrinsic + context), 0.5);
        const intrinsicShare = this.safeDiv(intrinsic, (intrinsic + context), 0.5);
        return {
            intrinsic,
            context,
            residual,
            intrinsic_share: intrinsicShare,
            context_share: contextShare
        };
    }

    buildValueDecomposition(player, fitScenario) {
        const scheme = this.getPlayerSchemeVector(player).vector || {};
        const blocks = [
            { key: 'on_ball_creation', label: 'On-Ball Creation', value: scheme.on_ball_creation || 0 },
            { key: 'off_ball_play', label: 'Off-Ball Play', value: scheme.off_ball_play || 0 },
            { key: 'spacing_gravity', label: 'Spacing Gravity', value: scheme.spacing_gravity || 0 },
            { key: 'rim_pressure', label: 'Finishing Pressure', value: scheme.rim_pressure || 0 },
            { key: 'defensive_disruption', label: 'POA/Disruption', value: scheme.defensive_disruption || 0 },
            { key: 'switchability', label: 'Switchability', value: scheme.switchability || 0 },
            { key: 'role_scalability', label: 'Role Scalability', value: scheme.role_scalability || 0 },
            { key: 'possession_stability', label: 'Possession Stability', value: scheme.possession_stability || 0 }
        ];
        const sb = fitScenario?.score_breakdown || {};
        const teamWeights = {
            on_ball_creation: sb.creation_fit || 0,
            off_ball_play: sb.offball_fit || 0,
            spacing_gravity: sb.shooting_fit || 0,
            rim_pressure: sb.rim_fit || 0,
            defensive_disruption: sb.defense_fit || 0,
            switchability: sb.switch_fit || 0,
            role_scalability: sb.role_fit || 0,
            possession_stability: sb.minutes_fit || 0
        };
        const weighted = blocks.map(b => ({
            ...b,
            team_weight: teamWeights[b.key] || 0,
            team_weighted_value: (b.value || 0) * (teamWeights[b.key] || 0)
        }));
        const baseScore = weighted.reduce((s, b) => s + b.value, 0) / Math.max(1, weighted.length);
        const teamWeightedScore = weighted.reduce((s, b) => s + b.team_weighted_value, 0) / Math.max(1e-6, weighted.reduce((s, b) => s + b.team_weight, 0));
        return {
            blocks: weighted,
            base_score: this.clampNum(baseScore, 0, 1),
            team_weighted_score: this.clampNum(teamWeightedScore, 0, 1)
        };
    }

    buildClampReport(ctx) {
        const {
            currentUsage,
            projectedUsage,
            projectedUsageCap,
            currentMinutes,
            projectedMinutes,
            projectedMinutesCap,
            constraints,
            defenseRole,
            roleExpansionSeverity,
            evidencePenalty,
            minutesGovernance
        } = ctx;

        const fired = [];
        const usageGap = Math.max(0, projectedUsageCap - projectedUsage);
        const minutesGap = Math.max(0, projectedMinutesCap - projectedMinutes);
        const usageCapHit = projectedUsage >= (projectedUsageCap - 0.001);
        const minutesCapHit = projectedMinutes >= (projectedMinutesCap - 0.08);

        if (usageCapHit) {
            fired.push({
                clamp: 'usage_cap',
                severity: this.clampNum(1 - this.safeDiv(usageGap, Math.max(0.0001, projectedUsageCap - currentUsage), 0), 0, 1),
                message: `Usage increase bounded near cap (${(projectedUsageCap * 100).toFixed(1)}%).`,
                unlock: 'Increase feasible on-ball share or reduce incumbent usage concentration.'
            });
        }
        if (minutesCapHit) {
            fired.push({
                clamp: 'minutes_cap',
                severity: this.clampNum(1 - this.safeDiv(minutesGap, Math.max(0.01, projectedMinutesCap - currentMinutes), 0), 0, 1),
                message: `Minutes increase bounded near cap (${projectedMinutesCap.toFixed(1)} MPG).`,
                unlock: 'Create rotation runway through role continuity and defensive survivability.'
            });
        }

        const shotConstraints = constraints?.shot_diet_constraints || [];
        if (shotConstraints.length) {
            fired.push({
                clamp: 'shot_mix_cap',
                severity: this.clampNum(0.35 + (shotConstraints.length / 10), 0.25, 0.75),
                message: `Shot diet constraints active (${shotConstraints.length} rules).`,
                unlock: 'Demonstrate in-role shot growth without violating diet constraints.'
            });
        }

        if ((defenseRole?.targeting_risk || 0) > 0.62 && roleExpansionSeverity > 0.52) {
            fired.push({
                clamp: 'defense_portability_cap',
                severity: this.clampNum(defenseRole.targeting_risk || 0, 0, 1),
                message: `Defensive targeting risk ${(100 * (defenseRole.targeting_risk || 0)).toFixed(0)}% constrained offensive expansion.`,
                unlock: 'Improve lineup protection, matchup insulation, and foul/size stability.'
            });
        }

        if (evidencePenalty > 0.04) {
            fired.push({
                clamp: 'evidence_quality_cap',
                severity: this.clampNum(evidencePenalty / 0.22, 0, 1),
                message: `Evidence quality penalty ${(evidencePenalty * 100).toFixed(1)}% reduced confidence ceiling.`,
                unlock: 'Increase observability coverage for missing priors/profile fields.'
            });
        }

        if ((minutesGovernance?.requires_rotation_promotion || false) || (minutesGovernance?.status === 'fail')) {
            fired.push({
                clamp: 'rotation_promotion_cap',
                severity: this.clampNum(minutesGovernance?.feasibility_penalty || 0.1, 0, 1),
                message: minutesGovernance?.reason || 'Low-minute baseline requires role promotion before validating breakout.',
                unlock: 'Sustain stable rotation role over larger sample and reduce promotion dependency.'
            });
        }

        return fired;
    }

    computeMinutesGovernance(ctx) {
        const {
            player,
            constraints,
            currentMinutes,
            projectedMinutes,
            minutesDelta,
            trustScore,
            mechanism,
            impactAudit
        } = ctx;
        const gp = Number(player.performance?.traditional?.games_played || 0);
        const mpg = Number(currentMinutes || 0);
        const constraintNotes = constraints?.constraints || [];
        const noteText = constraintNotes.join(' ').toLowerCase();
        const injuryFlag = /(injury|return|rehab|recovery|coming back|minutes restriction)/i.test(noteText);

        let status = mpg >= 16 ? 'pass' : (mpg >= 12 ? 'watch' : 'fail');
        if (gp < 25) {
            status = status === 'pass' ? 'watch' : 'fail';
        }
        if (status === 'fail' && injuryFlag) {
            status = 'watch';
        }

        let evidencePenalty = status === 'fail' ? 0.20 : (status === 'watch' ? 0.10 : 0.02);
        if (gp < 25) evidencePenalty += 0.06;
        evidencePenalty = this.clampNum(evidencePenalty, 0.02, 0.30);

        const requiresRotationPromotion =
            (mpg < 12 && projectedMinutes >= 18) ||
            (mpg < 16 && minutesDelta > 4.5) ||
            minutesDelta > 6.0;
        let feasibilityPenalty = requiresRotationPromotion
            ? this.clampNum((Math.max(0, minutesDelta - 3.5) / 8.0) + (Math.max(0, 16 - mpg) / 26.0), 0.06, 0.26)
            : 0;
        if (status === 'fail') feasibilityPenalty = Math.max(feasibilityPenalty, 0.09);

        // Visible-breakout carve-out.
        let visibleStatus = status;
        const priorAgree = impactAudit?.verdict === 'support' && (impactAudit?.disagreement ?? 1) <= 0.35;
        const lowClampPressure = ((mechanism?.clamp_severity || 1) + feasibilityPenalty) <= 0.42;
        if (mpg >= 12 && mpg < 16 && trustScore >= 75 && priorAgree && lowClampPressure) {
            visibleStatus = 'pass';
        }
        if (mpg < 12 && !injuryFlag) {
            visibleStatus = 'fail';
        }
        if (gp < 25 && visibleStatus === 'pass') {
            visibleStatus = 'watch';
        }

        const reason = visibleStatus === 'pass'
            ? `Minutes baseline supports visible validation (${mpg.toFixed(1)} MPG, ${gp} GP).`
            : (visibleStatus === 'watch'
                ? `Minutes are in watch zone (${mpg.toFixed(1)} MPG, ${gp} GP); scenario needs controlled promotion.`
                : `MPG/GP baseline too low for visible validation (${mpg.toFixed(1)} MPG, ${gp} GP).`);

        return {
            status,
            visible_status: visibleStatus,
            reason,
            games_played: gp,
            mpg,
            injury_flag: injuryFlag,
            evidence_penalty: evidencePenalty,
            evidence_score: this.clampNum(1 - evidencePenalty, 0, 1),
            feasibility_penalty: feasibilityPenalty,
            requires_rotation_promotion: requiresRotationPromotion
        };
    }

    computeMeanStd(values) {
        if (!values.length) return { mean: 0, std: 1 };
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((acc, v) => acc + ((v - mean) ** 2), 0) / values.length;
        const std = Math.sqrt(variance);
        return { mean, std: std > 1e-9 ? std : 1 };
    }

    computePlayerValueScores() {
        const epmVals = [];
        const lebronVals = [];

        for (const player of this.players) {
            const epm = this.getRawEpm(player);
            const lebron = this.getRawLebron(player);
            if (epm !== null) epmVals.push(epm);
            if (lebron !== null) lebronVals.push(lebron);
        }

        const epmStats = this.computeMeanStd(epmVals);
        const lebronStats = this.computeMeanStd(lebronVals);

        this.players = this.players.map(player => {
            const epm = this.getRawEpm(player);
            const lebron = this.getRawLebron(player);
            const zParts = [];

            if (epm !== null) zParts.push((epm - epmStats.mean) / epmStats.std);
            if (lebron !== null) zParts.push((lebron - lebronStats.mean) / lebronStats.std);

            let source = 'fallback';
            if (epm !== null && lebron !== null) source = 'epm_lebron';
            else if (epm !== null) source = 'epm_only';
            else if (lebron !== null) source = 'lebron_only';

            const compositeZ = zParts.length ? (zParts.reduce((a, b) => a + b, 0) / zParts.length) : 0;
            const valueScore = Math.max(0, Math.min(100, 50 + 15 * compositeZ));

            return {
                ...player,
                value_metrics: {
                    ...(player.value_metrics || {}),
                    epm_raw: epm,
                    lebron_raw: lebron,
                    player_value_score: Math.round(valueScore * 10) / 10,
                    player_value_source: source
                }
            };
        });
    }

    clampNum(x, lo, hi) {
        return Math.max(lo, Math.min(hi, x));
    }

    safeDiv(a, b, fallback = 0) {
        return b ? (a / b) : fallback;
    }

    weightedAverage(arr, valueFn, weightFn) {
        if (!arr.length) return 0;
        let numer = 0;
        let denom = 0;
        for (const item of arr) {
            const w = Math.max(0, weightFn(item));
            numer += valueFn(item) * w;
            denom += w;
        }
        return this.safeDiv(numer, denom, 0);
    }

    matchupVersatility(player) {
        const mp = player.defense_assessment?.matchup_profile || {};
        const g = Number(mp.vs_guards || 0);
        const w = Number(mp.vs_wings || 0);
        const b = Number(mp.vs_bigs || 0);
        const t = g + w + b;
        if (t <= 0) return 0.45;
        const probs = [g / t, w / t, b / t];
        let entropy = 0;
        for (const p of probs) {
            if (p > 0) entropy += -p * Math.log(p);
        }
        return this.clampNum(entropy / Math.log(3), 0, 1);
    }

    getPlayerSchemeVector(player) {
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const shot = player.shot_profile || {};
        const creation = player.creation_profile || {};
        const trust = player.v1_1_enhancements?.trust_assessment?.score || 60;
        const obs = player.defense_assessment?.visibility?.observability_score || 30;

        const usage = adv.usage_rate ?? 0.20;
        const ast = trad.assists_per_game || 0;
        const stocks = (trad.steals_per_game || 0) + (trad.blocks_per_game || 0);
        const mpg = trad.minutes_per_game || 0;
        const threePct = trad.three_point_pct || 0;
        const threeFreq = shot.three_point_frequency || 0.34;
        const rimFreq = shot.rim_frequency || 0.30;
        const drives = creation.drives_per_game || 0;
        const paintTouches = creation.paint_touches_per_game || 0;
        const assisted = creation.assisted_rate || 0.55;
        const iso = creation.isolation_frequency || 0;
        const pnr = creation.pick_and_roll_frequency || 0;
        const versatility = this.matchupVersatility(player);

        // Abstract skill dimensions (0-1).
        const vector = {
            on_ball_creation: this.clampNum((0.45 * this.clampNum(ast / 6.0, 0, 1)) + (0.35 * this.clampNum((pnr + iso) / 12.0, 0, 1)) + (0.20 * this.clampNum((usage - 0.16) / 0.14, 0, 1)), 0, 1),
            rim_pressure: this.clampNum((0.45 * this.clampNum(drives / 16.0, 0, 1)) + (0.30 * this.clampNum(paintTouches / 9.0, 0, 1)) + (0.25 * this.clampNum(rimFreq / 0.45, 0, 1)), 0, 1),
            spacing_gravity: this.clampNum((0.55 * this.clampNum(threeFreq / 0.55, 0, 1)) + (0.45 * this.clampNum((threePct - 0.30) / 0.12, 0, 1)), 0, 1),
            off_ball_play: this.clampNum((0.55 * this.clampNum(assisted, 0, 1)) + (0.25 * this.clampNum(threeFreq / 0.5, 0, 1)) + (0.20 * this.clampNum(rimFreq / 0.45, 0, 1)), 0, 1),
            defensive_disruption: this.clampNum((0.70 * this.clampNum(stocks / 2.2, 0, 1)) + (0.30 * this.clampNum(mpg / 32.0, 0, 1)), 0, 1),
            switchability: this.clampNum((0.65 * versatility) + (0.35 * this.clampNum(mpg / 30.0, 0, 1)), 0, 1),
            role_scalability: this.clampNum((0.40 * this.clampNum(assisted, 0, 1)) + (0.35 * this.clampNum(threeFreq / 0.55, 0, 1)) + (0.25 * this.clampNum((0.28 - usage) / 0.16, 0, 1)), 0, 1),
            possession_stability: this.clampNum((0.60 * this.clampNum(mpg / 30.0, 0, 1)) + (0.40 * this.clampNum(trust / 100.0, 0, 1)), 0, 1)
        };

        const confidence = this.clampNum((0.5 * (trust / 100.0)) + (0.3 * this.clampNum(mpg / 24.0, 0, 1)) + (0.2 * this.clampNum(obs / 100.0, 0, 1)), 0.45, 1.0);

        return { vector, confidence };
    }

    buildTeamProfiles() {
        const byTeam = new Map();
        for (const p of this.players) {
            const team = p.player?.team;
            if (!team) continue;
            if (!byTeam.has(team)) byTeam.set(team, []);
            byTeam.get(team).push(p);
        }

        const profiles = {};
        const teamRows = [];
        for (const [team, roster] of byTeam.entries()) {
            const rotation = roster.filter(p => (p.performance?.traditional?.minutes_per_game || 0) >= 18);
            const active = rotation.length ? rotation : roster;
            const playerSchemes = active.map(p => ({
                p,
                ...this.getPlayerSchemeVector(p),
                mpg: p.performance?.traditional?.minutes_per_game || 0
            }));

            const dimNames = [
                'on_ball_creation',
                'rim_pressure',
                'spacing_gravity',
                'off_ball_play',
                'defensive_disruption',
                'switchability',
                'role_scalability',
                'possession_stability'
            ];

            const scheme = {};
            for (const dim of dimNames) {
                scheme[dim] = this.weightedAverage(
                    playerSchemes,
                    x => x.vector[dim] || 0,
                    x => Math.max(8, x.mpg) * x.confidence
                );
            }

            const usageVals = active.map(p => p.performance?.advanced?.usage_rate ?? 0.2);
            const usageMean = usageVals.length ? usageVals.reduce((a, b) => a + b, 0) / usageVals.length : 0.2;
            const usageVar = usageVals.length ? usageVals.reduce((acc, v) => acc + ((v - usageMean) ** 2), 0) / usageVals.length : 0;

            const profile = {
                roster_size: roster.length,
                rotation_size: active.length,
                avg_usage: usageMean,
                usage_concentration: this.clampNum(Math.sqrt(usageVar) / 0.08, 0, 1),
                scheme
            };
            profiles[team] = profile;
            teamRows.push(profile);
        }

        const statKeys = ['rotation_size', 'avg_usage', 'usage_concentration'];
        const dimKeys = ['on_ball_creation', 'rim_pressure', 'spacing_gravity', 'off_ball_play', 'defensive_disruption', 'switchability', 'role_scalability', 'possession_stability'];

        this.teamFitBaselines = {};
        for (const key of statKeys) {
            const vals = teamRows.map(r => Number(r[key] || 0));
            this.teamFitBaselines[key] = this.computeMeanStd(vals);
        }
        for (const key of dimKeys) {
            const vals = teamRows.map(r => Number(r.scheme?.[key] || 0));
            this.teamFitBaselines[`scheme_${key}`] = this.computeMeanStd(vals);
        }

        // Team attractor bias: generic destination magnetism across the whole player pool.
        const playersScheme = this.players.map(p => this.getPlayerSchemeVector(p).vector);
        const fitWeight = {
            on_ball_creation: 0.20,
            rim_pressure: 0.12,
            spacing_gravity: 0.16,
            off_ball_play: 0.14,
            defensive_disruption: 0.14,
            switchability: 0.12,
            role_scalability: 0.12
        };
        const gapFromScheme = (teamScheme, dim) => this.clampNum((0.62 - (teamScheme?.[dim] || 0)) / 0.62, 0, 1);
        const attractorRaw = {};
        for (const [team, profile] of Object.entries(profiles)) {
            const teamScheme = profile.scheme || {};
            let sum = 0;
            for (const pv of playersScheme) {
                let v = 0;
                for (const [dim, w] of Object.entries(fitWeight)) {
                    v += w * gapFromScheme(teamScheme, dim) * (pv[dim] || 0);
                }
                sum += v;
            }
            attractorRaw[team] = playersScheme.length ? (sum / playersScheme.length) : 0;
        }
        const attractorVals = Object.values(attractorRaw);
        const attractorMean = attractorVals.reduce((a, b) => a + b, 0) / Math.max(1, attractorVals.length);
        const attractorStd = Math.sqrt(
            attractorVals.reduce((a, b) => a + ((b - attractorMean) ** 2), 0) / Math.max(1, attractorVals.length)
        ) || 1;
        for (const [team, raw] of Object.entries(attractorRaw)) {
            profiles[team].attractor_bias = this.clampNum((raw - attractorMean) / attractorStd, -2.5, 2.5);
        }

        return profiles;
    }

    getBestBreakoutFit(player) {
        const currentTeam = player.player?.team;
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const pos = String(player.player?.position || '').toLowerCase();
        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 60;
        const playerMpg = trad.minutes_per_game || 0;
        const playerUsage = adv.usage_rate ?? 0.2;
        const playerScheme = this.getPlayerSchemeVector(player);

        const baseline = this.teamFitBaselines || {};
        const zToNeed = (value, key) => {
            const stats = baseline[key] || { mean: 0, std: 1 };
            const z = (value - stats.mean) / (stats.std || 1);
            return this.clampNum((-z + 0.5) / 2.5, 0, 1);
        };

        let best = null;
        const scenarios = [];
        for (const [team, profile] of Object.entries(this.teamProfiles || {})) {
            if (!team) continue;

            const teamScheme = profile.scheme || {};
            const usageHeadroom = this.clampNum((0.245 - profile.avg_usage + (0.26 - playerUsage)) / 0.18, 0, 1);
            const minutesNeed = zToNeed(profile.rotation_size, 'rotation_size');
            const concentrationNeed = this.clampNum(0.65 - profile.usage_concentration, 0, 1);

            const gap = (dim) => this.clampNum((0.62 - (teamScheme[dim] || 0)) / 0.62, 0, 1);
            const fit = (dim) => playerScheme.vector[dim] || 0;

            const creationNeed = gap('on_ball_creation');
            const rimNeed = gap('rim_pressure');
            const spacingNeed = gap('spacing_gravity');
            const offBallNeed = gap('off_ball_play');
            const defenseNeed = gap('defensive_disruption');
            const switchNeed = gap('switchability');
            const scaleNeed = gap('role_scalability');
            const needs = [creationNeed, rimNeed, spacingNeed, offBallNeed, defenseNeed, switchNeed, scaleNeed];
            const needMean = needs.reduce((a, b) => a + b, 0) / needs.length;
            const needVar = needs.reduce((a, b) => a + ((b - needMean) ** 2), 0) / needs.length;
            const needStd = Math.sqrt(needVar);
            const genericNeedBreadth = (creationNeed + rimNeed + spacingNeed + offBallNeed + defenseNeed + switchNeed + scaleNeed) / 7.0;

            const creationFit = creationNeed * fit('on_ball_creation');
            const rimFit = rimNeed * fit('rim_pressure');
            const spacingFit = spacingNeed * fit('spacing_gravity');
            const offBallFit = offBallNeed * fit('off_ball_play');
            const defenseFit = defenseNeed * fit('defensive_disruption');
            const switchFit = switchNeed * fit('switchability');
            const scaleFit = scaleNeed * fit('role_scalability');

            const roleFit = pos.includes('wing')
                ? this.clampNum((0.4 * spacingFit) + (0.3 * switchFit) + (0.3 * offBallFit) + 0.2, 0, 1)
                : this.clampNum((0.45 * creationFit) + (0.25 * scaleFit) + (0.30 * offBallFit) + 0.15, 0, 1);

            const minutesOpportunity = this.clampNum((0.60 * minutesNeed) + (0.20 * concentrationNeed) + (0.20 * roleFit), 0, 1);

            const rawFit =
                0.18 * creationFit +
                0.12 * rimFit +
                0.14 * spacingFit +
                0.12 * offBallFit +
                0.12 * defenseFit +
                0.10 * switchFit +
                0.08 * scaleFit +
                0.14 * minutesOpportunity;

            const rosterConfidence = this.clampNum(profile.rotation_size / 8.0, 0.55, 1.0);
            const sampleConfidence = this.clampNum(playerMpg / 20.0, 0.55, 1.0);
            const dataConfidence = this.clampNum((trustScore / 100.0) * playerScheme.confidence, 0.5, 1.0);
            const confidence = this.clampNum((rosterConfidence * 0.30) + (sampleConfidence * 0.25) + (dataConfidence * 0.45), 0.5, 1.0);
            // Penalize teams that "need everything" since that can create generic false positives.
            const genericPenalty = this.clampNum((genericNeedBreadth - 0.45) * 0.45, 0, 0.20);
            // Penalize universal-absorber profiles (high average need + low specificity).
            const specializationPenalty = this.clampNum(((needMean - 0.36) * 0.35) + ((0.18 - needStd) * 0.40), 0, 0.18);
            const attractorPenalty = this.clampNum((profile.attractor_bias || 0) * 0.045, 0, 0.10);
            let fitScore = this.clampNum((rawFit * confidence) - genericPenalty - specializationPenalty - attractorPenalty, 0, 1);
            if (team === currentTeam) {
                // Continuity bonus to avoid noisy external-team over-selection in weak-signal cases.
                fitScore = this.clampNum(fitScore + 0.01, 0, 1);
            } else {
                // Small relocation friction for external scenarios.
                fitScore = this.clampNum(fitScore - 0.015, 0, 1);
            }

            const reasons = [];
            if (minutesOpportunity > 0.42) reasons.push(`Rotation runway and role scalability create stable minutes upside`);
            if (creationFit > 0.10) reasons.push(`On-ball creation gap match: player creation profile fills tactical initiator need`);
            if (spacingFit > 0.09) reasons.push(`Spacing gravity fit: team scheme lacks shooting gravity this player provides`);
            if (offBallFit > 0.09) reasons.push(`Off-ball scheme fit: assisted/off-ball profile aligns with team possession design`);
            if (defenseFit + switchFit > 0.16) reasons.push(`Defensive scheme compatibility: disruption + switchability improve lineup portability`);
            if (usageHeadroom > 0.38) reasons.push(`Usage headroom is available without overloading existing high-usage creators`);

            const scenario = {
                team,
                fit_score: fitScore,
                minutes_opportunity: minutesOpportunity,
                usage_opportunity: this.clampNum((creationFit * 0.45) + (usageHeadroom * 0.55), 0, 1),
                confidence,
                score_breakdown: {
                    minutes_fit: minutesOpportunity,
                    usage_headroom: usageHeadroom,
                    shooting_fit: spacingFit,
                    creation_fit: creationFit,
                    rim_fit: rimFit,
                    defense_fit: defenseFit,
                    switch_fit: switchFit,
                    offball_fit: offBallFit,
                    role_fit: roleFit,
                    scheme_complementarity: this.clampNum((creationFit + spacingFit + offBallFit + defenseFit + switchFit) / 5.0, 0, 1),
                    generic_need_penalty: genericPenalty,
                    specialization_penalty: specializationPenalty,
                    attractor_penalty: attractorPenalty
                },
                reasons: reasons.slice(0, 4)
            };
            scenarios.push(scenario);

            if (!best || scenario.fit_score > best.fit_score) best = scenario;
        }

        if (best) {
            const ranked = scenarios.sort((a, b) => b.fit_score - a.fit_score);
            const second = ranked[1] || ranked[0];
            const gap = best.fit_score - (second?.fit_score || 0);
            const weakSignal = best.fit_score < 0.05 || gap < 0.004;

            // In weak-signal regimes, avoid overconfident destination claims.
            if (weakSignal) {
                best = { ...best, team: 'No Clear Team Edge', is_no_clear_edge: true };
                best.reasons = [
                    'No strong external fit edge; treat destination as exploratory',
                    ...best.reasons
                ].slice(0, 4);
            }
            best.fit_gap = gap;
            best.signal_strength = weakSignal ? 'weak' : ((best.fit_score >= 0.08 && gap >= 0.01) ? 'strong' : 'moderate');
            best.alternatives = ranked
                .slice(0, 3)
                .map(s => ({ team: s.team, fit_score: s.fit_score }));
            return best;
        }

        return {
            team: currentTeam,
            fit_score: 0.5,
            minutes_opportunity: 0.4,
            usage_opportunity: 0.4,
            confidence: 0.6,
            score_breakdown: {
                minutes_fit: 0.4,
                usage_headroom: 0.4,
                shooting_fit: 0.4,
                creation_fit: 0.4,
                rim_fit: 0.4,
                defense_fit: 0.4,
                switch_fit: 0.4,
                offball_fit: 0.4,
                role_fit: 0.5,
                scheme_complementarity: 0.4,
                generic_need_penalty: 0
            },
            fit_gap: 0,
            signal_strength: 'weak',
            alternatives: [{ team: currentTeam, fit_score: 0.5 }],
            reasons: ['Role continuity scenario on current team']
        };
    }

    evaluateBreakoutScenario(player) {
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const constraints = player.v1_1_enhancements?.scenario_constraints || {};
        const fitScenario = this.getBestBreakoutFit(player);

        const age = player.player?.age || 25;
        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 50;
        const currentUsage = adv.usage_rate ?? 0.2;
        const currentMinutes = trad.minutes_per_game || 25;
        const currentPPG = trad.points_per_game || 10;
        const currentAPG = trad.assists_per_game || 2;
        const currentRPG = trad.rebounds_per_game || 5;

        const maxUsageIncrease = constraints?.feasible_changes?.max_usage_increase || 0.1;
        const maxMinutesIncrease = constraints?.feasible_changes?.max_minutes_increase || 5;
        const projectedUsageCap = constraints?.feasible_changes?.projected_usage_cap ?? (currentUsage + maxUsageIncrease);
        const projectedMinutesCap = constraints?.feasible_changes?.projected_minutes_cap ?? (currentMinutes + maxMinutesIncrease);

        const usageGainFactor = this.clampNum(
            0.35 +
            (fitScenario.score_breakdown?.usage_headroom || 0.4) * 0.40 +
            (fitScenario.score_breakdown?.creation_fit || 0.4) * 0.20 +
            (fitScenario.confidence || 0.6) * 0.20,
            0.25,
            1.0
        );
        const minutesGainFactor = this.clampNum(
            0.30 +
            (fitScenario.score_breakdown?.minutes_fit || 0.4) * 0.45 +
            (fitScenario.score_breakdown?.role_fit || 0.5) * 0.15 +
            (fitScenario.confidence || 0.6) * 0.20,
            0.25,
            1.0
        );
        const signalMultiplier = fitScenario.signal_strength === 'strong' ? 1.0 : (fitScenario.signal_strength === 'moderate' ? 0.85 : 0.70);
        let projectedUsage = Math.min(projectedUsageCap, currentUsage + (maxUsageIncrease * usageGainFactor * signalMultiplier));
        let projectedMinutes = Math.min(projectedMinutesCap, currentMinutes + (maxMinutesIncrease * minutesGainFactor * signalMultiplier));
        let usageDelta = Math.max(0, projectedUsage - currentUsage);
        let minutesDelta = Math.max(0, projectedMinutes - currentMinutes);
        const breakdown = fitScenario.score_breakdown || {};

        const usageMultiplier = currentUsage > 0 ? (projectedUsage / currentUsage) : 1.0;
        const minutesMultiplier = currentMinutes > 0 ? (projectedMinutes / currentMinutes) : 1.0;
        const overallMultiplier = Math.min(usageMultiplier * 0.7 + minutesMultiplier * 0.3, 1.5);

        let projectedPPG = currentPPG * overallMultiplier;
        let projectedAPG = currentAPG * overallMultiplier;
        let projectedRPG = currentRPG * Math.min(minutesMultiplier, 1.3);

        const ageCurve = this.clampNum(1 - Math.abs(age - 25) / 9, 0, 1);
        const usageUpside = this.clampNum(usageDelta / 0.10, 0, 1);
        const minutesUpside = this.clampNum(minutesDelta / 8.0, 0, 1);
        const fitEdge = this.clampNum((fitScenario.fit_score - 0.12) / 0.28, 0, 1);
        const trustNorm = this.clampNum(trustScore / 100, 0, 1);
        let mechanism = this.computeMechanismMetrics(player, {
            fitScenario,
            usageDelta,
            minutesDelta,
            maxUsageIncrease,
            maxMinutesIncrease
        });
        const defenseRole = this.inferDefenseRoleIdentity(player);

        // Defensive feasibility gate: high targeting risk reduces offensive role expansion realism.
        const roleExpansionSeverity = this.clampNum((usageDelta / 0.10) + (minutesDelta / 8.0), 0, 1);
        if (defenseRole.targeting_risk > 0.62 && roleExpansionSeverity > 0.52) {
            const offenseClamp = this.clampNum(1 - ((defenseRole.targeting_risk - 0.62) * 0.45), 0.78, 1.0);
            const minutesClamp = this.clampNum(1 - ((defenseRole.targeting_risk - 0.62) * 0.35), 0.84, 1.0);
            projectedUsage = currentUsage + (usageDelta * offenseClamp);
            projectedMinutes = currentMinutes + (minutesDelta * minutesClamp);
            usageDelta = Math.max(0, projectedUsage - currentUsage);
            minutesDelta = Math.max(0, projectedMinutes - currentMinutes);
            const usageMultAdj = currentUsage > 0 ? (projectedUsage / currentUsage) : 1.0;
            const minutesMultAdj = currentMinutes > 0 ? (projectedMinutes / currentMinutes) : 1.0;
            const overallAdj = Math.min(usageMultAdj * 0.7 + minutesMultAdj * 0.3, 1.5);
            projectedPPG = currentPPG * overallAdj;
            projectedAPG = currentAPG * overallAdj;
            projectedRPG = currentRPG * Math.min(minutesMultAdj, 1.3);
            mechanism = this.computeMechanismMetrics(player, {
                fitScenario,
                usageDelta,
                minutesDelta,
                maxUsageIncrease,
                maxMinutesIncrease
            });
        }

        // Team contribution channels (x, y, z) used to validate genuine breakout signals.
        const creationLift = this.clampNum(
            (0.55 * (breakdown.creation_fit || 0)) +
            (0.30 * (breakdown.usage_headroom || 0)) +
            (0.15 * usageUpside),
            0,
            1
        );
        const spacingLift = this.clampNum(
            (0.50 * (breakdown.shooting_fit || 0)) +
            (0.30 * (breakdown.offball_fit || 0)) +
            (0.20 * mechanism.spacing_gravity_skill),
            0,
            1
        );
        const defenseLift = this.clampNum(
            (0.45 * (breakdown.defense_fit || 0)) +
            (0.35 * (breakdown.switch_fit || 0)) +
            (0.20 * mechanism.portability_score),
            0,
            1
        );
        const roleStabilityLift = this.clampNum(
            (0.60 * (breakdown.role_fit || 0)) +
            (0.25 * (breakdown.minutes_fit || 0)) +
            (0.15 * (1 - mechanism.clamp_severity)),
            0,
            1
        );
        const contributionStrength = this.clampNum(
            (0.30 * creationLift) +
            (0.25 * spacingLift) +
            (0.25 * defenseLift) +
            (0.20 * roleStabilityLift),
            0,
            1
        );
        const contributionProfile = [
            { key: 'creation', label: 'Creation Lift', value: creationLift, note: 'on-ball creation + usage headroom conversion' },
            { key: 'spacing', label: 'Spacing Lift', value: spacingLift, note: 'shot gravity + off-ball scheme translation' },
            { key: 'defense', label: 'Defense Lift', value: defenseLift, note: 'defensive disruption + switch portability' },
            { key: 'stability', label: 'Role Stability', value: roleStabilityLift, note: 'minutes feasibility + role continuity' }
        ].sort((a, b) => b.value - a.value);
        const intrinsicContextLedger = this.computeIntrinsicContextLedger(
            player,
            fitScenario,
            mechanism,
            contributionStrength,
            usageUpside,
            minutesUpside,
            fitEdge
        );
        const valueDecomposition = this.buildValueDecomposition(player, fitScenario);
        const breakoutSignalNorm = this.clampNum(
            (0.45 * contributionStrength) + (0.30 * fitEdge) + (0.25 * (1 - mechanism.clamp_severity)),
            0,
            1
        );
        const impactAudit = this.computeImpactAudit(player, fitScenario, contributionStrength, breakoutSignalNorm);
        const evidence = this.computeEvidenceQuality(player, player.valuation);
        const minutesGovernance = this.computeMinutesGovernance({
            player,
            constraints,
            currentMinutes,
            projectedMinutes,
            minutesDelta,
            trustScore,
            mechanism,
            impactAudit
        });
        const evidencePenaltyTotal = this.clampNum((evidence.penalty || 0) + (minutesGovernance.evidence_penalty || 0), 0, 0.38);
        const effectiveEvidenceCoverage = this.clampNum((evidence.coverage || 0) - (minutesGovernance.evidence_penalty || 0) * 0.55, 0, 1);
        const agingSummary = this.getAgingAdjustmentSummary(player, player.valuation);
        const contractSummary = this.getContractSurplusSummary(player.valuation);
        const clampReport = this.buildClampReport({
            currentUsage,
            projectedUsage,
            projectedUsageCap,
            currentMinutes,
            projectedMinutes,
            projectedMinutesCap,
            constraints,
            defenseRole,
            roleExpansionSeverity,
            evidencePenalty: evidencePenaltyTotal,
            minutesGovernance
        });

        let breakoutScoreRaw = 100 * (
            0.18 * ageCurve +
            0.12 * usageUpside +
            0.10 * minutesUpside +
            0.20 * fitEdge +
            0.22 * contributionStrength +
            0.10 * mechanism.suppression_relief +
            0.08 * mechanism.portability_score
        );
        // Confidence governance penalties to prevent "usage-only" breakouts.
        breakoutScoreRaw -= (8 * mechanism.clamp_severity);
        breakoutScoreRaw -= (6 * mechanism.impact_prior_disagreement);
        breakoutScoreRaw -= (10 * evidencePenaltyTotal);
        breakoutScoreRaw -= (7 * (minutesGovernance.feasibility_penalty || 0));
        breakoutScoreRaw += impactAudit.bonus;
        if (contributionStrength < 0.36) breakoutScoreRaw -= 10;
        if (usageUpside > 0.65 && contributionStrength < 0.45) breakoutScoreRaw -= 10;
        if (fitScenario.signal_strength === 'weak') breakoutScoreRaw -= 8;
        breakoutScoreRaw += (3 * trustNorm);
        breakoutScoreRaw = this.clampNum(breakoutScoreRaw, 0, 100);

        const key = this.getPlayerKey(player);
        const relativeScore = this.breakoutScoreByKey?.[key] ?? breakoutScoreRaw;
        let breakoutScore = this.clampNum((0.72 * breakoutScoreRaw) + (0.28 * relativeScore), 0, 100);

        // Hard realism gates: weak contribution/fit cannot be "high breakout".
        const schemeComplementarity = breakdown.scheme_complementarity || 0;
        const weakFit = (fitScenario.fit_score || 0) < 0.07;
        const weakContribution = contributionStrength < 0.35;
        const weakScheme = schemeComplementarity < 0.08;
        const weakSignal = (fitScenario.signal_strength || 'weak') === 'weak';

        if ((weakFit && weakContribution) || (weakFit && weakScheme)) {
            breakoutScore = Math.min(breakoutScore, 44); // force Low
        } else if (weakContribution || weakScheme || weakSignal) {
            breakoutScore = Math.min(breakoutScore, 64); // cap to Medium ceiling
        }

        // Impact-audit governor: contradictory priors block promotion.
        let promotionBlocked = false;
        const promotionBlockReasons = [];
        if (impactAudit.verdict === 'contradict') {
            promotionBlocked = true;
            promotionBlockReasons.push('Impact audit contradicts prior anchor consensus.');
            breakoutScore = Math.min(breakoutScore, effectiveEvidenceCoverage < 0.78 ? 44 : 64);
        }
        if (effectiveEvidenceCoverage < 0.65) {
            promotionBlocked = true;
            promotionBlockReasons.push('Evidence coverage is too low for high-confidence promotion.');
            breakoutScore = Math.min(breakoutScore, 44);
        }

        // Likelihood signal is tied to team contribution quality, fit strength, clamp pressure, and confidence.
        const fitStrengthNorm = this.clampNum((fitScenario.fit_score || 0) / 0.10, 0, 1);
        const auditConfidenceAdj = impactAudit.verdict === 'support' ? 0.05 : (impactAudit.verdict === 'contradict' ? -0.08 : -0.03);
        const effectiveScenarioConfidence = this.clampNum(
            (fitScenario.confidence || 0.6) + auditConfidenceAdj - evidencePenaltyTotal - (minutesGovernance.feasibility_penalty || 0) * 0.4,
            0.2,
            0.95
        );
        const breakoutLikelihoodScore = this.clampNum(
            100 * (
                0.45 * contributionStrength +
                0.25 * fitStrengthNorm +
                0.15 * (1 - this.clampNum((mechanism.clamp_severity || 0) + (minutesGovernance.feasibility_penalty || 0), 0, 1)) +
                0.15 * effectiveScenarioConfidence
            ),
            0,
            100
        );

        // Final score reflects both raw breakout score and likelihood quality.
        breakoutScore = this.clampNum((0.78 * breakoutScore) + (0.22 * breakoutLikelihoodScore), 0, 100);

        // Ecosystem breakout carve-out: allow validated hidden-value path at low minutes.
        const hiddenSkillSignal = this.clampNum(
            (0.42 * mechanism.suppression_relief) +
            (0.28 * mechanism.spacing_gravity_skill) +
            (0.16 * intrinsicContextLedger.context_share) +
            (0.14 * mechanism.portability_score),
            0,
            1
        );
        const utilizationUplift = this.clampNum(
            (0.6 * intrinsicContextLedger.context_share) +
            (0.4 * fitStrengthNorm),
            0,
            1
        );
        const ecosystemValidated =
            hiddenSkillSignal >= 0.58 &&
            utilizationUplift >= 0.55 &&
            mechanism.portability_score >= 0.33 &&
            effectiveScenarioConfidence >= 0.60;

        if (minutesGovernance.visible_status === 'fail' && !ecosystemValidated) {
            promotionBlocked = true;
            promotionBlockReasons.push('Visible breakout blocked: low-minute baseline requires watchlist status.');
            breakoutScore = Math.min(breakoutScore, 44);
        } else if (minutesGovernance.visible_status === 'watch' && !ecosystemValidated) {
            breakoutScore = Math.min(breakoutScore, 64);
            if (!promotionBlocked) promotionBlockReasons.push('Visible breakout in watch zone until minute sample improves.');
        }

        let likelihood = 'Low';
        let likelihoodClass = 'low';
        if (breakoutScore >= 70) {
            likelihood = 'High';
            likelihoodClass = 'high';
        } else if (breakoutScore >= 45) {
            likelihood = 'Medium';
            likelihoodClass = 'medium';
        }

        const likelihoodPct = Math.round(this.clampNum(breakoutScore, 0, 100));
        const genuineBreakout =
            breakoutScore >= 70 &&
            breakoutLikelihoodScore >= 70 &&
            contributionStrength >= 0.55 &&
            (fitScenario.fit_score || 0) >= 0.08 &&
            (fitScenario.signal_strength || 'weak') !== 'weak' &&
            !promotionBlocked;
        const visibleBreakoutValidated = minutesGovernance.visible_status === 'pass' && !promotionBlocked;

        const reportContract = {
            identity_snapshot: true,
            value_decomposition: !!valueDecomposition,
            counterfactual_with_clamps: clampReport.length > 0,
            decision_layer: (!!contractSummary?.available) || !!agingSummary
        };

        return {
            fitScenario,
            age,
            trustScore,
            constraints,
            currentUsage,
            currentMinutes,
            currentPPG,
            currentAPG,
            currentRPG,
            projectedUsage,
            projectedMinutes,
            projectedPPG,
            projectedAPG,
            projectedRPG,
            usageDelta,
            minutesDelta,
            maxUsageIncrease,
            maxMinutesIncrease,
            contributionStrength,
            contributionProfile,
            mechanism,
            valueDecomposition,
            intrinsicContextLedger,
            clampReport,
            impactAudit,
            defenseRole,
            evidence: {
                ...evidence,
                coverage_effective: effectiveEvidenceCoverage,
                penalty_total: evidencePenaltyTotal
            },
            minutesGovernance,
            agingSummary,
            contractSummary,
            effectiveScenarioConfidence,
            promotionBlocked,
            promotionBlockReasons,
            reportContract,
            ecosystemValidated,
            visibleBreakoutValidated,
            hiddenSkillSignal,
            utilizationUplift,
            breakoutScore,
            breakoutScoreRaw,
            breakoutScoreRelative: relativeScore,
            breakoutLikelihoodScore,
            genuineBreakout,
            likelihood,
            likelihoodClass,
            likelihoodPct
        };
    }

    setupEventListeners() {
        // Search
        document.getElementById('searchInput').addEventListener('input', (e) => {
            this.filterPlayers();
        });

        // Filters
        document.getElementById('archetypeFilter').addEventListener('change', () => {
            this.filterPlayers();
        });

        document.getElementById('usageFilter').addEventListener('change', () => {
            this.filterPlayers();
        });

        document.getElementById('teamFilter').addEventListener('change', () => {
            this.filterPlayers();
        });

        // Sort
        document.getElementById('sortBy').addEventListener('change', (e) => {
            this.currentSort = e.target.value;
            this.sortPlayers();
            this.renderPlayers();
        });

        // Reset
        document.getElementById('resetFilters').addEventListener('click', () => {
            this.resetFilters();
        });

        // Modal
        const modal = document.getElementById('playerModal');
        const closeBtn = document.querySelector('.close');
        
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto'; // Re-enable scroll
        });

        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
                document.body.style.overflow = 'auto'; // Re-enable scroll
            }
        });

        // Card Pickup Overlay close handlers
        const pickupOverlay = document.getElementById('cardPickupOverlay');
        const pickupClose = document.getElementById('cardPickupClose');
        
        if (pickupClose) {
            pickupClose.addEventListener('click', () => this.putCardBack());
        }
        if (pickupOverlay) {
            pickupOverlay.addEventListener('click', (e) => {
                if (e.target === pickupOverlay) this.putCardBack();
            });
        }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && pickupOverlay?.classList.contains('active')) {
                this.putCardBack();
            }
        });

        window.addEventListener('resize', () => {
            if (this.distributionRaf) cancelAnimationFrame(this.distributionRaf);
            this.distributionRaf = requestAnimationFrame(() => this.drawValueDistributionChart());
        });
        window.addEventListener('themechange', () => {
            if (this.distributionRaf) cancelAnimationFrame(this.distributionRaf);
            this.distributionRaf = requestAnimationFrame(() => this.drawValueDistributionChart());
            if (this.currentModalPlayer) {
                this.drawShotChart(this.currentModalPlayer);
                this.drawValueDriversChart(this.currentModalPlayer);
            }
        });
        this.setupDistributionInteractions();
    }

    populateArchetypeFilter() {
        const archetypes = [...new Set(this.players.map(p => p.identity?.primary_archetype).filter(Boolean))].sort();
        const archetypeFilter = document.getElementById('archetypeFilter');
        
        archetypes.forEach(archetype => {
            const option = document.createElement('option');
            option.value = archetype;
            option.textContent = this.formatArchetype(archetype);
            archetypeFilter.appendChild(option);
        });
    }

    populateTeamFilter() {
        const teams = [...new Set(this.players.map(p => p.player.team))].sort();
        const teamFilter = document.getElementById('teamFilter');
        
        teams.forEach(team => {
            const option = document.createElement('option');
            option.value = team;
            option.textContent = team;
            teamFilter.appendChild(option);
        });
    }

    filterPlayers() {
        const searchTerm = document.getElementById('searchInput').value.toLowerCase();
        const archetypeFilter = document.getElementById('archetypeFilter').value;
        const usageFilter = document.getElementById('usageFilter').value;
        const teamFilter = document.getElementById('teamFilter').value;

        this.filteredPlayers = this.players.filter(player => {
            // Search filter
            const matchesSearch = !searchTerm || 
                player.player.name.toLowerCase().includes(searchTerm) ||
                player.player.team.toLowerCase().includes(searchTerm);

            // Archetype filter - handle data_sample format
            const playerArchetype = player.identity?.primary_archetype || '';
            const matchesArchetype = !archetypeFilter || 
                playerArchetype === archetypeFilter ||
                playerArchetype.includes(archetypeFilter);

            // Usage filter - handle data_sample format
            const playerUsage = player.identity?.usage_band || '';
            const matchesUsage = !usageFilter || 
                playerUsage === usageFilter ||
                playerUsage.toLowerCase() === usageFilter.toLowerCase();

            // Team filter
            const matchesTeam = !teamFilter || 
                player.player.team === teamFilter;

            return matchesSearch && matchesArchetype && matchesUsage && matchesTeam;
        });

        this.sortPlayers();
        this.renderPlayers();
        this.updateStats();
    }

    sortPlayers() {
        this.filteredPlayers.sort((a, b) => {
            switch (this.currentSort) {
                case 'name':
                    return a.player.name.localeCompare(b.player.name);
                
                case 'impact':
                    return this.getPlayerValue(b) - this.getPlayerValue(a);

                case 'team_fit':
                    return this.getTeamFitScore(b) - this.getTeamFitScore(a);
                
                case 'age':
                    return (a.player.age || 25) - (b.player.age || 25);
                
                case 'wins':
                    return (b.valuation?.impact?.wins_added || 0) - (a.valuation?.impact?.wins_added || 0);
                
                case 'breakout':
                    return this.evaluateBreakoutScenario(b).breakoutScore - this.evaluateBreakoutScenario(a).breakoutScore;
                
                default:
                    return 0;
            }
        });
    }

    updateStats() {
        // Stats widgets are optional in the current layout.
        const setText = (id, value) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        };

        const total = this.players.length;
        const visible = this.filteredPlayers.length;
        const breakoutCandidates = this.filteredPlayers.filter(p => {
            const s = this.evaluateBreakoutScenario(p);
            return s.likelihoodClass === 'high' && (s.breakoutScore || 0) >= 60;
        }).length;
        const highTrust = this.filteredPlayers.filter(
            p => Number(p.v1_1_enhancements?.trust_assessment?.score || 0) >= 75
        ).length;

        setText('totalPlayers', total.toLocaleString());
        setText('visiblePlayers', visible.toLocaleString());
        setText('breakoutCandidates', breakoutCandidates.toLocaleString());
        setText('highTrustPlayers', highTrust.toLocaleString());
    }

    getTeamFitScore(player) {
        if (!this.teamFitScoreCache) this.teamFitScoreCache = {};
        const key = this.getPlayerKey(player);
        if (this.teamFitScoreCache[key] !== undefined) return this.teamFitScoreCache[key];
        const fit = this.evaluateBreakoutScenario(player).fitScenario?.fit_score || 0;
        this.teamFitScoreCache[key] = fit;
        return fit;
    }

    getTeamFitPercentile(player) {
        if (!this.teamFitPercentileCache) {
            const rows = this.players.map(p => ({
                key: this.getPlayerKey(p),
                v: this.getTeamFitScore(p)
            }));
            const sorted = rows.map(r => r.v).sort((a, b) => a - b);
            const n = Math.max(1, sorted.length);
            const map = {};
            for (const r of rows) {
                let lo = 0;
                let hi = n;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (sorted[mid] <= r.v) lo = mid + 1;
                    else hi = mid;
                }
                map[r.key] = Math.round(this.clampNum((lo / n) * 100, 0, 100));
            }
            this.teamFitPercentileCache = map;
        }
        return this.teamFitPercentileCache[this.getPlayerKey(player)] ?? 50;
    }

    getDistributionMetricConfig() {
        const breakoutGetter = (p) => this.evaluateBreakoutScenario(p).breakoutScore;
        switch (this.currentSort) {
            case 'name':
                if (this.initialDistributionMetricKey === 'team_fit') {
                    return {
                        key: 'team_fit',
                        label: 'Showing: Team Fit Context (%)',
                        accessor: (p) => this.getTeamFitScore(p) * 100,
                        format: (v) => `${v.toFixed(1)}%`,
                        _initialOverride: true
                    };
                }
                return {
                    key: 'none',
                    label: 'Showing: Name sort selected (distribution hidden)',
                    accessor: null,
                    format: null
                };
            case 'impact':
                return {
                    key: 'impact',
                    label: 'Showing: Player Value',
                    accessor: (p) => this.getPlayerValue(p),
                    format: (v) => `${v.toFixed(1)}`
                };
            case 'team_fit':
                return {
                    key: 'team_fit',
                    label: 'Showing: Team Fit Context (%)',
                    accessor: (p) => this.getTeamFitScore(p) * 100,
                    format: (v) => `${v.toFixed(1)}%`
                };
            case 'age':
                return {
                    key: 'age',
                    label: 'Showing: Age',
                    accessor: (p) => Number(p.player?.age || 0),
                    format: (v) => `${v.toFixed(1)}`
                };
            case 'wins':
                return {
                    key: 'wins',
                    label: 'Showing: Wins Added',
                    accessor: (p) => Number(p.valuation?.impact?.wins_added || 0),
                    format: (v) => `${v.toFixed(1)}`,
                    transform: (v) => {
                        // Center-expanding signed log scale around 0.
                        // k<1 expands dense near-zero values and compresses tails.
                        const k = 0.6;
                        return Math.sign(v) * Math.log1p(Math.abs(v) / k);
                    }
                };
            case 'breakout':
                return {
                    key: 'breakout',
                    label: 'Showing: Breakout Score',
                    accessor: (p) => breakoutGetter(p),
                    format: (v) => `${v.toFixed(1)}`
                };
            default:
                return {
                    key: 'impact',
                    label: 'Showing: Player Value',
                    accessor: (p) => this.getPlayerValue(p),
                    format: (v) => `${v.toFixed(1)}`
                };
        }
    }

    renderPlayers() {
        const container = document.getElementById('playerCards');
        const noResults = document.getElementById('noResults');

        if (this.filteredPlayers.length === 0) {
            container.innerHTML = '';
            noResults.style.display = 'block';
            this.drawValueDistributionChart();
            return;
        }

        noResults.style.display = 'none';

        // Render set filter pills
        this.renderSetFilterPills();

        // Render cards in store display case format
        container.innerHTML = this.filteredPlayers.map(player => this.createStoreCard(player)).join('');

        // Add click listeners for card pickup
        document.querySelectorAll('.store-card-slot').forEach((slot, index) => {
            slot.addEventListener('click', () => {
                this.pickUpCard(this.filteredPlayers[index]);
            });
        });

        this.drawValueDistributionChart();
    }

    renderSetFilterPills() {
        const row = document.getElementById('setFilterRow');
        if (!row) return;
        const allSets = typeof CARD_SETS !== 'undefined' ? CARD_SETS : [];
        const pills = [
            `<button class="set-filter-pill active" data-set="all">All Sets</button>`
        ];
        allSets.forEach(s => {
            pills.push(`<button class="set-filter-pill" data-set="${s.id}">${s.brand} ${s.rarity}</button>`);
        });
        row.innerHTML = pills.join('');

        row.querySelectorAll('.set-filter-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                row.querySelectorAll('.set-filter-pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                const setId = pill.getAttribute('data-set');
                this.filterBySet(setId);
            });
        });
    }

    filterBySet(setId) {
        const slots = document.querySelectorAll('.store-card-slot');
        slots.forEach(slot => {
            if (setId === 'all') {
                slot.style.display = '';
            } else {
                slot.style.display = slot.getAttribute('data-set-id') === setId ? '' : 'none';
            }
        });
    }

    createStoreCard(player) {
        const set = typeof getCardSet === 'function' ? getCardSet(player.player?.name || 'Unknown') : null;
        const headshotUrl = this.getPlayerHeadshotUrl(player);
        const teamLogoUrl = this.getTeamLogoUrl(player.player?.team);
        const playerValue = this.getPlayerValue(player);
        const monogram = this.getPlayerMonogram(player.player?.name || 'NA');
        const shimmerClass = set && typeof getSetShimmerClass === 'function' ? getSetShimmerClass(set) : '';

        const borderBg = set ? set.borderStyle : 'linear-gradient(135deg, #334155, #475569)';
        const cardBg = set ? set.bgStyle : 'linear-gradient(160deg, #1a1a2e, #16213e)';
        const textCol = set ? set.textColor : '#e2e8f0';
        const nameCol = set ? set.nameColor : '#ffffff';
        const accentCol = set ? set.accentColor : '#94a3b8';
        const frameW = set ? set.frameWidth : '3px';
        const frameCol = set ? set.frameColor : 'rgba(148,163,184,0.4)';
        const labelBg = set ? set.labelBg : 'rgba(148,163,184,0.12)';
        const brand = set ? set.brand : 'NBA';
        const rarity = set ? set.rarity : 'Base';
        const setId = set ? set.id : 'base';

        const valueColor = playerValue >= 60 ? '#86efac' : (playerValue <= 40 ? '#fca5a5' : accentCol);

        return `
            <div class="store-card-slot" data-player="${player.player.name}" data-set-id="${setId}">
                <div class="store-card ${shimmerClass}"
                     style="background: ${borderBg}; padding: ${frameW}; border-radius: 14px;">
                    <div style="background: ${cardBg}; border-radius: 11px; overflow: hidden; display: flex; flex-direction: column; height: 100%; position: relative;">
                    ${teamLogoUrl ? `<div class="store-card-team-logo" style="background-image:url('${teamLogoUrl}');"></div>` : ''}
                    <div class="store-card-brand" style="background:${labelBg}; color:${accentCol};">${brand}</div>
                    <div class="store-card-rarity" style="background:${labelBg}; color:${accentCol};">${rarity}</div>
                    <div class="store-card-value" style="background:rgba(0,0,0,0.5); color:${valueColor};">${playerValue.toFixed(1)}</div>
                    <div class="store-card-photo">
                        <img src="${headshotUrl}"
                             alt="${player.player.name}"
                             loading="lazy"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
                        <div class="store-card-photo-fallback" style="display:none; color:${accentCol};">${monogram}</div>
                    </div>
                    <div class="store-card-info" style="color:${textCol};">
                        <div class="store-card-name" style="color:${nameCol};">${player.player.name}</div>
                        <div class="store-card-meta">
                            <span>${player.player.team}</span>
                            <span>${player.player.position || ''}</span>
                        </div>
                    </div>
                    </div>
                </div>
            </div>
        `;
    }

    pickUpCard(player) {
        const overlay = document.getElementById('cardPickupOverlay');
        const cardEl = document.getElementById('cardPickupCard');
        const detailEl = document.getElementById('cardPickupDetailInner');
        if (!overlay || !cardEl || !detailEl) return;

        // Render the enlarged card
        const set = typeof getCardSet === 'function' ? getCardSet(player.player?.name || 'Unknown') : null;
        cardEl.innerHTML = this.createPickupCardInner(player, set);
        cardEl.style.background = set ? set.bgStyle : 'linear-gradient(160deg, #1a1a2e, #16213e)';
        cardEl.style.border = `${set ? set.frameWidth : '3px'} solid ${set ? set.frameColor : 'rgba(148,163,184,0.4)'}`;

        // Render the detail panel
        this.currentModalPlayer = player;
        this.shotChartMode = this.shotChartMode || 'volume';
        detailEl.innerHTML = this.createPlayerDetail(player);

        // Show overlay
        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Setup tabs and charts after visible
        setTimeout(() => {
            this.setupTabs();
            this.setupShotModeToggle();
            this.drawShotChart(player);
            this.drawValueDriversChart(player);
        }, 80);
    }

    createPickupCardInner(player, set) {
        const headshotUrl = this.getPlayerHeadshotUrl(player);
        const teamLogoUrl = this.getTeamLogoUrl(player.player?.team);
        const playerValue = this.getPlayerValue(player);
        const monogram = this.getPlayerMonogram(player.player?.name || 'NA');
        const evalOut = this.evaluateBreakoutScenario(player);
        const breakoutStyle = this.getBreakoutTierStyle(evalOut.likelihoodClass, evalOut.likelihoodPct);
        const isRookie = this.isLikelyRookie(player);

        const textCol = set ? set.textColor : '#e2e8f0';
        const nameCol = set ? set.nameColor : '#ffffff';
        const accentCol = set ? set.accentColor : '#94a3b8';
        const labelBg = set ? set.labelBg : 'rgba(148,163,184,0.12)';
        const brand = set ? set.brand : 'NBA';
        const rarity = set ? set.rarity : 'Base';
        const valueColor = playerValue >= 60 ? '#86efac' : (playerValue <= 40 ? '#fca5a5' : accentCol);
        const shimmerClass = set && typeof getSetShimmerClass === 'function' ? getSetShimmerClass(set) : '';

        return `
            <div class="${shimmerClass}" style="position:absolute;inset:0;border-radius:16px;overflow:hidden;pointer-events:none;z-index:4;"></div>
            ${teamLogoUrl ? `<div style="position:absolute;right:-8%;top:8%;width:50%;aspect-ratio:1;background:url('${teamLogoUrl}') center/contain no-repeat;opacity:0.15;mix-blend-mode:screen;z-index:0;pointer-events:none;"></div>` : ''}
            <div style="position:absolute;top:10px;left:10px;z-index:5;font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;padding:3px 8px;border-radius:5px;background:${labelBg};color:${accentCol};">${brand}</div>
            <div style="position:absolute;top:10px;right:10px;z-index:5;font-family:'Barlow Condensed',sans-serif;font-size:0.6rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;padding:3px 8px;border-radius:5px;background:${labelBg};color:${accentCol};">${rarity}</div>
            <div style="position:relative;flex:1;overflow:hidden;z-index:1;">
                <img src="${headshotUrl}" alt="${player.player.name}" style="width:100%;height:100%;object-fit:cover;object-position:top center;filter:saturate(1.1) contrast(1.05);"
                     onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
                <div style="display:none;width:100%;height:100%;align-items:center;justify-content:center;font-size:3.5rem;font-weight:800;font-family:'Barlow Condensed',sans-serif;color:${accentCol};opacity:0.6;">${monogram}</div>
                <div style="position:absolute;inset:0;background:linear-gradient(180deg,transparent 30%,rgba(0,0,0,0.75) 100%);z-index:1;pointer-events:none;"></div>
                ${isRookie ? `<div style="position:absolute;top:38px;right:10px;z-index:5;background:linear-gradient(135deg,#facc15,#ca8a04);color:#1f2937;font-family:'Barlow Condensed',sans-serif;font-size:0.7rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;padding:3px 10px;border-radius:999px;">ROOKIE</div>` : ''}
                <div style="position:absolute;bottom:8px;left:10px;right:10px;z-index:5;text-align:center;background:${breakoutStyle.bg};color:${breakoutStyle.fg};font-weight:800;letter-spacing:0.08em;text-transform:uppercase;font-size:0.7rem;padding:4px 8px;border-radius:8px;text-shadow:0 1px 0 rgba(0,0,0,0.2);">${breakoutStyle.label} BREAKOUT &bull; ${Math.round(evalOut.likelihoodPct || 0)}%</div>
            </div>
            <div style="position:relative;z-index:2;padding:0.75rem 0.8rem 0.85rem;color:${textCol};">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;font-weight:700;letter-spacing:0.02em;text-transform:uppercase;line-height:1.1;margin-bottom:0.25rem;color:${nameCol};">${player.player.name}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;font-size:0.72rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;opacity:0.85;">
                    <span>${player.player.team} &bull; ${player.player.position || ''} &bull; Age ${Math.round(player.player.age) || '?'}</span>
                    <span style="color:${valueColor};font-weight:800;">VAL ${playerValue.toFixed(1)}</span>
                </div>
            </div>
        `;
    }

    putCardBack() {
        const overlay = document.getElementById('cardPickupOverlay');
        if (overlay) {
            overlay.classList.remove('active');
            document.body.style.overflow = 'auto';
        }
    }

    drawValueDistributionChart() {
        const canvas = document.getElementById('valueDistributionChart');
        if (!canvas) return;
        const metricLabelEl = document.getElementById('distributionMetricLabel');
        const config = this.getDistributionMetricConfig();
        this.distributionMetricConfig = config;
        if (config && config._initialOverride) {
            this.initialDistributionMetricKey = null;
        }
        if (metricLabelEl) metricLabelEl.textContent = config.label;

        const countEl = document.getElementById('distributionCount');
        const total = this.filteredPlayers.length;
        if (countEl) {
            countEl.textContent = `${total} player${total === 1 ? '' : 's'}`;
        }

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const cssWidth = Math.max(300, Math.floor(rect.width || 860));
        const cssHeight = Math.max(150, Math.floor(rect.height || 180));
        canvas.width = Math.floor(cssWidth * dpr);
        canvas.height = Math.floor(cssHeight * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssWidth, cssHeight);
        this.distributionPoints = [];

        // Background
        const bg = ctx.createLinearGradient(0, 0, cssWidth, cssHeight);
        bg.addColorStop(0, 'rgba(10, 14, 28, 0.24)');
        bg.addColorStop(1, 'rgba(8, 11, 23, 0.08)');
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, cssWidth, cssHeight);

        const pad = { left: 44, right: 20, top: 18, bottom: 30 };
        const plotW = Math.max(1, cssWidth - pad.left - pad.right);
        const centerY = Math.round((cssHeight - pad.top - pad.bottom) * 0.50 + pad.top);
        const amp = Math.max(20, Math.floor((cssHeight - pad.top - pad.bottom) * 0.48));

        // Axis
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, centerY);
        ctx.lineTo(cssWidth - pad.right, centerY);
        ctx.stroke();

        if (!total) {
            ctx.fillStyle = 'rgba(255,255,255,0.82)';
            ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No players match current filters', cssWidth / 2, centerY - 8);
            return;
        }

        if (!config.accessor) {
            ctx.fillStyle = 'rgba(255,255,255,0.82)';
            ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Choose a metric sort to render distribution nodes', cssWidth / 2, centerY - 8);
            return;
        }

        // Keep distribution geometry stable regardless of card sort order.
        // Card sort can change this.filteredPlayers sequence; chart should not.
        const distributionPlayers = [...this.filteredPlayers].sort((a, b) =>
            (a.player?.name || '').localeCompare(b.player?.name || '')
        );

        const values = distributionPlayers
            .map(p => config.accessor(p))
            .filter(v => Number.isFinite(v));
        if (!values.length) return;

        const xTransform = typeof config.transform === 'function' ? config.transform : (v) => v;
        const transformedValues = values.map(v => xTransform(v));
        const minV = Math.min(...values);
        const maxV = Math.max(...values);
        const minT = Math.min(...transformedValues);
        const maxT = Math.max(...transformedValues);
        const span = Math.max(0.0001, maxV - minV);
        const spanT = Math.max(0.0001, maxT - minT);
        const toX = (v) => {
            const tv = xTransform(v);
            return pad.left + ((tv - minT) / spanT) * plotW;
        };
        const q = (arr, p) => {
            const s = [...arr].sort((a, b) => a - b);
            const idx = Math.min(s.length - 1, Math.max(0, Math.floor(p * (s.length - 1))));
            return s[idx];
        };
        const q1 = q(values, 0.25);
        const q2 = q(values, 0.50);
        const q3 = q(values, 0.75);

        // Ticks
        const ticks = [minV, q1, q2, q3, maxV];
        ctx.fillStyle = 'rgba(255,255,255,0.65)';
        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
        ctx.textAlign = 'center';
        for (const t of ticks) {
            const x = toX(t);
            ctx.strokeStyle = 'rgba(255,255,255,0.18)';
            ctx.beginPath();
            ctx.moveTo(x, centerY - amp - 7);
            ctx.lineTo(x, centerY + amp + 7);
            ctx.stroke();
            ctx.fillText((config.format ? config.format(t) : t.toFixed(1)), x, cssHeight - 8);
        }

        // Beeswarm-like mirrored jitter by x-bins with adaptive spacing.
        // Dense metrics (e.g. Wins Added around 0) can cluster heavily, so we:
        // 1) increase bin resolution, 2) adapt vertical layer spacing,
        // 3) add small deterministic x-jitter to avoid rectangular blocks.
        const binCount = Math.max(48, Math.min(96, Math.floor(plotW / 11)));
        const bins = new Map();
        const rawPoints = distributionPlayers.map((p, idx) => {
            const v = config.accessor(p);
            const x = toX(v);
            const b = Math.floor(((x - pad.left) / plotW) * binCount);
            const key = Math.max(0, Math.min(binCount - 1, b));
            const stack = bins.get(key) || 0;
            bins.set(key, stack + 1);
            return { player: p, x, v, idx, key, stack };
        });

        let maxLayer = 1;
        for (const c of bins.values()) {
            maxLayer = Math.max(maxLayer, Math.floor((c - 1) / 2) + 1);
        }
        const layerStep = Math.max(2.0, Math.min(4.8, amp / (maxLayer + 1)));

        const points = rawPoints.map((pt) => {
            const count = bins.get(pt.key) || 1;
            const layer = Math.floor(pt.stack / 2) + 1;
            const sign = pt.stack % 2 === 0 ? -1 : 1;
            const rawOffset = layer * layerStep;
            // Use tanh squash instead of hard clipping to prevent flat "box" edges.
            const yOffset = amp * Math.tanh(rawOffset / Math.max(amp, 1));

            // Small deterministic x-jitter for dense bins to improve distinguishability.
            const denseFactor = this.clampNum((count - 6) / 26, 0, 1);
            const phase = (Math.floor(pt.stack / 2) % 3) + 1;
            const seed = (((pt.idx * 1103515245 + 12345) >>> 16) & 1023) / 1023; // deterministic 0..1
            const jitterMag = (0.35 + 0.95 * denseFactor) * phase;
            const jitterSign = pt.stack % 2 === 0 ? -1 : 1;
            const xJitter = jitterSign * jitterMag + (seed - 0.5) * 0.65;

            return {
                player: pt.player,
                x: pt.x + xJitter,
                y: centerY + sign * yOffset,
                v: pt.v,
                idx: pt.idx
            };
        });

        points.sort((a, b) => a.v - b.v);
        const mean = values.reduce((s, v) => s + v, 0) / values.length;
        for (const pt of points) {
            const t = this.clampNum((pt.v - mean) / (span * 0.55) + 0.5, 0, 1);
            const r = Math.round(203 + (63 - 203) * t);
            const g = Math.round(92 + (198 - 92) * t);
            const b = Math.round(103 + (138 - 103) * t);
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.78)`;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 2.7, 0, Math.PI * 2);
            ctx.fill();
        }

        this.distributionPoints = points.map(pt => ({
            x: pt.x,
            y: pt.y,
            radius: 4.5,
            name: pt.player?.player?.name || 'Unknown',
            value: pt.v
        }));
    }

    setupDistributionInteractions() {
        const canvas = document.getElementById('valueDistributionChart');
        const tooltip = document.getElementById('distributionTooltip');
        if (!canvas || !tooltip) return;

        const hideTooltip = () => {
            tooltip.style.display = 'none';
        };

        canvas.addEventListener('mouseleave', hideTooltip);
        canvas.addEventListener('mousemove', (e) => {
            if (!this.distributionPoints?.length || !this.distributionMetricConfig?.format) {
                hideTooltip();
                return;
            }
            const rect = canvas.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            let hit = null;
            let bestD2 = Number.POSITIVE_INFINITY;
            for (const p of this.distributionPoints) {
                const dx = px - p.x;
                const dy = py - p.y;
                const d2 = dx * dx + dy * dy;
                if (d2 <= (p.radius * p.radius) && d2 < bestD2) {
                    bestD2 = d2;
                    hit = p;
                }
            }
            if (!hit) {
                hideTooltip();
                return;
            }

            tooltip.innerHTML = `<strong>${hit.name}</strong><br>${this.distributionMetricConfig.label.replace('Showing: ', '')}: ${this.distributionMetricConfig.format(hit.value)}`;
            tooltip.style.display = 'block';
            // Edge-aware positioning so tooltip never gets clipped at chart boundaries.
            const tipW = tooltip.offsetWidth || 160;
            const tipH = tooltip.offsetHeight || 44;
            let left = px + 10;
            let top = py + 10;

            if (left + tipW > rect.width - 6) left = px - tipW - 10;
            if (left < 6) left = 6;
            if (top + tipH > rect.height - 6) top = py - tipH - 10;
            if (top < 6) top = 6;

            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
            tooltip.style.display = 'block';
        });
    }

    createPlayerCard(player) {
        // Extract data from data_sample format
        const playerValue = this.getPlayerValue(player);
        const impactClass = playerValue >= 60 ? 'positive' : (playerValue <= 40 ? 'negative' : '');
        const evalOut = this.evaluateBreakoutScenario(player);
        const typeCardProfile = this.getBasketballTypeProfile(player);
        const typeCardHeadshotUrl = this.getPlayerHeadshotUrl(player);
        const typeCardTeamLogoUrl = this.getTeamLogoUrl(player.player?.team);
        const breakoutStyle = this.getBreakoutTierStyle(evalOut.likelihoodClass, evalOut.likelihoodPct);
        const isRookie = this.isLikelyRookie(player);
        const typeCardMonogram = this.getPlayerMonogram(player.player?.name || 'NA');

        return `
            <div class="player-card type-card trading-card" data-player="${player.player.name}" style="--type-primary:${typeCardProfile.primaryColor};--type-secondary:${typeCardProfile.secondaryColor};--type-glow:${typeCardProfile.glowColor};--breakout-bg:${breakoutStyle.bg};--breakout-fg:${breakoutStyle.fg};--breakout-glow:${breakoutStyle.glow};">
                <div class="type-card-hero trading-hero">
                    ${typeCardTeamLogoUrl ? `<div class="type-card-team-logo" style="background-image:url('${typeCardTeamLogoUrl}');"></div>` : ''}
                    <img
                        class="type-card-headshot"
                        src="${typeCardHeadshotUrl}"
                        alt="${player.player.name} headshot"
                        loading="lazy"
                        onerror="this.style.display='none'; this.closest('.type-card-hero').classList.add('no-headshot');"
                    />
                    <div class="type-card-monogram">${typeCardMonogram}</div>
                    <div class="trading-topline">
                        <div class="trading-name">${player.player.name}</div>
                    </div>
                    <div class="trading-position-ribbon">${player.player.position || 'N/A'}</div>
                    <div class="trading-breakout-strip">${breakoutStyle.label} BREAKOUT &bull; ${Math.round(evalOut.likelihoodPct || 0)}%</div>
                    ${isRookie ? `<div class="trading-rookie-badge">ROOKIE</div>` : ''}
                    <div class="trading-team-ribbon">${player.player.team}</div>
                    <div class="trading-card-badge ${impactClass}">Value ${playerValue.toFixed(1)}</div>
                    <div class="trading-subtype">${typeCardProfile.typeLabel}</div>
                    <div class="trading-photo-vignette"></div>
                    <div class="trading-photo-grain"></div>
                    <div class="trading-photo-frame"></div>
                    <div class="trading-photo-corner"></div>
                    <div class="trading-photo-corner bottom"></div>
                    <div class="trading-photo-border"></div>
                    <div class="trading-photo-shadow"></div>
                    <div class="trading-photo-light"></div>
                    <div class="trading-photo-band">
                        <div class="trading-photo-band-text">${typeCardProfile.subtitle}</div>
                    </div>
                </div>
            </div>
        `;
        
        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 0;
        const fitScorePct = (evalOut.fitScenario?.fit_score || 0) * 100;
        const fitPercentile = this.getTeamFitPercentile(player);
        const breakoutScore = evalOut.breakoutScore || 0;
        const scenarioSummary = this.buildCardScenarioSummary(player, evalOut, mechanism);

        // Get usage rate (stored as decimal, e.g., 0.2 = 20%)
        const usageRate = player.performance?.advanced?.usage_rate ?? 0;
        const usageDisplay = (usageRate * 100).toFixed(1) + '%';

        // Tags
        const tags = [];
        if (trustScore >= 75) tags.push('<span class="tag high-impact">High Trust</span>');
        if (playerValue >= 70) tags.push('<span class="tag high-impact">High Value</span>');
        if (mechanism.portability_score >= 0.62) tags.push('<span class="tag high-portability">Portable Defense</span>');
        if ((evalOut.fitScenario?.signal_strength || 'weak') === 'strong') tags.push('<span class="tag high-impact">Strong Fit Signal</span>');
        if (evalOut.genuineBreakout) tags.push('<span class="tag high-impact">Genuine Breakout Signal</span>');
        if (evalOut.promotionBlocked) tags.push('<span class="tag breakout">Promotion Blocked</span>');
        
        // High usage tag (usage > 25%)
        if (usageRate >= 0.25) tags.push('<span class="tag breakout">High Usage</span>');

        return `
            <div class="player-card" data-player="${player.player.name}">
                <div class="card-header">
                    <div class="player-name">${player.player.name}</div>
                    <div class="player-meta">
                        <span>${player.player.team}</span>
                        <span>•</span>
                        <span>${player.player.position || 'N/A'}</span>
                        <span>•</span>
                        <span>Age ${Math.round(player.player.age) || 'N/A'}</span>
                    </div>
                </div>
                <div class="card-body">
                    <div class="archetype-badge">
                        ${this.formatArchetype(player.identity?.primary_archetype || 'unknown')}
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-label">Player Value</div>
                            <div class="metric-value ${impactClass}">${playerValue.toFixed(1)}</div>
                            ${this.createPercentileIndicator(playerValue, 0, 100)}
                        </div>
                        <div class="metric">
                            <div class="metric-label">Breakout Score</div>
                            <div class="metric-value">${breakoutScore.toFixed(1)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Team Fit Context</div>
                            <div class="metric-value">${fitScorePct.toFixed(0)}% <span style="font-size:0.72rem;color:#6b7280;">(p${fitPercentile})</span></div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Suppression Relief</div>
                            <div class="metric-value">${(mechanism.suppression_relief * 100).toFixed(0)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Portability</div>
                            <div class="metric-value">${(mechanism.portability_score * 100).toFixed(0)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Clamp Pressure</div>
                            <div class="metric-value">${(mechanism.clamp_severity * 100).toFixed(0)}%</div>
                        </div>
                    </div>

                    <div class="card-mechanism-note">
                        ${scenarioSummary}
                    </div>

                    ${tags.length > 0 ? `<div class="tags">${tags.join('')}</div>` : ''}
                </div>
            </div>
        `;
    }

    createPercentileIndicator(value, min, max) {
        const percentage = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
        
        let fillClass = 'bottom-50';
        if (percentage >= 90) fillClass = 'top-10';
        else if (percentage >= 75) fillClass = 'top-25';
        else if (percentage >= 50) fillClass = 'top-50';

        return `
            <div class="percentile-indicator">
                <div class="percentile-bar">
                    <div class="percentile-fill ${fillClass}" style="width: ${percentage}%"></div>
                </div>
                <div class="percentile-label">${percentage.toFixed(0)}%</div>
            </div>
        `;
    }

    showPlayerDetail(player) {
        const modal = document.getElementById('playerModal');
        const modalBody = document.getElementById('modalBody');
        this.currentModalPlayer = player;
        this.shotChartMode = this.shotChartMode || 'volume';

        modalBody.innerHTML = this.createPlayerDetail(player);
        modal.style.display = 'block';
        
        // Disable body scroll
        document.body.style.overflow = 'hidden';

        // Setup tab switching
        this.setupTabs();

        // Draw charts after modal is visible (only for visuals tab)
        setTimeout(() => {
            this.setupShotModeToggle();
            this.drawShotChart(player);
            this.drawValueDriversChart(player);
        }, 50);
    }

    setupShotModeToggle() {
        const container = document.getElementById('shotModeToggle');
        if (!container) return;
        const buttons = container.querySelectorAll('.mode-btn');
        buttons.forEach(btn => {
            const mode = btn.getAttribute('data-shot-mode');
            btn.classList.toggle('active', mode === this.shotChartMode);
            btn.onclick = () => {
                this.shotChartMode = mode;
                buttons.forEach(b => b.classList.toggle('active', b === btn));
                if (this.currentModalPlayer) {
                    this.drawShotChart(this.currentModalPlayer);
                }
            };
        });
    }

    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));

                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(`tab-${tabId}`).classList.add('active');
            });
        });
    }

    createPlayerDetail(player) {
        const valuation = player.valuation;
        const perf = player.performance || {};
        const trad = perf.traditional || {};
        const adv = perf.advanced || {};
        const evalOut = this.evaluateBreakoutScenario(player);
        const mechanism = this.computeMechanismMetrics(player, evalOut);
        const trust = player.v1_1_enhancements?.trust_assessment || {};
        const uncertainty = player.v1_1_enhancements?.uncertainty_estimates || {};
        const constraints = player.v1_1_enhancements?.scenario_constraints || {};

        return `
            <div class="modal-header">
                <h2>${player.player.name}</h2>
                <div style="margin-top: 0.5rem; opacity: 0.9;">
                    ${player.player.team} • ${player.player.position || 'N/A'} • 
                    Age ${Math.round(player.player.age) || 'N/A'} • 
                    ${this.formatArchetype(player.identity?.primary_archetype || 'unknown')}
                </div>
            </div>

            <!-- Tab Navigation -->
            <div class="tab-navigation">
                <button class="tab-button active" data-tab="visuals">Visuals</button>
                <button class="tab-button" data-tab="metrics">Metrics</button>
                <button class="tab-button" data-tab="archetype">Archetype & Similarity</button>
                <button class="tab-button" data-tab="breakout">Breakout Scenario</button>
            </div>

            <div class="modal-body">
                <!-- Visuals Tab -->
                <div class="tab-content active" id="tab-visuals">
                    <div class="visual-grid">
                        <div class="visual-card">
                            <h4>Shot Distribution</h4>
                            <div class="chart-mode-toggle" id="shotModeToggle">
                                <button class="mode-btn active" data-shot-mode="volume">Volume</button>
                                <button class="mode-btn" data-shot-mode="fg">FG%</button>
                                <button class="mode-btn" data-shot-mode="value">Shot Value</button>
                                <button class="mode-btn" data-shot-mode="trueimpact">True Impact</button>
                            </div>
                            <canvas id="shotChart" width="400" height="470"></canvas>
                        </div>

                        <div class="visual-card">
                            <h4>Archetype Value Drivers</h4>
                            <canvas id="valueDriversChart" width="450" height="400"></canvas>
                        </div>

                        <div class="visual-card">
                            <h4>Archetype Uniqueness</h4>
                            <div class="uniqueness-content">
                                ${this.generateUniquenessInsights(player)}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Metrics Tab -->
                <div class="tab-content" id="tab-metrics">
                    ${this.renderMetricsTab(player, evalOut, mechanism, valuation, trad, adv, trust, uncertainty)}
                </div>

                <!-- Archetype & Similarity Tab -->
                <div class="tab-content" id="tab-archetype">
                    <div class="detail-section">
                        <h3>Archetype Identity</h3>
                        <div class="archetype-info">
                            <div class="archetype-primary">
                                <div class="archetype-badge-large">
                                    ${this.formatArchetype(player.identity?.primary_archetype || 'unknown')}
                                </div>
                                <div class="archetype-confidence">
                                    Confidence: ${((player.identity?.archetype_confidence || 0) * 100).toFixed(0)}%
                                </div>
                                <div class="archetype-description">
                                    ${player.identity?.role_description || 'No description available'}
                                </div>
                            </div>

                            ${player.identity?.secondary_archetypes && player.identity.secondary_archetypes.length > 0 ? `
                            <div class="secondary-archetypes">
                                <h4>Secondary Archetypes</h4>
                                <div class="secondary-list">
                                    ${player.identity.secondary_archetypes.map(arch => `
                                        <div class="secondary-item">
                                            <span class="secondary-name">${this.formatArchetype(arch.name)}</span>
                                            <span class="secondary-prob">${(arch.probability * 100).toFixed(0)}%</span>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            ` : ''}
                        </div>
                    </div>

                    ${player.comparables?.similar_players && player.comparables.similar_players.length > 0 ? `
                    <div class="detail-section">
                        <h3>Similar Players</h3>
                        <div class="comparables-grid">
                            ${player.comparables.similar_players.slice(0, 5).map(comp => `
                                <div class="comparable-card">
                                    <div class="comparable-header">
                                        <div class="comparable-name">${comp.name}</div>
                                        <div class="comparable-similarity">${(comp.similarity_score * 100).toFixed(0)}% similar</div>
                                    </div>
                                    <div class="comparable-meta">${comp.team} • ${this.formatArchetype(comp.archetype)}</div>
                                    <div class="comparable-stats">
                                        ${comp.stats.points.toFixed(1)} PPG • 
                                        ${comp.stats.assists.toFixed(1)} APG • 
                                        ${comp.stats.rebounds.toFixed(1)} RPG
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>

                <!-- Breakout Scenario Tab -->
                <div class="tab-content" id="tab-breakout">
                    ${this.generateBreakoutScenario(player, constraints, trad, adv)}
                </div>
            </div>
        `;
    }

    renderMetricsTab(player, evalOut, mechanism, valuation, trad, adv, trust, uncertainty) {
        const fit = evalOut.fitScenario || {};
        const fitPercentile = this.getTeamFitPercentile(player);
        const usage = ((adv.usage_rate ?? 0) * 100).toFixed(1);
        const ledger = evalOut.intrinsicContextLedger || { intrinsic: 0.5, context: 0.5, residual: 0, intrinsic_share: 0.5, context_share: 0.5 };
        const valueDecomposition = evalOut.valueDecomposition || { blocks: [], base_score: 0, team_weighted_score: 0 };
        const clampReport = evalOut.clampReport || [];
        const impactAudit = evalOut.impactAudit || { verdict: 'insufficient_data', disagreement: 1, required_justification: [] };
        const defenseRole = evalOut.defenseRole || { primary_role: 'Unknown', targeting_risk: 0.5, warnings: [] };
        const evidence = evalOut.evidence || { coverage: 0.7, coverage_effective: 0.7, missing: [], penalty: 0, penalty_total: 0, grade: 'moderate' };
        const minutesGov = evalOut.minutesGovernance || { visible_status: 'watch', reason: 'No minutes governance output', evidence_penalty: 0, feasibility_penalty: 0, mpg: 0, games_played: 0 };
        const agingSummary = evalOut.agingSummary || this.getAgingAdjustmentSummary(player, valuation);
        const contractSummary = evalOut.contractSummary || this.getContractSurplusSummary(valuation);
        const reportContract = evalOut.reportContract || {};
        const mechanismRows = [
            {
                label: 'Suppression Relief',
                value: `${(mechanism.suppression_relief * 100).toFixed(0)}%`,
                meaning: 'Estimated value unlocked when role + scheme friction is removed.',
                interpretation: mechanism.suppression_relief >= 0.6 ? 'High unlock potential in better context.' : (mechanism.suppression_relief < 0.4 ? 'Limited latent value unlock.' : 'Moderate contextual unlock.')
            },
            {
                label: 'Spacing Gravity Skill',
                value: `${(mechanism.spacing_gravity_skill * 100).toFixed(0)}%`,
                meaning: 'Off-ball defensive pull generated by shooting profile/deployment.',
                interpretation: mechanism.spacing_gravity_skill >= 0.6 ? 'Defenses must account for this spacing threat.' : 'Spacing impact is currently secondary.'
            },
            {
                label: 'Defense Portability',
                value: `${(mechanism.portability_score * 100).toFixed(0)}%`,
                meaning: 'How well defensive role survives matchup targeting and scheme shifts.',
                interpretation: mechanism.portability_score >= 0.6 ? 'Playoff-resilient defensive translation.' : (mechanism.portability_score < 0.45 ? 'Likely requires tactical protection.' : 'Situationally stable.')
            },
            {
                label: 'Clamp Severity',
                value: `${(mechanism.clamp_severity * 100).toFixed(0)}%`,
                meaning: 'Constraint pressure from role/usage/minutes feasibility limits.',
                interpretation: mechanism.clamp_severity <= 0.4 ? 'Scenario is loosely constrained.' : (mechanism.clamp_severity >= 0.65 ? 'Projection heavily constrained by feasibility.' : 'Moderate constraint pressure.')
            },
            {
                label: 'Prior Disagreement',
                value: `${(mechanism.impact_prior_disagreement * 100).toFixed(0)}%`,
                meaning: 'Distance between internal value signals and external impact priors.',
                interpretation: mechanism.impact_prior_disagreement <= 0.35 ? 'Signals align well with priors.' : 'Disagreement requires caution and review.'
            },
            {
                label: 'Comp Density',
                value: `${(mechanism.comp_density * 100).toFixed(0)}%`,
                meaning: 'How dense/anchored this profile is among similar players.',
                interpretation: mechanism.comp_density < 0.5 ? 'Unique profile; wider uncertainty band.' : 'Comparable archetype support is decent.'
            }
        ];

        return `
            <div class="detail-section">
                <h3>Mechanism Dashboard</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Player Value</div>
                        <div class="detail-item-value">${this.getPlayerValue(player).toFixed(1)}</div>
                        <div class="detail-item-note">Uniformly scaled intrinsic value proxy (EPM + LEBRON blend).</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Breakout Score</div>
                        <div class="detail-item-value">${evalOut.breakoutScore.toFixed(1)}</div>
                        <div class="detail-item-note">Context-aware breakout likelihood, calibrated by role constraints.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Team Fit Context</div>
                        <div class="detail-item-value">${((fit.fit_score || 0) * 100).toFixed(0)}% (p${fitPercentile})</div>
                        <div class="detail-item-note">How strongly team tactical needs match this player profile; percentile is league-relative.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Scenario Confidence</div>
                        <div class="detail-item-value">${(mechanism.confidence * 100).toFixed(0)}%</div>
                        <div class="detail-item-note">Confidence governance combining data quality, comp density, and agreement.</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>What The Complex Metrics Mean</h3>
                <div class="mechanism-explainer-grid">
                    ${mechanismRows.map(row => `
                        <div class="mechanism-explainer-card">
                            <div class="mechanism-explainer-head">
                                <div class="mechanism-explainer-label">${row.label}</div>
                                <div class="mechanism-explainer-value">${row.value}</div>
                            </div>
                            <div class="mechanism-explainer-meaning">${row.meaning}</div>
                            <div class="mechanism-explainer-read">${row.interpretation}</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Value Decomposition Vector</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Base Vector Score</div>
                        <div class="detail-item-value">${(100 * (valueDecomposition.base_score || 0)).toFixed(0)}%</div>
                        <div class="detail-item-note">Unweighted player mechanism profile across offense/defense role blocks.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Team-Weighted Vector Score</div>
                        <div class="detail-item-value">${(100 * (valueDecomposition.team_weighted_score || 0)).toFixed(0)}%</div>
                        <div class="detail-item-note">Mechanism profile re-weighted by team context fit dimensions.</div>
                    </div>
                </div>
                <div class="mechanism-explainer-grid" style="margin-top: 0.8rem;">
                    ${(valueDecomposition.blocks || []).map(b => `
                        <div class="mechanism-explainer-card">
                            <div class="mechanism-explainer-head">
                                <div class="mechanism-explainer-label">${b.label}</div>
                                <div class="mechanism-explainer-value">${(100 * (b.value || 0)).toFixed(0)}%</div>
                            </div>
                            <div class="mechanism-explainer-meaning">Team weight ${(100 * (b.team_weight || 0)).toFixed(0)}%</div>
                            <div class="mechanism-explainer-read">Weighted contribution ${(100 * (b.team_weighted_value || 0)).toFixed(0)}%.</div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Intrinsic vs Context Ledger</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Intrinsic Component</div>
                        <div class="detail-item-value">${(ledger.intrinsic * 100).toFixed(0)}%</div>
                        <div class="detail-item-note">Player-owned signal from value priors, spacing skill, and defensive portability.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Context Component</div>
                        <div class="detail-item-value">${(ledger.context * 100).toFixed(0)}%</div>
                        <div class="detail-item-note">System translation signal from fit edge, usage/minutes runway, and scheme complementarity.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Intrinsic Share</div>
                        <div class="detail-item-value">${(ledger.intrinsic_share * 100).toFixed(0)}%</div>
                        <div class="detail-item-note">Share of explainable breakout path attributable to player-owned factors.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Context Share</div>
                        <div class="detail-item-value">${(ledger.context_share * 100).toFixed(0)}%</div>
                        <div class="detail-item-note">Share of explainable breakout path attributable to team/deployment context.</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Clamps Fired</h3>
                <div class="constraints-list">
                    ${clampReport.length ? clampReport.map(c => `
                        <div class="constraint-item">
                            <span class="constraint-icon">⛓️</span>
                            <span class="constraint-text"><strong>${c.clamp}</strong> (${(100 * (c.severity || 0)).toFixed(0)}%): ${c.message} Unlock path: ${c.unlock}</span>
                        </div>
                    `).join('') : `
                        <div class="constraint-item">
                            <span class="constraint-icon">✅</span>
                            <span class="constraint-text">No explicit clamp fire events were triggered for this scenario.</span>
                        </div>
                    `}
                </div>
            </div>

            <div class="detail-section">
                <h3>Minutes Governance</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Minutes Gate</div>
                        <div class="detail-item-value">${String(minutesGov.visible_status || 'watch').toUpperCase()}</div>
                        <div class="detail-item-note">${minutesGov.reason || 'Minutes governance reason unavailable.'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">MPG / GP</div>
                        <div class="detail-item-value">${(minutesGov.mpg || 0).toFixed(1)} / ${(minutesGov.games_played || 0).toFixed(0)}</div>
                        <div class="detail-item-note">Low minutes are treated as confidence + feasibility clamps, not hard exclusion.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Minutes Evidence Penalty</div>
                        <div class="detail-item-value">${((minutesGov.evidence_penalty || 0) * 100).toFixed(1)}%</div>
                        <div class="detail-item-note">Penalty applied to confidence governance for low-minute or low-game evidence.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Minutes Feasibility Penalty</div>
                        <div class="detail-item-value">${((minutesGov.feasibility_penalty || 0) * 100).toFixed(1)}%</div>
                        <div class="detail-item-note">Penalty applied when scenario requires aggressive rotation promotion.</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Impact Audit (Adult Supervision)</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Audit Verdict</div>
                        <div class="detail-item-value">${String(impactAudit.verdict || 'insufficient_data').replace('_', ' ').toUpperCase()}</div>
                        <div class="detail-item-note">Compares breakout signal quality vs EPM/LEBRON prior anchor consensus.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Prior Consensus</div>
                        <div class="detail-item-value">${impactAudit.consensus === null || impactAudit.consensus === undefined ? 'N/A' : impactAudit.consensus.toFixed(2)}</div>
                        <div class="detail-item-note">Positive values support strong projected impact; negative values increase scrutiny.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Prior Disagreement</div>
                        <div class="detail-item-value">${(100 * (impactAudit.disagreement ?? mechanism.impact_prior_disagreement ?? 0)).toFixed(0)}%</div>
                        <div class="detail-item-note">Distance between priors; high disagreement lowers confidence.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Required Justification</div>
                        <div class="detail-item-value">${impactAudit.required_justification?.length ? impactAudit.required_justification.length : 0}</div>
                        <div class="detail-item-note">${impactAudit.required_justification?.length ? impactAudit.required_justification[0] : 'No mandatory contradiction note triggered.'}</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Defense Role Identity</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Primary Defensive Role</div>
                        <div class="detail-item-value">${defenseRole.primary_role || 'Unknown'}</div>
                        <div class="detail-item-note">Role identity proxy from matchup profile distribution + position context.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Playoff Targeting Risk</div>
                        <div class="detail-item-value">${(100 * (defenseRole.targeting_risk || 0)).toFixed(0)}%</div>
                        <div class="detail-item-note">Higher risk automatically clamps aggressive offensive role expansion.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Role Mix</div>
                        <div class="detail-item-value">G ${(100 * (defenseRole.role_mix?.guards || 0)).toFixed(0)} / W ${(100 * (defenseRole.role_mix?.wings || 0)).toFixed(0)} / B ${(100 * (defenseRole.role_mix?.bigs || 0)).toFixed(0)}</div>
                        <div class="detail-item-note">Guard/Wing/Big matchup distribution used to infer portability stress.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Breakpoint Warning</div>
                        <div class="detail-item-value">${defenseRole.warnings?.length ? 'ACTIVE' : 'CLEAR'}</div>
                        <div class="detail-item-note">${defenseRole.warnings?.[0] || 'No major defensive breakpoint warning triggered.'}</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Observed Box & Role Context</h3>
                <div class="detail-grid">
                    ${this.formatMetricWithPercentile(player, 'Usage Rate', usage + '%', 'usage')}
                    ${this.formatMetricWithPercentile(player, 'Points Per Game', (trad.points_per_game || 0).toFixed(1), 'points')}
                    ${this.formatMetricWithPercentile(player, 'Assists Per Game', (trad.assists_per_game || 0).toFixed(1), 'assists')}
                    ${this.formatMetricWithPercentile(player, 'Rebounds Per Game', (trad.rebounds_per_game || 0).toFixed(1), 'rebounds')}
                    ${this.formatMetricWithPercentile(player, 'Rim Frequency', ((player.shot_profile?.rim_frequency || 0) * 100).toFixed(0) + '%', 'rim_freq')}
                    ${this.formatMetricWithPercentile(player, 'Three-Point Frequency', ((player.shot_profile?.three_point_frequency || 0) * 100).toFixed(0) + '%', 'three_freq')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Data Governance</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Trust Score</div>
                        <div class="detail-item-value">${(trust.score || 0).toFixed(0)}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Trust Level</div>
                        <div class="detail-item-value">${(trust.level || 'N/A').toUpperCase()}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Overall Uncertainty</div>
                        <div class="detail-item-value">${((uncertainty.overall_uncertainty || 0) * 100).toFixed(1)}%</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Signal Strength</div>
                        <div class="detail-item-value">${(fit.signal_strength || 'weak').toUpperCase()}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Effective Scenario Confidence</div>
                        <div class="detail-item-value">${(100 * (evalOut.effectiveScenarioConfidence || fit.confidence || 0)).toFixed(0)}%</div>
                        <div class="detail-item-note">Fit confidence after audit/evidence governance dampeners.</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Evidence Coverage</div>
                        <div class="detail-item-value">${(evidence.coverage_effective * 100).toFixed(0)}%</div>
                        <div class="detail-item-note">Observed field coverage driving confidence governance (${evidence.grade.toUpperCase()}).</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Missing Inputs</div>
                        <div class="detail-item-value">${evidence.missing.length}</div>
                        <div class="detail-item-note">${evidence.missing.length ? evidence.missing.slice(0, 3).join(', ') : 'No major required fields missing.'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Total Evidence Penalty</div>
                        <div class="detail-item-value">${((evidence.penalty_total || evidence.penalty || 0) * 100).toFixed(1)}%</div>
                        <div class="detail-item-note">Base missingness penalty plus minutes-evidence penalty.</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Report Contract</h3>
                <div class="detail-grid">
                    <div class="detail-item">
                        <div class="detail-item-label">Identity Snapshot</div>
                        <div class="detail-item-value">${reportContract.identity_snapshot ? 'PRESENT' : 'MISSING'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Value Decomposition</div>
                        <div class="detail-item-value">${reportContract.value_decomposition ? 'PRESENT' : 'MISSING'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Counterfactual + Clamps</div>
                        <div class="detail-item-value">${reportContract.counterfactual_with_clamps ? 'PRESENT' : 'MISSING'}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-item-label">Decision Layer</div>
                        <div class="detail-item-value">${reportContract.decision_layer ? 'PRESENT' : 'MISSING'}</div>
                    </div>
                </div>
            </div>

            ${valuation ? `
                <div class="detail-section">
                    <h3>Valuation Context</h3>
                    <div class="detail-grid">
                        ${this.formatMetricWithPercentile(player, 'Wins Added', valuation.impact.wins_added.toFixed(1), 'wins_added')}
                        ${this.formatMetricWithPercentile(player, 'Trade Value (Base)', valuation.trade_value.base.toFixed(1) + 'M', 'trade_value')}
                        <div class="detail-item">
                            <div class="detail-item-label">Contract NPV Surplus</div>
                            <div class="detail-item-value">${contractSummary.npv_surplus !== null && contractSummary.npv_surplus !== undefined ? contractSummary.npv_surplus.toFixed(2) + 'M' : 'N/A'}</div>
                            <div class="detail-item-note">Projected value minus contract cost over control horizon.</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Aging Phase</div>
                            <div class="detail-item-value">${this.formatAgingPhase(valuation.aging.current_phase)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Peak Age</div>
                            <div class="detail-item-value">${valuation.aging.peak_age.toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Aging Adjustment</div>
                            <div class="detail-item-value">${agingSummary.pct >= 0 ? '+' : ''}${agingSummary.pct.toFixed(1)}%</div>
                            <div class="detail-item-note">Current age vs archetype/valuation curve baseline (${agingSummary.source}).</div>
                        </div>
                    </div>
                </div>
            ` : ''}
        `;
    }

    calculateArchetypePercentile(player, metric) {
        // Get all players with same archetype
        const archetype = player.identity?.primary_archetype;
        const archetypePlayers = this.players.filter(p => 
            p.identity?.primary_archetype === archetype
        );

        if (archetypePlayers.length < 2) return null;

        // Get metric value for current player
        let playerValue;
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const shot = player.shot_profile || {};
        const creation = player.creation_profile || {};
        const defense = player.defense_assessment || {};

        switch(metric) {
            case 'player_value':
                playerValue = this.getPlayerValue(player);
                break;
            case 'points':
                playerValue = trad.points_per_game || 0;
                break;
            case 'assists':
                playerValue = trad.assists_per_game || 0;
                break;
            case 'rebounds':
                playerValue = trad.rebounds_per_game || 0;
                break;
            case 'fg_pct':
                playerValue = trad.field_goal_pct || 0;
                break;
            case 'three_pct':
                playerValue = trad.three_point_pct || 0;
                break;
            case 'usage':
                playerValue = adv.usage_rate ?? 0;
                break;
            case 'steals':
                playerValue = trad.steals_per_game || 0;
                break;
            case 'blocks':
                playerValue = trad.blocks_per_game || 0;
                break;
            case 'rim_freq':
                playerValue = shot.rim_frequency || 0;
                break;
            case 'three_freq':
                playerValue = shot.three_point_frequency || 0;
                break;
            case 'drives':
                playerValue = creation.drives_per_game || 0;
                break;
            case 'paint_touches':
                playerValue = creation.paint_touches_per_game || 0;
                break;
            case 'assisted_rate':
                playerValue = creation.assisted_rate || 0;
                break;
            case 'def_rating':
                playerValue = defense.estimated_metrics?.defensive_rating || 0;
                break;
            case 'wins_added':
                playerValue = player.valuation?.impact?.wins_added || 0;
                break;
            case 'trade_value':
                playerValue = player.valuation?.trade_value?.base || 0;
                break;
            default:
                return null;
        }

        // Get all values for this metric
        const allValues = archetypePlayers.map(p => {
            const t = p.performance?.traditional || {};
            const a = p.performance?.advanced || {};
            const s = p.shot_profile || {};
            const c = p.creation_profile || {};
            const d = p.defense_assessment || {};
            
            switch(metric) {
                case 'player_value': return this.getPlayerValue(p);
                case 'points': return t.points_per_game || 0;
                case 'assists': return t.assists_per_game || 0;
                case 'rebounds': return t.rebounds_per_game || 0;
                case 'fg_pct': return t.field_goal_pct || 0;
                case 'three_pct': return t.three_point_pct || 0;
                case 'usage': return a.usage_rate ?? 0;
                case 'steals': return t.steals_per_game || 0;
                case 'blocks': return t.blocks_per_game || 0;
                case 'rim_freq': return s.rim_frequency || 0;
                case 'three_freq': return s.three_point_frequency || 0;
                case 'drives': return c.drives_per_game || 0;
                case 'paint_touches': return c.paint_touches_per_game || 0;
                case 'assisted_rate': return c.assisted_rate || 0;
                case 'def_rating': return d.estimated_metrics?.defensive_rating || 0;
                case 'wins_added': return p.valuation?.impact?.wins_added || 0;
                case 'trade_value': return p.valuation?.trade_value?.base || 0;
                default: return 0;
            }
        }).sort((a, b) => a - b);

        // Calculate percentile
        const rank = allValues.filter(v => v < playerValue).length;
        const percentile = (rank / allValues.length) * 100;

        return Math.round(percentile);
    }

    formatMetricWithPercentile(player, label, value, metric) {
        const percentile = this.calculateArchetypePercentile(player, metric);
        const percentileClass = percentile >= 75 ? 'percentile-high' : percentile >= 50 ? 'percentile-mid' : 'percentile-low';
        
        return `
            <div class="detail-item">
                <div class="detail-item-label">${label}</div>
                <div class="detail-item-value">${value}</div>
                ${percentile !== null ? `
                    <div class="detail-item-percentile ${percentileClass}">
                        ${percentile}th percentile
                    </div>
                ` : ''}
            </div>
        `;
    }

    resetFilters() {
        document.getElementById('searchInput').value = '';
        document.getElementById('archetypeFilter').value = '';
        document.getElementById('usageFilter').value = '';
        document.getElementById('teamFilter').value = '';
        document.getElementById('sortBy').value = 'name';
        this.currentSort = 'name';
        // Restore landing behavior: cards by name, distribution by team fit.
        this.initialDistributionMetricKey = 'team_fit';
        this.filterPlayers();
    }

    formatArchetype(archetype) {
        return archetype.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    getPlayerMonogram(name) {
        const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
        if (!parts.length) return 'NA';
        if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
        return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
    }

    getPlayerHeadshotUrl(player) {
        const id = String(player.player?.id || '').trim();
        return `https://cdn.nba.com/headshots/nba/latest/1040x760/${id}.png`;
    }

    getTeamLogoUrl(team) {
        const codeMap = {
            ATL: 'atl', BOS: 'bos', BKN: 'bkn', CHA: 'cha', CHI: 'chi',
            CLE: 'cle', DAL: 'dal', DEN: 'den', DET: 'det', GSW: 'gs',
            HOU: 'hou', IND: 'ind', LAC: 'lac', LAL: 'lal', MEM: 'mem',
            MIA: 'mia', MIL: 'mil', MIN: 'min', NOP: 'no', NYK: 'ny',
            OKC: 'okc', ORL: 'orl', PHI: 'phi', PHX: 'phx', POR: 'por',
            SAC: 'sac', SAS: 'sa', TOR: 'tor', UTA: 'uta', WAS: 'wsh'
        };
        const clean = String(team || '').toUpperCase().trim();
        const espnCode = codeMap[clean];
        return espnCode ? `https://a.espncdn.com/i/teamlogos/nba/500/${espnCode}.png` : '';
    }

    getBreakoutTierStyle(likelihoodClass, likelihoodPct) {
        const cls = String(likelihoodClass || '').toLowerCase();
        const pct = Number(likelihoodPct || 0);
        if (cls === 'high' || pct >= 70) {
            return { label: 'HIGH', bg: '#16a34a', fg: '#ecfdf5', glow: 'rgba(22,163,74,0.45)' };
        }
        if (cls === 'medium' || pct >= 45) {
            return { label: 'MEDIUM', bg: '#d97706', fg: '#fffbeb', glow: 'rgba(217,119,6,0.42)' };
        }
        return { label: 'LOW', bg: '#dc2626', fg: '#fef2f2', glow: 'rgba(220,38,38,0.42)' };
    }

    isLikelyRookie(player) {
        const candidateFlags = [
            player?.player?.rookie,
            player?.player?.is_rookie,
            player?.player?.isRookie,
            player?.identity?.rookie,
            player?.identity?.is_rookie,
            player?.identity?.isRookie
        ];
        for (const flag of candidateFlags) {
            if (flag === true) return true;
            if (String(flag || '').toLowerCase() === 'true') return true;
        }

        const expCandidates = [
            player?.player?.years_experience,
            player?.player?.experience,
            player?.player?.nba_experience,
            player?.identity?.years_experience,
            player?.identity?.experience
        ];
        for (const exp of expCandidates) {
            const n = Number(exp);
            if (Number.isFinite(n)) return n <= 0;
            const s = String(exp || '').trim().toLowerCase();
            if (s === 'r' || s === 'rookie' || s === '0') return true;
        }

        // Conservative fallback for this dataset shape.
        const age = Number(player?.player?.age);
        return Number.isFinite(age) ? age <= 20 : false;
    }

    getBasketballTypeProfile(player) {
        const archetype = String(player.identity?.primary_archetype || '').toLowerCase();
        const map = {
            lead_guard: {
                typeLabel: 'Lead Engine',
                subtitle: 'Tempo Creator',
                styleLine: 'Primary organizer who bends defenses with initiation volume, touch creation, and pace control.',
                primaryColor: '#8f1dff',
                secondaryColor: '#1d8dff',
                glowColor: 'rgba(143, 29, 255, 0.32)'
            },
            versatile_wing: {
                typeLabel: 'Two-Way Wing',
                subtitle: 'Switch Connector',
                styleLine: 'Flexible wing profile built for off-ball scaling, matchup switching, and lineup glue value.',
                primaryColor: '#0057d9',
                secondaryColor: '#00a8a8',
                glowColor: 'rgba(0, 87, 217, 0.30)'
            },
            stretch_big: {
                typeLabel: 'Stretch Anchor',
                subtitle: 'Paint-to-Perimeter Big',
                styleLine: 'Frontcourt spacer that preserves interior gravity while stretching coverage past the arc.',
                primaryColor: '#d97706',
                secondaryColor: '#f59e0b',
                glowColor: 'rgba(217, 119, 6, 0.34)'
            },
            rim_pressure_guard: {
                typeLabel: 'Pressure Guard',
                subtitle: 'Rim Collapse Driver',
                styleLine: 'Downhill guard archetype that collapses shell coverages and creates chain-reaction passing lanes.',
                primaryColor: '#dc2626',
                secondaryColor: '#f97316',
                glowColor: 'rgba(220, 38, 38, 0.30)'
            },
            '3_and_d_wing': {
                typeLabel: '3-and-D Wing',
                subtitle: 'Spacing Stopper',
                styleLine: 'Role-optimized wing focused on floor spacing, shot discipline, and possession-level defensive containment.',
                primaryColor: '#0f766e',
                secondaryColor: '#22c55e',
                glowColor: 'rgba(15, 118, 110, 0.30)'
            }
        };

        return map[archetype] || {
            typeLabel: this.formatArchetype(archetype || 'balanced role'),
            subtitle: 'Balanced Contributor',
            styleLine: 'Hybrid profile with mixed on-ball and off-ball utility that adapts to lineup context.',
            primaryColor: '#334155',
            secondaryColor: '#64748b',
            glowColor: 'rgba(71, 85, 105, 0.30)'
        };
    }

    formatRole(role) {
        return role.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    formatAgingPhase(phase) {
        return phase.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    drawShotChart(player) {
        const canvas = document.getElementById('shotChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Get shot frequencies and percentages
        const rimFreq = player.shot_profile?.rim_frequency || 0;
        const midFreq = player.shot_profile?.mid_range_frequency || 0;
        const threeFreq = player.shot_profile?.three_point_frequency || 0;
        const fgPct = player.performance?.traditional?.field_goal_pct || 0;
        const threePct = player.performance?.traditional?.three_point_pct || 0;
        const assistedRate = this.clampNum(Number(player.creation_profile?.assisted_rate ?? 0.56), 0.1, 0.95);
        const mode = this.shotChartMode || 'volume';
        const contextMods = this.getTrueImpactContext(player);

        // Draw court image as background (preserve aspect ratio; no stretching).
        if (this.courtImage && this.courtImage.complete) {
            const imgW = this.courtImage.naturalWidth || this.courtImage.width || width;
            const imgH = this.courtImage.naturalHeight || this.courtImage.height || height;
            const imgAspect = imgW / imgH;
            const canvasAspect = width / height;
            let drawW;
            let drawH;
            if (imgAspect > canvasAspect) {
                drawW = width;
                drawH = width / imgAspect;
            } else {
                drawH = height;
                drawW = height * imgAspect;
            }

            ctx.save();
            ctx.translate(width / 2, height / 2);
            ctx.rotate(Math.PI);
            ctx.drawImage(this.courtImage, -drawW / 2, -drawH / 2, drawW, drawH);
            ctx.restore();
        } else {
            // Fallback: light background
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, width, height);
        }

        // Hexagon parameters
        const hexSize = 28;
        
        // Court dimensions
        const courtWidth = width;
        const courtHeight = height - 50;
        const centerX = courtWidth / 2;
        const baselineY = courtHeight;

        // Define shot zones with hexagon coverage and shooting percentages
        const shotZones = this.generateShotZonesWithPercentages(
            centerX, baselineY, rimFreq, midFreq, threeFreq, fgPct, threePct, assistedRate, contextMods
        );

        // Draw hexagons
        shotZones.forEach(zone => {
            const intensity = mode === 'volume' ? (zone.intensity ?? 0) : (zone.metricIntensity ?? 0);
            const volumeVisibility = zone.intensity ?? 0;
            const blendedVisibility = mode === 'volume'
                ? intensity
                : this.clampNum((0.60 * intensity) + (0.40 * volumeVisibility), 0, 1);
            
            if (blendedVisibility > 0.03) {
                // Color based on current mode and metric value.
                const baseColor = this.getShotHexColor(zone, mode);

                // Alpha combines metric intensity + confidence, so uncertain zones fade out.
                const zoneConfidence = zone.confidence ?? 0.65;
                const alpha = this.clampNum((0.14 + (blendedVisibility * 0.66)) * (0.45 + 0.55 * zoneConfidence), 0.10, 0.88);
                
                ctx.fillStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${alpha})`;
                ctx.strokeStyle = `rgba(255, 255, 255, 0.3)`;
                ctx.lineWidth = 1;

                this.drawHexagon(ctx, zone.x, zone.y, hexSize);
                ctx.fill();
                ctx.stroke();

                // Draw labels only for non-volume modes to avoid implying FG% in a volume chart.
                if ((mode !== 'volume') && (blendedVisibility > 0.2) && (zone.shootingPct > 0) && ((zone.frequency || 0) >= 0.0075)) {
                    ctx.fillStyle = '#1f2937';
                    ctx.font = 'bold 10px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    const label = mode === 'fg'
                        ? `${(zone.shootingPct * 100).toFixed(0)}%`
                        : (mode === 'value'
                            ? `${zone.pointsPerShot.toFixed(2)}`
                            : (mode === 'trueimpact'
                                ? `${zone.trueImpactValue.toFixed(2)}`
                                : `${(zone.shootingPct * 100).toFixed(0)}%`)
                        );
                    ctx.fillText(label, zone.x, zone.y);
                }
            }
        });

        // Draw readable legend strip at bottom so labels are never lost on the image.
        const footerTop = height - 52;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
        ctx.fillRect(0, footerTop, width, height - footerTop);
        ctx.strokeStyle = 'rgba(44, 62, 80, 0.12)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, footerTop);
        ctx.lineTo(width, footerTop);
        ctx.stroke();

        // Draw legend at bottom
        const legendY = height - 35;
        const zoneSummary = this.estimateZoneShooting(rimFreq, midFreq, threeFreq, fgPct, threePct);
        const legendItems = mode === 'volume'
            ? [
                { color: 'rgba(255, 107, 107, 0.7)', label: `Rim: ${(rimFreq * 100).toFixed(0)}%` },
                { color: 'rgba(255, 217, 61, 0.7)', label: `Mid: ${(midFreq * 100).toFixed(0)}%` },
                { color: 'rgba(107, 207, 127, 0.7)', label: `3PT: ${(threeFreq * 100).toFixed(0)}%` }
            ]
            : (mode === 'fg'
                ? [
                    { color: 'rgba(220, 38, 38, 0.7)', label: 'Lower FG%', pct: 'Below average zone efficiency' },
                    { color: 'rgba(245, 158, 11, 0.7)', label: 'Average FG%', pct: 'Near player baseline' },
                    { color: 'rgba(22, 163, 74, 0.7)', label: 'Higher FG%', pct: 'Above average zone efficiency' }
                ]
                : (mode === 'trueimpact'
                    ? [
                        { color: 'rgba(220, 38, 38, 0.7)', label: 'Low True Impact', pct: '< 1.00 adj pts/shot' },
                        { color: 'rgba(245, 158, 11, 0.7)', label: 'Medium Impact', pct: '1.00 - 1.20 adj pts/shot' },
                        { color: 'rgba(22, 163, 74, 0.7)', label: 'High True Impact', pct: '> 1.20 adj pts/shot' }
                    ]
                    : [
                        { color: 'rgba(220, 38, 38, 0.7)', label: 'Low Shot Value', pct: '< 0.95 pts/shot' },
                        { color: 'rgba(245, 158, 11, 0.7)', label: 'Medium Value', pct: '0.95 - 1.15 pts/shot' },
                        { color: 'rgba(22, 163, 74, 0.7)', label: 'High Value', pct: '> 1.15 pts/shot' }
                    ]));

        // Center the legend
        const totalLegendWidth = legendItems.length * 130;
        let legendX = (width - totalLegendWidth) / 2;

        legendItems.forEach(item => {
            // Color box with shadow
            ctx.shadowColor = 'rgba(0, 0, 0, 0.15)';
            ctx.shadowBlur = 3;
            ctx.shadowOffsetY = 2;
            
            ctx.fillStyle = item.color;
            ctx.fillRect(legendX, legendY, 22, 22);
            
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetY = 0;
            
            ctx.strokeStyle = '#dee2e6';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(legendX, legendY, 22, 22);

            // Label
            ctx.fillStyle = '#1f2937';
            ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(item.label, legendX + 28, legendY + 10);
            
            // Optional second line (not shown in volume mode).
            if (item.pct) {
                ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
                ctx.fillStyle = '#374151';
                ctx.fillText(item.pct, legendX + 28, legendY + 20);
            }

            legendX += 130;
        });

        // Data source note
        ctx.fillStyle = '#374151';
        ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
        ctx.textAlign = 'center';
        const note = mode === 'volume'
            ? 'Modeled shot volume from rim/mid/3 frequencies (no XY logs).'
            : (mode === 'trueimpact'
                ? 'True Impact adjusts shot value by rebound/transition context; lower-confidence zones are faded.'
                : 'FG%/Value are quantitatively estimated from aggregate shooting + zone priors.');
        ctx.fillText(note, width / 2, height - 8);
    }

    getTrueImpactContext(player) {
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const shot = player.shot_profile || {};
        const creation = player.creation_profile || {};
        const trust = (player.v1_1_enhancements?.trust_assessment?.score || 60) / 100;
        const team = player.player?.team;
        const teamProfile = this.teamProfiles?.[team] || {};
        const teamScheme = teamProfile.scheme || {};

        const rebounds = trad.rebounds_per_game || 0;
        const steals = trad.steals_per_game || 0;
        const usage = adv.usage_rate ?? 0.2;
        const drives = creation.drives_per_game || 0;
        const rimFreq = shot.rim_frequency || 0.3;
        const threeFreq = shot.three_point_frequency || 0.34;

        const orProxy = this.clampNum(
            (0.45 * this.clampNum(rebounds / 10.0, 0, 1)) +
            (0.35 * this.clampNum(rimFreq / 0.5, 0, 1)) +
            (0.20 * this.clampNum((teamScheme.rim_pressure || 0.4), 0, 1)),
            0, 1
        );
        const transitionProxy = this.clampNum(
            (0.35 * this.clampNum(drives / 15.0, 0, 1)) +
            (0.25 * this.clampNum(steals / 1.5, 0, 1)) +
            (0.20 * this.clampNum((teamScheme.on_ball_creation || 0.4), 0, 1)) +
            (0.20 * this.clampNum((threeFreq + usage) / 0.7, 0, 1)),
            0, 1
        );

        return {
            offensive_rebound_factor: orProxy,
            transition_factor: transitionProxy,
            confidence_base: this.clampNum(
                (0.60 * trust) +
                (0.20 * this.clampNum((player.defense_assessment?.visibility?.observability_score || 30) / 100, 0, 1)) +
                (0.20 * this.clampNum((trad.games_played || 0) / 82, 0, 1)),
                0.35,
                0.95
            )
        };
    }

    estimateZoneShooting(rimFreq, midFreq, threeFreq, fgPct, threePct) {
        // 82games anchored priors (locations + assisted study).
        // locations.htm FG%: corner3 .425, wing3 .349, straight3 .388,
        // baseline2 .439, wing2 .385, straight2 .453, high paint .450, low paint .600.
        const priors = {
            three_avg: 0.388,
            two_avg: 0.448,
            rim_fg: 0.600,
            mid_fg: 0.420
        };

        const threeFG = this.clampNum(threePct || priors.three_avg, 0.26, 0.50);
        const nonThreeFreq = Math.max(0.0001, rimFreq + midFreq);
        const targetNonThree = this.safeDiv((fgPct - (threeFreq * threeFG)), nonThreeFreq, priors.two_avg);

        // Solve broad rim/mid targets first; subzone values are set later.
        let rimFG = this.clampNum(priors.rim_fg * this.safeDiv(targetNonThree, priors.two_avg, 1), 0.50, 0.76);
        let midFG = this.safeDiv((targetNonThree * nonThreeFreq) - (rimFreq * rimFG), Math.max(midFreq, 0.0001), priors.mid_fg);
        if (!Number.isFinite(midFG)) midFG = priors.mid_fg;
        midFG = this.clampNum(midFG, 0.30, 0.56);

        return { rim_fg: rimFG, mid_fg: midFG, three_fg: threeFG };
    }

    getShotHexColor(zone, mode) {
        if (mode === 'volume') {
            if (zone.type === 'rim') return { r: 255, g: 107, b: 107 };
            if (zone.type === 'paint') return { r: 255, g: 217, b: 61 };
            if (zone.type === 'mid') return { r: 255, g: 179, b: 71 };
            if (zone.type === 'three') return { r: 107, g: 207, b: 127 };
            return { r: 78, g: 205, b: 196 };
        }

        const value = mode === 'fg'
            ? zone.shootingPct
            : (mode === 'trueimpact' ? zone.trueImpactValue : zone.pointsPerShot);
        const min = mode === 'fg' ? 0.32 : (mode === 'trueimpact' ? 0.90 : 0.82);
        const max = mode === 'fg' ? 0.62 : (mode === 'trueimpact' ? 1.40 : 1.35);
        const t = this.clampNum((value - min) / (max - min), 0, 1);

        // Red -> Yellow -> Green gradient
        let r, g, b;
        if (t < 0.5) {
            const k = t / 0.5;
            r = 220 + (245 - 220) * k;
            g = 38 + (158 - 38) * k;
            b = 38 + (11 - 38) * k;
        } else {
            const k = (t - 0.5) / 0.5;
            r = 245 + (22 - 245) * k;
            g = 158 + (163 - 158) * k;
            b = 11 + (74 - 11) * k;
        }
        return { r: Math.round(r), g: Math.round(g), b: Math.round(b) };
    }

    drawHexagon(ctx, x, y, size) {
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i;
            const hx = x + size * Math.cos(angle);
            const hy = y + size * Math.sin(angle);
            if (i === 0) {
                ctx.moveTo(hx, hy);
            } else {
                ctx.lineTo(hx, hy);
            }
        }
        ctx.closePath();
    }
    generateShotZonesWithPercentages(centerX, baselineY, rimFreq, midFreq, threeFreq, fgPct, threePct, assistedRateInput = 0.56, contextMods = null) {
        const candidates = [];
        const hexSize = 28;
        const hexWidth = hexSize * 2;
        const hexHeight = hexSize * Math.sqrt(3);

        // Hoop location aligned with court image orientation (bottom half-court)
        const rimX = centerX;
        const rimY = baselineY * 0.86;

        const getDistance = (x, y) => {
            return Math.sqrt(Math.pow(x - rimX, 2) + Math.pow(y - rimY, 2));
        };

        // Generate hexagon grid
        const rows = 14;
        const cols = 12;
        const gridShiftX = 10; // slight right shift to better center the hex map over the court

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const xOffset = (row % 2) * (hexWidth * 0.75);
                const x = 20 + gridShiftX + col * (hexWidth * 1.5) + xOffset;
                const y = 30 + row * (hexHeight * 0.75);

                // Skip if outside court bounds
                if (x < 10 || x > centerX * 2 - 10 || y > baselineY) continue;

                const distance = getDistance(x, y);
                const angleFromCenter = Math.atan2(y - rimY, x - centerX);

                // Determine zone type, relative weight, and shooting percentage.
                // 3Steps-style location taxonomy is used implicitly:
                // S3-LC/S3-L/S3-C/S3-R/S3-RC and S2-LM/S2-CM/S2-RM/S2-LC/S2-CC/S2-RC.
                let type, subtype, weight, shootingPct;
                if (y < baselineY * 0.34) continue; // suppress unrealistic half-court zones

                if (distance < 54) {
                    type = 'rim';
                    subtype = 'low_paint';
                    weight = Math.max(0.25, 1.2 - distance / 62);
                    shootingPct = fgPct * 1.15;
                } else if (distance < 96 && y > baselineY * 0.52) {
                    type = 'paint';
                    subtype = 'high_paint';
                    weight = Math.max(0.15, 1.0 - distance / 142);
                    shootingPct = fgPct * 1.05;
                } else if (distance < 168) {
                    type = 'mid';
                    const absA = Math.abs(angleFromCenter);
                    if (absA > 1.05) subtype = 'baseline2';
                    else if (absA > 0.55) subtype = 'wing2';
                    else subtype = 'straight2';
                    weight = Math.max(0.1, 1.0 - Math.abs(distance - 132) / 78);
                    shootingPct = fgPct * 0.9;
                } else if (distance >= 160 && distance <= 208) {
                    const isCorner = (x < 58 || x > centerX * 2 - 58) && y > baselineY * 0.64;
                    const arcBand = this.clampNum(1 - (Math.abs(distance - 184) / 24), 0.08, 1.0);
                    const topPenalty = this.clampNum((y - (baselineY * 0.44)) / (baselineY * 0.24), 0.05, 1.0);

                    if (isCorner) {
                        type = 'three';
                        subtype = 'corner3';
                        weight = 1.08 * this.clampNum(0.70 + (0.30 * topPenalty), 0.5, 1.0);
                        shootingPct = threePct * 1.02 * this.clampNum(0.90 + (0.10 * topPenalty), 0.82, 1.02);
                    } else {
                        type = 'three';
                        const normalizedAngle = Math.abs(angleFromCenter);
                        const isWing = normalizedAngle > 0.5 && normalizedAngle < 1.5;
                        subtype = isWing ? 'wing3' : 'straight3';
                        weight = (isWing ? 1.0 : 0.9) * arcBand * topPenalty;
                        shootingPct = threePct * (isWing ? 0.98 : 0.94) * (0.72 + (0.28 * arcBand)) * (0.84 + (0.16 * topPenalty));
                    }
                } else {
                    continue;
                }

                // Ensure shooting percentage is realistic
                shootingPct = Math.min(Math.max(shootingPct, 0.2), 0.75);

                candidates.push({
                    x,
                    y,
                    row,
                    col,
                    type,
                    subtype,
                    weight,
                    distance,
                    shootingPct
                });
            }
        }

        const zoneFG = this.estimateZoneShooting(rimFreq, midFreq, threeFreq, fgPct, threePct);
        const locPriorFG = {
            corner3: 0.425,
            wing3: 0.349,
            straight3: 0.388,
            baseline2: 0.439,
            wing2: 0.385,
            straight2: 0.453,
            high_paint: 0.450,
            low_paint: 0.600
        };
        const locPriorPPP = {
            corner3: 1.188,
            wing3: 0.887,
            straight3: 1.053,
            baseline2: 0.819,
            wing2: 0.686,
            straight2: 0.857,
            high_paint: 0.852,
            low_paint: 1.171
        };

        // Assisted effect priors from assisted.htm:
        // +3.7% (3PT), +9.5% (2PT jumpers), +12.6% (close shots), +7.0% (dunks), all +8.1%.
        const assistedRate = this.clampNum(Number(assistedRateInput ?? 0.56), 0.1, 0.95);
        const assistedDelta = assistedRate - 0.56;

        const mods = contextMods || { offensive_rebound_factor: 0.5, transition_factor: 0.5, confidence_base: 0.6 };
        const groupKey = (type) => {
            if (type === 'rim') return 'rim';
            if (type === 'three') return 'three';
            return 'mid';
        };

        const targetFreq = { rim: rimFreq, mid: midFreq, three: threeFreq };
        const totalWeight = { rim: 0, mid: 0, three: 0 };
        candidates.forEach((z) => {
            totalWeight[groupKey(z.type)] += z.weight;
        });

        const zones = candidates.map((z) => {
            const group = groupKey(z.type);
            const groupTotal = totalWeight[group] || 1;
            const frequency = targetFreq[group] * (z.weight / groupTotal);
            const broadTarget = group === 'rim' ? zoneFG.rim_fg : (group === 'three' ? zoneFG.three_fg : zoneFG.mid_fg);
            const sub = z.subtype || (group === 'three' ? 'straight3' : (group === 'rim' ? 'low_paint' : 'wing2'));
            const baseFG = locPriorFG[sub] ?? broadTarget;

            // Scale subtype FG to player profile while preserving 82games relative shape.
            let shootingPct = this.clampNum(baseFG * this.safeDiv(broadTarget, (group === 'three' ? 0.388 : (group === 'rim' ? 0.600 : 0.420)), 1), 0.25, 0.78);

            // Assisted profile adjustment by shot type (marginal effect vs league baseline).
            const assistCoeff = group === 'three' ? 0.037 : (group === 'mid' ? 0.095 : 0.110);
            shootingPct = this.clampNum(shootingPct + (assistedDelta * assistCoeff), 0.25, 0.80);

            const pointsPerShot = shootingPct * (group === 'three' ? 3.0 : 2.0);
            const secondChanceBonus =
                (group === 'rim' ? 0.12 : group === 'three' ? 0.05 : 0.08) * mods.offensive_rebound_factor;
            const transitionBonus =
                (group === 'three' ? 0.08 : group === 'rim' ? 0.06 : 0.03) * mods.transition_factor;
            const locShapeMultiplier = this.clampNum((locPriorPPP[sub] || 0.936) / 0.936, 0.75, 1.25);
            const trueImpactValue = (pointsPerShot * (0.78 + 0.22 * locShapeMultiplier)) + secondChanceBonus + transitionBonus;
            const confidence = this.clampNum(
                mods.confidence_base * (0.45 + 0.55 * Math.sqrt(this.clampNum(frequency / Math.max(0.0001, targetFreq[group] || 0.001), 0, 1))),
                0.25,
                0.98
            );
            return {
                ...z,
                frequency,
                shootingPct,
                pointsPerShot,
                trueImpactValue,
                confidence
            };
        });

        const maxFrequency = Math.max(...zones.map((z) => z.frequency), 0.0001);
        const minFG = Math.min(...zones.map((z) => z.shootingPct), 0.0001);
        const maxFG = Math.max(...zones.map((z) => z.shootingPct), 0.0001);
        const minPPS = Math.min(...zones.map((z) => z.pointsPerShot), 0.0001);
        const maxPPS = Math.max(...zones.map((z) => z.pointsPerShot), 0.0001);
        const minTI = Math.min(...zones.map((z) => z.trueImpactValue), 0.0001);
        const maxTI = Math.max(...zones.map((z) => z.trueImpactValue), 0.0001);
        return zones.map((z) => ({
            ...z,
            intensity: z.frequency / maxFrequency,
            fgIntensity: (z.shootingPct - minFG) / Math.max(0.0001, (maxFG - minFG)),
            valueIntensity: (z.pointsPerShot - minPPS) / Math.max(0.0001, (maxPPS - minPPS)),
            trueImpactIntensity: (z.trueImpactValue - minTI) / Math.max(0.0001, (maxTI - minTI)),
            metricIntensity: this.shotChartMode === 'fg'
                ? ((z.shootingPct - minFG) / Math.max(0.0001, (maxFG - minFG)))
                : (this.shotChartMode === 'value'
                    ? ((z.pointsPerShot - minPPS) / Math.max(0.0001, (maxPPS - minPPS)))
                    : (this.shotChartMode === 'trueimpact'
                        ? ((z.trueImpactValue - minTI) / Math.max(0.0001, (maxTI - minTI)))
                        : (z.frequency / maxFrequency)))
        }));
    }

    drawValueDriversChart(player) {
        const canvas = document.getElementById('valueDriversChart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const paddingX = 56;
        const topPadding = 34;
        const bottomPadding = 24;
        const barHeight = 42;
        const rowGap = 24;
        const styles = getComputedStyle(document.documentElement);
        const textPrimary = (styles.getPropertyValue('--text-primary') || '#111827').trim();
        const textSecondary = (styles.getPropertyValue('--text-secondary') || '#6b7280').trim();
        const isDark = document.documentElement.classList.contains('theme-dark') || document.body.classList.contains('theme-dark');

        // Transparent background: just clear the canvas.
        ctx.clearRect(0, 0, width, height);

        // Calculate value drivers
        const drivers = this.calculateValueDrivers(player);

        // Better color palette
        const colors = [
            { start: '#667eea', end: '#764ba2' }, // Purple gradient
            { start: '#f093fb', end: '#f5576c' }, // Pink gradient
            { start: '#4facfe', end: '#00f2fe' }, // Blue gradient
            { start: '#43e97b', end: '#38f9d7' }, // Green gradient
            { start: '#fa709a', end: '#fee140' }  // Orange gradient
        ];

        // Assign colors to drivers
        drivers.forEach((driver, i) => {
            driver.gradient = colors[i % colors.length];
        });

        const maxValue = Math.max(...drivers.map(d => d.value), 1);
        const barWidth = Math.max(120, width - (paddingX * 2));
        const totalRowsHeight = drivers.length * barHeight + Math.max(0, drivers.length - 1) * rowGap;
        const availableHeight = Math.max(0, height - topPadding - bottomPadding);
        const startY = topPadding + Math.max(0, (availableHeight - totalRowsHeight) / 2);

        drivers.forEach((driver, index) => {
            const y = startY + (index * (barHeight + rowGap));
            const fillWidth = (driver.value / maxValue) * barWidth;

            // Draw shadow
            ctx.shadowColor = isDark ? 'rgba(0, 0, 0, 0.32)' : 'rgba(0, 0, 0, 0.12)';
            ctx.shadowBlur = 7;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 3;

            // Background bar (subtle translucent track)
            ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(15, 23, 42, 0.12)';
            ctx.beginPath();
            ctx.roundRect(paddingX, y, barWidth, barHeight, 8);
            ctx.fill();

            // Reset shadow
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;

            // Value bar with gradient
            const gradient = ctx.createLinearGradient(paddingX, 0, paddingX + fillWidth, 0);
            gradient.addColorStop(0, driver.gradient.start);
            gradient.addColorStop(1, driver.gradient.end);
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.roundRect(paddingX, y, fillWidth, barHeight, 8);
            ctx.fill();

            // Label inside track for stable readability in both themes.
            ctx.fillStyle = textPrimary;
            ctx.font = 'bold 14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(driver.label, paddingX + 12, y + (barHeight / 2));

            // Value (inside bar if it fits, otherwise outside)
            const valueText = driver.value.toFixed(1);
            ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            const textWidth = ctx.measureText(valueText).width;

            if (fillWidth > textWidth + 34) {
                // Inside bar near right edge.
                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'right';
                ctx.fillText(valueText, paddingX + fillWidth - 12, y + (barHeight / 2));
            } else {
                // Outside bar, clamped so text never falls out of bounds.
                ctx.fillStyle = textSecondary;
                ctx.textAlign = 'left';
                const outsideX = Math.min(paddingX + barWidth - textWidth - 6, paddingX + fillWidth + 8);
                ctx.fillText(valueText, outsideX, y + (barHeight / 2));
            }
        });
    }

    calculateValueDrivers(player) {
        const archetype = player.identity?.primary_archetype || 'unknown';
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const creation = player.creation_profile || {};
        const shot = player.shot_profile || {};

        // Base drivers that apply to all archetypes
        const drivers = [];

        switch (archetype) {
            case 'defensive_anchor':
                drivers.push(
                    { label: 'Rim Protection', value: (trad.blocks_per_game || 0) * 10, color: '#e74c3c' },
                    { label: 'Rebounding', value: (trad.rebounds_per_game || 0), color: '#9b59b6' },
                    { label: 'Interior Presence', value: (shot.rim_frequency || 0) * 20, color: '#3498db' },
                    { label: 'Player Value', value: this.getPlayerValue(player) / 5, color: '#2ecc71' }
                );
                break;

            case 'rim_runner':
                drivers.push(
                    { label: 'Finishing at Rim', value: (shot.rim_frequency || 0) * 20, color: '#e74c3c' },
                    { label: 'Efficiency (FG%)', value: (trad.field_goal_pct || 0) * 20, color: '#f39c12' },
                    { label: 'Paint Touches', value: (creation.paint_touches_per_game || 0), color: '#9b59b6' },
                    { label: 'Scoring Output', value: (trad.points_per_game || 0), color: '#2ecc71' }
                );
                break;

            case 'stretch_big':
                drivers.push(
                    { label: 'Three-Point Shooting', value: (shot.three_point_frequency || 0) * 30, color: '#3498db' },
                    { label: '3P% Efficiency', value: (trad.three_point_pct || 0) * 30, color: '#1abc9c' },
                    { label: 'Spacing Value', value: (trad.points_per_game || 0), color: '#f39c12' },
                    { label: 'Rebounding', value: (trad.rebounds_per_game || 0), color: '#9b59b6' }
                );
                break;

            case 'versatile_wing':
                drivers.push(
                    { label: 'Scoring Versatility', value: (trad.points_per_game || 0), color: '#e74c3c' },
                    { label: 'Playmaking', value: (trad.assists_per_game || 0) * 2, color: '#3498db' },
                    { label: 'Defense (Stls+Blks)', value: ((trad.steals_per_game || 0) + (trad.blocks_per_game || 0)) * 5, color: '#9b59b6' },
                    { label: 'Usage Rate', value: (adv.usage_rate ?? 0) * 50, color: '#f39c12' }
                );
                break;

            default:
                drivers.push(
                    { label: 'Scoring', value: (trad.points_per_game || 0), color: '#e74c3c' },
                    { label: 'Playmaking', value: (trad.assists_per_game || 0) * 2, color: '#3498db' },
                    { label: 'Rebounding', value: (trad.rebounds_per_game || 0), color: '#9b59b6' },
                    { label: 'Player Value', value: this.getPlayerValue(player) / 5, color: '#2ecc71' }
                );
        }

        return drivers.sort((a, b) => b.value - a.value);
    }

    generateUniquenessInsights(player) {
        const archetype = player.identity?.primary_archetype || 'unknown';
        const trad = player.performance?.traditional || {};
        const adv = player.performance?.advanced || {};
        const creation = player.creation_profile || {};
        const shot = player.shot_profile || {};
        const defense = player.defense_assessment || {};

        const insights = [];

        // Archetype-specific uniqueness
        switch (archetype) {
            case 'defensive_anchor':
                if ((trad.blocks_per_game || 0) > 1.5) {
                    insights.push({ icon: '🛡️', text: `Elite rim protector with ${trad.blocks_per_game.toFixed(1)} BPG` });
                }
                if ((trad.rebounds_per_game || 0) > 10) {
                    insights.push({ icon: '💪', text: `Dominant rebounder at ${trad.rebounds_per_game.toFixed(1)} RPG` });
                }
                if ((shot.rim_frequency || 0) > 0.6) {
                    insights.push({ icon: '🎯', text: `${((shot.rim_frequency || 0) * 100).toFixed(0)}% of shots at the rim` });
                }
                break;

            case 'rim_runner':
                if ((trad.field_goal_pct || 0) > 0.55) {
                    insights.push({ icon: '🔥', text: `Efficient finisher at ${((trad.field_goal_pct || 0) * 100).toFixed(1)}% FG` });
                }
                if ((creation.paint_touches_per_game || 0) > 8) {
                    insights.push({ icon: '🏀', text: `High paint activity: ${creation.paint_touches_per_game.toFixed(1)} touches/game` });
                }
                if ((creation.assisted_rate || 0) > 0.7) {
                    insights.push({ icon: '🤝', text: `Team-oriented: ${((creation.assisted_rate || 0) * 100).toFixed(0)}% assisted` });
                }
                break;

            case 'stretch_big':
                if ((trad.three_point_pct || 0) > 0.35) {
                    insights.push({ icon: '🎯', text: `Reliable shooter at ${((trad.three_point_pct || 0) * 100).toFixed(1)}% from three` });
                }
                if ((shot.three_point_frequency || 0) > 0.3) {
                    insights.push({ icon: '📊', text: `Modern big: ${((shot.three_point_frequency || 0) * 100).toFixed(0)}% of shots from three` });
                }
                if ((trad.rebounds_per_game || 0) > 8) {
                    insights.push({ icon: '💪', text: `Two-way value with ${trad.rebounds_per_game.toFixed(1)} RPG` });
                }
                break;

            case 'versatile_wing':
                if ((trad.points_per_game || 0) > 15 && (trad.assists_per_game || 0) > 3) {
                    insights.push({ icon: '⭐', text: `Dual threat: ${trad.points_per_game.toFixed(1)} PPG, ${trad.assists_per_game.toFixed(1)} APG` });
                }
                if ((trad.steals_per_game || 0) > 1.0) {
                    insights.push({ icon: '🛡️', text: `Defensive playmaker with ${trad.steals_per_game.toFixed(1)} SPG` });
                }
                if ((adv.usage_rate ?? 0) > 0.25) {
                    insights.push({ icon: '🎯', text: `High usage player at ${((adv.usage_rate ?? 0) * 100).toFixed(1)}%` });
                }
                break;
        }

        // General uniqueness factors
        const playerValue = this.getPlayerValue(player);
        if (playerValue >= 70) {
            insights.push({ icon: '📈', text: `Elite player value score: ${playerValue.toFixed(1)}` });
        } else if (playerValue < 40) {
            insights.push({ icon: '📉', text: `Low player value score: ${playerValue.toFixed(1)}` });
        }

        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 0;
        if (trustScore >= 80) {
            insights.push({ icon: '✅', text: `High data reliability (${trustScore.toFixed(0)} trust score)` });
        } else if (trustScore < 60) {
            insights.push({ icon: '⚠️', text: `Limited data (${trustScore.toFixed(0)} trust score)` });
        }

        // Compare to archetype peers
        if (player.comparables?.similar_players?.length > 0) {
            const topComp = player.comparables.similar_players[0];
            insights.push({ icon: '👥', text: `Most similar to ${topComp.name} (${(topComp.similarity_score * 100).toFixed(0)}%)` });
        }

        if (insights.length === 0) {
            insights.push({ icon: '📊', text: 'Standard archetype profile' });
        }

        return insights.map(insight => `
            <div class="insight-item">
                <span class="insight-icon">${insight.icon}</span>
                <span class="insight-text">${insight.text}</span>
            </div>
        `).join('');
    }

    generateBreakoutScenario(player, constraints, trad, adv) {
        const evalOut = this.evaluateBreakoutScenario(player);
        const fitPercentile = this.getTeamFitPercentile(player);
        const {
            fitScenario,
            age,
            trustScore,
            currentUsage,
            currentMinutes,
            currentPPG,
            currentAPG,
            currentRPG,
            projectedUsage,
            projectedMinutes,
            projectedPPG,
            projectedAPG,
            projectedRPG,
            usageDelta,
            minutesDelta,
            maxUsageIncrease,
            maxMinutesIncrease,
            contributionStrength,
            contributionProfile,
            mechanism,
            intrinsicContextLedger,
            clampReport,
            impactAudit,
            defenseRole,
            evidence,
            minutesGovernance,
            breakoutScore,
            likelihood,
            likelihoodClass,
            likelihoodPct,
            breakoutLikelihoodScore,
            genuineBreakout,
            promotionBlocked,
            promotionBlockReasons,
            valueDecomposition,
            ecosystemValidated,
            visibleBreakoutValidated,
            hiddenSkillSignal,
            utilizationUplift
        } = evalOut;

        const usageImpactWeight = this.clampNum((usageDelta * 100) / 12, 0.1, 1.0);
        const minutesImpactWeight = this.clampNum(minutesDelta / 8, 0.1, 1.0);
        const ppgBoostPct = (usageImpactWeight * 0.65 + minutesImpactWeight * 0.35) * 100;
        const apgBoostPct = (usageImpactWeight * 0.55 + minutesImpactWeight * 0.45) * 100;
        const rpgBoostPct = (minutesImpactWeight * 0.8 + usageImpactWeight * 0.2) * 100;

        return `
            <div class="detail-section">
                <h3>Breakout Potential</h3>
                <div class="breakout-header">
                    <div class="breakout-likelihood ${likelihoodClass}">
                        <div class="likelihood-label">Breakout Likelihood</div>
                        <div class="likelihood-value">${likelihood}</div>
                        <div class="likelihood-label">Score: ${breakoutScore.toFixed(1)}</div>
                        <div class="likelihood-label">Likelihood Signal: ${breakoutLikelihoodScore.toFixed(0)}</div>
                        <div class="likelihood-label">${genuineBreakout ? 'Genuine Breakout Signal: YES' : 'Genuine Breakout Signal: NO'}</div>
                        <div class="likelihood-label">${promotionBlocked ? 'Promotion Gate: BLOCKED' : 'Promotion Gate: OPEN'}</div>
                        <div class="likelihood-label">${visibleBreakoutValidated ? 'Visible Validation: PASS' : `Visible Validation: ${String(minutesGovernance.visible_status || 'watch').toUpperCase()}`}</div>
                        <div class="likelihood-label">${ecosystemValidated ? 'Ecosystem Validation: PASS' : 'Ecosystem Validation: NOT VALIDATED'}</div>
                        <div class="likelihood-bar">
                            <div class="likelihood-fill" style="width: ${likelihoodPct}%"></div>
                        </div>
                    </div>
                </div>
            </div>

            ${promotionBlocked ? `
            <div class="detail-section">
                <h3>Promotion Block</h3>
                <div class="constraints-list">
                    ${(promotionBlockReasons || []).map(r => `
                        <div class="constraint-item">
                            <span class="constraint-icon">⛔</span>
                            <span class="constraint-text">${r}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <div class="detail-section">
                <h3>Minutes Gate</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">⏱️</span>
                        <span class="constraint-text"><strong>Status:</strong> ${String(minutesGovernance.visible_status || 'watch').toUpperCase()} (${(minutesGovernance.mpg || 0).toFixed(1)} MPG, ${(minutesGovernance.games_played || 0).toFixed(0)} GP).</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">🧪</span>
                        <span class="constraint-text"><strong>Evidence penalty:</strong> ${((minutesGovernance.evidence_penalty || 0) * 100).toFixed(1)}%, <strong>feasibility penalty:</strong> ${((minutesGovernance.feasibility_penalty || 0) * 100).toFixed(1)}%.</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">📝</span>
                        <span class="constraint-text">${minutesGovernance.reason || 'Minutes gate reason unavailable.'}</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Best Team Scenario</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">🏀</span>
                        <span class="constraint-text">${fitScenario.signal_strength === 'weak'
                            ? `${fitScenario.team === 'No Clear Team Edge'
                                ? `No clear destination edge (fit score ${(fitScenario.fit_score * 100).toFixed(0)}%, confidence ${(fitScenario.confidence * 100).toFixed(0)}%)`
                                : `Exploratory destination: <strong>${fitScenario.team}</strong> (fit score ${(fitScenario.fit_score * 100).toFixed(0)}%, confidence ${(fitScenario.confidence * 100).toFixed(0)}%)`
                            }`
                            : `<strong>${fitScenario.team}</strong> is the strongest breakout context (fit score ${(fitScenario.fit_score * 100).toFixed(0)}%, confidence ${(fitScenario.confidence * 100).toFixed(0)}%)`
                        }</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">🧪</span>
                        <span class="constraint-text">Signal strength: <strong>${(fitScenario.signal_strength || 'weak').toUpperCase()}</strong> (edge over next option ${(100 * (fitScenario.fit_gap || 0)).toFixed(1)}%)</span>
                    </div>
                    ${fitScenario.reasons.map(reason => `
                        <div class="constraint-item">
                            <span class="constraint-icon">✓</span>
                            <span class="constraint-text">${reason}</span>
                        </div>
                    `).join('')}
                    <div class="constraint-item">
                        <span class="constraint-icon">🧩</span>
                        <span class="constraint-text">Scheme complementarity ${(100 * (fitScenario.score_breakdown?.scheme_complementarity || 0)).toFixed(0)}% (creation, spacing, off-ball, and defensive role alignment)</span>
                    </div>
                    ${fitScenario.alternatives && fitScenario.alternatives.length > 1 ? `
                        <div class="constraint-item">
                            <span class="constraint-icon">📊</span>
                            <span class="constraint-text">Next best fits: ${fitScenario.alternatives.slice(1).map(a => `${a.team} (${(a.fit_score * 100).toFixed(0)}%)`).join(', ')}</span>
                        </div>
                    ` : ''}
                </div>
            </div>

            <div class="detail-section">
                <h3>Team Contribution Channels (X / Y / Z)</h3>
                <div class="factors-grid">
                    ${contributionProfile.slice(0, 3).map((c, i) => `
                        <div class="factor-item ${c.value >= 0.62 ? 'positive' : c.value < 0.42 ? 'negative' : 'neutral'}">
                            <div class="factor-label">Channel ${i + 1}: ${c.label}</div>
                            <div class="factor-value">${(c.value * 100).toFixed(0)}%</div>
                            <div class="factor-assessment">${c.note}</div>
                        </div>
                    `).join('')}
                </div>
                <div class="constraints-list" style="margin-top: 0.75rem;">
                    <div class="constraint-item">
                        <span class="constraint-icon">🧭</span>
                        <span class="constraint-text">Composite contribution strength: <strong>${(contributionStrength * 100).toFixed(0)}%</strong> for ${fitScenario.team === 'No Clear Team Edge' ? 'exploratory context' : fitScenario.team}.</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">🛡️</span>
                        <span class="constraint-text">Portability ${(mechanism.portability_score * 100).toFixed(0)}% and clamp pressure ${(mechanism.clamp_severity * 100).toFixed(0)}% shape whether this value translates in-game.</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Intrinsic vs Context Ledger</h3>
                <div class="factors-grid">
                    <div class="factor-item ${(intrinsicContextLedger.intrinsic_share || 0.5) >= 0.5 ? 'positive' : 'neutral'}">
                        <div class="factor-label">Intrinsic Share</div>
                        <div class="factor-value">${(100 * (intrinsicContextLedger.intrinsic_share || 0.5)).toFixed(0)}%</div>
                        <div class="factor-assessment">Player-owned contribution path (skills + priors + portability).</div>
                    </div>
                    <div class="factor-item ${(intrinsicContextLedger.context_share || 0.5) >= 0.5 ? 'positive' : 'neutral'}">
                        <div class="factor-label">Context Share</div>
                        <div class="factor-value">${(100 * (intrinsicContextLedger.context_share || 0.5)).toFixed(0)}%</div>
                        <div class="factor-assessment">Team/deployment translation path (fit + role runway).</div>
                    </div>
                    <div class="factor-item ${(intrinsicContextLedger.residual || 0) <= 0.25 ? 'positive' : 'negative'}">
                        <div class="factor-label">Residual Unexplained</div>
                        <div class="factor-value">${(100 * (intrinsicContextLedger.residual || 0)).toFixed(0)}%</div>
                        <div class="factor-assessment">Lower is better; high residual implies weaker explanation completeness.</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Value Decomposition (Team-Weighted)</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">🧮</span>
                        <span class="constraint-text">Base vector ${(100 * (valueDecomposition.base_score || 0)).toFixed(0)}% vs team-weighted ${(100 * (valueDecomposition.team_weighted_score || 0)).toFixed(0)}%.</span>
                    </div>
                    ${(valueDecomposition.blocks || []).slice(0, 4).map(b => `
                        <div class="constraint-item">
                            <span class="constraint-icon">•</span>
                            <span class="constraint-text"><strong>${b.label}</strong>: ${(100 * (b.value || 0)).toFixed(0)}% with team weight ${(100 * (b.team_weight || 0)).toFixed(0)}%.</span>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="detail-section">
                <h3>Mechanism Trace (Why, Not Just What)</h3>
                <div class="factors-grid">
                    <div class="factor-item ${mechanism.suppression_relief >= 0.6 ? 'positive' : mechanism.suppression_relief < 0.4 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Suppression Relief</div>
                        <div class="factor-value">${(mechanism.suppression_relief * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">Expected role unlock from better scheme + usage fit.</div>
                    </div>
                    <div class="factor-item ${mechanism.spacing_gravity_skill >= 0.6 ? 'positive' : mechanism.spacing_gravity_skill < 0.4 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Spacing Gravity Skill</div>
                        <div class="factor-value">${(mechanism.spacing_gravity_skill * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">How much off-ball spacing pressure this player creates.</div>
                    </div>
                    <div class="factor-item ${mechanism.portability_score >= 0.6 ? 'positive' : mechanism.portability_score < 0.45 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Defense Portability</div>
                        <div class="factor-value">${(mechanism.portability_score * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">Survivability across matchup targeting and switch stress.</div>
                    </div>
                    <div class="factor-item ${mechanism.clamp_severity <= 0.4 ? 'positive' : mechanism.clamp_severity >= 0.65 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Clamp Severity</div>
                        <div class="factor-value">${(mechanism.clamp_severity * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">How hard feasibility constraints limit scenario upside.</div>
                    </div>
                    <div class="factor-item ${mechanism.impact_prior_disagreement <= 0.35 ? 'positive' : mechanism.impact_prior_disagreement >= 0.6 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Prior Disagreement</div>
                        <div class="factor-value">${(mechanism.impact_prior_disagreement * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">Lower is better; high disagreement reduces claim strength.</div>
                    </div>
                    <div class="factor-item ${mechanism.comp_density >= 0.65 ? 'positive' : mechanism.comp_density < 0.5 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Comp Density</div>
                        <div class="factor-value">${(mechanism.comp_density * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">Lower density means more uniqueness and wider uncertainty.</div>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Impact Audit Verdict</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">🔎</span>
                        <span class="constraint-text"><strong>Verdict:</strong> ${String(impactAudit.verdict || 'insufficient_data').replace('_', ' ').toUpperCase()} (prior disagreement ${(100 * (impactAudit.disagreement ?? mechanism.impact_prior_disagreement ?? 0)).toFixed(0)}%).</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">🧾</span>
                        <span class="constraint-text"><strong>Consensus prior:</strong> ${impactAudit.consensus === null || impactAudit.consensus === undefined ? 'N/A' : impactAudit.consensus.toFixed(2)} from available EPM/LEBRON anchors.</span>
                    </div>
                    ${impactAudit.required_justification && impactAudit.required_justification.length ? impactAudit.required_justification.map(j => `
                        <div class="constraint-item">
                            <span class="constraint-icon">⚠️</span>
                            <span class="constraint-text">${j}</span>
                        </div>
                    `).join('') : `
                        <div class="constraint-item">
                            <span class="constraint-icon">✅</span>
                            <span class="constraint-text>No mandatory contradiction justification triggered.</span>
                        </div>
                    `}
                </div>
            </div>

            <div class="detail-section">
                <h3>Defense Role Identity & Breakpoints</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">🛡️</span>
                        <span class="constraint-text"><strong>${defenseRole.primary_role || 'Unknown'}</strong> profile with targeting risk ${(100 * (defenseRole.targeting_risk || 0)).toFixed(0)}% (G ${(100 * (defenseRole.role_mix?.guards || 0)).toFixed(0)} / W ${(100 * (defenseRole.role_mix?.wings || 0)).toFixed(0)} / B ${(100 * (defenseRole.role_mix?.bigs || 0)).toFixed(0)}).</span>
                    </div>
                    ${defenseRole.warnings && defenseRole.warnings.length ? defenseRole.warnings.map(w => `
                        <div class="constraint-item">
                            <span class="constraint-icon">⚠️</span>
                            <span class="constraint-text">${w}</span>
                        </div>
                    `).join('') : ''}
                </div>
            </div>

            <div class="detail-section">
                <h3>Evidence Coverage & Missingness</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">📚</span>
                        <span class="constraint-text">Evidence coverage ${(100 * (evidence.coverage_effective || evidence.coverage || 0)).toFixed(0)}% (${String(evidence.grade || 'moderate').toUpperCase()}); confidence penalty ${(100 * (evidence.penalty_total || evidence.penalty || 0)).toFixed(1)}%.</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">🧩</span>
                        <span class="constraint-text">${evidence.missing && evidence.missing.length ? `Missing inputs: ${evidence.missing.slice(0, 6).join(', ')}` : 'No major required fields missing for mechanism audit.'}</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Ecosystem Carve-Out</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">🌐</span>
                        <span class="constraint-text"><strong>${ecosystemValidated ? 'Validated ecosystem breakout path.' : 'Ecosystem path not validated.'}</strong> Hidden skill ${(100 * (hiddenSkillSignal || 0)).toFixed(0)}%, utilization uplift ${(100 * (utilizationUplift || 0)).toFixed(0)}%.</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Clamps Fired</h3>
                <div class="constraints-list">
                    ${clampReport && clampReport.length ? clampReport.map(c => `
                        <div class="constraint-item">
                            <span class="constraint-icon">⛓️</span>
                            <span class="constraint-text"><strong>${c.clamp}</strong> (${(100 * (c.severity || 0)).toFixed(0)}%): ${c.message} Unlock path: ${c.unlock}</span>
                        </div>
                    `).join('') : `
                        <div class="constraint-item">
                            <span class="constraint-icon">✅</span>
                            <span class="constraint-text">No explicit clamp events fired in this scenario.</span>
                        </div>
                    `}
                </div>
            </div>

            <div class="detail-section">
                <h3>Causal Chain: Why This Breakout Happens</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">1</span>
                        <span class="constraint-text"><strong>Skill signal:</strong> spacing gravity ${(mechanism.spacing_gravity_skill * 100).toFixed(0)}% + portability ${(mechanism.portability_score * 100).toFixed(0)}% create scalable role fit.</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">2</span>
                        <span class="constraint-text"><strong>Context translation:</strong> suppression relief ${(mechanism.suppression_relief * 100).toFixed(0)}% implies this team can convert hidden skills into usable possessions.</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">3</span>
                        <span class="constraint-text"><strong>Feasibility check:</strong> clamp pressure ${(mechanism.clamp_severity * 100).toFixed(0)}% limits upside if role/minutes/usage caps are tight.</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">4</span>
                        <span class="constraint-text"><strong>Claim strength:</strong> prior disagreement ${(mechanism.impact_prior_disagreement * 100).toFixed(0)}% and confidence ${(mechanism.confidence * 100).toFixed(0)}% determine reliability of this scenario.</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Current vs Projected Stats</h3>
                <div class="projection-grid">
                    <div class="projection-card">
                        <div class="projection-label">Points Per Game</div>
                        <div class="projection-values">
                            <span class="current-value">${currentPPG.toFixed(1)}</span>
                            <span class="projection-arrow">→</span>
                            <span class="projected-value">${projectedPPG.toFixed(1)}</span>
                        </div>
                        <div class="projection-change">+${(projectedPPG - currentPPG).toFixed(1)} PPG</div>
                    </div>

                    <div class="projection-card">
                        <div class="projection-label">Assists Per Game</div>
                        <div class="projection-values">
                            <span class="current-value">${currentAPG.toFixed(1)}</span>
                            <span class="projection-arrow">→</span>
                            <span class="projected-value">${projectedAPG.toFixed(1)}</span>
                        </div>
                        <div class="projection-change">+${(projectedAPG - currentAPG).toFixed(1)} APG</div>
                    </div>

                    <div class="projection-card">
                        <div class="projection-label">Rebounds Per Game</div>
                        <div class="projection-values">
                            <span class="current-value">${currentRPG.toFixed(1)}</span>
                            <span class="projection-arrow">→</span>
                            <span class="projected-value">${projectedRPG.toFixed(1)}</span>
                        </div>
                        <div class="projection-change">+${(projectedRPG - currentRPG).toFixed(1)} RPG</div>
                    </div>

                    <div class="projection-card">
                        <div class="projection-label">Usage Rate</div>
                        <div class="projection-values">
                            <span class="current-value">${(currentUsage * 100).toFixed(1)}%</span>
                            <span class="projection-arrow">→</span>
                            <span class="projected-value">${(projectedUsage * 100).toFixed(1)}%</span>
                        </div>
                        <div class="projection-change">+${(usageDelta * 100).toFixed(1)}%</div>
                    </div>

                    <div class="projection-card">
                        <div class="projection-label">Minutes Per Game</div>
                        <div class="projection-values">
                            <span class="current-value">${currentMinutes.toFixed(1)}</span>
                            <span class="projection-arrow">→</span>
                            <span class="projected-value">${projectedMinutes.toFixed(1)}</span>
                        </div>
                        <div class="projection-change">+${minutesDelta.toFixed(1)} MPG</div>
                    </div>
                </div>
            </div>

            ${constraints?.constraints && constraints.constraints.length > 0 ? `
            <div class="detail-section">
                <h3>Scenario Constraints</h3>
                <div class="constraints-list">
                    ${constraints.constraints.map(constraint => `
                        <div class="constraint-item">
                            <span class="constraint-icon">⚠️</span>
                            <span class="constraint-text">${constraint}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            ${constraints?.shot_diet_constraints && constraints.shot_diet_constraints.length > 0 ? `
            <div class="detail-section">
                <h3>Shot Diet Constraints</h3>
                <div class="constraints-list">
                    ${constraints.shot_diet_constraints.map(constraint => `
                        <div class="constraint-item">
                            <span class="constraint-icon">🎯</span>
                            <span class="constraint-text">${constraint}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <div class="detail-section">
                <h3>Why The Numbers Improve</h3>
                <div class="constraints-list">
                    <div class="constraint-item">
                        <span class="constraint-icon">📈</span>
                        <span class="constraint-text">PPG rises by ${((projectedPPG - currentPPG)).toFixed(1)} from combined usage/minutes lift (impact mix ${ppgBoostPct.toFixed(0)}%).</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">🧠</span>
                        <span class="constraint-text">APG increases by ${((projectedAPG - currentAPG)).toFixed(1)} because this team fit creates more on-ball possessions (${(usageDelta * 100).toFixed(1)}% usage gain).</span>
                    </div>
                    <div class="constraint-item">
                        <span class="constraint-icon">💪</span>
                        <span class="constraint-text">RPG improves by ${((projectedRPG - currentRPG)).toFixed(1)} mainly from court-time increase (+${minutesDelta.toFixed(1)} MPG).</span>
                    </div>
                </div>
            </div>

            <div class="detail-section">
                <h3>Breakout Factors</h3>
                <div class="factors-grid">
                    <div class="factor-item ${age < 25 ? 'positive' : age > 28 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Age</div>
                        <div class="factor-value">${age.toFixed(1)}</div>
                        <div class="factor-assessment">${age < 25 ? 'Prime development window' : age > 28 ? 'Limited upside' : 'Entering prime'}</div>
                    </div>

                    <div class="factor-item ${trustScore > 70 ? 'positive' : trustScore < 60 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Data Reliability</div>
                        <div class="factor-value">${trustScore.toFixed(0)}</div>
                        <div class="factor-assessment">${trustScore > 70 ? 'High confidence' : trustScore < 60 ? 'Limited sample' : 'Moderate confidence'}</div>
                    </div>

                    <div class="factor-item ${usageDelta > 0.08 ? 'positive' : usageDelta < 0.05 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Usage Upside</div>
                        <div class="factor-value">+${(usageDelta * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">${usageDelta > 0.08 ? 'Significant room' : usageDelta < 0.05 ? 'Limited room' : 'Moderate room'}</div>
                    </div>

                    <div class="factor-item ${minutesDelta > 5 ? 'positive' : minutesDelta < 3 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Minutes Upside</div>
                        <div class="factor-value">+${minutesDelta.toFixed(1)}</div>
                        <div class="factor-assessment">${minutesDelta > 5 ? 'Significant room' : minutesDelta < 3 ? 'Limited room' : 'Moderate room'}</div>
                    </div>

                    <div class="factor-item ${fitScenario.fit_score >= 0.09 ? 'positive' : fitScenario.fit_score >= 0.06 ? 'neutral' : 'negative'}">
                        <div class="factor-label">Team Fit Context</div>
                        <div class="factor-value">${(fitScenario.fit_score * 100).toFixed(0)}% (p${fitPercentile})</div>
                        <div class="factor-assessment">${fitScenario.fit_score >= 0.09 ? 'Amazing fit path' : fitScenario.fit_score >= 0.06 ? 'Good fit path' : 'Weak fit path'} • relative league rank p${fitPercentile}</div>
                    </div>

                    <div class="factor-item ${fitScenario.confidence > 0.72 ? 'positive' : fitScenario.confidence < 0.58 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Scenario Confidence</div>
                        <div class="factor-value">${(fitScenario.confidence * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">${fitScenario.confidence > 0.72 ? 'High confidence' : fitScenario.confidence < 0.58 ? 'Low confidence' : 'Moderate confidence'}</div>
                    </div>
                </div>
            </div>
        `;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new PlayerCardsApp();
});
