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
        this.courtImage = null;
        this.distributionRaf = null;
        
        this.init();
    }

    async init() {
        await this.loadData();
        await this.loadCourtImage();
        this.setupEventListeners();
        this.populateArchetypeFilter();
        this.populateTeamFilter();
        this.renderPlayers();
        this.updateStats();
        this.drawValueDistributionChart();
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
        const projectedUsage = Math.min(projectedUsageCap, currentUsage + (maxUsageIncrease * usageGainFactor * signalMultiplier));
        const projectedMinutes = Math.min(projectedMinutesCap, currentMinutes + (maxMinutesIncrease * minutesGainFactor * signalMultiplier));
        const usageDelta = Math.max(0, projectedUsage - currentUsage);
        const minutesDelta = Math.max(0, projectedMinutes - currentMinutes);

        const usageMultiplier = currentUsage > 0 ? (projectedUsage / currentUsage) : 1.0;
        const minutesMultiplier = currentMinutes > 0 ? (projectedMinutes / currentMinutes) : 1.0;
        const overallMultiplier = Math.min(usageMultiplier * 0.7 + minutesMultiplier * 0.3, 1.5);

        const projectedPPG = currentPPG * overallMultiplier;
        const projectedAPG = currentAPG * overallMultiplier;
        const projectedRPG = currentRPG * Math.min(minutesMultiplier, 1.3);

        const ageCurve = this.clampNum(1 - Math.abs(age - 25) / 9, 0, 1);
        const usageUpside = this.clampNum(usageDelta / 0.10, 0, 1);
        const minutesUpside = this.clampNum(minutesDelta / 8.0, 0, 1);
        const fitEdge = this.clampNum((fitScenario.fit_score - 0.12) / 0.28, 0, 1);
        const trustNorm = this.clampNum(trustScore / 100, 0, 1);

        let breakoutScoreRaw = 100 * (
            0.30 * ageCurve +
            0.25 * usageUpside +
            0.20 * minutesUpside +
            0.15 * trustNorm +
            0.10 * fitEdge
        );
        if (fitScenario.signal_strength === 'weak') breakoutScoreRaw -= 6;
        breakoutScoreRaw = this.clampNum(breakoutScoreRaw, 0, 100);

        const key = this.getPlayerKey(player);
        const breakoutScore = this.breakoutScoreByKey?.[key] ?? breakoutScoreRaw;

        let likelihood = 'Low';
        let likelihoodClass = 'low';
        if (breakoutScore >= 70) {
            likelihood = 'High';
            likelihoodClass = 'high';
        } else if (breakoutScore >= 45) {
            likelihood = 'Medium';
            likelihoodClass = 'medium';
        }

        const likelihoodPct = Math.round(this.clampNum(breakoutScore, 20, 85));

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
            breakoutScore,
            breakoutScoreRaw,
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

        window.addEventListener('resize', () => {
            if (this.distributionRaf) cancelAnimationFrame(this.distributionRaf);
            this.distributionRaf = requestAnimationFrame(() => this.drawValueDistributionChart());
        });
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
        container.innerHTML = this.filteredPlayers.map(player => this.createPlayerCard(player)).join('');

        // Add click listeners
        document.querySelectorAll('.player-card').forEach((card, index) => {
            card.addEventListener('click', () => {
                this.showPlayerDetail(this.filteredPlayers[index]);
            });
        });

        this.drawValueDistributionChart();
    }

    drawValueDistributionChart() {
        const canvas = document.getElementById('valueDistributionChart');
        if (!canvas) return;

        const countEl = document.getElementById('distributionCount');
        const total = this.filteredPlayers.length;
        if (countEl) {
            countEl.textContent = `${total} player${total === 1 ? '' : 's'}`;
        }

        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const cssWidth = Math.max(300, Math.floor(rect.width || 860));
        const cssHeight = Math.max(130, Math.floor(rect.height || 160));
        canvas.width = Math.floor(cssWidth * dpr);
        canvas.height = Math.floor(cssHeight * dpr);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssWidth, cssHeight);

        // Background
        const bg = ctx.createLinearGradient(0, 0, cssWidth, cssHeight);
        bg.addColorStop(0, 'rgba(10, 14, 28, 0.24)');
        bg.addColorStop(1, 'rgba(8, 11, 23, 0.08)');
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, cssWidth, cssHeight);

        const pad = { left: 44, right: 20, top: 20, bottom: 28 };
        const plotW = Math.max(1, cssWidth - pad.left - pad.right);
        const centerY = Math.round((cssHeight - pad.top - pad.bottom) * 0.52 + pad.top);
        const amp = Math.max(14, Math.floor((cssHeight - pad.top - pad.bottom) * 0.38));

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

        const values = this.filteredPlayers
            .map(p => this.getPlayerValue(p))
            .filter(v => Number.isFinite(v));
        if (!values.length) return;

        const minV = Math.min(...values);
        const maxV = Math.max(...values);
        const span = Math.max(0.0001, maxV - minV);
        const toX = (v) => pad.left + ((v - minV) / span) * plotW;
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
            ctx.fillText(t.toFixed(1), x, cssHeight - 8);
        }

        // Beeswarm-like mirrored jitter by x-bins.
        const bins = new Map();
        const points = this.filteredPlayers.map((p, idx) => {
            const v = this.getPlayerValue(p);
            const x = toX(v);
            const b = Math.floor(((x - pad.left) / plotW) * 42);
            const key = Math.max(0, Math.min(41, b));
            const stack = bins.get(key) || 0;
            bins.set(key, stack + 1);
            const layer = Math.floor(stack / 2) + 1;
            const sign = stack % 2 === 0 ? -1 : 1;
            const y = centerY + sign * Math.min(amp, layer * 3.3);
            return { player: p, x, y, v, idx };
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
    }

    createPlayerCard(player) {
        // Extract data from data_sample format
        const playerValue = this.getPlayerValue(player);
        const impactClass = playerValue >= 60 ? 'positive' : (playerValue <= 40 ? 'negative' : '');
        
        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 0;
        const trustLevel = player.v1_1_enhancements?.trust_assessment?.level || 'low';
        
        const winsAdded = player.valuation?.impact?.wins_added || 0;
        
        const points = player.performance?.traditional?.points_per_game || 0;
        const assists = player.performance?.traditional?.assists_per_game || 0;
        const rebounds = player.performance?.traditional?.rebounds_per_game || 0;
        
        // Get usage rate (stored as decimal, e.g., 0.2 = 20%)
        const usageRate = player.performance?.advanced?.usage_rate ?? 0;
        const usageDisplay = (usageRate * 100).toFixed(1) + '%';

        // Tags
        const tags = [];
        if (trustScore >= 75) tags.push('<span class="tag high-impact">High Trust</span>');
        if (playerValue >= 70) tags.push('<span class="tag high-impact">High Value</span>');
        
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
                            <div class="metric-label">Usage</div>
                            <div class="metric-value">${usageDisplay}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Points</div>
                            <div class="metric-value">${points.toFixed(1)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Trust Score</div>
                            <div class="metric-value">${trustScore.toFixed(0)}</div>
                            ${this.createPercentileIndicator(trustScore, 0, 100)}
                        </div>
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
                    <!-- Performance Metrics -->
                    <div class="detail-section">
                        <h3>Performance Metrics</h3>
                        <div class="detail-grid">
                            ${this.formatMetricWithPercentile(player, 'Player Value', this.getPlayerValue(player).toFixed(1), 'player_value')}
                            ${this.formatMetricWithPercentile(player, 'Points Per Game', (trad.points_per_game || 0).toFixed(1), 'points')}
                            ${this.formatMetricWithPercentile(player, 'Assists Per Game', (trad.assists_per_game || 0).toFixed(1), 'assists')}
                            ${this.formatMetricWithPercentile(player, 'Rebounds Per Game', (trad.rebounds_per_game || 0).toFixed(1), 'rebounds')}
                        </div>
                    </div>

                    ${player.possession_decomposition ? `
                    <!-- Advanced Impact Metrics -->
                    <div class="detail-section">
                        <h3>Advanced Impact Analysis</h3>
                        <div class="impact-explanation">
                            <p>${player.possession_decomposition.interpretation_summary || 'Advanced possession-based impact analysis'}</p>
                        </div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-item-label">Intrinsic Offense</div>
                                <div class="detail-item-value">${(player.possession_decomposition.intrinsic_offense || 0).toFixed(1)}</div>
                                <div class="detail-item-note">Raw offensive ability per 100 poss</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-item-label">Adjusted Value</div>
                                <div class="detail-item-value">${(player.possession_decomposition.adjusted_value || 0).toFixed(1)}</div>
                                <div class="detail-item-note">Context-adjusted impact</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-item-label">Opponent Defense</div>
                                <div class="detail-item-value">${((player.possession_decomposition.opponent_defense_context || 0) * 100).toFixed(0)}%</div>
                                <div class="detail-item-note">${(player.possession_decomposition.opponent_defense_context || 0) > 0.5 ? 'Easier defenses' : 'Tougher defenses'}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-item-label">Team Context</div>
                                <div class="detail-item-value">${(player.possession_decomposition.team_context_adjustment || 0) > 0 ? '+' : ''}${((player.possession_decomposition.team_context_adjustment || 0) * 100).toFixed(1)}%</div>
                                <div class="detail-item-note">${(player.possession_decomposition.team_context_adjustment || 0) > 0
                                    ? `Team environment boosts expected output by ${((player.possession_decomposition.team_context_adjustment || 0) * 100).toFixed(1)}%`
                                    : (player.possession_decomposition.team_context_adjustment || 0) < 0
                                        ? `Team environment suppresses expected output by ${Math.abs((player.possession_decomposition.team_context_adjustment || 0) * 100).toFixed(1)}%`
                                        : 'Neutral team context (no material boost or drag)'
                                }</div>
                            </div>
                        </div>
                    </div>
                    ` : ''}

                    <!-- Shot Profile -->
                    <div class="detail-section">
                        <h3>Shot Profile</h3>
                        <div class="detail-grid">
                            ${this.formatMetricWithPercentile(player, 'Rim Frequency', ((player.shot_profile?.rim_frequency || 0) * 100).toFixed(0) + '%', 'rim_freq')}
                            ${this.formatMetricWithPercentile(player, 'Three-Point Frequency', ((player.shot_profile?.three_point_frequency || 0) * 100).toFixed(0) + '%', 'three_freq')}
                            ${this.formatMetricWithPercentile(player, 'FG%', ((trad.field_goal_pct || 0) * 100).toFixed(1) + '%', 'fg_pct')}
                            ${this.formatMetricWithPercentile(player, '3P%', ((trad.three_point_pct || 0) * 100).toFixed(1) + '%', 'three_pct')}
                        </div>
                    </div>

                    <!-- Creation Profile -->
                    <div class="detail-section">
                        <h3>Creation Profile</h3>
                        <div class="detail-grid">
                            ${this.formatMetricWithPercentile(player, 'Drives Per Game', (player.creation_profile?.drives_per_game || 0).toFixed(1), 'drives')}
                            ${this.formatMetricWithPercentile(player, 'Paint Touches', (player.creation_profile?.paint_touches_per_game || 0).toFixed(1), 'paint_touches')}
                            ${this.formatMetricWithPercentile(player, 'Assisted Rate', ((player.creation_profile?.assisted_rate || 0) * 100).toFixed(0) + '%', 'assisted_rate')}
                            ${this.formatMetricWithPercentile(player, 'Usage Rate', ((adv.usage_rate ?? 0) * 100).toFixed(1) + '%', 'usage')}
                        </div>
                    </div>

                    <!-- Defense Assessment -->
                    <div class="detail-section">
                        <h3>Defense Assessment</h3>
                        <div class="detail-grid">
                            ${this.formatMetricWithPercentile(player, 'Steals Per Game', (trad.steals_per_game || 0).toFixed(1), 'steals')}
                            ${this.formatMetricWithPercentile(player, 'Blocks Per Game', (trad.blocks_per_game || 0).toFixed(1), 'blocks')}
                            ${this.formatMetricWithPercentile(player, 'Defensive Rating', (player.defense_assessment?.estimated_metrics?.defensive_rating || 0).toFixed(1), 'def_rating')}
                            <div class="detail-item">
                                <div class="detail-item-label">Observability</div>
                                <div class="detail-item-value">${(player.defense_assessment?.visibility?.observability_level || 'N/A').toUpperCase()}</div>
                            </div>
                        </div>
                    </div>

                    ${valuation ? `
                    <!-- Valuation -->
                    <div class="detail-section">
                        <h3>Valuation & Contract</h3>
                        <div class="detail-grid">
                            ${this.formatMetricWithPercentile(player, 'Wins Added', valuation.impact.wins_added.toFixed(1), 'wins_added')}
                            ${this.formatMetricWithPercentile(player, 'Trade Value (Base)', valuation.trade_value.base.toFixed(1) + 'M', 'trade_value')}
                            <div class="detail-item">
                                <div class="detail-item-label">Aging Phase</div>
                                <div class="detail-item-value">${this.formatAgingPhase(valuation.aging.current_phase)}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-item-label">Peak Age</div>
                                <div class="detail-item-value">${valuation.aging.peak_age.toFixed(1)}</div>
                            </div>
                        </div>
                    </div>
                    ` : ''}

                    <!-- Trust & Uncertainty -->
                    <div class="detail-section">
                        <h3>Data Quality & Trust</h3>
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
                                <div class="detail-item-label">Confidence Level</div>
                                <div class="detail-item-value">${(uncertainty.confidence_level || 'N/A').toUpperCase()}</div>
                            </div>
                        </div>
                    </div>
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
        this.filterPlayers();
    }

    formatArchetype(archetype) {
        return archetype.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
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
        const padding = 60;
        const barHeight = 45;
        const barSpacing = 20;

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
        const barWidth = width - padding * 2;
        const chartHeight = drivers.length * (barHeight + barSpacing);
        const startY = (height - chartHeight) / 2;

        drivers.forEach((driver, index) => {
            const y = startY + (index * (barHeight + barSpacing));
            const fillWidth = (driver.value / maxValue) * barWidth;

            // Draw shadow
            ctx.shadowColor = 'rgba(0, 0, 0, 0.1)';
            ctx.shadowBlur = 8;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 4;

            // Background bar (subtle translucent track)
            ctx.fillStyle = 'rgba(31, 41, 55, 0.10)';
            ctx.beginPath();
            ctx.roundRect(padding, y, barWidth, barHeight, 8);
            ctx.fill();

            // Reset shadow
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;

            // Value bar with gradient
            const gradient = ctx.createLinearGradient(padding, 0, padding + fillWidth, 0);
            gradient.addColorStop(0, driver.gradient.start);
            gradient.addColorStop(1, driver.gradient.end);
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.roundRect(padding, y, fillWidth, barHeight, 8);
            ctx.fill();

            // Label (above bar) with stronger contrast.
            ctx.fillStyle = '#111827';
            ctx.font = 'bold 14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(driver.label, padding, y - 8);

            // Value (inside bar if it fits, otherwise outside)
            const valueText = driver.value.toFixed(1);
            ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            const textWidth = ctx.measureText(valueText).width;

            if (fillWidth > textWidth + 20) {
                // Inside bar (white text)
                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'right';
                ctx.fillText(valueText, padding + fillWidth - 12, y + barHeight / 2 + 6);
            } else {
                // Outside bar (dark text)
                ctx.fillStyle = '#111827';
                ctx.textAlign = 'left';
                ctx.fillText(valueText, padding + fillWidth + 8, y + barHeight / 2 + 6);
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
            breakoutScore,
            likelihood,
            likelihoodClass,
            likelihoodPct
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
                        <div class="likelihood-bar">
                            <div class="likelihood-fill" style="width: ${likelihoodPct}%"></div>
                        </div>
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
                        <div class="factor-value">${(fitScenario.fit_score * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">${fitScenario.fit_score >= 0.09 ? 'Amazing fit path' : fitScenario.fit_score >= 0.06 ? 'Good fit path' : 'Weak fit path'}</div>
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
