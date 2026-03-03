// NBA-VAR Player Cards Web App

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
            const cardsResponse = await fetch('data/cards.json');
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
                const valuationsResponse = await fetch('data/valuations.json');
                const valuationsData = await valuationsResponse.json();
                this.valuations = Array.isArray(valuationsData) ? valuationsData : [valuationsData];
                console.log('Valuations loaded:', this.valuations.length);
            } catch (e) {
                console.log('Valuations not available:', e);
                this.valuations = [];
            }

            // Merge valuation data into players
            this.mergePlayerData();
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
                    // Use plus_minus from performance.advanced
                    const aImpact = a.performance?.advanced?.plus_minus || 0;
                    const bImpact = b.performance?.advanced?.plus_minus || 0;
                    return bImpact - aImpact;
                
                case 'age':
                    return (a.player.age || 25) - (b.player.age || 25);
                
                case 'wins':
                    return (b.valuation?.impact?.wins_added || 0) - (a.valuation?.impact?.wins_added || 0);
                
                case 'breakout':
                    // Use trust score as proxy for breakout potential
                    const aTrust = a.v1_1_enhancements?.trust_assessment?.score || 0;
                    const bTrust = b.v1_1_enhancements?.trust_assessment?.score || 0;
                    return bTrust - aTrust;
                
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
    }

    createPlayerCard(player) {
        // Extract data from data_sample format
        const plusMinus = player.performance?.advanced?.plus_minus || 0;
        const impactClass = plusMinus > 0 ? 'positive' : (plusMinus < 0 ? 'negative' : '');
        
        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 0;
        const trustLevel = player.v1_1_enhancements?.trust_assessment?.level || 'low';
        
        const winsAdded = player.valuation?.impact?.wins_added || 0;
        
        const points = player.performance?.traditional?.points_per_game || 0;
        const assists = player.performance?.traditional?.assists_per_game || 0;
        const rebounds = player.performance?.traditional?.rebounds_per_game || 0;

        // Tags
        const tags = [];
        if (trustScore >= 75) tags.push('<span class="tag high-impact">High Trust</span>');
        if (Math.abs(plusMinus) >= 3) tags.push('<span class="tag high-impact">High Impact</span>');
        if (player.identity?.usage_band === 'high') tags.push('<span class="tag breakout">High Usage</span>');

        return `
            <div class="player-card" data-player="${player.player.name}">
                <div class="card-header">
                    <div class="player-name">${player.player.name}</div>
                    <div class="player-meta">
                        <span>${player.player.team}</span>
                        <span>•</span>
                        <span>${player.player.position || 'N/A'}</span>
                        <span>•</span>
                        <span>Age ${player.player.age?.toFixed(1) || 'N/A'}</span>
                    </div>
                </div>
                <div class="card-body">
                    <div class="archetype-badge">
                        ${this.formatArchetype(player.identity?.primary_archetype || 'unknown')}
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-label">Plus/Minus</div>
                            <div class="metric-value ${impactClass}">${plusMinus.toFixed(1)}</div>
                            ${this.createPercentileIndicator(plusMinus, -5, 10)}
                        </div>
                        <div class="metric">
                            <div class="metric-label">Usage</div>
                            <div class="metric-value">${(player.identity?.usage_band || 'low').toUpperCase()}</div>
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

        modalBody.innerHTML = this.createPlayerDetail(player);
        modal.style.display = 'block';
        
        // Disable body scroll
        document.body.style.overflow = 'hidden';

        // Setup tab switching
        this.setupTabs();

        // Draw charts after modal is visible (only for visuals tab)
        setTimeout(() => {
            this.drawShotChart(player);
            this.drawValueDriversChart(player);
        }, 50);
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
                    Age ${player.player.age?.toFixed(1) || 'N/A'} • 
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
                            ${this.formatMetricWithPercentile(player, 'Plus/Minus', (adv.plus_minus || 0).toFixed(1), 'plus_minus')}
                            ${this.formatMetricWithPercentile(player, 'Points Per Game', (trad.points_per_game || 0).toFixed(1), 'points')}
                            ${this.formatMetricWithPercentile(player, 'Assists Per Game', (trad.assists_per_game || 0).toFixed(1), 'assists')}
                            ${this.formatMetricWithPercentile(player, 'Rebounds Per Game', (trad.rebounds_per_game || 0).toFixed(1), 'rebounds')}
                        </div>
                    </div>

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
                            ${this.formatMetricWithPercentile(player, 'Usage Rate', ((adv.usage_rate || 0) * 100).toFixed(1) + '%', 'usage')}
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
            case 'plus_minus':
                playerValue = adv.plus_minus || 0;
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
                playerValue = adv.usage_rate || 0;
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
                case 'plus_minus': return a.plus_minus || 0;
                case 'points': return t.points_per_game || 0;
                case 'assists': return t.assists_per_game || 0;
                case 'rebounds': return t.rebounds_per_game || 0;
                case 'fg_pct': return t.field_goal_pct || 0;
                case 'three_pct': return t.three_point_pct || 0;
                case 'usage': return a.usage_rate || 0;
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

        // Draw court image as background (flipped 180 degrees)
        if (this.courtImage && this.courtImage.complete) {
            ctx.save();
            ctx.translate(width / 2, height / 2);
            ctx.rotate(Math.PI); // 180 degree rotation
            ctx.drawImage(this.courtImage, -width / 2, -height / 2, width, height);
            ctx.restore();
        } else {
            // Fallback: light background
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, width, height);
        }

        // Hexagon parameters
        const hexSize = 28;
        
        // Court dimensions (flipped)
        const courtWidth = width;
        const courtHeight = height - 50;
        const centerX = courtWidth / 2;
        const baselineY = courtHeight;

        // Define shot zones with hexagon coverage and shooting percentages
        const shotZones = this.generateShotZonesWithPercentages(centerX, baselineY, rimFreq, midFreq, threeFreq, fgPct, threePct);

        // Draw hexagons
        shotZones.forEach(zone => {
            const intensity = zone.frequency;
            
            if (intensity > 0.01) {
                // Color based on zone type
                let baseColor;
                if (zone.type === 'rim') baseColor = { r: 255, g: 107, b: 107 };
                else if (zone.type === 'paint') baseColor = { r: 255, g: 217, b: 61 };
                else if (zone.type === 'mid') baseColor = { r: 255, g: 179, b: 71 };
                else if (zone.type === 'three') baseColor = { r: 107, g: 207, b: 127 };
                else baseColor = { r: 78, g: 205, b: 196 };

                // Alpha based on frequency
                const alpha = 0.25 + (intensity * 0.65);
                
                ctx.fillStyle = `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${alpha})`;
                ctx.strokeStyle = `rgba(255, 255, 255, 0.3)`;
                ctx.lineWidth = 1;

                this.drawHexagon(ctx, zone.x, zone.y, hexSize);
                ctx.fill();
                ctx.stroke();

                // Draw shooting percentage if significant frequency
                if (intensity > 0.15 && zone.shootingPct > 0) {
                    ctx.fillStyle = '#2c3e50';
                    ctx.font = 'bold 10px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(`${(zone.shootingPct * 100).toFixed(0)}%`, zone.x, zone.y);
                }
            }
        });

        // Draw legend at bottom
        const legendY = height - 35;
        const legendItems = [
            { color: 'rgba(255, 107, 107, 0.7)', label: `Rim: ${(rimFreq * 100).toFixed(0)}%`, pct: `${(fgPct * 100).toFixed(1)}% FG` },
            { color: 'rgba(255, 217, 61, 0.7)', label: `Mid: ${(midFreq * 100).toFixed(0)}%`, pct: `${(fgPct * 100).toFixed(1)}% FG` },
            { color: 'rgba(107, 207, 127, 0.7)', label: `3PT: ${(threeFreq * 100).toFixed(0)}%`, pct: `${(threePct * 100).toFixed(1)}%` }
        ];

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
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(item.label, legendX + 28, legendY + 10);
            
            // Shooting percentage
            ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.fillStyle = '#6b7280';
            ctx.fillText(item.pct, legendX + 28, legendY + 20);

            legendX += 130;
        });
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

    generateShotZonesWithPercentages(centerX, baselineY, rimFreq, midFreq, threeFreq, fgPct, threePct) {
        const zones = [];
        const hexSize = 28;
        const hexWidth = hexSize * 2;
        const hexHeight = hexSize * Math.sqrt(3);

        // Helper function to calculate distance from rim (now at top since court is flipped)
        const rimX = centerX;
        const rimY = baselineY * 0.12; // Flipped position
        
        const getDistance = (x, y) => {
            return Math.sqrt(Math.pow(x - rimX, 2) + Math.pow(y - rimY, 2));
        };

        // Generate hexagon grid
        const rows = 14;
        const cols = 12;

        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                const xOffset = (row % 2) * (hexWidth * 0.75);
                const x = 20 + col * (hexWidth * 1.5) + xOffset;
                const y = 30 + row * (hexHeight * 0.75);

                // Skip if outside court bounds
                if (x < 10 || x > centerX * 2 - 10 || y > baselineY) continue;

                const distance = getDistance(x, y);
                const angleFromCenter = Math.atan2(y - rimY, x - centerX);
                
                // Determine zone type, frequency, and shooting percentage
                let type, frequency, shootingPct;

                // Rim zone (within ~60 pixels of rim)
                if (distance < 65) {
                    type = 'rim';
                    frequency = rimFreq * (1.2 - distance / 100);
                    shootingPct = fgPct * 1.15; // Rim shots typically higher %
                }
                // Paint/short mid-range
                else if (distance < 120 && y < baselineY * 0.45) {
                    type = 'paint';
                    frequency = midFreq * 0.8;
                    shootingPct = fgPct * 1.05;
                }
                // Mid-range (120-200 pixels from rim)
                else if (distance < 200) {
                    type = 'mid';
                    frequency = midFreq * 0.6;
                    shootingPct = fgPct * 0.9; // Mid-range typically lower %
                }
                // Three-point zone
                else if (distance >= 200 && distance < 280) {
                    const isCorner = (x < 60 || x > centerX * 2 - 60) && y < baselineY * 0.35;
                    
                    if (isCorner) {
                        type = 'corner';
                        frequency = threeFreq * 0.4;
                        shootingPct = threePct * 1.05; // Corner 3s typically better
                    } else {
                        type = 'three';
                        const normalizedAngle = Math.abs(angleFromCenter);
                        const isWing = normalizedAngle > 0.5 && normalizedAngle < 1.5;
                        frequency = threeFreq * (isWing ? 0.7 : 0.5);
                        shootingPct = threePct;
                    }
                }
                // Deep three / beyond arc
                else if (distance >= 280 && y < baselineY * 0.5) {
                    type = 'three';
                    frequency = threeFreq * 0.2;
                    shootingPct = threePct * 0.85; // Deep 3s typically lower %
                }
                else {
                    continue;
                }

                // Add some randomness for visual variety (±15%)
                frequency *= (0.85 + Math.random() * 0.3);
                frequency = Math.min(frequency, 1.0);
                
                // Ensure shooting percentage is realistic
                shootingPct = Math.min(Math.max(shootingPct, 0.2), 0.75);

                zones.push({ x, y, type, frequency, shootingPct });
            }
        }

        return zones;
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

        // Clear canvas with light background
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, width, height);

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

            // Background bar (light gray)
            ctx.fillStyle = '#e9ecef';
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

            // Label (above bar)
            ctx.fillStyle = '#2c3e50';
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
                ctx.fillStyle = '#2c3e50';
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
                    { label: 'Plus/Minus Impact', value: Math.abs(adv.plus_minus || 0), color: '#2ecc71' }
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
                    { label: 'Usage Rate', value: (adv.usage_rate || 0) * 50, color: '#f39c12' }
                );
                break;

            default:
                drivers.push(
                    { label: 'Scoring', value: (trad.points_per_game || 0), color: '#e74c3c' },
                    { label: 'Playmaking', value: (trad.assists_per_game || 0) * 2, color: '#3498db' },
                    { label: 'Rebounding', value: (trad.rebounds_per_game || 0), color: '#9b59b6' },
                    { label: 'Impact', value: Math.abs(adv.plus_minus || 0), color: '#2ecc71' }
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
                if ((adv.usage_rate || 0) > 0.25) {
                    insights.push({ icon: '🎯', text: `High usage player at ${((adv.usage_rate || 0) * 100).toFixed(1)}%` });
                }
                break;
        }

        // General uniqueness factors
        if ((adv.plus_minus || 0) > 5) {
            insights.push({ icon: '📈', text: `Elite impact: +${adv.plus_minus.toFixed(1)} plus/minus` });
        } else if ((adv.plus_minus || 0) < -3) {
            insights.push({ icon: '📉', text: `Negative impact: ${adv.plus_minus.toFixed(1)} plus/minus` });
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
        const currentUsage = adv.usage_rate || 0.2;
        const currentMinutes = trad.minutes_per_game || 25;
        const currentPPG = trad.points_per_game || 10;
        const currentAPG = trad.assists_per_game || 2;
        const currentRPG = trad.rebounds_per_game || 5;

        // Get constraints
        const applicable = constraints?.applicable !== false;
        const maxUsageIncrease = constraints?.feasible_changes?.max_usage_increase || 0.1;
        const maxMinutesIncrease = constraints?.feasible_changes?.max_minutes_increase || 5;
        const projectedUsageCap = constraints?.feasible_changes?.projected_usage_cap || (currentUsage + 0.1);
        const projectedMinutesCap = constraints?.feasible_changes?.projected_minutes_cap || (currentMinutes + 5);

        // Calculate breakout projections
        const usageMultiplier = projectedUsageCap / currentUsage;
        const minutesMultiplier = projectedMinutesCap / currentMinutes;
        const overallMultiplier = Math.min(usageMultiplier * 0.7 + minutesMultiplier * 0.3, 1.5);

        const projectedPPG = currentPPG * overallMultiplier;
        const projectedAPG = currentAPG * overallMultiplier;
        const projectedRPG = currentRPG * Math.min(minutesMultiplier, 1.3);

        // Determine breakout likelihood
        const age = player.player.age || 25;
        const trustScore = player.v1_1_enhancements?.trust_assessment?.score || 50;
        const archetype = player.identity?.primary_archetype || 'unknown';

        let likelihood = 'Medium';
        let likelihoodClass = 'medium';
        let likelihoodPct = 50;

        if (age < 24 && trustScore > 70 && maxUsageIncrease > 0.08) {
            likelihood = 'High';
            likelihoodClass = 'high';
            likelihoodPct = 75;
        } else if (age > 28 || trustScore < 60 || maxUsageIncrease < 0.05) {
            likelihood = 'Low';
            likelihoodClass = 'low';
            likelihoodPct = 25;
        }

        return `
            <div class="detail-section">
                <h3>Breakout Potential</h3>
                <div class="breakout-header">
                    <div class="breakout-likelihood ${likelihoodClass}">
                        <div class="likelihood-label">Breakout Likelihood</div>
                        <div class="likelihood-value">${likelihood}</div>
                        <div class="likelihood-bar">
                            <div class="likelihood-fill" style="width: ${likelihoodPct}%"></div>
                        </div>
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
                            <span class="projected-value">${(projectedUsageCap * 100).toFixed(1)}%</span>
                        </div>
                        <div class="projection-change">+${(maxUsageIncrease * 100).toFixed(1)}%</div>
                    </div>

                    <div class="projection-card">
                        <div class="projection-label">Minutes Per Game</div>
                        <div class="projection-values">
                            <span class="current-value">${currentMinutes.toFixed(1)}</span>
                            <span class="projection-arrow">→</span>
                            <span class="projected-value">${projectedMinutesCap.toFixed(1)}</span>
                        </div>
                        <div class="projection-change">+${maxMinutesIncrease.toFixed(1)} MPG</div>
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

                    <div class="factor-item ${maxUsageIncrease > 0.08 ? 'positive' : maxUsageIncrease < 0.05 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Usage Upside</div>
                        <div class="factor-value">+${(maxUsageIncrease * 100).toFixed(0)}%</div>
                        <div class="factor-assessment">${maxUsageIncrease > 0.08 ? 'Significant room' : maxUsageIncrease < 0.05 ? 'Limited room' : 'Moderate room'}</div>
                    </div>

                    <div class="factor-item ${maxMinutesIncrease > 5 ? 'positive' : maxMinutesIncrease < 3 ? 'negative' : 'neutral'}">
                        <div class="factor-label">Minutes Upside</div>
                        <div class="factor-value">+${maxMinutesIncrease.toFixed(1)}</div>
                        <div class="factor-assessment">${maxMinutesIncrease > 5 ? 'Significant room' : maxMinutesIncrease < 3 ? 'Limited room' : 'Moderate room'}</div>
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
