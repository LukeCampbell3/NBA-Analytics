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
        
        this.init();
    }

    async init() {
        await this.loadData();
        this.setupEventListeners();
        this.populateArchetypeFilter();
        this.populateTeamFilter();
        this.renderPlayers();
        this.updateStats();
    }

    async loadData() {
        try {
            // Load player cards from data_sample format
            const cardsResponse = await fetch('data/cards.json');
            const cardsData = await cardsResponse.json();
            
            // Handle both array and object formats
            this.players = Array.isArray(cardsData) ? cardsData : [cardsData];

            // Load valuations if available
            try {
                const valuationsResponse = await fetch('data/valuations.json');
                const valuationsData = await valuationsResponse.json();
                this.valuations = Array.isArray(valuationsData) ? valuationsData : [valuationsData];
            } catch (e) {
                console.log('Valuations not available');
                this.valuations = [];
            }

            // Merge valuation data into players
            this.mergePlayerData();
            this.filteredPlayers = [...this.players];
            
            document.getElementById('loading').style.display = 'none';
        } catch (error) {
            console.error('Error loading data:', error);
            document.getElementById('loading').innerHTML = '<p>Error loading player data. Please ensure data files are available.</p>';
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
        });

        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
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

        // Draw charts after modal is visible
        setTimeout(() => {
            this.drawShotChart(player);
            this.drawValueDriversChart(player);
        }, 50);
    }

    createPlayerDetail(player) {
        const valuation = player.valuation;
        const perf = player.performance || {};
        const trad = perf.traditional || {};
        const adv = perf.advanced || {};
        const trust = player.v1_1_enhancements?.trust_assessment || {};
        const uncertainty = player.v1_1_enhancements?.uncertainty_estimates || {};

        return `
            <div class="modal-header">
                <h2>${player.player.name}</h2>
                <div style="margin-top: 0.5rem; opacity: 0.9;">
                    ${player.player.team} • ${player.player.position || 'N/A'} • 
                    Age ${player.player.age?.toFixed(1) || 'N/A'} • 
                    ${this.formatArchetype(player.identity?.primary_archetype || 'unknown')}
                </div>
            </div>

            <div class="modal-body">
                <!-- Visual Analytics Section -->
                <div class="detail-section">
                    <h3>Visual Analytics</h3>
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

                <div class="modal-body">
                <!-- Performance Metrics -->
                <div class="detail-section">
                    <h3>Performance Metrics</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-item-label">Plus/Minus</div>
                            <div class="detail-item-value">${(adv.plus_minus || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Points Per Game</div>
                            <div class="detail-item-value">${(trad.points_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Assists Per Game</div>
                            <div class="detail-item-value">${(trad.assists_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Rebounds Per Game</div>
                            <div class="detail-item-value">${(trad.rebounds_per_game || 0).toFixed(1)}</div>
                        </div>
                    </div>
                </div>

                <!-- Shot Profile -->
                <div class="detail-section">
                    <h3>Shot Profile</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-item-label">Rim Frequency</div>
                            <div class="detail-item-value">${((player.shot_profile?.rim_frequency || 0) * 100).toFixed(0)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Three-Point Frequency</div>
                            <div class="detail-item-value">${((player.shot_profile?.three_point_frequency || 0) * 100).toFixed(0)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">FG%</div>
                            <div class="detail-item-value">${((trad.field_goal_pct || 0) * 100).toFixed(1)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">3P%</div>
                            <div class="detail-item-value">${((trad.three_point_pct || 0) * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>

                <!-- Creation Profile -->
                <div class="detail-section">
                    <h3>Creation Profile</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-item-label">Drives Per Game</div>
                            <div class="detail-item-value">${(player.creation_profile?.drives_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Paint Touches</div>
                            <div class="detail-item-value">${(player.creation_profile?.paint_touches_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Assisted Rate</div>
                            <div class="detail-item-value">${((player.creation_profile?.assisted_rate || 0) * 100).toFixed(0)}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Usage Rate</div>
                            <div class="detail-item-value">${((adv.usage_rate || 0) * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>

                <!-- Defense Assessment -->
                <div class="detail-section">
                    <h3>Defense Assessment</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-item-label">Steals Per Game</div>
                            <div class="detail-item-value">${(trad.steals_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Blocks Per Game</div>
                            <div class="detail-item-value">${(trad.blocks_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Defensive Rating</div>
                            <div class="detail-item-value">${(player.defense_assessment?.estimated_metrics?.defensive_rating || 0).toFixed(1)}</div>
                        </div>
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
                        <div class="detail-item">
                            <div class="detail-item-label">Wins Added</div>
                            <div class="detail-item-value">${valuation.impact.wins_added.toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Trade Value (Base)</div>
                            <div class="detail-item-value">$${valuation.trade_value.base.toFixed(1)}M</div>
                        </div>
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
                    
                    ${trust.factors && trust.factors.length > 0 ? `
                    <div style="margin-top: 1rem;">
                        <h4 style="font-size: 0.9rem; margin-bottom: 0.5rem; color: var(--text-secondary);">Trust Factors:</h4>
                        <ul style="list-style: none; padding: 0;">
                            ${trust.factors.map(([factor, value]) => `
                                <li style="padding: 0.25rem 0; font-size: 0.85rem;">
                                    <strong>${this.formatFactor(factor)}:</strong> ${value}
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>

                <!-- Comparables -->
                ${player.comparables?.similar_players && player.comparables.similar_players.length > 0 ? `
                <div class="detail-section">
                    <h3>Similar Players</h3>
                    <div style="display: grid; gap: 0.75rem;">
                        ${player.comparables.similar_players.slice(0, 5).map(comp => `
                            <div style="background: var(--bg-secondary); padding: 0.75rem; border-radius: 6px;">
                                <div style="font-weight: 600; margin-bottom: 0.25rem;">
                                    ${comp.name} (${comp.team}) - ${(comp.similarity_score * 100).toFixed(0)}% similar
                                </div>
                                <div style="font-size: 0.85rem; color: var(--text-secondary);">
                                    ${comp.stats.points.toFixed(1)} PPG, 
                                    ${comp.stats.assists.toFixed(1)} APG, 
                                    ${comp.stats.rebounds.toFixed(1)} RPG
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}

                <!-- Sample Info -->
                <div class="detail-section">
                    <h3>Sample Information</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <div class="detail-item-label">Games Played</div>
                            <div class="detail-item-value">${trad.games_played || 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Minutes Per Game</div>
                            <div class="detail-item-value">${(trad.minutes_per_game || 0).toFixed(1)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Data Version</div>
                            <div class="detail-item-value">${player.metadata?.data_version || 'N/A'}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-item-label">Archetype Confidence</div>
                            <div class="detail-item-value">${((player.identity?.archetype_confidence || 0) * 100).toFixed(0)}%</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    formatFactor(factor) {
        return factor.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    updateStats() {
        // Stats summary removed from UI
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

        // Get shot frequencies
        const rimFreq = player.shot_profile?.rim_frequency || 0;
        const midFreq = player.shot_profile?.mid_range_frequency || 0;
        const threeFreq = player.shot_profile?.three_point_frequency || 0;

        // Scale factors
        const courtWidth = width - 40;
        const courtHeight = height - 60;
        const offsetX = 20;
        const offsetY = 20;

        // Draw court background
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(offsetX, offsetY, courtWidth, courtHeight);

        // Draw court lines
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;

        // Baseline
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY + courtHeight);
        ctx.lineTo(offsetX + courtWidth, offsetY + courtHeight);
        ctx.stroke();

        // Sidelines
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY);
        ctx.lineTo(offsetX, offsetY + courtHeight);
        ctx.moveTo(offsetX + courtWidth, offsetY);
        ctx.lineTo(offsetX + courtWidth, offsetY + courtHeight);
        ctx.stroke();

        // Free throw line (19 feet from baseline, ~40% of half court)
        const ftLineY = offsetY + courtHeight - (courtHeight * 0.4);
        ctx.beginPath();
        ctx.moveTo(offsetX + courtWidth * 0.2, ftLineY);
        ctx.lineTo(offsetX + courtWidth * 0.8, ftLineY);
        ctx.stroke();

        // Paint (16 feet wide, ~43% of court width)
        const paintWidth = courtWidth * 0.43;
        const paintX = offsetX + (courtWidth - paintWidth) / 2;
        ctx.strokeRect(paintX, offsetY + courtHeight - (courtHeight * 0.4), paintWidth, courtHeight * 0.4);

        // Free throw circle
        ctx.beginPath();
        ctx.arc(offsetX + courtWidth / 2, ftLineY, courtWidth * 0.1, 0, Math.PI * 2);
        ctx.stroke();

        // Three-point arc (23.75 feet, ~50% of half court depth)
        const threePointRadius = courtWidth * 0.45;
        ctx.beginPath();
        ctx.arc(offsetX + courtWidth / 2, offsetY + courtHeight, threePointRadius, Math.PI * 1.15, Math.PI * 1.85);
        ctx.stroke();

        // Three-point corners
        ctx.beginPath();
        ctx.moveTo(offsetX, offsetY + courtHeight - courtHeight * 0.15);
        ctx.lineTo(offsetX, offsetY + courtHeight);
        ctx.moveTo(offsetX + courtWidth, offsetY + courtHeight - courtHeight * 0.15);
        ctx.lineTo(offsetX + courtWidth, offsetY + courtHeight);
        ctx.stroke();

        // Rim
        ctx.beginPath();
        ctx.arc(offsetX + courtWidth / 2, offsetY + courtHeight - courtHeight * 0.08, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6b6b';
        ctx.fill();
        ctx.stroke();

        // Heat zones with better colors
        const zones = [
            // Rim zone (restricted area)
            {
                type: 'circle',
                x: offsetX + courtWidth / 2,
                y: offsetY + courtHeight - courtHeight * 0.08,
                radius: courtWidth * 0.15,
                freq: rimFreq,
                color: '#ff6b6b',
                label: 'Rim'
            },
            // Mid-range zones (paint + mid-range)
            {
                type: 'rect',
                x: paintX,
                y: ftLineY,
                width: paintWidth,
                height: courtHeight * 0.4,
                freq: midFreq * 0.6,
                color: '#ffd93d',
                label: 'Paint'
            },
            {
                type: 'arc',
                x: offsetX + courtWidth / 2,
                y: offsetY + courtHeight,
                innerRadius: courtWidth * 0.15,
                outerRadius: threePointRadius - 10,
                startAngle: Math.PI * 1.15,
                endAngle: Math.PI * 1.85,
                freq: midFreq * 0.4,
                color: '#ffb347',
                label: 'Mid-Range'
            },
            // Three-point zones
            {
                type: 'arc',
                x: offsetX + courtWidth / 2,
                y: offsetY + courtHeight,
                innerRadius: threePointRadius - 10,
                outerRadius: threePointRadius + 40,
                startAngle: Math.PI * 1.15,
                endAngle: Math.PI * 1.85,
                freq: threeFreq * 0.7,
                color: '#6bcf7f',
                label: 'Three'
            },
            // Corners
            {
                type: 'rect',
                x: offsetX,
                y: offsetY + courtHeight - courtHeight * 0.15,
                width: 40,
                height: courtHeight * 0.15,
                freq: threeFreq * 0.15,
                color: '#4ecdc4',
                label: 'Corner'
            },
            {
                type: 'rect',
                x: offsetX + courtWidth - 40,
                y: offsetY + courtHeight - courtHeight * 0.15,
                width: 40,
                height: courtHeight * 0.15,
                freq: threeFreq * 0.15,
                color: '#4ecdc4',
                label: 'Corner'
            }
        ];

        // Draw heat zones
        zones.forEach(zone => {
            const intensity = Math.min(zone.freq * 2, 1); // Scale up for visibility
            const alpha = 0.2 + (intensity * 0.6);

            ctx.globalAlpha = alpha;
            ctx.fillStyle = zone.color;

            if (zone.type === 'circle') {
                ctx.beginPath();
                ctx.arc(zone.x, zone.y, zone.radius, 0, Math.PI * 2);
                ctx.fill();
            } else if (zone.type === 'rect') {
                ctx.fillRect(zone.x, zone.y, zone.width, zone.height);
            } else if (zone.type === 'arc') {
                ctx.beginPath();
                ctx.arc(zone.x, zone.y, zone.outerRadius, zone.startAngle, zone.endAngle);
                ctx.arc(zone.x, zone.y, zone.innerRadius, zone.endAngle, zone.startAngle, true);
                ctx.closePath();
                ctx.fill();
            }
        });

        ctx.globalAlpha = 1;

        // Draw legend at bottom
        const legendY = offsetY + courtHeight + 25;
        const legendItems = [
            { color: '#ff6b6b', label: `Rim: ${(rimFreq * 100).toFixed(0)}%` },
            { color: '#ffd93d', label: `Mid: ${(midFreq * 100).toFixed(0)}%` },
            { color: '#6bcf7f', label: `3PT: ${(threeFreq * 100).toFixed(0)}%` }
        ];

        let legendX = offsetX;
        legendItems.forEach(item => {
            // Color box
            ctx.fillStyle = item.color;
            ctx.fillRect(legendX, legendY, 20, 20);
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.strokeRect(legendX, legendY, 20, 20);

            // Label
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 13px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(item.label, legendX + 25, legendY + 15);

            legendX += 120;
        });
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
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new PlayerCardsApp();
});
