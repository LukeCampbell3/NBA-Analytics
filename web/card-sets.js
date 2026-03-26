// Card Set System - Random "trading card set" styles inspired by real card brands
// Each player gets randomly assigned to a set, giving visual variety like a real card store

const CARD_SETS = [
    {
        id: 'prizm-silver',
        name: 'Prizm Silver',
        brand: 'PRIZM',
        borderStyle: 'linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 25%, #a0a0a0 50%, #d4d4d4 75%, #b0b0b0 100%)',
        bgStyle: 'linear-gradient(160deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        accentColor: '#c0c0c0',
        textColor: '#f0f0f0',
        nameColor: '#ffffff',
        shimmer: true,
        holoEffect: 'silver',
        frameWidth: '4px',
        frameColor: 'rgba(192,192,192,0.6)',
        cornerAccent: '#e8e8e8',
        labelBg: 'rgba(192,192,192,0.15)',
        rarity: 'Silver'
    },
    {
        id: 'prizm-gold',
        name: 'Prizm Gold',
        brand: 'PRIZM',
        borderStyle: 'linear-gradient(135deg, #ffd700 0%, #ffec80 25%, #daa520 50%, #f5d060 75%, #b8860b 100%)',
        bgStyle: 'linear-gradient(160deg, #1a0a00 0%, #2d1810 50%, #3d1c00 100%)',
        accentColor: '#ffd700',
        textColor: '#fff8dc',
        nameColor: '#ffd700',
        shimmer: true,
        holoEffect: 'gold',
        frameWidth: '4px',
        frameColor: 'rgba(255,215,0,0.5)',
        cornerAccent: '#ffd700',
        labelBg: 'rgba(255,215,0,0.12)',
        rarity: 'Gold'
    },
    {
        id: 'topps-chrome',
        name: 'Topps Chrome',
        brand: 'TOPPS',
        borderStyle: 'linear-gradient(180deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%)',
        bgStyle: 'linear-gradient(170deg, #0c1445 0%, #162055 40%, #1e3a8a 100%)',
        accentColor: '#3b82f6',
        textColor: '#dbeafe',
        nameColor: '#93c5fd',
        shimmer: false,
        holoEffect: 'chrome',
        frameWidth: '3px',
        frameColor: 'rgba(59,130,246,0.5)',
        cornerAccent: '#60a5fa',
        labelBg: 'rgba(59,130,246,0.12)',
        rarity: 'Chrome'
    },
    {
        id: 'topps-heritage',
        name: 'Topps Heritage',
        brand: 'TOPPS',
        borderStyle: 'linear-gradient(180deg, #fbbf24 0%, #f59e0b 50%, #d97706 100%)',
        bgStyle: 'linear-gradient(170deg, #fef3c7 0%, #fde68a 40%, #fcd34d 100%)',
        accentColor: '#d97706',
        textColor: '#451a03',
        nameColor: '#78350f',
        shimmer: false,
        holoEffect: null,
        frameWidth: '5px',
        frameColor: 'rgba(217,119,6,0.6)',
        cornerAccent: '#f59e0b',
        labelBg: 'rgba(217,119,6,0.1)',
        rarity: 'Heritage'
    },
    {
        id: 'select-courtside',
        name: 'Select Courtside',
        brand: 'SELECT',
        borderStyle: 'linear-gradient(135deg, #7c3aed 0%, #a78bfa 30%, #6d28d9 60%, #8b5cf6 100%)',
        bgStyle: 'linear-gradient(160deg, #0f0520 0%, #1e1045 50%, #2e1065 100%)',
        accentColor: '#a78bfa',
        textColor: '#ede9fe',
        nameColor: '#c4b5fd',
        shimmer: true,
        holoEffect: 'purple',
        frameWidth: '3px',
        frameColor: 'rgba(167,139,250,0.45)',
        cornerAccent: '#a78bfa',
        labelBg: 'rgba(167,139,250,0.12)',
        rarity: 'Courtside'
    },
    {
        id: 'mosaic-reactive',
        name: 'Mosaic Reactive',
        brand: 'MOSAIC',
        borderStyle: 'linear-gradient(135deg, #06b6d4 0%, #22d3ee 30%, #0891b2 60%, #67e8f9 100%)',
        bgStyle: 'linear-gradient(160deg, #042f2e 0%, #064e3b 50%, #0f766e 100%)',
        accentColor: '#22d3ee',
        textColor: '#cffafe',
        nameColor: '#67e8f9',
        shimmer: true,
        holoEffect: 'teal',
        frameWidth: '3px',
        frameColor: 'rgba(34,211,238,0.4)',
        cornerAccent: '#22d3ee',
        labelBg: 'rgba(34,211,238,0.1)',
        rarity: 'Reactive'
    },
    {
        id: 'optic-holo',
        name: 'Optic Holo',
        brand: 'OPTIC',
        borderStyle: 'linear-gradient(135deg, #f43f5e 0%, #fb7185 30%, #e11d48 60%, #fda4af 100%)',
        bgStyle: 'linear-gradient(160deg, #1a0008 0%, #350012 50%, #4c0519 100%)',
        accentColor: '#fb7185',
        textColor: '#ffe4e6',
        nameColor: '#fda4af',
        shimmer: true,
        holoEffect: 'rose',
        frameWidth: '3px',
        frameColor: 'rgba(251,113,133,0.4)',
        cornerAccent: '#fb7185',
        labelBg: 'rgba(251,113,133,0.1)',
        rarity: 'Holo'
    },
    {
        id: 'flux-supernova',
        name: 'Flux Supernova',
        brand: 'FLUX',
        borderStyle: 'linear-gradient(135deg, #f97316 0%, #fb923c 30%, #ea580c 60%, #fdba74 100%)',
        bgStyle: 'linear-gradient(160deg, #1c0800 0%, #431407 50%, #7c2d12 100%)',
        accentColor: '#fb923c',
        textColor: '#fff7ed',
        nameColor: '#fdba74',
        shimmer: true,
        holoEffect: 'orange',
        frameWidth: '3px',
        frameColor: 'rgba(251,146,60,0.4)',
        cornerAccent: '#fb923c',
        labelBg: 'rgba(251,146,60,0.1)',
        rarity: 'Supernova'
    }
];

// Deterministic "random" set assignment based on player name hash
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
}

function getCardSet(playerName) {
    const hash = hashString(playerName);
    return CARD_SETS[hash % CARD_SETS.length];
}

// Build the shimmer/holo CSS class for a given set
function getSetShimmerClass(set) {
    if (!set.shimmer) return '';
    return `card-shimmer card-shimmer-${set.holoEffect || 'silver'}`;
}
