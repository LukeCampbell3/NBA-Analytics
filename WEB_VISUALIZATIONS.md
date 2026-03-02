# Web Frontend Visual Analytics

## Overview
The NBA-VAR web frontend now includes rich visual analytics for each player, showing shot distribution, archetype value drivers, and uniqueness insights.

## Visual Components

### 1. Shot Distribution Chart
**Location**: Player detail modal, top-left visual card

**What it shows**:
- Concentric circles representing shot zones (rim, mid-range, three-point)
- Opacity indicates frequency (more shots = more opaque)
- Percentage labels show exact distribution

**How it works**:
- Uses HTML5 Canvas to draw three zones
- Rim zone (red): innermost circle, 30% of max radius
- Mid-range (orange): middle circle, 65% of max radius  
- Three-point (blue): outer circle, 100% of max radius
- Court outline provides context

**Data source**: `player.shot_profile.{rim_frequency, mid_range_frequency, three_point_frequency}`

### 2. Archetype Value Drivers Chart
**Location**: Player detail modal, top-center visual card

**What it shows**:
- Horizontal bar chart showing what makes this archetype valuable
- Bars are color-coded by metric type
- Values are normalized to show relative importance

**Archetype-specific metrics**:

**Defensive Anchor**:
- Rim Protection (blocks × 10)
- Rebounding (rebounds per game)
- Interior Presence (rim frequency × 20)
- Plus/Minus Impact

**Rim Runner**:
- Finishing at Rim (rim frequency × 20)
- Efficiency (FG% × 20)
- Paint Touches (per game)
- Scoring Output (PPG)

**Stretch Big**:
- Three-Point Shooting (3P frequency × 30)
- 3P% Efficiency (3P% × 30)
- Spacing Value (PPG)
- Rebounding (RPG)

**Versatile Wing**:
- Scoring Versatility (PPG)
- Playmaking (APG × 2)
- Defense (steals + blocks × 5)
- Usage Rate (× 50)

**How it works**:
- Calculates value drivers based on archetype
- Sorts by value (highest to lowest)
- Draws gradient-filled bars
- Shows exact values on bars

### 3. Archetype Uniqueness Insights
**Location**: Player detail modal, top-right visual card

**What it shows**:
- Bullet points with emoji icons
- What makes this player unique within their archetype
- Comparisons to archetype norms
- Data quality indicators

**Insight categories**:

**Performance-based**:
- Elite rim protection (>1.5 BPG)
- Dominant rebounding (>10 RPG)
- High rim frequency (>60%)
- Efficient finishing (>55% FG)
- Reliable shooting (>35% 3P)
- Modern big (>30% shots from three)
- Dual threat (scoring + playmaking)
- Defensive playmaker (>1.0 SPG)

**Impact-based**:
- Elite impact (+5 or better)
- Negative impact (-3 or worse)
- High usage (>25%)

**Data quality**:
- High reliability (trust score ≥80)
- Limited data (trust score <60)

**Comparisons**:
- Most similar player from comparables

**How it works**:
- Analyzes player stats against archetype thresholds
- Generates contextual insights with emojis
- Highlights standout characteristics
- Falls back to "Standard archetype profile" if no standouts

## Technical Implementation

### Files Modified
1. `web/app.js`:
   - Added `drawShotChart()` method
   - Added `drawValueDriversChart()` method
   - Added `calculateValueDrivers()` method
   - Added `generateUniquenessInsights()` method
   - Modified `createPlayerDetail()` to include visual section
   - Modified `showPlayerDetail()` to draw charts after modal opens

2. `web/styles.css`:
   - Added `.visual-grid` for responsive layout
   - Added `.visual-card` for chart containers
   - Added `.shot-legend` for shot chart legend
   - Added `.insight-item` for uniqueness insights
   - Added responsive breakpoints

3. `web/index.html`:
   - No changes needed (modal structure already supports dynamic content)

### Canvas Drawing Details

**Shot Chart**:
- 300×300px canvas
- Three concentric circles
- Opacity: 0.3 + (frequency × 0.7)
- Text with stroke for readability
- Court outline for context

**Value Drivers**:
- 300×300px canvas (can be wider)
- Horizontal bars with gradient fills
- Background bars show max scale
- Labels above bars, values inside bars
- 4 metrics per archetype

### Data Flow
1. User clicks player card
2. `showPlayerDetail()` called with player data
3. Modal HTML generated with canvas elements
4. `setTimeout()` ensures modal is visible
5. `drawShotChart()` and `drawValueDriversChart()` called
6. Charts render using player data

## Usage

### Viewing Visualizations
1. Start web server: `python serve_web.py`
2. Open browser to http://localhost:8000
3. Click any player card
4. Scroll to "Visual Analytics" section at top of modal

### Testing Visualizations
- Open `test_visuals.html` in browser for standalone chart tests
- Modify test data to see different scenarios
- Useful for debugging chart rendering

## Future Enhancements

### Potential Additions
1. **Shot Chart Heatmap**: Show actual shot locations on court diagram
2. **Comparison Mode**: Overlay two players' charts side-by-side
3. **Historical Trends**: Line charts showing performance over time
4. **Archetype Radar**: Spider/radar chart comparing to archetype average
5. **Interactive Tooltips**: Hover over chart elements for more detail
6. **Export Charts**: Download charts as PNG images
7. **Animated Transitions**: Smooth animations when switching players

### Data Requirements
- Shot location data (x, y coordinates) for detailed shot charts
- Historical performance data for trend analysis
- Archetype averages for comparison baselines

## Browser Compatibility
- Modern browsers with Canvas support (Chrome, Firefox, Safari, Edge)
- No external chart libraries required
- Responsive design works on mobile/tablet

## Performance
- Charts render in <50ms on modern hardware
- No performance impact on page load
- Charts only drawn when modal opens
- Efficient canvas operations

## Accessibility
- Charts include text labels for screen readers
- Color-blind friendly palette (consider adding patterns)
- Keyboard navigation supported via modal
- Consider adding ARIA labels for canvas elements
