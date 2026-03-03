# NBA-VAR Web Frontend

A clean, dynamic web interface for browsing and analyzing NBA player cards.

## Features

- **Browse Players**: View all player cards in a responsive grid layout
- **Search**: Search by player name or team
- **Filter**: Filter by archetype, usage band, or team
- **Sort**: Sort by name, impact, age, wins added, or breakout score
- **Percentile Indicators**: Visual indicators showing player percentiles
- **Detailed View**: Click any card to see comprehensive player details
- **Responsive Design**: Works on desktop, tablet, and mobile

## Quick Start

### 1. Prepare Data

First, generate player cards, analyses, and valuations:

```bash
# From root directory
python create_cards.py --input data/stats.csv --output data/cards
python analyze_players.py --cards data/cards --output analysis
python value_players.py --cards data/cards --output valuations
```

### 2. Prepare Web Data

Convert the data for web consumption:

```bash
python prepare_web_data.py --cards data/cards --analysis analysis --valuations valuations --output web/data
```

### 3. Serve the Web App

Option A - Python HTTP Server:
```bash
python -m http.server 8000 --directory web
```

Option B - Node.js HTTP Server:
```bash
npx http-server web -p 8000
```

Option C - Open directly:
```bash
# Open web/index.html in your browser
# Note: Some browsers may block local file access
```

### 4. Access the App

Open your browser and navigate to:
```
http://localhost:8000
```

## File Structure

```
web/
├── index.html          # Main HTML structure
├── styles.css          # Styling and layout
├── app.js              # Application logic
├── data/               # Data files (generated)
│   ├── cards.json      # Player cards
│   ├── analyses.json   # Player analyses
│   ├── valuations.json # Player valuations
│   └── summary.json    # Summary statistics
└── README.md           # This file
```

## Features in Detail

### Player Cards

Each card displays:
- Player name, team, position, age
- Primary archetype
- Impact metrics with percentile indicators
- Usage band
- Wins added
- Breakout score
- Tags (Breakout Candidate, High Impact, High Portability)

### Filters

- **Search**: Real-time search by player name or team
- **Archetype**: Filter by player archetype (7 types)
- **Usage**: Filter by usage band (high/med/low)
- **Team**: Filter by team
- **Sort**: Sort by various metrics

### Player Detail Modal

Click any card to see:
- Complete impact metrics
- Offensive profile (shot profile, creation, playmaking)
- Defensive profile (burden, stocks, portability)
- Valuation & contract (wins added, trade value, aging phase)
- Breakout analysis (opportunity, signal, confidence)
- Scouting report (strengths, weaknesses)
- Data quality metrics

### Percentile Indicators

Visual bars showing player percentiles:
- Green: Top 10%
- Blue: Top 25%
- Orange: Top 50%
- Red: Bottom 50%

### Stats Summary

Dashboard showing:
- Total players (filtered)
- Average impact
- Breakout candidates count
- High portability defenders count

## Customization

### Colors

Edit `styles.css` to customize colors:

```css
:root {
    --primary-color: #1e40af;    /* Main brand color */
    --secondary-color: #3b82f6;  /* Secondary brand color */
    --accent-color: #f59e0b;     /* Accent color */
    --success-color: #10b981;    /* Success/positive */
    --danger-color: #ef4444;     /* Danger/negative */
}
```

### Layout

Adjust grid columns in `styles.css`:

```css
.cards-container {
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
}
```

### Metrics

Add or modify metrics in `app.js`:

```javascript
createPlayerCard(player) {
    // Add your custom metrics here
}
```

## Data Format

### cards.json

Array of player card objects:

```json
[
  {
    "player": {
      "name": "Player Name",
      "team": "TEAM",
      "position": "PG",
      "age": 25.5
    },
    "identity": {
      "usage_band": "high",
      "primary_archetype": "initiator_creator"
    },
    "impact": {
      "net": 5.2,
      "offensive": 3.1,
      "defensive": 2.1
    },
    ...
  }
]
```

### analyses.json

Array of analysis objects:

```json
[
  {
    "scouting_report": { ... },
    "breakout_potential": { ... },
    "defense_portability": { ... },
    "impact_sanity": { ... }
  }
]
```

### valuations.json

Array of valuation objects:

```json
[
  {
    "player": { ... },
    "impact": { "wins_added": 4.2 },
    "trade_value": { "low": 3.2, "base": 4.5, "high": 5.8 },
    "aging": { "current_phase": "growth", "peak_age": 28.5 }
  }
]
```

## Browser Compatibility

- Chrome/Edge: ✓ Full support
- Firefox: ✓ Full support
- Safari: ✓ Full support
- Mobile browsers: ✓ Responsive design

## Performance

- Handles 500+ players smoothly
- Real-time filtering and search
- Lazy loading for modal content
- Optimized CSS animations

## Troubleshooting

### Data not loading

1. Check that data files exist in `web/data/`
2. Run `prepare_web_data.py` to generate data files
3. Check browser console for errors
4. Ensure you're serving via HTTP (not file://)

### Filters not working

1. Clear browser cache
2. Check browser console for JavaScript errors
3. Verify data format matches expected structure

### Styling issues

1. Clear browser cache
2. Check that `styles.css` is loading
3. Verify CSS file path in `index.html`

## Development

### Adding New Features

1. Edit `app.js` for new functionality
2. Edit `styles.css` for new styling
3. Edit `index.html` for new structure
4. Test in multiple browsers

### Adding New Metrics

1. Ensure data is in player cards
2. Add display in `createPlayerCard()` method
3. Add to detail modal in `createPlayerDetail()` method
4. Update filters if needed

## Production Deployment

### Static Hosting

Deploy to any static hosting service:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront

### Steps

1. Generate all data files
2. Run `prepare_web_data.py`
3. Upload `web/` directory to hosting
4. Configure custom domain (optional)

### Example: GitHub Pages

```bash
# Create gh-pages branch
git checkout -b gh-pages

# Copy web files to root
cp -r web/* .

# Commit and push
git add .
git commit -m "Deploy web app"
git push origin gh-pages
```

## License

MIT License

## Support

For issues or questions:
1. Check this README
2. Review browser console for errors
3. Verify data files are properly formatted
4. Check that all dependencies are loaded
