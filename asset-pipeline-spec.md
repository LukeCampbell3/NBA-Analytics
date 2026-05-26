# NBA Player Cards Asset Pipeline Specification

## Overview
Hybrid build-time rendering pipeline using Satori (structured layout) + Sharp (image compositing) to generate premium basketball card images as static assets.

## Architecture Goals
- **Static deployment**: Pre-render all cards at build time
- **Rich visual quality**: 5-8 layer compositing for realistic card feel
- **Performance**: Frontend only displays pre-rendered images
- **Maintainability**: Clear separation between data, templates, and rendering

## Folder Structure

```
/
├── dist/                          # Static site output (deployable)
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   ├── assets/
│   │   ├── cards/                # Generated card images (PNG/WebP)
│   │   │   ├── thumbnails/       # 300x420px
│   │   │   └── full/             # 600x840px
│   │   ├── cutouts/              # Player image cutouts
│   │   ├── overlays/             # Card material layers
│   │   └── team-logos/
│   └── data/
│       ├── cards.json            # Player data
│       └── valuations.json       # Valuation data
├── web/                          # Source frontend
│   ├── index.html
│   ├── app.js                    # Modified to use pre-rendered images
│   ├── styles.css
│   └── templates/
│       └── card-template.jsx     # Satori JSX template
├── src/
│   ├── asset-pipeline/
│   │   ├── generate-cards.ts     # Main card generation script
│   │   ├── satori-renderer.ts    # Satori layout rendering
│   │   ├── sharp-compositor.ts   # Sharp image compositing
│   │   ├── cutout-processor.ts   # Player image preprocessing
│   │   └── asset-manager.ts      # Asset organization
│   ├── data/
│   │   ├── card-assigner.ts      # Assign card families/materials
│   │   └── player-processor.ts   # Process player data
│   └── utils/
│       └── image-utils.ts        # Image utilities
├── assets/
│   ├── source/
│   │   ├── player-images/        # Raw player photos
│   │   └── action-shots/         # Action photos for cutouts
│   ├── processed/
│   │   ├── cutouts/              # Transparent PNG cutouts
│   │   └── silhouettes/          # Secondary silhouette layers
│   └── card-materials/
│       ├── frames/               # Card frame templates
│       ├── textures/             # Material textures
│       ├── overlays/             # Foil/refractor overlays
│       ├── badges/               # Rookie/team badges
│       └── team-motifs/          # Team-specific backgrounds
├── scripts/
│   ├── build-cards.sh            # Full pipeline script
│   ├── preprocess-cutouts.sh     # Cutout preprocessing
│   └── deploy.sh                 # Deployment script
└── config/
    ├── card-families.yaml        # Card family definitions
    ├── material-assignments.yaml # Material assignment rules
    └── render-settings.yaml      # Rendering settings
```

## Card Layer Stack (5-8 Layers)

1. **Base Frame** - Card shape and border
2. **Material Finish** - Chrome, refractor, ice, manga, patch, auto
3. **Background Motif** - Team/city theme, diagonal stripes, arena lights
4. **Player Cutout** - Primary action/waist-up cutout
5. **Silhouette Layer** - Secondary depth shadow
6. **UI/Stat Layer** - Rendered by Satori (typography, stats, badges)
7. **Overlays** - Rookie badge, signature strip, patch window
8. **Glare/Specular** - Premium card shine

## Card Families & Materials

### Standard Family
- **Chrome** - Metallic base
- **Refractor** - Holographic prism effect

### Fit Family (Ice Variants)
- **Blue Ice** - High fit score
- **Zebra Ice** - Medium-high fit
- **Tiger Ice** - Medium fit
- **Black Ice** - Low fit

### Value Family
- **Auto** - Autograph style
- **Patch** - Jersey patch window
- **Manga** - Comic book style
- **Auto-Patch** - Combined auto + patch

## Build Pipeline Steps

### 1. Data Preparation
- Load player cards from `data/processed/player_cards/*_final.json`
- Load valuations from `data/valuations/`
- Process and merge data
- Assign card families based on value/fit scores

### 2. Asset Preprocessing
- Process player images → transparent cutouts (U-2-Net)
- Generate silhouette layers
- Prepare card material assets

### 3. Card Layout Rendering (Satori)
- Render structured UI layer to SVG:
  - Player name, team, position
  - Value/fit scores
  - Stat strips
  - Labels and badges
- Output: SVG for each player

### 4. Image Compositing (Sharp)
- Composite 5-8 layers:
  1. Base frame
  2. Material texture
  3. Background motif
  4. Player cutout
  5. Silhouette layer
  6. Satori SVG (UI layer)
  7. Overlays (badges, patches)
  8. Glare/specular layer
- Export: PNG and WebP formats
- Generate thumbnails (300x420px) and full size (600x840px)

### 5. Static Site Generation
- Update frontend to use pre-rendered images
- Generate `dist/` with all assets
- Create JSON manifest of card-image mappings

## Player Data Structure

```json
{
  "player": {
    "id": "player_123",
    "name": "LeBron James",
    "team": "LAL",
    "position": "F",
    "season": 2025,
    "age": 40
  },
  "value_metrics": {
    "player_value_score": 85.5,
    "player_value_score_raw": 82.3
  },
  "card_assignment": {
    "family": "value",
    "material": "auto-patch",
    "color_scheme": "#8B0000,#FFD700",
    "overlays": ["rookie_badge", "signature_strip"]
  },
  "image_assets": {
    "cutout": "assets/processed/cutouts/lebron_james_2025.png",
    "silhouette": "assets/processed/silhouettes/lebron_james_2025.png",
    "card_front": "assets/cards/full/lebron_james_LAL_2025.webp",
    "card_thumb": "assets/cards/thumbnails/lebron_james_LAL_2025.webp"
  }
}
```

## Frontend Modifications

### Current (Runtime HTML)
```javascript
// Current: HTML + CSS rendering at runtime
createPlayerCard(player) {
  return `<div class="player-card">...HTML/CSS...</div>`;
}
```

### New (Pre-rendered Images)
```javascript
// New: Display pre-rendered images
createPlayerCard(player) {
  return `
    <div class="player-card" data-player-id="${player.id}">
      <img 
        src="${player.image_assets.card_thumb}" 
        alt="${player.player.name}"
        class="card-image"
        data-full-size="${player.image_assets.card_front}"
      />
      <div class="card-overlay">...</div>
    </div>
  `;
}
```

## Build Scripts

### `scripts/build-cards.sh`
```bash
#!/bin/bash
# Full card generation pipeline

echo "=== NBA Player Cards Asset Pipeline ==="

# 1. Prepare data
echo "Step 1: Preparing player data..."
node src/asset-pipeline/generate-cards.ts --step prepare

# 2. Preprocess images (if needed)
echo "Step 2: Preprocessing player images..."
node src/asset-pipeline/cutout-processor.ts

# 3. Render card layouts
echo "Step 3: Rendering card layouts with Satori..."
node src/asset-pipeline/satori-renderer.ts

# 4. Composite final images
echo "Step 4: Compositing final card images with Sharp..."
node src/asset-pipeline/sharp-compositor.ts

# 5. Build static site
echo "Step 5: Building static site..."
python build_static_site.py

echo "=== Pipeline Complete ==="
echo "Generated cards in: dist/assets/cards/"
echo "Total players processed: $(find dist/assets/cards/ -name '*.webp' | wc -l)"
```

### `src/asset-pipeline/generate-cards.ts` (Sample Architecture)
```typescript
import { readPlayerData } from './data/player-processor';
import { assignCardFamilies } from './data/card-assigner';
import { renderCardLayouts } from './satori-renderer';
import { compositeCardImages } from './sharp-compositor';
import { generateManifest } from './asset-manager';

interface PlayerCard {
  player: any;
  value_metrics: any;
  card_assignment: CardAssignment;
  image_assets: ImageAssets;
}

interface CardAssignment {
  family: 'standard' | 'fit' | 'value';
  material: string;
  color_scheme: [string, string];
  overlays: string[];
}

interface ImageAssets {
  cutout: string;
  silhouette: string;
  card_front: string;
  card_thumb: string;
}

async function generateCards() {
  console.log('Starting card generation pipeline...');
  
  // 1. Load and process data
  const players = await readPlayerData();
  console.log(`Loaded ${players.length} players`);
  
  // 2. Assign card families and materials
  const assignedPlayers = assignCardFamilies(players);
  
  // 3. Render SVG layouts with Satori
  const svgOutputs = await renderCardLayouts(assignedPlayers);
  
  // 4. Composite final images with Sharp
  const imageOutputs = await compositeCardImages(assignedPlayers, svgOutputs);
  
  // 5. Generate manifest and update frontend data
  await generateManifest(assignedPlayers, imageOutputs);
  
  console.log('Card generation complete!');
  console.log(`Generated ${imageOutputs.length} card images`);
}

// Run pipeline
generateCards().catch(console.error);
```

## Dependencies

### Core Dependencies
```json
{
  "dependencies": {
    "satori": "^0.1.0",
    "sharp": "^0.33.0",
    "jsdom": "^24.0.0",
    "canvas": "^3.0.0"
  },
  "devDependencies": {
    "@types/sharp": "^0.33.0",
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0"
  }
}
```

### Optional: Cutout Processing
- **U-2-Net**: Offline background removal (Python)
- **@imgly/background-removal**: Node.js background removal (AGPL-3.0)

## Performance Considerations

### Build Time Optimization
- Parallel processing of players
- Cache intermediate assets
- Incremental builds

### Output Optimization
- WebP format for modern browsers
- PNG fallback for compatibility
- Responsive images with srcset

### Frontend Performance
- Lazy loading card images
- Intersection Observer for viewport loading
- Preload critical cards

## Deployment

### Static Hosting
- **Netlify**: Connect to GitHub, auto-build on push
- **Vercel**: Zero-config deployment
- **GitHub Pages**: Free static hosting
- **AWS S3 + CloudFront**: Scalable CDN

### Build Triggers
- GitHub Actions on data updates
- Scheduled nightly builds
- Manual trigger via webhook

## Migration Plan

### Phase 1: Foundation
1. Set up folder structure
2. Create build scripts
3. Implement Satori template
4. Test with sample player

### Phase 2: Core Pipeline
1. Implement Sharp compositing
2. Create asset management
3. Generate full pipeline
4. Test with 10 players

### Phase 3: Production
1. Process all players
2. Optimize performance
3. Update frontend
4. Deploy to staging

### Phase 4: Enhancement
1. Add cutout preprocessing
2. Implement material variations
3. Add animation effects
4. Optimize loading

## Success Metrics

- **Build time**: < 5 minutes for 500 players
- **Image quality**: 90+ on Lighthouse performance
- **File size**: < 100KB per card (WebP)
- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3s

## Next Steps

1. **Review this spec** and provide feedback
2. **Set up initial structure** with sample implementation
3. **Test pipeline** with 5-10 players
4. **Iterate on design** based on output quality
5. **Scale to full dataset** and deploy

This pipeline transforms the current runtime HTML cards into premium, pre-rendered basketball card images while maintaining static deployment and excellent performance.