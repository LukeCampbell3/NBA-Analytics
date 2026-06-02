# NBA Player Cards Asset Pipeline Architecture

## Overview
Production-ready static asset pipeline for generating basketball card images at build time using Satori + Sharp. Replaces runtime HTML/CSS cards with pre-rendered PNG/WebP images.

## Folder Structure

```
/
├── assets/                          # Source art assets
│   ├── textures/                    # Card material textures
│   │   ├── chrome/
│   │   ├── refractor/
│   │   ├── ice/
│   │   ├── manga/
│   │   └── auto/
│   ├── overlays/                    # Card overlays
│   │   ├── rookie-badge.png
│   │   ├── signature-strip.png
│   │   ├── patch-window.png
│   │   └── glare-specular.png
│   ├── team-motifs/                 # Team background motifs
│   │   ├── DEN.png
│   │   ├── LAL.png
│   │   └── etc...
│   └── cutouts/                     # Player image cutouts
│       ├── Aaron_Gordon_DEN.png
│       ├── LeBron_James_LAL.png
│       └── etc...
├── scripts/                         # Build scripts
│   ├── generate-cards.ts            # Main card generation script
│   ├── preprocess-cutouts.py        # Background removal preprocessing
│   └── assign-card-families.ts      # Card family assignment logic
├── templates/                       # Satori card templates
│   ├── base-card.tsx               # Base card layout
│   ├── standard-family.tsx         # Chrome/Refractor templates
│   ├── fit-family.tsx              # Ice variants templates
│   └── value-family.tsx            # Auto/Patch/Manga templates
├── dist/                           # Final static site (unchanged)
├── web/                           # Frontend (updated to use images)
│   ├── data/
│   │   ├── cards.json             # Player data
│   │   ├── card-images.json       # Image path mapping
│   │   └── valuations.json        # Valuation data
│   └── images/                    # Generated card images
│       ├── thumbnails/            # 300x420px thumbnails
│       │   ├── Aaron_Gordon_DEN_2025_standard_thumb.webp
│       │   └── etc...
│       └── full/                  # 600x840px full resolution
│           ├── Aaron_Gordon_DEN_2025_standard_full.webp
│           └── etc...
└── package.json                   # Node.js dependencies
```

## Build Pipeline Steps

### 1. Preprocessing Phase
- **Input**: Raw player photos from data/raw/
- **Process**: Background removal using U-2-Net or IMG.LY
- **Output**: Transparent PNG cutouts in `assets/cutouts/`

### 2. Card Family Assignment
- **Input**: Player data + valuations from `web/data/`
- **Process**: Apply same logic as current `getCardFamilyProfile()`
- **Output**: Card family mapping JSON

### 3. Card Image Generation
- **Process**: For each player:
  - Load player data
  - Determine card family (standard/fit/value)
  - Select appropriate textures/overlays
  - Render layout with Satori (SVG)
  - Composite with Sharp (player cutout + overlays)
  - Export WebP + PNG
- **Output**: Card images in `web/images/`

### 4. Frontend Integration
- **Update**: Modify `createPlayerCard()` to use `<img>` tags
- **Data**: Generate `card-images.json` mapping player → image paths
- **Performance**: Lazy loading, responsive images

## Card Layer Stack (5-8 layers)

1. **Base Frame**: Card shape with rounded corners
2. **Material Finish**: Chrome/Refractor/Ice/Manga/Auto texture
3. **Team Motif**: Subtle team background (arena lights, city map, etc.)
4. **Player Cutout**: Transparent PNG of player (action pose)
5. **Secondary Silhouette**: Depth layer behind player
6. **UI Layer**: Stats, badges, labels (rendered by Satori)
7. **Insert Overlays**: Rookie badge, signature strip, patch window
8. **Glare/Specular**: Premium card shine effect

## Technical Stack

### Core Dependencies
```json
{
  "dependencies": {
    "satori": "^0.0.50",      // SVG rendering from JSX
    "sharp": "^0.33.0",       // Image compositing
    "react": "^18.2.0",       // JSX for Satori
    "@types/react": "^18.2.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "tsx": "^4.0.0"           // TypeScript execution
  }
}
```

### Card Generation Script (`scripts/generate-cards.ts`)

```typescript
interface CardGenerationConfig {
  inputData: string;          // Path to cards.json
  outputDir: string;          // Output directory
  assetDir: string;           // Asset directory
  formats: ('webp' | 'png')[]; // Output formats
  sizes: {
    thumbnail: { width: number; height: number };
    full: { width: number; height: number };
  };
}

interface PlayerCard {
  player: {
    id: string;
    name: string;
    team: string;
    season: number;
    position: string;
  };
  // ... existing card data
}

interface CardFamily {
  key: string;        // 'standard', 'fit', 'value'
  variant: string;    // 'chrome', 'blue-ice', 'auto', etc.
  textures: string[]; // Texture file paths
  overlays: string[]; // Overlay file paths
}
```

## Implementation Priority

### Phase 1: Core Pipeline
1. Set up folder structure
2. Install dependencies
3. Create basic Satori template
4. Generate simple test cards

### Phase 2: Realism Enhancements
1. Implement layer compositing with Sharp
2. Add material textures
3. Add overlays (badges, signatures)
4. Add team motifs

### Phase 3: Production Optimization
1. Batch processing
2. Caching
3. Error handling
4. Progress reporting

### Phase 4: Frontend Integration
1. Update app.js to use images
2. Add lazy loading
3. Add fallbacks
4. Performance testing

## Performance Considerations

- **Build Time**: ~1-2 seconds per card (500 cards = 10-15 minutes)
- **File Size**: ~50-100KB per card (WebP)
- **CDN Ready**: Static assets can be served from CDN
- **Caching**: Images are cacheable forever (content hash)

## Migration Strategy

1. Keep current HTML cards as fallback
2. Generate images for subset of players first
3. A/B test performance
4. Gradually migrate all cards
5. Remove HTML card generation code

## Next Steps

1. Create `package.json` with dependencies
2. Create folder structure
3. Write `generate-cards.ts` skeleton
4. Test with 5 sample players
5. Integrate with existing build process