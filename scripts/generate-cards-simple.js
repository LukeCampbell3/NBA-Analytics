#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('=== NBA Player Cards Asset Pipeline (Simple) ===\n');

// Configuration
const config = {
  inputData: path.join(__dirname, '../web/data/cards.json'),
  outputDir: path.join(__dirname, '../web/images'),
  assetDir: path.join(__dirname, '../assets'),
  maxPlayers: 10, // Test with 10 players first
  cardWidth: 600,
  cardHeight: 840,
  thumbnailWidth: 300,
  thumbnailHeight: 420
};

// Create output directories
function createDirectories() {
  console.log('Creating directories...');
  
  const dirs = [
    config.outputDir,
    path.join(config.outputDir, 'thumbnails'),
    path.join(config.outputDir, 'full'),
    path.join(config.assetDir, 'cutouts'),
    path.join(config.assetDir, 'silhouettes'),
    path.join(config.assetDir, 'textures'),
    path.join(config.assetDir, 'overlays')
  ];
  
  for (const dir of dirs) {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      console.log(`  Created: ${dir}`);
    }
  }
}

// Load player data
function loadPlayerData() {
  console.log('Loading player data...');
  
  if (!fs.existsSync(config.inputData)) {
    throw new Error(`Data file not found: ${config.inputData}`);
  }
  
  const data = JSON.parse(fs.readFileSync(config.inputData, 'utf8'));
  const players = Array.isArray(data) ? data : [data];
  
  console.log(`  Loaded ${players.length} players`);
  
  // Limit for testing
  if (config.maxPlayers && players.length > config.maxPlayers) {
    console.log(`  Testing with first ${config.maxPlayers} players`);
    return players.slice(0, config.maxPlayers);
  }
  
  return players;
}

// Assign card families (simplified logic)
function assignCardFamilies(players) {
  console.log('Assigning card families...');
  
  const teamColors = {
    'LAL': ['#552583', '#FDB927'], // Lakers
    'BOS': ['#007A33', '#BA9653'], // Celtics
    'GSW': ['#1D428A', '#FFC72C'], // Warriors
    'default': ['#1E3A8A', '#DC2626'] // Blue/Red
  };
  
  return players.map((player, index) => {
    const families = ['standard', 'fit', 'value'];
    const family = families[index % families.length];
    
    const materials = {
      standard: ['chrome', 'refractor'],
      fit: ['blue-ice', 'zebra-ice', 'tiger-ice', 'black-ice'],
      value: ['auto', 'patch', 'manga', 'auto-patch']
    };
    
    const material = materials[family][index % materials[family].length];
    const team = player.player?.team || 'default';
    const color_scheme = teamColors[team] || teamColors.default;
    
    return {
      ...player,
      card_assignment: {
        family,
        material,
        color_scheme,
        overlays: index % 3 === 0 ? ['rookie_badge'] : []
      }
    };
  });
}

// Generate placeholder images and manifest
function generatePlaceholderAssets(players) {
  console.log('Generating placeholder assets...');
  
  const manifest = [];
  
  for (const player of players) {
    const playerName = player.player?.name?.replace(/\s+/g, '_') || 'Unknown';
    const team = player.player?.team || 'UNK';
    const season = player.player?.season || 2025;
    const family = player.card_assignment.family;
    const material = player.card_assignment.material;
    
    const imagePaths = {
      card_front: `images/full/${playerName}_${team}_${season}_${family}_${material}_full.png`,
      card_thumb: `images/thumbnails/${playerName}_${team}_${season}_${family}_${material}_thumb.png`
    };
    
    manifest.push({
      player: player.player,
      card_assignment: player.card_assignment,
      image_assets: imagePaths
    });
    
    // Create placeholder image files
    createPlaceholderImage(
      path.join(config.outputDir, 'full', `${playerName}_${team}_${season}_${family}_${material}_full.png`),
      config.cardWidth,
      config.cardHeight,
      player.player?.name || 'Player',
      team,
      family,
      material
    );
    
    createPlaceholderImage(
      path.join(config.outputDir, 'thumbnails', `${playerName}_${team}_${season}_${family}_${material}_thumb.png`),
      config.thumbnailWidth,
      config.thumbnailHeight,
      player.player?.name || 'Player',
      team,
      family,
      material
    );
  }
  
  // Save manifest
  const manifestPath = path.join(config.outputDir, '..', 'card-images.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`  Saved manifest: ${manifestPath}`);
  
  return manifest.length;
}

// Create a simple placeholder image using HTML/CSS
function createPlaceholderImage(filePath, width, height, name, team, family, material) {
  const html = `
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      background: #f0f0f0;
    }
    .card {
      width: ${width}px;
      height: ${height}px;
      background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
      border-radius: 16px;
      border: 4px solid #444;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      color: white;
      font-family: Arial, sans-serif;
      text-align: center;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .name {
      font-size: ${width / 15}px;
      font-weight: bold;
      margin-bottom: 10px;
      color: #fff;
    }
    .team {
      font-size: ${width / 20}px;
      color: #ffcc00;
      margin-bottom: 10px;
    }
    .family {
      font-size: ${width / 25}px;
      color: #66ccff;
      margin-bottom: 5px;
    }
    .material {
      font-size: ${width / 30}px;
      color: #99ff99;
      margin-bottom: 20px;
    }
    .placeholder {
      font-size: ${width / 40}px;
      color: #cccccc;
      opacity: 0.7;
    }
    .size {
      font-size: ${width / 50}px;
      color: #999;
      margin-top: 20px;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="name">${name}</div>
    <div class="team">${team}</div>
    <div class="family">${family.toUpperCase()} SERIES</div>
    <div class="material">${material.replace('-', ' ').toUpperCase()}</div>
    <div class="placeholder">[Placeholder for generated card image]</div>
    <div class="size">${width}x${height}px</div>
  </div>
</body>
</html>
  `;
  
  // For now, just create a text file with the HTML
  // In production, this would use Satori + Sharp to generate actual images
  fs.writeFileSync(filePath.replace('.png', '.html'), html);
  
  // Create a simple text file as placeholder
  fs.writeFileSync(filePath, `Placeholder for: ${name} (${team})\nFamily: ${family}\nMaterial: ${material}\nSize: ${width}x${height}px\n\nThis would be a generated PNG/WebP image in production.`);
}

// Update frontend to use generated images
function updateFrontend() {
  console.log('Updating frontend configuration...');
  
  // Create a simple script to show how frontend would be updated
  const updateScript = `
// Frontend update for pre-rendered cards
// Replace the createPlayerCard function to use pre-rendered images

// Load card image mappings
let cardImageCache = {};
try {
  const response = await fetch('data/card-images.json');
  cardImageCache = await response.json();
} catch (error) {
  console.warn('Could not load card images, falling back to HTML rendering');
}

// Modified createPlayerCard function
function createPlayerCardWithImages(player, options = {}) {
  const playerKey = \`\${player.player?.name}|\${player.player?.team}\`;
  const imageInfo = cardImageCache.find(card => 
    card.player.name === player.player?.name && 
    card.player.team === player.player?.team
  );
  
  if (imageInfo && imageInfo.image_assets) {
    // Use pre-rendered image
    return \`
      <div class="player-card-image">
        <img 
          src="\${imageInfo.image_assets.card_thumb}" 
          alt="\${player.player?.name}"
          class="card-image"
          data-full="\${imageInfo.image_assets.card_front}"
          loading="lazy"
        />
        <div class="card-overlay">
          <div class="player-name">\${player.player?.name}</div>
          <div class="player-team">\${player.player?.team}</div>
        </div>
      </div>
    \`;
  } else {
    // Fallback to HTML rendering
    return originalCreatePlayerCard(player, options);
  }
}

console.log('Frontend ready for image-based cards');
  `;
  
  const scriptPath = path.join(config.outputDir, '..', 'frontend-update.js');
  fs.writeFileSync(scriptPath, updateScript);
  console.log(`  Created frontend update example: ${scriptPath}`);
}

// Main function
async function main() {
  try {
    createDirectories();
    
    const players = loadPlayerData();
    const assignedPlayers = assignCardFamilies(players);
    
    const count = generatePlaceholderAssets(assignedPlayers);
    
    updateFrontend();
    
    console.log('\n✅ Asset pipeline setup complete!');
    console.log(`📊 Processed ${count} players`);
    console.log(`📁 Output directory: ${config.outputDir}`);
    console.log(`📁 Assets directory: ${config.assetDir}`);
    console.log('\n📝 Next steps:');
    console.log('1. Install Satori and Sharp for actual image generation');
    console.log('2. Implement actual card rendering with team colors and materials');
    console.log('3. Process player images for cutouts');
    console.log('4. Update frontend to use generated images');
    
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

// Run the script
main();