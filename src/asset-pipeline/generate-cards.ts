#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';
import chalk from 'chalk';
import ora from 'ora';

// Types
interface PlayerCard {
  player: {
    id?: string;
    name: string;
    team: string;
    season: number;
    position: string;
    age?: number;
  };
  value_metrics?: {
    player_value_score?: number;
    player_value_score_raw?: number;
  };
  // Add other fields as needed
}

interface CardAssignment {
  family: 'standard' | 'fit' | 'value';
  material: string;
  color_scheme: [string, string];
  overlays: string[];
}

interface ImageAssets {
  cutout?: string;
  silhouette?: string;
  card_front: string;
  card_thumb: string;
}

interface CardGenerationConfig {
  inputData: string;
  outputDir: string;
  assetDir: string;
  formats: ('webp' | 'png')[];
  sizes: {
    thumbnail: { width: number; height: number };
    full: { width: number; height: number };
  };
  maxPlayers?: number; // For testing
}

class CardGenerator {
  private config: CardGenerationConfig;
  private spinner = ora();

  constructor(config: CardGenerationConfig) {
    this.config = {
      inputData: config.inputData,
      outputDir: config.outputDir,
      assetDir: config.assetDir,
      formats: config.formats || ['webp'],
      sizes: config.sizes || {
        thumbnail: { width: 300, height: 420 },
        full: { width: 600, height: 840 }
      },
      maxPlayers: config.maxPlayers
    };
  }

  async generateCards(): Promise<void> {
    this.spinner.start('Starting card generation pipeline...');
    
    try {
      // 1. Load player data
      this.spinner.text = 'Loading player data...';
      const players = await this.loadPlayerData();
      
      if (this.config.maxPlayers && players.length > this.config.maxPlayers) {
        players.length = this.config.maxPlayers;
        this.spinner.text = `Testing with ${players.length} players...`;
      }
      
      this.spinner.succeed(`Loaded ${players.length} players`);
      
      // 2. Assign card families
      this.spinner.start('Assigning card families...');
      const assignedPlayers = this.assignCardFamilies(players);
      this.spinner.succeed(`Assigned card families to ${assignedPlayers.length} players`);
      
      // 3. Create output directories
      this.spinner.start('Creating output directories...');
      this.createOutputDirectories();
      this.spinner.succeed('Created output directories');
      
      // 4. Generate card images (placeholder for now)
      this.spinner.start('Generating card images...');
      await this.generateCardImages(assignedPlayers);
      this.spinner.succeed(`Generated images for ${assignedPlayers.length} players`);
      
      // 5. Generate manifest
      this.spinner.start('Generating manifest...');
      await this.generateManifest(assignedPlayers);
      this.spinner.succeed('Generated manifest file');
      
      console.log(chalk.green('\n✅ Card generation complete!'));
      console.log(chalk.blue(`📁 Output directory: ${this.config.outputDir}`));
      
    } catch (error) {
      this.spinner.fail('Card generation failed');
      console.error(chalk.red('Error:'), error);
      process.exit(1);
    }
  }

  private async loadPlayerData(): Promise<PlayerCard[]> {
    const dataPath = path.resolve(this.config.inputData);
    
    if (!fs.existsSync(dataPath)) {
      throw new Error(`Data file not found: ${dataPath}`);
    }
    
    const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
    
    // Handle both array and single object formats
    return Array.isArray(data) ? data : [data];
  }

  private assignCardFamilies(players: PlayerCard[]): Array<PlayerCard & { card_assignment: CardAssignment }> {
    // Simplified card family assignment logic
    // In production, this would use the same logic as getCardFamilyProfile() in app.js
    
    return players.map((player, index) => {
      // Simple assignment based on index for testing
      const families: CardAssignment['family'][] = ['standard', 'fit', 'value'];
      const family = families[index % families.length];
      
      const materials = {
        standard: ['chrome', 'refractor'],
        fit: ['blue-ice', 'zebra-ice', 'tiger-ice', 'black-ice'],
        value: ['auto', 'patch', 'manga', 'auto-patch']
      };
      
      const material = materials[family][index % materials[family].length];
      
      // Simple color schemes based on team/position
      const colorSchemes: Record<string, [string, string]> = {
        'LAL': ['#552583', '#FDB927'], // Lakers
        'BOS': ['#007A33', '#BA9653'], // Celtics
        'GSW': ['#1D428A', '#FFC72C'], // Warriors
        'default': ['#1E3A8A', '#DC2626'] // Blue/Red
      };
      
      const team = player.player.team || 'default';
      const color_scheme = colorSchemes[team] || colorSchemes.default;
      
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

  private createOutputDirectories(): void {
    const dirs = [
      this.config.outputDir,
      path.join(this.config.outputDir, 'thumbnails'),
      path.join(this.config.outputDir, 'full'),
      path.join(this.config.assetDir, 'cutouts'),
      path.join(this.config.assetDir, 'silhouettes'),
      path.join(this.config.assetDir, 'textures'),
      path.join(this.config.assetDir, 'overlays')
    ];
    
    for (const dir of dirs) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  }

  private async generateCardImages(players: Array<PlayerCard & { card_assignment: CardAssignment }>): Promise<void> {
    // Placeholder for actual image generation
    // In production, this would:
    // 1. Render SVG layouts with Satori
    // 2. Composite images with Sharp
    // 3. Save in multiple formats and sizes
    
    console.log(chalk.yellow('⚠  Image generation is a placeholder'));
    console.log(chalk.yellow('   In production, this would use Satori + Sharp'));
    
    // Create placeholder manifest for now
    const manifest: any[] = [];
    
    for (const player of players) {
      const playerName = player.player.name.replace(/\s+/g, '_');
      const team = player.player.team || 'UNK';
      const season = player.player.season || 2025;
      const family = player.card_assignment.family;
      const material = player.card_assignment.material;
      
      const imagePaths = {
        card_front: `images/full/${playerName}_${team}_${season}_${family}_${material}_full.webp`,
        card_thumb: `images/thumbnails/${playerName}_${team}_${season}_${family}_${material}_thumb.webp`
      };
      
      manifest.push({
        player: player.player,
        card_assignment: player.card_assignment,
        image_assets: imagePaths
      });
    }
    
    // Save placeholder manifest
    const manifestPath = path.join(this.config.outputDir, '..', 'card-images.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  }

  private async generateManifest(players: Array<PlayerCard & { card_assignment: CardAssignment }>): Promise<void> {
    // Already done in generateCardImages for now
    // In production, this would create a comprehensive manifest
  }
}

// Main execution
async function main() {
  console.log(chalk.blue('=== NBA Player Cards Asset Pipeline ===\n'));
  
  const config: CardGenerationConfig = {
    inputData: path.join(__dirname, '../../web/data/cards.json'),
    outputDir: path.join(__dirname, '../../web/images'),
    assetDir: path.join(__dirname, '../../assets'),
    formats: ['webp'],
    sizes: {
      thumbnail: { width: 300, height: 420 },
      full: { width: 600, height: 840 }
    },
    maxPlayers: 5 // Test with 5 players first
  };
  
  const generator = new CardGenerator(config);
  await generator.generateCards();
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { CardGenerator, CardGenerationConfig };