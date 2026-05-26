#!/usr/bin/env ts-node

import * as fs from 'fs';
import * as path from 'path';
import satori from 'satori';
import { Resvg } from '@resvg/resvg-js';
import React from 'react';

interface PlayerCard {
  player: {
    name: string;
    team: string;
    position: string;
    season: number;
    age?: number;
  };
  value_metrics?: {
    player_value_score?: number;
    player_value_score_raw?: number;
  };
}

interface CardAssignment {
  family: string;
  material: string;
  color_scheme: [string, string];
  overlays: string[];
}

interface SatoriRendererConfig {
  width: number;
  height: number;
  fonts: Array<{
    name: string;
    data: Buffer;
    weight: number;
    style: 'normal' | 'italic';
  }>;
}

class SatoriRenderer {
  private config: SatoriRendererConfig;

  constructor(config: SatoriRendererConfig) {
    this.config = config;
  }

  async renderCard(player: PlayerCard, assignment: CardAssignment): Promise<Buffer> {
    // Create React element for the card
    const element = this.createCardElement(player, assignment);
    
    // Render to SVG
    const svg = await satori(element, {
      width: this.config.width,
      height: this.config.height,
      fonts: this.config.fonts
    });
    
    // Convert SVG to PNG
    const resvg = new Resvg(svg, {
      fitTo: {
        mode: 'width',
        value: this.config.width
      }
    });
    
    const pngData = resvg.render();
    return pngData.asPng();
  }

  private createCardElement(player: PlayerCard, assignment: CardAssignment): React.ReactElement {
    const [primaryColor, secondaryColor] = assignment.color_scheme;
    
    return React.createElement(
      'div',
      {
        style: {
          display: 'flex',
          flexDirection: 'column',
          width: '100%',
          height: '100%',
          backgroundColor: '#1a1a1a',
          color: 'white',
          fontFamily: 'Inter, sans-serif',
          borderRadius: '16px',
          padding: '24px',
          border: `2px solid ${primaryColor}`,
          boxShadow: `0 8px 32px rgba(0, 0, 0, 0.3), inset 0 0 0 1px ${secondaryColor}20`,
          position: 'relative',
          overflow: 'hidden'
        }
      },
      [
        // Background gradient
        React.createElement('div', {
          key: 'bg',
          style: {
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `linear-gradient(135deg, ${primaryColor}20, ${secondaryColor}20)`,
            zIndex: 0
          }
        }),
        
        // Content
        React.createElement('div', {
          key: 'content',
          style: {
            position: 'relative',
            zIndex: 1,
            display: 'flex',
            flexDirection: 'column',
            height: '100%'
          }
        }, [
          // Header
          React.createElement('div', {
            key: 'header',
            style: {
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: '16px'
            }
          }, [
            React.createElement('div', {
              key: 'name',
              style: {
                fontSize: '28px',
                fontWeight: 'bold',
                color: 'white',
                textShadow: `0 2px 4px rgba(0, 0, 0, 0.5)`
              }
            }, player.player.name),
            
            React.createElement('div', {
              key: 'team',
              style: {
                fontSize: '20px',
                fontWeight: '600',
                color: secondaryColor,
                backgroundColor: `${primaryColor}40`,
                padding: '4px 12px',
                borderRadius: '8px',
                border: `1px solid ${primaryColor}`
              }
            }, player.player.team)
          ]),
          
          // Position and season
          React.createElement('div', {
            key: 'details',
            style: {
              display: 'flex',
              gap: '12px',
              marginBottom: '24px',
              fontSize: '16px',
              color: '#cccccc'
            }
          }, [
            React.createElement('div', {
              key: 'position'
            }, `Position: ${player.player.position}`),
            
            React.createElement('div', {
              key: 'season'
            }, `Season: ${player.player.season}`),
            
            player.player.age && React.createElement('div', {
              key: 'age'
            }, `Age: ${player.player.age}`)
          ]),
          
          // Value metrics
          player.value_metrics && React.createElement('div', {
            key: 'metrics',
            style: {
              display: 'flex',
              gap: '16px',
              marginBottom: '24px'
            }
          }, [
            player.value_metrics.player_value_score && React.createElement('div', {
              key: 'value',
              style: {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                backgroundColor: `${primaryColor}30`,
                padding: '12px',
                borderRadius: '8px',
                border: `1px solid ${primaryColor}`
              }
            }, [
              React.createElement('div', {
                key: 'label',
                style: {
                  fontSize: '14px',
                  color: '#aaaaaa',
                  marginBottom: '4px'
                }
              }, 'Value Score'),
              React.createElement('div', {
                key: 'score',
                style: {
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: secondaryColor
                }
              }, player.value_metrics.player_value_score.toFixed(1))
            ]),
            
            player.value_metrics.player_value_score_raw && React.createElement('div', {
              key: 'raw',
              style: {
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                backgroundColor: `${secondaryColor}20`,
                padding: '12px',
                borderRadius: '8px',
                border: `1px solid ${secondaryColor}`
              }
            }, [
              React.createElement('div', {
                key: 'label',
                style: {
                  fontSize: '14px',
                  color: '#aaaaaa',
                  marginBottom: '4px'
                }
              }, 'Raw Value'),
              React.createElement('div', {
                key: 'score',
                style: {
                  fontSize: '24px',
                  fontWeight: 'bold',
                  color: primaryColor
                }
              }, player.value_metrics.player_value_score_raw.toFixed(1))
            ])
          ]),
          
          // Card family info
          React.createElement('div', {
            key: 'family',
            style: {
              marginTop: 'auto',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              paddingTop: '16px',
              borderTop: `1px solid ${primaryColor}40`
            }
          }, [
            React.createElement('div', {
              key: 'family-label',
              style: {
                fontSize: '18px',
                fontWeight: '600',
                color: secondaryColor
              }
            }, `${assignment.family.toUpperCase()} SERIES`),
            
            React.createElement('div', {
              key: 'material',
              style: {
                fontSize: '16px',
                color: '#cccccc',
                textTransform: 'capitalize'
              }
            }, assignment.material.replace('-', ' '))
          ]),
          
          // Footer
          React.createElement('div', {
            key: 'footer',
            style: {
              fontSize: '12px',
              color: '#888888',
              textAlign: 'center',
              marginTop: '8px'
            }
          }, 'NBA Analytics Engine • Generated Card')
        ])
      ]
    );
  }
}

// Test function
async function testRenderer() {
  console.log('Testing Satori renderer...');
  
  // Load a font (using default for now)
  // In production, you would load actual font files
  const fonts = [{
    name: 'Inter',
    data: Buffer.from(''), // Empty for now
    weight: 400,
    style: 'normal' as const
  }];
  
  const renderer = new SatoriRenderer({
    width: 600,
    height: 840,
    fonts
  });
  
  const testPlayer: PlayerCard = {
    player: {
      name: 'LeBron James',
      team: 'LAL',
      position: 'F',
      season: 2025,
      age: 40
    },
    value_metrics: {
      player_value_score: 85.5,
      player_value_score_raw: 82.3
    }
  };
  
  const testAssignment: CardAssignment = {
    family: 'value',
    material: 'auto-patch',
    color_scheme: ['#8B0000', '#FFD700'], // Dark red and gold
    overlays: ['rookie_badge', 'signature_strip']
  };
  
  try {
    console.log('Rendering test card...');
    const pngBuffer = await renderer.renderCard(testPlayer, testAssignment);
    
    // Save test output
    const outputPath = path.join(__dirname, '../../test-card.png');
    fs.writeFileSync(outputPath, pngBuffer);
    
    console.log(`✅ Test card saved to: ${outputPath}`);
    console.log('Note: Font rendering may be basic without actual font files');
    
  } catch (error) {
    console.error('Error rendering card:', error);
  }
}

// Run if called directly
if (require.main === module) {
  testRenderer().catch(console.error);
}

export { SatoriRenderer, SatoriRendererConfig };