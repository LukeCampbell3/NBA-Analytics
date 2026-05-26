#!/usr/bin/env ts-node

console.log('=== NBA Player Cards Asset Pipeline Test ===');
console.log('Testing pipeline setup...');

// Test basic imports
try {
  console.log('✓ TypeScript/Node.js environment is working');
  
  // Check if we can read sample data
  const fs = require('fs');
  const path = require('path');
  
  const sampleDataPath = path.join(__dirname, '../../web/data/cards.json');
  if (fs.existsSync(sampleDataPath)) {
    console.log(`✓ Found cards data at: ${sampleDataPath}`);
    
    const data = JSON.parse(fs.readFileSync(sampleDataPath, 'utf8'));
    console.log(`✓ Loaded ${Array.isArray(data) ? data.length : 1} player cards`);
    
    // Show first player as sample
    const samplePlayer = Array.isArray(data) ? data[0] : data;
    console.log(`✓ Sample player: ${samplePlayer.player?.name || 'Unknown'} (${samplePlayer.player?.team || 'No team'})`);
  } else {
    console.log(`⚠ Cards data not found at: ${sampleDataPath}`);
  }
  
} catch (error) {
  console.error('✗ Error during test:', error);
  process.exit(1);
}

console.log('=== Test Complete ===');
console.log('Next steps:');
console.log('1. Install dependencies: npm install');
console.log('2. Run card generation: npm run build-cards');
console.log('3. Check generated assets in web/images/');