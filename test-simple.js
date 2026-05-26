// Simple test to check if Node.js is working
console.log('Node.js version:', process.version);
console.log('Testing asset pipeline setup...');

const fs = require('fs');
const path = require('path');

console.log('Current directory:', process.cwd());

// Check if web/data directory exists
const dataPath = path.join(process.cwd(), 'web', 'data');
console.log('Data path exists:', fs.existsSync(dataPath));

// Check if cards.json exists
const cardsPath = path.join(dataPath, 'cards.json');
if (fs.existsSync(cardsPath)) {
  console.log('cards.json exists');
  const data = JSON.parse(fs.readFileSync(cardsPath, 'utf8'));
  console.log(`Found ${Array.isArray(data) ? data.length : 1} player cards`);
} else {
  console.log('cards.json not found');
}

console.log('Test complete');