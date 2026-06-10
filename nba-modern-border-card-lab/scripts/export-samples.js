const fs = require('fs');
const path = require('path');
const builder = require('../src/modern-border-card-builder.js');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'output');
fs.mkdirSync(out, { recursive: true });
for (const f of fs.readdirSync(out)) {
  if (f.endsWith('.svg') || f.endsWith('.png')) fs.rmSync(path.join(out, f), { force: true });
}
const player = {
  playerName: 'MARCUS REED',
  teamName: 'NASHVILLE STARS',
  jerseyNumber: '23',
  headshotData: '__demo_vector__',
  stats: builder.DEFAULT_STATS
};
const keys = [
  'base-paper',
  'prism-edge',
  'blue-prism-edge',
  'wave-border-r',
  'gold-wave-edge',
  'mojo-border',
  'checker-border',
  'chrome-full',
  'cracked-ice-edge',
  'black-gold-edge'
];
fs.writeFileSync(path.join(out, 'modern-border-refractor-board.svg'), builder.renderBoardSvg({ ...player, variants: keys }));
for (const key of keys) {
  fs.writeFileSync(path.join(out, `${key}.svg`), builder.renderCardSvg({ ...player, variant: key }, keys.indexOf(key)));
}
console.log('SVG exports written:', keys.length + 1);
