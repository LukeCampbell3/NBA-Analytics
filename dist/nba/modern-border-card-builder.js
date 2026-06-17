const CARD_W = 360;
const CARD_H = 504;

const DEFAULT_STATS = [
  { value: '21.4', label: 'PTS' },
  { value: '5.8', label: 'REB' },
  { value: '6.9', label: 'AST' },
  { value: '+4.1', label: 'VAR' }
];

const DEFAULT_PLAYER = {
  playerName: 'MARCUS REED',
  teamName: 'NASHVILLE STARS',
  jerseyNumber: '23',
  headshot: '__demo_vector__',
  variant: 'prism-edge',
  stats: DEFAULT_STATS
};

const VARIANTS = {
  'base-paper': {
    label: 'BASE PAPER',
    note: 'quiet stock, clean border, headshot-forward base card',
    mode: 'border',
    palette: ['#f8f5ec', '#d9c492', '#0d1726', '#ffffff', '#b99a50', '#141414']
  },
  'prism-edge': {
    label: 'PRISM EDGE',
    note: 'neutral field with refractor color concentrated in the rails',
    mode: 'border',
    palette: ['#eef4f8', '#cfd6dc', '#00d7ff', '#eb4dff', '#ffe66e', '#08111d']
  },
  'blue-prism-edge': {
    label: 'BLUE PRISM EDGE',
    note: 'cool blue/cyan prism spectrum on the border shell',
    mode: 'border',
    palette: ['#eef8ff', '#cdeaff', '#00b7ff', '#2c6bff', '#bdf8ff', '#071426']
  },
  'wave-border-r': {
    label: 'PURPLE WAVE EDGE',
    note: 'wavy violet/indigo refractor contained to the frame',
    mode: 'border',
    rookie: true,
    palette: ['#11182d', '#4e2ee8', '#43d9ff', '#a855f7', '#ffdb6b', '#050914']
  },
  'gold-wave-edge': {
    label: 'GOLD WAVE EDGE',
    note: 'warm gold/orange wave spectrum without changing the photo field',
    mode: 'border',
    palette: ['#211203', '#f59e0b', '#ffd56a', '#ff6b35', '#fff2b2', '#100702']
  },
  'mojo-border': {
    label: 'MOJO CIRCUIT EDGE',
    note: 'interlocking mojo lens/circuit geometry on rails and lower frame',
    mode: 'border',
    palette: ['#07111d', '#1f8fff', '#10f0c5', '#b45cff', '#ffd45d', '#041017']
  },
  'checker-border': {
    label: 'CHECKER EDGE',
    note: 'controlled teal-violet-amber checker spectrum, edge-led only',
    mode: 'border',
    palette: ['#0b1020', '#21d4fd', '#38f8c7', '#8b5cf6', '#ffce5c', '#ffffff']
  },
  'chrome-full': {
    label: 'CHROME FULL',
    note: 'whole-card chromium sheen with broad refractor hits',
    mode: 'full',
    palette: ['#e9eef3', '#b9c8d3', '#00eaff', '#ff4ede', '#fff26b', '#07101a']
  },
  'cracked-ice-edge': {
    label: 'CRACKED ICE',
    note: 'ice-crystal material masked away from the headshot zone',
    mode: 'full',
    palette: ['#f5fdff', '#d8f7ff', '#8ceeff', '#ffffff', '#b7f4ff', '#0b2233']
  },
  'black-gold-edge': {
    label: 'BLACK FINITE 1/1',
    note: 'black finite chase-card material with gold/chrome trim',
    mode: 'full',
    oneOfOne: true,
    palette: ['#020202', '#111111', '#d4af37', '#fff0a8', '#6b4a12', '#000000']
  }
};
function esc(v = '') {
  return String(v)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function rng(seed = 1) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function starPath(cx, cy, outer = 15, inner = 6) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const r = i % 2 ? inner : outer;
    pts.push(`${(cx + Math.cos(a) * r).toFixed(1)},${(cy + Math.sin(a) * r).toFixed(1)}`);
  }
  return pts.join(' ');
}

function defs(id, variant) {
  const p = variant.palette;
  return `
  <defs>
    <clipPath id="${id}-cardClip"><rect x="0" y="0" width="360" height="504" rx="26"/></clipPath>
    <clipPath id="${id}-portraitSafe"><path d="M30 48 H330 V352 C292 344 244 340 180 340 C116 340 68 344 30 352 Z"/></clipPath>
    <filter id="${id}-shadow" x="-35%" y="-35%" width="170%" height="175%"><feDropShadow dx="0" dy="20" stdDeviation="18" flood-color="#000" flood-opacity=".72"/></filter>
    <filter id="${id}-cardGrain" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency=".74" numOctaves="3" seed="12" result="noise"/><feColorMatrix in="noise" type="saturate" values="0" result="gray"/><feComponentTransfer><feFuncA type="table" tableValues="0 .05"/></feComponentTransfer></filter>
    <filter id="${id}-iceBump" x="-5%" y="-5%" width="110%" height="110%"><feTurbulence type="fractalNoise" baseFrequency=".028 .05" numOctaves="4" seed="18" result="noise"/><feSpecularLighting in="noise" result="spec" surfaceScale="6" specularConstant=".95" specularExponent="18" lighting-color="#ffffff"><feDistantLight azimuth="235" elevation="55"/></feSpecularLighting><feComposite in="spec" in2="SourceAlpha" operator="in" result="specOut"/><feMerge><feMergeNode in="SourceGraphic"/><feMergeNode in="specOut"/></feMerge></filter>
    <linearGradient id="${id}-paper" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f9f7ef"/><stop offset=".52" stop-color="#e8e5dc"/><stop offset="1" stop-color="#f5f7fb"/></linearGradient>
    <radialGradient id="${id}-field" cx="50%" cy="38%" r="70%"><stop offset="0" stop-color="#ffffff"/><stop offset=".38" stop-color="#eef2f4"/><stop offset="1" stop-color="#cfd6dc"/></radialGradient>
    <linearGradient id="${id}-darkField" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#050505"/><stop offset=".44" stop-color="#0c0c0a"/><stop offset="1" stop-color="#000000"/></linearGradient>
    <radialGradient id="${id}-blackGlow" cx="50%" cy="36%" r="72%"><stop offset="0" stop-color="#2a2a2a"/><stop offset=".46" stop-color="#070707"/><stop offset="1" stop-color="#000000"/></radialGradient>
    <linearGradient id="${id}-borderFoil" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${p[2]}"/><stop offset=".12" stop-color="#fff"/><stop offset=".22" stop-color="${p[3]}"/><stop offset=".37" stop-color="${p[4]}"/><stop offset=".51" stop-color="#fff"/><stop offset=".69" stop-color="${p[2]}"/><stop offset=".84" stop-color="${p[3]}"/><stop offset="1" stop-color="#fff"/></linearGradient>
    <linearGradient id="${id}-metalSweep" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#f7f7f7"/><stop offset=".18" stop-color="#7b8791"/><stop offset=".33" stop-color="#ffffff"/><stop offset=".51" stop-color="${p[2]}"/><stop offset=".66" stop-color="${p[3]}"/><stop offset=".82" stop-color="#fff7aa"/><stop offset="1" stop-color="#dce7ef"/></linearGradient>
    <linearGradient id="${id}-goldSweep" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5e3a07"/><stop offset=".24" stop-color="#ffdf82"/><stop offset=".44" stop-color="#b87b15"/><stop offset=".58" stop-color="#fff3b8"/><stop offset=".76" stop-color="#d5a342"/><stop offset="1" stop-color="#5a3605"/></linearGradient>
    <linearGradient id="${id}-namePlate" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f9f9f5"/><stop offset="1" stop-color="#dde3e8"/></linearGradient>
    <pattern id="${id}-mojoTile" width="46" height="46" patternUnits="userSpaceOnUse">
      <rect width="46" height="46" fill="#02070c" opacity=".25"/>
      <path d="M5 5 H19 L23 9 L27 5 H41 V19 L37 23 L41 27 V41 H27 L23 37 L19 41 H5 V27 L9 23 L5 19 Z" fill="none" stroke="${p[2]}" stroke-width="2.2" stroke-opacity=".74"/>
      <path d="M13 13 H33 V33 H13 Z" fill="none" stroke="${p[4]}" stroke-width="1.45" stroke-opacity=".62"/>
      <path d="M5 23 H15 M31 23 H41 M23 5 V15 M23 31 V41" stroke="${p[3]}" stroke-width="2" stroke-opacity=".70" stroke-linecap="round"/>
      <circle cx="23" cy="23" r="5.5" fill="${p[4]}" opacity=".22" stroke="#fff" stroke-width=".8" stroke-opacity=".45"/>
      <circle cx="9" cy="9" r="1.8" fill="#fff" opacity=".46"/><circle cx="37" cy="37" r="1.8" fill="#fff" opacity=".36"/>
    </pattern>
    <pattern id="${id}-checker" width="34" height="34" patternUnits="userSpaceOnUse">
      <rect width="17" height="17" fill="${p[2]}" opacity=".50"/><rect x="17" y="0" width="17" height="17" fill="${p[3]}" opacity=".42"/>
      <rect x="0" y="17" width="17" height="17" fill="${p[4]}" opacity=".44"/><rect x="17" y="17" width="17" height="17" fill="#ffffff" opacity=".28"/>
      <path d="M0 34 L34 0 M-8 26 L26 -8 M8 42 L42 8" stroke="${p[4]}" stroke-width="1.0" opacity=".20"/>
      <path d="M0 0 H34 V34 H0 Z" fill="none" stroke="#fff" stroke-width=".7" stroke-opacity=".20"/>
    </pattern>
    <pattern id="${id}-finiteTile" width="28" height="28" patternUnits="userSpaceOnUse">
      <rect width="28" height="28" fill="#010101"/>
      <path d="M0 28 L28 0 M-7 21 L21 -7 M7 35 L35 7" stroke="#242424" stroke-width="1.15" opacity=".72"/>
      <path d="M0 0 H28 V28 H0 Z" fill="none" stroke="#0f0f0f" stroke-width="1"/>
      <path d="M0 14 H28 M14 0 V28" stroke="#151515" stroke-width=".75" opacity=".58"/>
      <circle cx="14" cy="14" r="1.4" fill="#2f2f2f" opacity=".65"/>
    </pattern>
    <pattern id="${id}-microLines" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(25)"><path d="M0 0 H8" stroke="#fff" stroke-width=".55" opacity=".12"/></pattern>
    <clipPath id="${id}-borderClip" clipPathUnits="userSpaceOnUse"><path d="M0 0 H360 V78 H282 L256 52 H216 L202 68 H158 L144 52 H104 L78 78 H0 Z"/><path d="M0 0 H70 V78 L50 104 V266 L70 302 V504 H0 Z"/><path d="M360 0 H290 V78 L310 104 V266 L290 302 V504 H360 Z"/><path d="M0 302 H360 V504 H0 Z"/></clipPath>
    <clipPath id="${id}-edgeClip" clipPathUnits="userSpaceOnUse"><rect x="0" y="0" width="360" height="22"/><rect x="0" y="482" width="360" height="22"/><rect x="0" y="0" width="22" height="504"/><rect x="338" y="0" width="22" height="504"/></clipPath>
    <mask id="${id}-iceAvoidPortrait" maskUnits="userSpaceOnUse"><rect x="0" y="0" width="360" height="504" fill="#fff"/><ellipse cx="180" cy="166" rx="112" ry="143" fill="#000"/><path d="M70 256 C110 228 250 228 290 256 L318 346 C272 334 226 329 180 329 C134 329 88 334 42 346 Z" fill="#000"/></mask>
  </defs>`;
}

function baseSurface(id, key, variant) {
  const blackGold = key === 'black-gold-edge';
  const chrome = key === 'chrome-full';
  const ice = key === 'cracked-ice-edge';
  if (blackGold) {
    return `<rect width="360" height="504" rx="26" fill="#010101"/>
      <rect width="360" height="504" rx="26" fill="url(#${id}-finiteTile)" opacity=".92"/>
      <rect width="360" height="504" rx="26" fill="url(#${id}-blackGlow)" opacity=".78"/>
      <path d="M-28 68 C72 18 134 48 206 18 C274 -10 320 2 388 -18 V104 C320 70 262 92 202 126 C134 164 58 134 -28 188 Z" fill="#ffffff" opacity=".055"/>
      <path d="M-20 366 C70 324 132 346 206 312 C274 280 320 288 392 250 V338 C318 354 260 374 194 406 C122 442 48 428 -20 462 Z" fill="url(#${id}-goldSweep)" opacity=".10"/>
      <rect width="360" height="504" rx="26" fill="url(#${id}-borderFoil)" opacity=".035"/>`;
  }
  if (chrome) {
    return `<rect width="360" height="504" rx="26" fill="url(#${id}-field)"/><rect width="360" height="504" rx="26" fill="url(#${id}-borderFoil)" opacity=".20"/><path d="M-30 80 C90 15 158 42 240 0 C283 -22 325 -18 392 4 V120 C308 70 250 94 190 126 C126 160 72 138 -30 196 Z" fill="url(#${id}-borderFoil)" opacity=".23"/><path d="M-22 376 C84 302 142 327 230 290 C274 271 322 274 382 246 V342 C300 348 243 371 181 402 C110 436 44 428 -22 461 Z" fill="#fff" opacity=".17"/>`;
  }
  if (ice) {
    return `<rect width="360" height="504" rx="26" fill="#f4fdff"/><rect width="360" height="504" rx="26" fill="url(#${id}-field)" opacity=".65"/>`;
  }
  return `<rect width="360" height="504" rx="26" fill="#d1d5db"/><rect width="360" height="504" rx="26" fill="url(#${id}-paper)" opacity=".56"/>`;
}

function polygonCell(points, fill, opacity, stroke = '#fff', so = .45, sw = .9) {
  return `<polygon points="${points}" fill="${fill}" opacity="${opacity}" stroke="${stroke}" stroke-opacity="${so}" stroke-width="${sw}"/>`;
}

function iceCells(id, full = true) {
  const r = rng(99);
  const cells = [];
  const cols = [22, 88, 154, 218, 288, 350];
  const rows = [20, 82, 145, 210, 280, 350, 430, 498];
  for (let y = 0; y < rows.length - 1; y++) {
    for (let x = 0; x < cols.length - 1; x++) {
      const x0 = cols[x] + (r() - .5) * 20;
      const x1 = cols[x + 1] + (r() - .5) * 20;
      const y0 = rows[y] + (r() - .5) * 17;
      const y1 = rows[y + 1] + (r() - .5) * 17;
      const cx = (x0 + x1) / 2 + (r() - .5) * 16;
      const cy = (y0 + y1) / 2 + (r() - .5) * 16;
      const p1 = `${x0.toFixed(1)},${y0.toFixed(1)} ${x1.toFixed(1)},${(y0 + (r()-.5)*10).toFixed(1)} ${cx.toFixed(1)},${cy.toFixed(1)}`;
      const p2 = `${x1.toFixed(1)},${(y0 + (r()-.5)*10).toFixed(1)} ${x1.toFixed(1)},${y1.toFixed(1)} ${cx.toFixed(1)},${cy.toFixed(1)}`;
      const p3 = `${x1.toFixed(1)},${y1.toFixed(1)} ${x0.toFixed(1)},${y1.toFixed(1)} ${cx.toFixed(1)},${cy.toFixed(1)}`;
      const p4 = `${x0.toFixed(1)},${y1.toFixed(1)} ${x0.toFixed(1)},${y0.toFixed(1)} ${cx.toFixed(1)},${cy.toFixed(1)}`;
      const alpha = full ? (.16 + r()*.22).toFixed(2) : (.24 + r()*.30).toFixed(2);
      cells.push(polygonCell(p1, '#ffffff', alpha, '#fff', .58, .9));
      cells.push(polygonCell(p2, '#c7f4ff', (.12 + r()*.18).toFixed(2), '#7ee7ff', .36, .8));
      cells.push(polygonCell(p3, '#eafdff', (.13 + r()*.18).toFixed(2), '#fff', .44, .8));
      cells.push(polygonCell(p4, '#8feaff', (.09 + r()*.15).toFixed(2), '#5edfff', .30, .8));
    }
  }
  for (let i=0; i<28; i++) {
    const x1 = 12 + r()*336, y1 = 18 + r()*466;
    const x2 = x1 + (r()-.5)*120, y2 = y1 + (r()-.5)*80;
    cells.push(`<path d="M${x1.toFixed(1)} ${y1.toFixed(1)} L${x2.toFixed(1)} ${y2.toFixed(1)}" stroke="#ffffff" stroke-width="${(.8 + r()*1.6).toFixed(2)}" stroke-opacity="${(.22 + r()*.45).toFixed(2)}"/>`);
  }
  return `<g clip-path="url(#${id}-cardClip)" mask="url(#${id}-iceAvoidPortrait)" filter="url(#${id}-iceBump)"><rect width="360" height="504" fill="#eafdff" opacity=".22"/>${cells.join('')}</g><g clip-path="url(#${id}-cardClip)" mask="url(#${id}-iceAvoidPortrait)"><path d="M-40 96 C80 52 118 96 190 64 C252 37 305 32 402 76" stroke="#ffffff" stroke-width="18" stroke-opacity=".16" fill="none"/><path d="M-30 382 C90 318 154 356 222 314 C290 272 331 288 396 250" stroke="#d6fbff" stroke-width="24" stroke-opacity=".16" fill="none"/></g>`;
}

function prismFacets(id, mask = 'borderMask', seed = 13, density = 56) {
  const r = rng(seed);
  const out = [];
  for (let i=0; i<density; i++) {
    const x = -20 + r()*400;
    const y = -20 + r()*545;
    const w = 32 + r()*94;
    const h = 18 + r()*70;
    const s = -38 + r()*76;
    const pts = `${x.toFixed(1)},${y.toFixed(1)} ${(x+w).toFixed(1)},${(y+s*.35).toFixed(1)} ${(x+w+s*.22).toFixed(1)},${(y+h).toFixed(1)} ${(x+s*.15).toFixed(1)},${(y+h+s*.15).toFixed(1)}`;
    out.push(`<polygon points="${pts}" fill="url(#${id}-borderFoil)" opacity="${(.10 + r()*.24).toFixed(2)}" stroke="#fff" stroke-opacity="${(.06 + r()*.24).toFixed(2)}"/>`);
  }
  return `<g clip-path="url(#${id}-cardClip)"><g clip-path="url(#${id}-${mask === 'edgeMask' ? 'edgeClip' : 'borderClip'})">${out.join('')}</g></g>`;
}

function wavePaths(id, mask='borderMask') {
  const lines = [];
  for (let i=0; i<28; i++) {
    const y = 18 + i*17;
    lines.push(`<path d="M-42 ${y} C42 ${y-36} 92 ${y+38} 176 ${y} C248 ${y-32} 306 ${y+34} 402 ${y-8}" fill="none" stroke="url(#${id}-borderFoil)" stroke-width="${1.6 + (i%4)*.55}" stroke-opacity="${.22 + (i%5)*.055}"/>`);
  }
  return `<g clip-path="url(#${id}-cardClip)"><g clip-path="url(#${id}-${mask === 'edgeMask' ? 'edgeClip' : 'borderClip'})">${lines.join('')}</g></g>`;
}

function mojoDots(id, mask='borderMask') {
  return `<g clip-path="url(#${id}-cardClip)"><g clip-path="url(#${id}-${mask === 'edgeMask' ? 'edgeClip' : 'borderClip'})">
    <rect width="360" height="504" fill="#020810" opacity=".72"/>
    <rect width="360" height="504" fill="url(#${id}-mojoTile)" opacity=".94"/>
    <rect width="360" height="504" fill="url(#${id}-borderFoil)" opacity=".18"/>
    <path d="M10 72 C72 58 120 76 176 50 C236 22 282 46 350 26" stroke="#fff" stroke-width="5" stroke-opacity=".16" fill="none"/>
    <path d="M0 424 C78 398 132 422 190 392 C248 364 300 382 360 348" stroke="#fff" stroke-width="6" stroke-opacity=".12" fill="none"/>
  </g></g>`;
}

function checkerTiles(id, mask='borderMask') {
  return `<g clip-path="url(#${id}-cardClip)"><g clip-path="url(#${id}-${mask === 'edgeMask' ? 'edgeClip' : 'borderClip'})"><rect width="360" height="504" fill="url(#${id}-checker)" opacity=".85"/><rect width="360" height="504" fill="url(#${id}-borderFoil)" opacity=".17"/></g></g>`;
}

function tigerEdge(id) {
  const stripes = [];
  const shadow = [];
  const top = [20,48,80,112,148,182,214,248,282,320];
  top.forEach((x, i) => {
    stripes.push(`<path d="M${x} 0 C ${x-18} 24 ${x+22} 42 ${x+6} 76" stroke="#1a0b04" stroke-width="${12 + (i%3)*3}" stroke-linecap="round" stroke-opacity=".92" fill="none"/>`);
    shadow.push(`<path d="M${x+4} 0 C ${x-10} 24 ${x+28} 42 ${x+12} 76" stroke="#ffad21" stroke-width="${15 + (i%3)*3}" stroke-linecap="round" stroke-opacity=".42" fill="none"/>`);
  });
  const sideY = [96,136,182,228,276,324,372,420];
  sideY.forEach((y, i) => {
    stripes.push(`<path d="M0 ${y} C 22 ${y-12} 30 ${y+14} 62 ${y+6}" stroke="#170904" stroke-width="${12 + (i%2)*3}" stroke-linecap="round" stroke-opacity=".92" fill="none"/>`);
    stripes.push(`<path d="M360 ${y} C 338 ${y-10} 330 ${y+15} 298 ${y+5}" stroke="#170904" stroke-width="${12 + (i%2)*3}" stroke-linecap="round" stroke-opacity=".92" fill="none"/>`);
    shadow.push(`<path d="M0 ${y-2} C 24 ${y-16} 32 ${y+10} 64 ${y+2}" stroke="#ffb428" stroke-width="${15 + (i%2)*3}" stroke-linecap="round" stroke-opacity=".38" fill="none"/>`);
    shadow.push(`<path d="M360 ${y-2} C 336 ${y-16} 328 ${y+10} 296 ${y+2}" stroke="#ffb428" stroke-width="${15 + (i%2)*3}" stroke-linecap="round" stroke-opacity=".38" fill="none"/>`);
  });
  const bottom = [26,60,98,138,176,212,246,284,322];
  bottom.forEach((x, i) => {
    stripes.push(`<path d="M${x} 504 C ${x+10} 478 ${x-20} 458 ${x+8} 430" stroke="#150903" stroke-width="${12 + (i%2)*3}" stroke-linecap="round" stroke-opacity=".94" fill="none"/>`);
    shadow.push(`<path d="M${x+3} 504 C ${x+14} 480 ${x-16} 460 ${x+12} 432" stroke="#ffba33" stroke-width="${15 + (i%2)*3}" stroke-linecap="round" stroke-opacity=".32" fill="none"/>`);
  });
  return `<g clip-path="url(#${id}-cardClip)"><g clip-path="url(#${id}-borderClip)"><rect width="360" height="504" fill="#ff9b15" opacity=".92"/><rect width="360" height="504" fill="url(#${id}-microLines)" opacity=".18"/>${shadow.join('')}${stripes.join('')}<rect width="360" height="504" fill="url(#${id}-borderFoil)" opacity=".14"/></g></g>`;
}

function laserEdge(id) {
  const r = rng(77); const lines=[];
  for (let i=0;i<42;i++) {
    const side = i%2;
    const x1 = side ? 265+r()*70 : 25+r()*70;
    const y1 = 30+r()*430;
    const x2 = x1 + (r()-.5)*120;
    const y2 = y1 + (r()-.5)*100;
    lines.push(`<path d="M${x1.toFixed(1)} ${y1.toFixed(1)} L${x2.toFixed(1)} ${y2.toFixed(1)}" stroke="url(#${id}-borderFoil)" stroke-opacity="${(.35+r()*.45).toFixed(2)}" stroke-width="${(.65+r()*1.2).toFixed(2)}"/>`);
    if (i%5===0) lines.push(`<circle cx="${x2.toFixed(1)}" cy="${y2.toFixed(1)}" r="${(1.7+r()*2.8).toFixed(1)}" fill="url(#${id}-borderFoil)" opacity=".55"/>`);
  }
  return `<g clip-path="url(#${id}-cardClip)"><g clip-path="url(#${id}-borderClip)"><rect width="360" height="504" fill="#010407"/><rect width="360" height="504" fill="url(#${id}-microLines)" opacity=".44"/>${lines.join('')}</g></g>`;
}

function borderShell(id, key, variant) {
  const p = variant.palette;
  const gold = key === 'black-gold-edge' || key === 'base-paper' || key === 'gold-wave-edge';
  const stroke = gold ? `url(#${id}-goldSweep)` : `url(#${id}-metalSweep)`;
  let effect = '';
  if (key === 'prism-edge' || key === 'blue-prism-edge') effect = prismFacets(id, 'borderMask', key === 'blue-prism-edge' ? 41 : 9, 46);
  if (key === 'wave-border-r' || key === 'gold-wave-edge') effect = wavePaths(id, 'borderMask');
  if (key === 'mojo-border') effect = mojoDots(id, 'borderMask');
  if (key === 'checker-border') effect = checkerTiles(id, 'borderMask');
  if (key === 'chrome-full') effect = `${prismFacets(id, 'edgeMask', 24, 45)}<rect width="360" height="504" rx="26" fill="url(#${id}-borderFoil)" opacity=".08"/>`;
  if (key === 'cracked-ice-edge') effect = iceCells(id, true);
  if (key === 'black-gold-edge') effect = `<g clip-path="url(#${id}-cardClip)">
      <rect width="360" height="504" fill="url(#${id}-finiteTile)" opacity=".60"/>
      <g clip-path="url(#${id}-borderClip)">
        <rect width="360" height="504" fill="#030303" opacity=".84"/>
        <rect width="360" height="504" fill="url(#${id}-goldSweep)" opacity=".19"/>
        <rect width="360" height="504" fill="url(#${id}-microLines)" opacity=".24"/>
      </g>
      <path d="M28 110 L88 48 H272 L332 110 V396 L276 456 H84 L28 396 Z" fill="none" stroke="url(#${id}-goldSweep)" stroke-opacity=".70" stroke-width="2.4"/>
      <path d="M38 98 L108 62 H252 L322 98 M38 406 L108 442 H252 L322 406" stroke="url(#${id}-goldSweep)" stroke-opacity=".72" stroke-width="1.9" fill="none"/>
      <path d="M56 126 H304 M56 388 H304" stroke="#ffffff" stroke-opacity=".13" stroke-width="1"/>
      <text x="318" y="307" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="22" font-weight="900" fill="url(#${id}-goldSweep)" opacity=".92">1/1</text>
    </g>`;
  const rails = `
    ${effect}
    <g clip-path="url(#${id}-cardClip)">
      <rect x="7" y="7" width="346" height="490" rx="24" fill="none" stroke="${stroke}" stroke-width="2.7" stroke-opacity=".85"/>
      <rect x="16" y="16" width="328" height="472" rx="19" fill="none" stroke="#fff" stroke-width=".9" stroke-opacity=".40"/>
      <rect x="24" y="24" width="312" height="456" rx="16" fill="none" stroke="${stroke}" stroke-width="1.2" stroke-opacity=".56"/>
      <path d="M21 89 L66 43 H132 L146 58 H214 L228 43 H294 L339 89" fill="none" stroke="${stroke}" stroke-width="2" stroke-opacity=".72"/>
      <path d="M21 415 L66 462 H132 L146 447 H214 L228 462 H294 L339 415" fill="none" stroke="${stroke}" stroke-width="2" stroke-opacity=".72"/>
      <path d="M43 100 L43 300 L72 332 M317 100 L317 300 L288 332" fill="none" stroke="${stroke}" stroke-width="2.4" stroke-opacity=".46"/>
      <path d="M32 121 L64 92 V244 L44 286 L32 278 Z M328 121 L296 92 V244 L316 286 L328 278 Z" fill="${key === 'base-paper' ? '#f9f0d7' : '#05080e'}" opacity=".20" stroke="${stroke}" stroke-opacity=".34"/>
    </g>`;
  return rails;
}

function stage(id, key) {
  const dark = key === 'black-gold-edge';
  return `<g clip-path="url(#${id}-cardClip)"><ellipse cx="180" cy="178" rx="118" ry="132" fill="${dark ? '#ffffff' : '#ffffff'}" opacity="${dark ? .04 : .20}"/><ellipse cx="180" cy="308" rx="132" ry="42" fill="${dark ? '#ffffff' : '#d9e1e7'}" opacity="${dark ? .035 : .20}"/><path d="M38 326 C120 304 240 304 322 326" fill="none" stroke="${dark ? '#fff' : '#07111b'}" stroke-opacity="${dark ? .05 : .07}" stroke-width="12"/><path d="M56 76 C110 52 250 52 304 76" fill="none" stroke="${dark ? '#fff' : '#07111b'}" stroke-opacity="${dark ? .03 : .055}" stroke-width="8"/></g>`;
}

function demoVectorHeadshot() {
  return `<g transform="translate(23 44) scale(.35)" filter="url(#demoShadow)">
    <defs>
      <radialGradient id="demoSkin" cx="50%" cy="32%" r="58%"><stop offset="0" stop-color="#d99a68"/><stop offset=".52" stop-color="#b66e3f"/><stop offset="1" stop-color="#74401f"/></radialGradient>
      <linearGradient id="demoJersey" x1="290" y1="760" x2="625" y2="1060" gradientUnits="userSpaceOnUse"><stop stop-color="#10243c"/><stop offset="1" stop-color="#050910"/></linearGradient>
      <filter id="demoShadow" x="-30%" y="-30%" width="160%" height="160%"><feDropShadow dx="0" dy="22" stdDeviation="24" flood-color="#000" flood-opacity="0.34"/></filter>
    </defs>
    <path d="M188 1115c17-180 112-300 262-300s245 120 262 300H188Z" fill="url(#demoJersey)"/>
    <path d="M362 790c3 58 30 102 88 102s85-44 88-102H362Z" fill="#9d5b31"/>
    <path d="M292 904c55 75 111 110 158 110s103-35 158-110l34 211H258l34-211Z" fill="#e9edf4"/>
    <path d="M325 905c45 60 87 88 125 88s80-28 125-88l24 210H301l24-210Z" fill="url(#demoJersey)"/>
    <ellipse cx="246" cy="506" rx="58" ry="90" fill="#a46137"/>
    <ellipse cx="654" cy="506" rx="58" ry="90" fill="#a46137"/>
    <path d="M287 404c0-142 66-240 163-240s163 98 163 240v145c0 131-74 247-163 247S287 680 287 549V404Z" fill="url(#demoSkin)"/>
    <path d="M290 394c26-78 95-126 161-126 70 0 136 50 161 126-16-112-75-192-162-192-86 0-145 80-160 192Z" fill="#0b0d0f"/>
    <path d="M269 385c7-120 71-226 181-226s174 106 181 226c-54-41-87-78-181-78s-127 37-181 78Z" fill="#111316"/>
    <circle cx="311" cy="244" r="58" fill="#0d0f12"/><circle cx="375" cy="198" r="70" fill="#111316"/><circle cx="454" cy="185" r="82" fill="#0e1013"/><circle cx="535" cy="201" r="70" fill="#111316"/><circle cx="592" cy="254" r="55" fill="#0e1013"/>
    <ellipse cx="390" cy="514" rx="22" ry="15" fill="#171717" opacity=".84"/>
    <ellipse cx="515" cy="514" rx="22" ry="15" fill="#171717" opacity=".84"/>
    <path d="M455 520c-11 44-8 68 18 76" stroke="#7e4322" stroke-width="14" stroke-linecap="round" opacity=".7"/>
    <path d="M375 646c46 35 106 35 151 0" stroke="#31180d" stroke-width="18" stroke-linecap="round"/>
    <path d="M397 660c34 15 71 15 105 0" stroke="#f2d4c0" stroke-width="8" stroke-linecap="round" opacity=".95"/>
    <path d="M322 448c43-39 97-42 139-19" stroke="#0b0d0f" stroke-width="18" stroke-linecap="round"/>
    <path d="M451 428c43-22 96-19 138 20" stroke="#0b0d0f" stroke-width="18" stroke-linecap="round"/>
    <path d="M294 404c50-35 86-78 128-126 54 53 110 70 190 117-12-122-80-213-166-213-94 0-152 86-152 222Z" fill="#0c0e11" opacity=".96"/>
  </g>`;
}

function portrait(id, headshot) {
  const imageLayer = headshot === '__demo_vector__'
    ? demoVectorHeadshot()
    : `<image href="${headshot}" x="29" y="58" width="302" height="372" preserveAspectRatio="xMidYMid meet"/>`;
  return `<g clip-path="url(#${id}-cardClip)">
    <ellipse cx="180" cy="335" rx="102" ry="24" fill="#000" opacity=".23" filter="url(#${id}-shadow)"/>
    ${imageLayer}
    <path d="M0 300 C75 282 128 287 180 294 C231 301 282 286 360 304 V352 H0 Z" fill="#fff" opacity=".04"/>
  </g>`;
}

function badges(id, variant, jerseyNumber) {
  const p = variant.palette;
  const rookie = variant.rookie;
  const rBadge = rookie ? `<rect x="312" y="36" width="24" height="24" rx="6" fill="#080b10" stroke="url(#${id}-borderFoil)" stroke-width="1.4"/><text x="324" y="54" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="17" font-weight="900" fill="${p[4]}">R</text>` : `<polygon points="${starPath(323,47,16,6.5)}" fill="url(#${id}-goldSweep)" stroke="#fff" stroke-opacity=".4"/>`;
  return `<g><circle cx="43" cy="45" r="23" fill="#050912" stroke="url(#${id}-borderFoil)" stroke-width="2"/><circle cx="43" cy="45" r="28" fill="none" stroke="#fff" stroke-opacity=".20"/><text x="43" y="51" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="17" font-weight="900" fill="#fff">#${esc(jerseyNumber)}</text>${rBadge}</g>`;
}

function identity(id, key, playerName, teamName) {
  const dark = key === 'laser-edge';
  const gold = key === 'black-gold-edge';
  const nameFill = gold ? `url(#${id}-goldSweep)` : (dark ? '#f5f7fb' : '#111923');
  const plateFill = gold ? '#050505' : (dark ? '#050912' : `url(#${id}-namePlate)`);
  const teamFill = gold ? '#f5cd71' : (dark ? '#f4f8ff' : '#102038');
  return `<g>
    <path d="M41 318 H319 L307 341 H53 Z" fill="${dark ? '#07101a' : '#142338'}" opacity=".96" stroke="url(#${id}-borderFoil)" stroke-opacity=".50"/>
    <text x="180" y="334" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="10.2" font-weight="900" letter-spacing="2" fill="${teamFill}">${esc(teamName)}</text>
    <path d="M37 344 H323 L311 399 H49 Z" fill="${plateFill}" opacity=".96" stroke="${gold ? `url(#${id}-goldSweep)` : `url(#${id}-borderFoil)`}" stroke-opacity="${gold ? '.74' : '.58'}"/>
    <path d="M55 344 L82 399 M305 344 L278 399" stroke="url(#${id}-borderFoil)" stroke-opacity=".25"/>
    <text x="180" y="384" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="31" font-weight="900" letter-spacing="0" fill="${nameFill}" stroke="${dark ? '#000' : '#fff'}" stroke-opacity=".18" stroke-width="1">${esc(playerName)}</text>
  </g>`;
}

function stats(id, key, stats) {
  const dark = key === 'laser-edge';
  const gold = key === 'black-gold-edge';
  return `<g>${stats.slice(0,4).map((s,i) => {
    const x = 36 + i*72;
    return `<g transform="translate(${x} 418)"><path d="M0 0 H64 L58 54 H6 Z" fill="${dark ? '#020304' : '#08111c'}" opacity=".95" stroke="url(#${id}-borderFoil)" stroke-opacity=".58"/><path d="M7 8 H57 M7 47 H57" stroke="#fff" stroke-opacity=".10"/><text x="32" y="24" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="16" font-weight="900" fill="${gold ? '#f1cd74' : '#f8fbff'}">${esc(s.value)}</text><text x="32" y="42" text-anchor="middle" font-family="Manrope,Arial,sans-serif" font-size="9.5" font-weight="900" letter-spacing=".5" fill="${gold ? '#dcb760' : '#dce5ef'}">${esc(s.label)}</text></g>`;
  }).join('')}</g>`;
}

function renderCardSvg(input = {}, index = 0) {
  const p = { ...DEFAULT_PLAYER, ...input };
  const key = p.variant || 'prism-edge';
  const variant = VARIANTS[key] || VARIANTS['prism-edge'];
  const id = `card_${index}_${key.replace(/[^a-z0-9]/g, '')}`;
  const headshot = p.headshotData || p.headshot || DEFAULT_PLAYER.headshot;
  const statsList = p.stats && p.stats.length ? p.stats : DEFAULT_STATS;
  return `<svg class="modern-border-card" width="${CARD_W}" height="${CARD_H}" viewBox="0 0 ${CARD_W} ${CARD_H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(p.playerName)} ${esc(variant.label)} card">
    <title>${esc(p.playerName)} ${esc(variant.label)}</title>
    ${defs(id, variant)}
    <g filter="url(#${id}-shadow)">
      ${baseSurface(id, key, variant)}
      ${stage(id, key)}
      ${borderShell(id, key, variant)}
      ${portrait(id, headshot)}
      ${key === 'chrome-full' ? `<g clip-path="url(#${id}-cardClip)" opacity=".18">${prismFacets(id, 'edgeMask', 71, 24)}</g>` : ''}
      ${badges(id, variant, p.jerseyNumber)}
      ${identity(id, key, p.playerName, p.teamName)}
      ${stats(id, key, statsList)}
      <rect x="0" y="0" width="360" height="504" rx="26" fill="none" stroke="#fff" stroke-opacity=".24"/>
      <rect width="360" height="504" rx="26" filter="url(#${id}-cardGrain)" opacity=".55"/>
    </g>
  </svg>`;
}

function renderBoardSvg(input = {}) {
  const variants = input.variants || Object.keys(VARIANTS);
  const player = { ...DEFAULT_PLAYER, ...input };
  const cols = 5;
  const cardScale = 1;
  const gapX = 410;
  const gapY = 650;
  const startX = 54;
  const startY = 230;
  const rows = Math.ceil(variants.length / cols);
  const W = 2160;
  const H = 270 + rows * gapY + 150;
  const cards = variants.map((key, i) => {
    const col = i % cols, row = Math.floor(i / cols);
    const x = startX + col*gapX, y = startY + row*gapY;
    const v = VARIANTS[key];
    const inner = renderCardSvg({ ...player, variant: key, rookie: v.rookie }, i).replace(/<svg[^>]*>|<\/svg>/g, '');
    return `<g transform="translate(${x} ${y}) scale(${cardScale})">${inner}<g transform="translate(0 532)"><circle cx="16" cy="14" r="12" fill="url(#card_${i}_${key.replace(/[^a-z0-9]/g, '')}-borderFoil)"/><text x="42" y="17" font-family="Manrope,Arial,sans-serif" font-size="15" font-weight="900" letter-spacing="1.4" fill="#fff">${esc(v.label)}</text><text x="42" y="42" font-family="Manrope,Arial,sans-serif" font-size="13" fill="#b9c0ca">${esc(v.note)}</text></g></g>`;
  }).join('');
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    <defs><linearGradient id="boardBg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#050507"/><stop offset=".6" stop-color="#0b1018"/><stop offset="1" stop-color="#030405"/></linearGradient><pattern id="boardGrid" width="88" height="88" patternUnits="userSpaceOnUse"><path d="M88 0H0V88" fill="none" stroke="#fff" stroke-opacity=".04"/></pattern></defs>
    <rect width="${W}" height="${H}" fill="url(#boardBg)"/><rect width="${W}" height="${H}" fill="url(#boardGrid)"/>
    <text x="54" y="66" font-family="Manrope,Arial,sans-serif" font-size="14" font-weight="900" letter-spacing="2" fill="#d6aa48">MODERN SPORTS-CARD FRONTEND LAB</text>
    <text x="54" y="136" font-family="Manrope,Arial,sans-serif" font-size="62" font-weight="900" fill="#fbf8ef">Border-Led Refractor System</text>
    <text x="54" y="178" font-family="Manrope,Arial,sans-serif" font-size="17" fill="#c5cbd4">Modern direction: quiet player field, no headshot box, border-led parallels. Chrome, ice, and black finite use full-card material; checker/wave/mojo keep the effect mostly on the border shell.</text>
    ${cards}
    <g transform="translate(54 ${H-95})"><rect width="2052" height="66" rx="18" fill="#05070b" stroke="#d6aa48" stroke-opacity=".32"/><text x="24" y="28" font-family="Manrope,Arial,sans-serif" font-size="14" font-weight="900" letter-spacing="1.5" fill="#d6aa48">FRONTEND STACK</text><text x="24" y="52" font-family="Manrope,Arial,sans-serif" font-size="14" fill="#d9dde5">SVG shell + masks for border-only treatments, full-card material modules for chrome/ice/black, headshot cutout layer, typography plates, stat rails, CSS/SVG gradients, patterns, turbulence, and specular lighting.</text></g>
  </svg>`;
}

function mountCard(target, input = {}) {
  const el = typeof target === 'string' ? document.querySelector(target) : target;
  if (!el) throw new Error('target not found');
  el.innerHTML = renderCardSvg(input);
}
function mountBoard(target, input = {}) {
  const el = typeof target === 'string' ? document.querySelector(target) : target;
  if (!el) throw new Error('target not found');
  el.innerHTML = renderBoardSvg(input);
}

const api = { CARD_W, CARD_H, VARIANTS, DEFAULT_PLAYER, DEFAULT_STATS, renderCardSvg, renderBoardSvg, mountCard, mountBoard };
if (typeof module !== 'undefined') module.exports = api;
if (typeof window !== 'undefined') window.ModernBorderCardBuilder = api;
