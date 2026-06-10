# NBA Modern Border Card Lab

A reusable HTML/CSS/JS/SVG frontend builder for modern, headshot-only sports card fronts.

This version follows a border-led modern card direction:

- Most parallels modify the border/rails only, keeping the player field clean.
- Chrome remains a full-card chromium/refractor treatment.
- Cracked ice uses code-generated crystal cells but masks the ice away from the headshot protection zone.
- Black finite is a black 1/1-inspired chase-card material with dark finite texture, gold/chrome trim, and a small `1/1` stamp. It is original and does not copy official logos or protected layouts.
- Checker edge uses a controlled teal/violet/amber spectrum instead of a loud full-rainbow palette.
- Mojo is now a distinct interlocking lens/circuit geometry instead of generic dots.
- Tiger was removed.

The card is built entirely from frontend code: SVG geometry, masks, clip paths, gradients, procedural pattern fills, turbulence/specular filters, and reusable JS rendering functions. No generated-image card art is required.

## Files

- `index.html` - interactive frontend card lab
- `src/modern-border-card-builder.js` - reusable renderer
- `styles/card-lab.css` - page styling
- `scripts/export-samples.js` - exports SVG samples
- `scripts/rasterize.py` - converts SVG outputs to PNG
- `output/` - rendered samples

## Variants

- `base-paper`
- `prism-edge`
- `blue-prism-edge`
- `wave-border-r`
- `gold-wave-edge`
- `mojo-border`
- `checker-border`
- `chrome-full`
- `cracked-ice-edge`
- `black-gold-edge` (`BLACK FINITE 1/1`)

## Headshots

The demo uses `__demo_vector__`. Replace `headshot` or `headshotData` with a transparent PNG/SVG path or data URI for real players. The portrait is not boxed; it is layered over the card surface as a clean cutout.
