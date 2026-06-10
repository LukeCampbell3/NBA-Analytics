from pathlib import Path
import cairosvg
root = Path(__file__).resolve().parents[1]
out = root / 'output'
for svg in out.glob('*.svg'):
    png = svg.with_suffix('.png')
    if svg.name == 'modern-border-refractor-board.svg':
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=2160)
    else:
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=900)
print('PNG exports written')
