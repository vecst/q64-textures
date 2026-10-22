"""Contact sheets of decoded textures with index labels. usage: sheets.py <list.json|-> <out_prefix> [cell] [cols]
'-' = ui_textures + art_textures minus text."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
R = Path.home() / "games/q64-textures"
meta = json.loads((R / "decoded/meta.json").read_text())
if sys.argv[1] == "-":
    text = set(json.loads((R / "text_textures.json").read_text()))
    hs = sorted((set(json.loads((R / "ui_textures.json").read_text())) | set(json.loads((R / "art_textures.json").read_text()))) - text)
else:
    hs = json.loads(Path(sys.argv[1]).read_text())
cell = int(sys.argv[3]) if len(sys.argv) > 3 else 96
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 12
per = cols * 8
Path(sys.argv[2] + "_index.json").write_text(json.dumps(hs))
for s in range(0, len(hs), per):
    chunk = hs[s:s + per]
    rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 12)), (255, 0, 255))
    d = ImageDraw.Draw(sheet)
    for i, h in enumerate(chunk):
        im = Image.open(R / "decoded" / f"{h}.png").convert("RGBA")
        sc = min(cell / im.width, cell / im.height)
        sc = max(1, int(sc)) if sc >= 1 else sc
        im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.NEAREST)
        bg = Image.new("RGBA", im.size, (60, 60, 70, 255)); bg.alpha_composite(im)
        x, y = (i % cols) * cell, (i // cols) * (cell + 12)
        sheet.paste(bg.convert("RGB"), (x, y + 12))
        m = meta[h]
        d.text((x + 1, y), f"{s+i} {m['w']}x{m['h']}", fill=(255, 255, 255))
    sheet.save(f"{sys.argv[2]}_{s // per:02d}.png")
print(len(hs), "textures")
