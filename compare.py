"""Side-by-side: original | current NXbrz | xBRZ | vector trace, at 8x. usage: compare.py out.png idx..."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import textui
R = Path.home() / "games/q64-textures"
NX = Path.home() / "games/q64-mods/archive/png-packs/Quest64HD-HDCube3"
text = json.loads((R / "text_textures.json").read_text())
ui = json.loads(Path("/tmp/q64sheets/ui_index.json").read_text())
CROP = 56
ALPHAMAX = float(__import__("os").environ.get("ALPHAMAX", "1.0"))
BG = (38, 40, 92, 255)
picks = []
for a in sys.argv[2:]:
    kind, i = a[0], int(a[1:])
    h = (text if kind == "t" else ui)[i]
    src = R / ("decoded_text" if kind == "t" and (R / "decoded_text" / f"{h}.png").exists() else "decoded") / f"{h}.png"
    picks.append((h, src))
rows = []
for h, src in picks:
    rgba = np.asarray(Image.open(src).convert("RGBA"))
    rgba = rgba[:, :CROP]
    w8, h8 = rgba.shape[1] * 8, rgba.shape[0] * 8
    cells = [textui.upscale(rgba, "nearest")]
    nx = NX / f"{h}.png"
    if nx.exists():
        im = Image.open(nx).convert("RGBA")
        im = im.crop((0, 0, min(im.width, CROP * im.width // max(1, Image.open(src).width)), im.height))
        cells.append(np.asarray(im.resize((w8, h8), Image.BICUBIC)))
    else:
        cells.append(np.zeros((h8, w8, 4), np.uint8))
    cells.append(textui.upscale(rgba, "xbrz"))
    cells.append(textui.upscale(rgba, "hybrid"))
    rows.append(cells)
W = max(r[0].shape[1] for r in rows)
H = sum(r[0].shape[0] + 8 for r in rows) + 20
sheet = Image.new("RGBA", (4 * (W + 8), H), BG)
d = ImageDraw.Draw(sheet)
for c, name in enumerate(["original", "NXbrz (current)", "xBRZ", "hybrid"]):
    d.text((c * (W + 8) + 4, 4), name, fill=(255, 255, 255, 255))
y = 20
for r in rows:
    for c, cell in enumerate(r):
        sheet.alpha_composite(Image.fromarray(np.ascontiguousarray(cell), "RGBA"), (c * (W + 8), y))
    y += r[0].shape[0] + 8
sheet.convert("RGB").save(sys.argv[1])
print("saved", sys.argv[1], sheet.size)
