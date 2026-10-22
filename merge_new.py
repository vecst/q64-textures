"""Merge newly decoded textures (decoded_new) into decoded/ and list them. usage: merge_new.py"""
import json, shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, ".")
import upscale as up
R = Path(".")
old = json.loads((R / "decoded/meta.json").read_text())
new = json.loads((R / "decoded_new/meta.json").read_text())
fresh = sorted(h for h in new if h not in old)
shutil.copy2(R / "decoded/meta.json", R / "backup/meta.json.before-newdumps") if (R / "backup").exists() else None
for h in fresh:
    shutil.copy2(R / "decoded_new" / f"{h}.png", R / "decoded" / f"{h}.png")
    old[h] = new[h]
(R / "decoded/meta.json").write_text(json.dumps(old))
(R / "new_textures.json").write_text(json.dumps(fresh, indent=0))
units = up.group_strips(fresh, old)
units.sort(key=lambda u: -len(u))
ims = []
for n, u in enumerate(units):
    w = max(x + old[h]["w"] for h, x, _ in u); hh = max(y + old[h]["h"] for h, _, y in u)
    c = Image.new("RGBA", (w, hh), (0, 0, 0, 0))
    for h, x, y in u:
        c.alpha_composite(Image.open(R / "decoded" / f"{h}.png").convert("RGBA"), (x, y))
    k = max(1, min(3, 600 // max(w, 1)))
    c = c.resize((w * k, hh * k), Image.NEAREST)
    if c.width > 700: c.thumbnail((700, 700))
    ims.append((n, len(u), c))
cols_w = 720
sheets, cur, y = [], [], 0
for n, cnt, c in ims:
    if y + c.height + 14 > 2000 and cur:
        sheets.append(cur); cur, y = [], 0
    cur.append((n, cnt, c, y)); y += c.height + 16
sheets.append(cur)
for i, sh in enumerate(sheets):
    H = max(yy + c.height + 16 for _, _, c, yy in sh)
    s = Image.new("RGBA", (cols_w, H), (70, 70, 90, 255)); d = ImageDraw.Draw(s)
    for n, cnt, c, yy in sh:
        d.text((2, yy), f"unit {n} ({cnt} pieces)", fill=(255, 255, 0)); s.alpha_composite(c, (10, yy + 12))
    s.save(f"/tmp/q64sheets/new_{i}.png")
print(len(fresh), "merged;", len(units), "units;", len(sheets), "sheets")
