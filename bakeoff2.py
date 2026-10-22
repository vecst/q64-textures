#!/usr/bin/env python3
"""Bake-off across texture categories: HDCube 3 vs UltraSharp V2 vs PBRify V4 (text excluded; it uses NXbrz)."""
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

R = Path.home() / "games/q64-textures"
PY = str(Path.home() / "comfy/ComfyUI/.venv/bin/python")
MODELS = {"HDCube3": "4xHDcube3_500k.safetensors", "UltraSharpV2": "4x-UltraSharpV2.safetensors",
          "PBRifyV4": "4x-PBRify_UpscalerV4.pth"}
meta = json.loads((R / "decoded/meta.json").read_text())
text = set(json.loads((R / "text_textures.json").read_text()))
ui = set(json.loads((R / "ui_textures.json").read_text()))


def kind(h):
    a = np.asarray(Image.open(R / "decoded" / f"{h}.png"))[..., 3]
    return "sky" if a.max() == 0 else "opaque" if a.min() == 255 else "cut"


random.seed(21)
pool = [h for h, v in meta.items() if h not in text and min(v["w"], v["h"]) >= 16]
cats = {"world": [], "ui": [], "sprite": [], "effect": []}
for h in random.sample(pool, len(pool)):
    k, v = kind(h), meta[h]
    if k == "sky":
        continue
    c = ("ui" if h in ui else "effect" if v["fmt"] in (3, 4) else "world" if k == "opaque" else "sprite")
    if len(cats[c]) < {"world": 8, "ui": 4, "sprite": 5, "effect": 3}[c]:
        cats[c].append(h)
picks = [h for c in cats.values() for h in c]
only = ",".join(picks)
for name, model in MODELS.items():
    subprocess.run([PY, "upscale.py", "decoded", f"Bake{name}", "--model", model, "--mods-dir", "/tmp/bake2",
                    "--only", only], cwd=R, check=True, stdout=subprocess.DEVNULL)
    print(name, "done", flush=True)

cell, gap = 210, 6
cols = ["Original"] + list(MODELS)
order = [(c, h) for c, hs in cats.items() for h in hs]
sheet = Image.new("RGB", (len(cols) * (cell + gap) + gap + 60, len(order) * (cell + gap) + 26), (30, 30, 38))
d = ImageDraw.Draw(sheet)
for i, name in enumerate(cols):
    d.text((60 + gap + i * (cell + gap), 6), name, fill=(255, 220, 120))
for r, (c, h) in enumerate(order):
    y = 22 + r * (cell + gap)
    d.text((4, y + 4), c, fill=(190, 190, 200))
    srcs = [R / "decoded" / f"{h}.png"] + [Path(f"/tmp/bake2/Quest64HD-Bake{n}/{h}.png") for n in MODELS]
    for i, src in enumerate(srcs):
        im = Image.open(src).convert("RGBA")
        k = min(cell / im.width, cell / im.height)
        im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.NEAREST if i == 0 else Image.LANCZOS)
        bg = Image.new("RGBA", im.size, (30, 30, 38, 255))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (60 + gap + i * (cell + gap), y))
sheet.save(R / "bakeoff" / "compare_categories.png")
print("sheet done")
