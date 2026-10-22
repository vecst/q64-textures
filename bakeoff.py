#!/usr/bin/env python3
"""Model bake-off on the reassembled title-screen background images.

usage: bakeoff.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upscale as up  # noqa: E402

R = Path.home() / "games/q64-textures"
HD = Path.home() / ".config/Quest64Recompiled/mods/Quest64HD-HDCube3"
OUT = R / "bakeoff"
MODELS = ["4xHDcube3_500k.safetensors", "4xHDCube4Compact.safetensors", "4x-UltraSharpV2.safetensors",
          "4x-PBRify_UpscalerV4.pth", "4x-NXbrz.safetensors", "RealESRGAN_x4plus.safetensors"]


def title_groups(meta):
    groups = {}
    for h, v in meta.items():
        load = v.get("load")
        if load and load["width"] >= 300:
            groups.setdefault((load["addr"], load["width"], load["pal"]), []).append(h)
    return [sorted(g, key=lambda h: meta[h]["load"]["y"]) for g in groups.values() if len(g) >= 20]


def canvas_for(members, meta):
    unit = [(h, meta[h]["load"]["x"] - min(meta[m]["load"]["x"] for m in members),
             meta[h]["load"]["y"] - min(meta[m]["load"]["y"] for m in members)) for h in members]
    rgba, info = up.assemble(unit, R / "decoded", meta)
    return unit, rgba, info


def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((R / "decoded/meta.json").read_text())
    groups = title_groups(meta)
    print(f"{len(groups)} wide strip images:", [(meta[g[0]]["load"]["width"], len(g)) for g in groups])
    results = {}
    for gi, members in enumerate(groups):
        unit, rgba, info = canvas_for(members, meta)
        Image.fromarray(rgba).save(OUT / f"g{gi}_original.png")
        for model in MODELS:
            comfy = up.Comfy("http://127.0.0.1:8188", str(Path(__file__).resolve().parent / "workflow_api.json"), model)
            img, alpha, pad = up.prepare(rgba, info)
            full = up.finish(comfy.result(comfy.submit(f"bake_g{gi}.png", img)), alpha, pad, rgba[..., 3])
            full.save(OUT / f"g{gi}_{Path(model).stem}.png")
            print(f"  group {gi} {model} done", flush=True)
        results[gi] = rgba.shape

    # Comparison: a 160x110 crop from the middle of each image, shown 4x.
    for gi, (h, w, _) in results.items():
        cx, cy, cw, ch = w // 2 - 80, h // 2 - 55, 160, 110
        tiles = [("Original", Image.open(OUT / f"g{gi}_original.png").crop((cx, cy, cx + cw, cy + ch))
                  .resize((cw * 4, ch * 4), Image.NEAREST))]
        for model in MODELS:
            im = Image.open(OUT / f"g{gi}_{Path(model).stem}.png")
            s = im.width / w
            tiles.append((Path(model).stem, im.crop((int(cx * s), int(cy * s), int((cx + cw) * s), int((cy + ch) * s)))
                          .resize((cw * 4, ch * 4), Image.LANCZOS)))
        cols = 2
        sheet = Image.new("RGB", (cols * (cw * 4 + 10) + 10, ((len(tiles) + 1) // 2) * (ch * 4 + 24) + 10), (20, 20, 26))
        d = ImageDraw.Draw(sheet)
        for i, (name, im) in enumerate(tiles):
            x, y = 10 + (i % cols) * (cw * 4 + 10), 10 + (i // cols) * (ch * 4 + 24)
            d.text((x, y), name, fill=(255, 220, 120))
            bg = Image.new("RGBA", im.size, (20, 20, 26, 255))
            bg.alpha_composite(im.convert("RGBA"))
            sheet.paste(bg.convert("RGB"), (x, y + 14))
        sheet.save(OUT / f"compare_g{gi}.png")
    print("done")


if __name__ == "__main__":
    main()
