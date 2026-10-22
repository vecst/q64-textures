#!/usr/bin/env python3
"""Re-render the stylized menu/HUD labels (Name, Agi, Status, LOAD, NEW, Miss! ...) as clean text.

Each label keeps its original fill colour (the brightest fifth of its pixels: white, gold, green) and gets a crisp
dark outline instead of the blurry shadow/gradient, so it stays readable over any background. Script-style labels
use Adwaita Sans Bold Italic, block labels Adwaita Sans ExtraBold. The word is fitted inside the original's ink box.

usage: from labeltext import render_labels; for h, rgba in render_labels(): ...
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
S = 6
SS = 4
UPRIGHT = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
ITALIC = "/usr/share/fonts/Adwaita/AdwaitaSans-Italic.ttf"
OUTLINE = 0.8          # outline width in original pixels
OUTLINE_COLOR = (24, 18, 14)
# (list, index, word, style) - index into text_textures.json ("t") or the sorted 2D candidate list ("u", same
# numbering as the contact sheets / edit/text-ui/ui file names)
LABELS = [
    ("t", 10, "PUSH START", "block"), ("t", 11, "Environment", "italic"), ("t", 12, "DELETE", "block"),
    ("t", 13, "Message", "italic"), ("t", 14, "max", "italic"), ("t", 16, "exp", "italic"), ("t", 17, "min", "italic"),
    ("t", 18, "MP", "block"), ("t", 19, "Hp", "italic"), ("t", 20, "HP", "block"),
    ("u", 16, "days", "italic"), ("u", 39, "Agi", "italic"), ("u", 47, "Name", "italic"), ("u", 56, "No.", "italic"),
    ("u", 82, "Item", "italic"), ("u", 115, "LOAD", "block"), ("u", 151, "Map", "italic"), ("u", 181, "No.", "italic"),
    ("u", 183, "Name", "italic"), ("u", 221, "BGM", "italic"), ("u", 224, "Mp", "italic"),
    ("u", 289, "Status", "italic"), ("u", 317, "NEW", "block"), ("u", 325, "Miss!", "block"),
    ("u", 342, "days", "italic"), ("u", 356, "Escape!", "block"), ("u", 400, "Def", "italic"),
    ("u", 403, "Elements", "italic"),
]


def lists():
    text = json.loads((R / "text_textures.json").read_text())
    cand = sorted((set(json.loads((R / "ui_textures.json").read_text()))
                   | set(json.loads((R / "art_textures.json").read_text()))) - set(text))
    return text, cand


def _font(style, size):
    f = ImageFont.truetype(ITALIC if style == "italic" else UPRIGHT, size)
    wt = 700 if style == "italic" else 800
    f.set_variation_by_axes([20 if a["name"] == b"Optical size" else wt if a["name"] == b"Weight" else a["default"]
                             for a in f.get_variation_axes()])
    return f


def _fill_color(rgba):
    op = rgba[..., 3] > 128
    v = rgba[op][:, :3].astype(np.float32)
    lum = v @ np.array([0.3, 0.59, 0.11], np.float32)
    return tuple(int(c) for c in v[lum >= np.percentile(lum, 80)].mean(0))


def render_label(rgba, word, style):
    h, w = rgba.shape[:2]
    ys, xs = np.where(rgba[..., 3] > 0)
    bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    fill = _fill_color(rgba)
    sw = int(round(OUTLINE * S * SS))
    probe = _font(style, 200)
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), word, font=probe, stroke_width=0)
    bw, bh = (bx1 - bx0) * S * SS - 2 * sw, (by1 - by0) * S * SS - 2 * sw
    size = int(200 * min(bw / (r - l), bh / (b - t)))
    font = _font(style, max(size, 8))
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), word, font=font, stroke_width=sw)
    big = Image.new("RGBA", (w * S * SS, h * S * SS), (0, 0, 0, 0))
    cx, cy = (bx0 + bx1) / 2 * S * SS, (by0 + by1) / 2 * S * SS
    ImageDraw.Draw(big).text((cx - (l + r) / 2, cy - (t + b) / 2), word, font=font, fill=fill + (255,),
                             stroke_width=sw, stroke_fill=OUTLINE_COLOR + (255,))
    small = big.convert("RGBa").resize((w * S, h * S), Image.BOX).convert("RGBA")  # premultiplied downsample
    return np.asarray(small)


def render_labels():
    text, cand = lists()
    for kind, i, word, style in LABELS:
        h = (text if kind == "t" else cand)[i]
        src = R / ("decoded_text" if kind == "t" else "decoded") / f"{h}.png"  # text: leftover rows cleaned
        rgba = np.asarray(Image.open(src).convert("RGBA"))
        yield h, render_label(rgba, word, style)


if __name__ == "__main__":
    for h, a in render_labels():
        Image.fromarray(a, "RGBA").save(f"/tmp/q64sheets/label_{h}.png")
    print("ok")
