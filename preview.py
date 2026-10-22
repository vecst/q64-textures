#!/usr/bin/env python3
"""Preview of the HD Text & UI pack without starting the game: dialog lines set from the real font-sheet textures,
plus HUD/menu labels and icons, original (point-sampled like the game) vs. HD, at the pack's 6x scale.

usage: preview.py <out.png> [pack_png_dir]     (default pack dir: staging/Quest64HD-TextUI)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import upscale as up  # noqa: E402

S = 6
PACK = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "staging/Quest64HD-TextUI"
EDIT = R / "edit/text-ui"
meta = json.loads((R / "decoded/meta.json").read_text())
text = json.loads((R / "text_textures.json").read_text())
cand = sorted((set(json.loads((R / "ui_textures.json").read_text()))
               | set(json.loads((R / "art_textures.json").read_text()))) - set(text))
BOX = (226, 211, 174, 255)  # parchment: the font is dark-on-light
BG = (14, 16, 40, 255)


def orig(h):
    src = R / ("decoded_text" if h in text else "decoded") / f"{h}.png"
    im = Image.open(src).convert("RGBA")
    return im.resize((im.width * S, im.height * S), Image.NEAREST)


def hd(h):
    im = Image.open(PACK / f"{h}.png").convert("RGBA")
    w, hh = meta[h]["w"] * S, meta[h]["h"] * S
    return im if im.size == (w, hh) else im.resize((w, hh), Image.BICUBIC)


# --- dialog font: 3 sheets on a 10-px grid
SHEETS = text[:3]  # digits/punctuation, lowercase, uppercase
CHARS = {}
for i, c in enumerate("0123456789!?"):
    CHARS[c] = (0, i)
for i in range(26):
    CHARS[chr(97 + i)] = (1, i)
    CHARS[chr(65 + i)] = (2, i)


def set_line(txt, getter):
    """The game's text engine is monospaced: each glyph is drawn from its cell's left edge, advancing 8 px."""
    sheets = [getter(h) for h in SHEETS]
    line = Image.new("RGBA", (len(txt) * 8 * S, 13 * S), (0, 0, 0, 0))
    for i, c in enumerate(txt):
        if c == " ":
            continue
        row, cell = CHARS[c]
        line.alpha_composite(sheets[row].crop((cell * 10 * S, 0, (cell * 10 + 8) * S, 13 * S)), (i * 8 * S, 0))
    return line


def group_image(member, getter):
    for u in up.group_strips(sorted(set(cand) - set(text)), meta):
        if member in [h for h, _, _ in u]:
            w = max(x + meta[h]["w"] for h, x, _ in u) * S
            hh = max(y + meta[h]["h"] for h, _, y in u) * S
            c = Image.new("RGBA", (w, hh), (0, 0, 0, 0))
            for h, x, y in u:
                c.alpha_composite(getter(h), (x * S, y * S))
            return c
    return getter(member)


DIALOG = ["Grand Abbott", "find yourself in danger as", "well  Will you go there now"]
LABELS = [("t", 20), ("t", 19), ("t", 18), ("u", 224), ("u", 39), ("u", 400), ("u", 16), ("t", 16), ("t", 14),
          ("t", 17), ("u", 47), ("u", 82), ("u", 289), ("u", 403), ("u", 151), ("u", 115), ("u", 317), ("u", 325),
          ("u", 356), ("t", 12), ("t", 13), ("t", 11)]
ICONS = [("u", 81), ("u", 110), ("u", 254), ("u", 256), ("u", 249), ("u", 0), ("u", 12), ("u", 122), ("u", 130),
         ("u", 274), ("u", 402), ("u", 95), ("u", 431), ("u", 14), ("u", 231), ("u", 156), ("u", 167), ("u", 382)]


def pick(k, i):
    return (text if k == "t" else cand)[i]


def flow(items, width, gap=4 * S):
    rows, row, x, rh = [], [], 0, 0
    for im in items:
        if x + im.width > width and row:
            rows.append((row, rh))
            row, x, rh = [], 0, 0
        row.append((im, x))
        x += im.width + gap
        rh = max(rh, im.height)
    if row:
        rows.append((row, rh))
    h = sum(rh + gap for _, rh in rows)
    out = Image.new("RGBA", (width, max(h, 1)), (0, 0, 0, 0))
    y = 0
    for row, rh in rows:
        for im, x in row:
            out.alpha_composite(im, (x, y + (rh - im.height) // 2))
        y += rh + gap
    return out


def panel(getter, title):
    width = 300 * S
    blocks = []
    lines = [set_line(t, getter) for t in DIALOG]
    box = Image.new("RGBA", (width, sum(l.height + 3 * S for l in lines) + 8 * S), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=4 * S, fill=BOX,
                                          outline=(120, 90, 60, 255), width=S)
    y = 4 * S
    for l in lines:
        box.alpha_composite(l, (6 * S, y))
        y += l.height + 3 * S
    blocks.append(box)
    blocks.append(flow([getter(pick(*k)) for k in LABELS], width))
    digits = [getter(text[8]), getter(text[9])]
    blocks.append(flow(digits + [group_image(pick("u", 214), getter)], width))
    blocks.append(flow([getter(pick(*k)) for k in ICONS], width))
    blocks.append(flow([group_image(pick("u", 72), getter), getter(text[10])], width))
    h = sum(b.height + 6 * S for b in blocks) + 14 * S
    out = Image.new("RGBA", (width + 12 * S, h), BG)
    ImageDraw.Draw(out).text((6 * S, 4 * S), title, fill=(255, 255, 255, 255), font_size=7 * S)
    y = 14 * S
    for b in blocks:
        out.alpha_composite(b, (6 * S, y))
        y += b.height + 6 * S
    return out


a, b = panel(orig, "ORIGINAL (as the game draws it)"), panel(hd, "HD Text & UI")
sheet = Image.new("RGBA", (a.width + b.width + 4 * S, max(a.height, b.height)), (0, 0, 0, 255))
sheet.alpha_composite(a, (0, 0))
sheet.alpha_composite(b, (a.width + 4 * S, 0))
sheet.convert("RGB").save(sys.argv[1])
# a 1:1 crop of the dialog box, the part where sharpness matters most
crop = Image.new("RGBA", (a.width, 70 * S), BG)
crop.alpha_composite(a.crop((0, 14 * S, a.width, 84 * S)), (0, 0))
crop2 = b.crop((0, 14 * S, b.width, 84 * S))
both = Image.new("RGB", (a.width, 140 * S + 2 * S))
both.paste(crop.convert("RGB"), (0, 0))
both.paste(crop2.convert("RGB"), (0, 70 * S + 2 * S))
both.save(sys.argv[1].replace(".png", "_dialog.png"))
print("saved", sys.argv[1], sheet.size)
