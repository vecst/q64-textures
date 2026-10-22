#!/usr/bin/env python3
"""Re-render the dialog font sheets from a real font: clean dark letters, no drop shadow.

The game draws dialog one glyph at a time from 3 font sheets (304x13, 10-px cells):
  digits/symbols  0-9 ! ? 「 」 『 』 . , … · : — ♪ ♡
  lowercase       a-z ’ ” “
  uppercase       A-Z &
Measured in-game (screenshot, 2026-10-18): the text engine is MONOSPACED - every glyph advances exactly 8 px and is
drawn from the left edge of its cell (a space is one more 8-px step). So the sheets are rendered with a monospace
font, each glyph's advance box centred in the 8-px slot, cap height 8 px on the original baseline (under row 7).
The corner brackets (no font glyph) are rebuilt from the original's dark pixels, shadow removed.

usage: from fontsheet import render_sheets; sheets = render_sheets(color=(26, 20, 16))
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
S = 6
SS = 4  # supersampling
FONT = "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf"
SYMBOL_FONT = "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
CAP, BASE, SLOT, CELL = 8, 8, 8, 10
WIDEN = 1.12  # monospace letters fill ~6.6 of the 8-px slot; widen a little so text doesn't look letter-spaced
ROWS = [
    list("0123456789!?") + ["BRK1", "BRK2", "BRK3", "BRK4", ".", ",", "…", "·", ":", "—", "♪", "♡"],
    list("abcdefghijklmnopqrstuvwxyz") + ["’", "”", "“"],
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["&"],
]
SYMBOLS = {"♪": 0.0, "♡": 0.3}  # from the symbol font; value = outline thickening (fraction of a stem)
BODY_MAX = 70  # original letter pixels are (49,49,49); shadow/highlight are brighter browns


def _glyph(ch, font, stroke=0):
    """Supersampled ink mask, ink offset from the pen origin (x, y relative to baseline), advance."""
    pad = font.size
    im = Image.new("L", (font.size * 3, font.size * 3), 0)
    ImageDraw.Draw(im).text((pad, pad * 2), ch, fill=255, font=font, anchor="ls", stroke_width=stroke,
                            stroke_fill=255)
    a = np.asarray(im)
    ys, xs = np.where(a > 0)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return a[box[1]:box[3], box[0]:box[2]], (box[0] - pad, box[1] - pad * 2), font.getlength(ch)


def _cap_font(path):
    """Font whose 'H' is exactly CAP*S*SS tall (supersampled)."""
    target = CAP * S * SS
    size = target
    for _ in range(6):
        m, _, _ = _glyph("H", ImageFont.truetype(path, size))
        size = int(round(size * target / m.shape[0]))
    return ImageFont.truetype(path, size)


def render_sheets(color=(26, 20, 16), font_path=FONT):
    import textui
    text = json.loads((R / "text_textures.json").read_text())
    font = _cap_font(font_path)
    sym = _cap_font(SYMBOL_FONT)
    stem = _glyph("l", font)[0].shape[1] * 0.35  # rough stem width, for thickening hairline symbols
    sheets = []
    for row, chars in enumerate(ROWS):
        src = np.asarray(Image.open(R / "decoded" / f"{text[row]}.png").convert("RGBA"))
        body = (src[..., 3] > 0) & (src[..., 0] < BODY_MAX)
        H, W = src.shape[:2]
        cov = np.zeros((H * S, W * S), np.float32)
        for cell, ch in enumerate(chars):
            c0 = cell * CELL * S
            if ch.startswith("BRK"):  # no font glyph: clean up the original shape, without its shadow
                m = np.zeros((H, CELL, 4), np.uint8)
                m[body[:, cell * CELL:(cell + 1) * CELL]] = (255, 255, 255, 255)
                up = textui.upscale(m, "hybrid", S)[..., 3].astype(np.float32) / 255
                cov[:, c0:c0 + CELL * S] = np.maximum(cov[:, c0:c0 + CELL * S], up)
                continue
            f = sym if ch in SYMBOLS else font
            mask, (ox, oy), adv = _glyph(ch, f, int(round(SYMBOLS.get(ch, 0) * stem)))
            g = Image.fromarray(mask)
            slot = SLOT * S * SS
            if ch in SYMBOLS:  # proportional symbol font: centre the ink itself
                ox, adv = 0, g.width
            else:
                g = g.resize((max(1, round(g.width * WIDEN)), g.height), Image.LANCZOS)
                ox, adv = ox * WIDEN, adv * WIDEN
            if g.width > slot - S * SS // 2:  # keep half a pixel of air inside the slot
                g = g.resize((slot - S * SS // 2, g.height), Image.LANCZOS)
                ox, adv = 0, g.width
            gw, gh = max(1, round(g.width / SS)), max(1, round(g.height / SS))
            g = np.asarray(g.resize((gw, gh), Image.BOX), np.float32) / 255  # box filter = exact coverage
            origin = c0 + (SLOT * S - adv / SS) / 2  # advance box centred in the 8-px slot
            left = int(round(origin + ox / SS))
            top = int(round(BASE * S + oy / SS))
            t0, l0 = max(top, 0), max(left, c0)
            t1, l1 = min(top + gh, H * S), min(left + gw, c0 + SLOT * S)
            cov[t0:t1, l0:l1] = np.maximum(cov[t0:t1, l0:l1], g[t0 - top:t1 - top, l0 - left:l1 - left])
        out = np.zeros((H * S, W * S, 4), np.uint8)
        out[..., :3] = color
        out[..., 3] = np.clip(cov * 255, 0, 255).round().astype(np.uint8)
        sheets.append(out)
    return text[:3], sheets


if __name__ == "__main__":
    for h, s in zip(*render_sheets()):
        Image.fromarray(s, "RGBA").save(f"/tmp/q64sheets/font_{h}.png")
    print("ok")
