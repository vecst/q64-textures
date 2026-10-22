#!/usr/bin/env python3
"""HD replacement for Tale's overlay font (Experience panel, spell preview, story arrow labels).

Tale's overlay is patched (patch_hud.py) to draw its 3x5 pixel font from a 512x8 I8 atlas with texture rectangles.
This module rebuilds that exact atlas, computes its RT64 texture-pack hash (rt64hash.py port, LoadBlock layout:
odd TMEM rows have their 32-bit words swapped), and renders a smooth replacement: Adwaita Mono Bold capitals, white
with alpha coverage (the overlay tints it with its own colours), 8x, same cell layout.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "games/q64-mods/q64-capture"))
import rt64hash  # noqa: E402

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ%:- /!.,?+'()&"
GLYPHS = [0x7B6F, 0x2492, 0x73E7, 0x73CF, 0x5BC9, 0x79CF, 0x79EF, 0x7249, 0x7BEF, 0x7BCF,
          0x2BED, 0x6BAE, 0x7927, 0x6B6E, 0x79E7, 0x79E4, 0x796F, 0x5BED, 0x7497, 0x124F,
          0x5BAD, 0x4927, 0x5FED, 0x5F6D, 0x7B6F, 0x7BE4, 0x7B7B, 0x7BAD, 0x79CF, 0x7492,
          0x5B6F, 0x5B6A, 0x5BFD, 0x5AAD, 0x5A92, 0x73A7, 0x52A5, 0x0410, 0x01C0, 0, 0x12A4,
          0x2482, 0x0002, 0x0014, 0x72C2, 0x05D0, 0x2400, 0x1491, 0x4494, 0x2AAB]
W, H, CELL = 512, 8, 6
SCALE = 8
SS = 4
FONT = "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf"
MAX_W = 3.6  # ink width in atlas px (glyph box is 3 px; the drawn cell has a 1-px margin each side)


def atlas():
    a = np.zeros((H, W), np.uint8)
    for i, bits in enumerate(GLYPHS):
        for row in range(5):
            for col in range(3):
                if bits & (1 << (14 - row * 3 - col)):
                    a[row + 1, i * CELL + col + 1] = 0xFF
    return a


def texture_hash():
    tmem = rt64hash.tmem_from_rgba16_rows(atlas().tobytes(), W, H, W // 8)
    tile = {"fmt": 4, "siz": 1, "line": W // 8, "tmem": 0, "palette": 0}
    return "%016x" % rt64hash.tmem_hash(tmem, tile, W, H, 0)


def render():
    k = SCALE * SS
    target = 5 * k  # cap height = the 5-px glyph height
    size = target
    for _ in range(5):
        f = ImageFont.truetype(FONT, size)
        _, t, _, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), "H", font=f)
        size = int(round(size * target / (b - t)))
    font = ImageFont.truetype(FONT, size)
    cov = Image.new("L", (W * k, H * k), 0)
    for i, ch in enumerate(ALPHABET):
        if ch == " ":
            continue
        tmp = Image.new("L", (size * 3, size * 3), 0)
        ImageDraw.Draw(tmp).text((size, size * 2), ch, font=font, fill=255, anchor="ls")
        bb = tmp.getbbox()
        g = tmp.crop(bb)
        if g.width > MAX_W * k:
            g = g.resize((int(MAX_W * k), g.height), Image.LANCZOS)
        cx = (i * CELL + 2.5) * k  # centre of the glyph box (atlas px 1..3)
        top = int(round(6 * k + (bb[1] - size * 2)))  # baseline under glyph row 5 (atlas row 6)
        cov.paste(g, (int(round(cx - g.width / 2)), top), g)
    cov = cov.resize((W * SCALE, H * SCALE), Image.BOX)
    out = np.zeros((H * SCALE, W * SCALE, 4), np.uint8)
    out[..., :3] = 255
    out[..., 3] = np.asarray(cov)
    return texture_hash(), out


if __name__ == "__main__":
    h, img = render()
    Image.fromarray(img, "RGBA").save(f"/tmp/q64sheets/hudfont_{h}.png")
    print(h)
