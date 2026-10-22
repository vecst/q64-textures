#!/usr/bin/env python3
"""Clean rendered numbers for the HUD/battle digits, plus the "Select an element" panel text.

Digits: the game loads each HUD digit as its own 8x16 texture (the glyph in the top rows, leftover TMEM below), so
one digit shows up under many hashes. Every variant is identified by matching its shape against the two digit rows
(8-px grid: 0-9 / % + - ± and :), then redrawn as Adwaita Sans Bold Italic in its own fill colour with a dark
outline, at one common size on one common baseline so all numbers line up.

Panel: the old yellow pixel lettering is erased (filled with parchment grain sampled from the same panel) before
the panel is upscaled, then "Select" / "an element" are drawn fresh on top at 6x.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

R = Path(__file__).resolve().parent
S = 6
SS = 4
ROW_CHARS = "0123456789/%+-±:"
CELL = 8
OUTLINE = 0.7
OUTLINE_COLOR = (24, 18, 14)


def _alpha(h, src="decoded_text"):
    p = R / src / f"{h}.png"
    if not p.exists():
        p = R / "decoded" / f"{h}.png"
    return np.asarray(Image.open(p).convert("RGBA"))


def templates(text):
    """char -> (mask 10xN, (x0, y0, x1, y1) ink box in cell coords) from the two digit rows."""
    out = {}
    for i in (8, 9):
        a = _alpha(text[i])
        m = a[..., 3] > 0
        for c, ch in enumerate(ROW_CHARS):
            cell = m[:, c * CELL:(c + 1) * CELL]
            if cell.shape[1] < CELL or not cell.any() or ch in out:
                continue
            ys, xs = np.where(cell)
            out[ch] = (cell[:10], (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return out


def match(mask, tpl):
    """Best (char, dx, dy, score) of a glyph mask against the templates (small shifts allowed)."""
    best = (None, 0, 0, -1.0)
    g = np.zeros((14, 12), bool)
    g[2:2 + min(10, mask.shape[0]), 2:2 + min(8, mask.shape[1])] = mask[:10, :8]
    for ch, (t, _) in tpl.items():
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                tt = np.zeros_like(g)
                tt[2 + dy:2 + dy + t.shape[0], 2 + dx:2 + dx + t.shape[1]] = t[:, :8]
                inter, union = (g & tt).sum(), (g | tt).sum()
                if union and inter / union > best[3]:
                    best = (ch, dx, dy, inter / union)
    return best


def fill_color(rgba):
    op = rgba[..., 3] > 128
    v = rgba[op][:, :3].astype(np.float32)
    lum = v @ np.array([0.3, 0.59, 0.11], np.float32)
    return tuple(int(c) for c in v[lum >= np.percentile(lum, 75)].mean(0))


def draw_glyphs(size, items, font, outline=OUTLINE, outline_color=OUTLINE_COLOR, margin=0.35):
    """items: (text, (cell_x0, cell_x1, baseline) in source px, fill). Returns an RGBA array at 6x.
    The game samples each glyph from its own cell, so ink (outline included) is kept inside the cell: centred,
    and narrowed if it would cross the cell edge (otherwise it is clipped and spills into the neighbour)."""
    w, h = size
    k = S * SS
    out = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    sw = int(round(outline * k))
    for txt, (x0, x1, base), fill in items:
        tmp = Image.new("RGBA", (font.size * 4, font.size * 3), (0, 0, 0, 0))
        ox, oy = font.size, font.size * 2
        ImageDraw.Draw(tmp).text((ox, oy), txt, font=font, anchor="ls", fill=fill + (255,), stroke_width=sw,
                                 stroke_fill=outline_color + (255,))
        bb = tmp.getbbox()
        if not bb:
            continue
        g = tmp.crop(bb)
        room = int((x1 - x0 - 2 * margin) * k)
        if g.width > room:
            g = g.resize((room, g.height), Image.LANCZOS)
        left = int(round((x0 + x1) / 2 * k - g.width / 2))
        top = int(round(base * k + (bb[1] - oy)))
        out.alpha_composite(g, (left, max(top, 0)))
    return np.asarray(out.convert("RGBa").resize((w * S, h * S), Image.BOX).convert("RGBA"))


def digit_font(height_px):
    """Bold italic font whose digit '0' (plus outline) is height_px source pixels tall."""
    import labeltext
    target = (height_px - 2 * OUTLINE) * S * SS
    size = int(target)
    for _ in range(5):
        f = labeltext._font("italic", size)
        l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), "0", font=f)
        size = int(round(size * target / (b - t)))
    return labeltext._font("italic", size)


def render_digits():
    """Yields (hash, rgba 6x) for the digit rows, the single-glyph 8x10 symbols and every 8x16 digit variant."""
    text = json.loads((R / "text_textures.json").read_text())
    meta = json.loads((R / "decoded/meta.json").read_text())
    tpl = templates(text)
    height = int(np.median([b[3] - b[1] for _, b in tpl.values()]))
    base = int(np.median([b[3] for _, b in tpl.values()]))
    font = digit_font(height)
    report = {}
    # the two rows themselves
    for i in (8, 9):
        h = text[i]
        a = _alpha(h)
        items = []
        row_fill = fill_color(a)  # one colour per row (a stray red line tints some cells of the second row)
        for c, ch in enumerate(ROW_CHARS):
            if c * CELL + CELL > a.shape[1] or not (a[:, c * CELL:(c + 1) * CELL, 3] > 0).any():
                continue
            x0, _, x1, _ = tpl[ch][1] if ch in tpl else (0, 0, CELL, 0)
            items.append((ch, (c * CELL, (c + 1) * CELL, base), row_fill))
        yield h, draw_glyphs((a.shape[1], a.shape[0]), items, font)
    # single glyph textures: 8x10 symbols and 8x16 digits
    for h in text:
        v = meta[h]
        if (v["w"], v["h"]) not in ((8, 16), (8, 10)):
            continue
        a = _alpha(h)
        ch, dx, dy, score = match(a[..., 3] > 0, tpl)
        if score < 0.5 or ch == ":":  # ':' only exists inside the row; lone 8x10 marks are cross icons, not text
            continue
        x0, _, x1, _ = tpl[ch][1]
        report[ch] = report.get(ch, 0) + 1
        yield h, draw_glyphs((v["w"], v["h"]), [(ch, (0, CELL, base + dy), fill_color(a[:10]))], font)
    print("digits matched: " + " ".join(f"{k}x{n}" for k, n in sorted(report.items())))


# ---------------------------------------------------------------- "Select an element" panel
PANEL_TOP = "68b52875cbd118f3"  # 65x21 piece with "Select" (the text continues into the piece below it)
PANEL_LINES = ["Select", "an element"]


def _text_mask(c):
    """Yellow letter pixels, and everything within 3 px of them (their dark outline and its soft edge)."""
    f = c.astype(np.int32)
    yellow = (f[..., 0] > 150) & (f[..., 1] > 120) & (f[..., 2] < 100) & (f[..., 0] - f[..., 2] > 90)
    near = yellow.copy()
    for _ in range(3):  # 8-neighbour dilation
        g = np.pad(near, 1)
        near = np.max([g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return near, yellow


def clean_panel(canvas):
    """Erase the panel lettering in an assembled group canvas (source px). Returns (canvas, line boxes, colour)."""
    c = canvas.copy()
    region = np.zeros(c.shape[:2], bool)
    region[2:40, 3:63] = True
    mask, yellow = _text_mask(c)
    mask &= region
    yellow &= region
    f = c[..., :3].astype(np.float32)
    par = region & ~mask
    mean = f[par].mean(0)
    pool = c[par & (np.abs(f - mean).sum(-1) < 60)]
    rng = np.random.default_rng(64)
    c[mask] = pool[rng.integers(0, len(pool), mask.sum())]
    rows = np.where(yellow.any(1))[0]
    lines, start = [], rows[0]
    for a, b in zip(rows, rows[1:]):
        if b > a + 1:
            lines.append((start, a + 1))
            start = b
    lines.append((start, rows[-1] + 1))
    boxes = []
    for y0, y1 in lines:
        xs = np.where(yellow[y0:y1].any(0))[0]
        boxes.append((xs.min(), y0, xs.max() + 1, y1))
    col = tuple(int(v) for v in canvas[yellow][:, :3].mean(0))
    return c, boxes, col


def draw_panel_text(big, boxes, color):
    """Draw PANEL_LINES (block bold, yellow, dark outline) into the 6x canvas over the cleaned panel."""
    import labeltext
    h, w = big.shape[0] // S, big.shape[1] // S
    out = Image.fromarray(big, "RGBA")
    for (x0, y0, x1, y1), word in zip(boxes, PANEL_LINES):
        cap = (y1 - y0) - 2 * OUTLINE  # glyph rows include the outline
        probe = labeltext._font("block", 200)
        d = ImageDraw.Draw(Image.new("L", (1, 1)))
        _, t, _, b = d.textbbox((0, 0), "Selct", font=probe, anchor="ls")
        size = int(200 * cap * S * SS / (b - t))
        f = labeltext._font("block", size)
        l, _, r, _ = d.textbbox((0, 0), word, font=f, anchor="ls")
        maxw = (x1 - x0 + 2) * S * SS
        if r - l > maxw:
            f = labeltext._font("block", int(size * maxw / (r - l)))
        layer = draw_glyphs((w, h), [(word, (x0 - 3, x1 + 3, y1 - OUTLINE - 0.3), color)], f, margin=0)
        out.alpha_composite(Image.fromarray(layer, "RGBA"))
    return np.asarray(out)


# ---------------------------------------------------------------- 7x9 menu font ("WIND CUTTER LV1", "EMPTY")
# Each glyph is its own 7x9 texture cut from a 258-px-wide sheet on a 6-px grid in ASCII order from '0'
# (cell = ord(char) - 48: 1 -> cell 1, A -> cell 17, Y -> cell 41). Rows 0-6 letter, row 7 shadow, row 8 junk.
MENU_FONT = "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf"
MENU_CAP, MENU_CELL = 7, 6
MENU_MAX_LUM = 95  # grey variants are darkened to this luminance so they read on parchment (hue kept)


def _menu_color(rgba):
    """Letter colour of a 7x9 glyph: the most common colour in its letter rows, darkened if too light."""
    a = rgba[:MENU_CAP, :MENU_CELL]
    v = a[a[..., 3] > 0][:, :3]
    cols, n = np.unique(v, axis=0, return_counts=True)
    c = cols[np.argmax(n)].astype(np.float32)
    if tuple(int(x) for x in c) == (49, 49, 49):
        return (26, 20, 16)  # same rich dark as the dialog font
    lum = float(c @ np.array([0.3, 0.59, 0.11]))
    if lum > MENU_MAX_LUM:
        c *= MENU_MAX_LUM / lum
    return tuple(int(x) for x in c)


def render_menu_font():
    from PIL import ImageFont
    text = json.loads((R / "text_textures.json").read_text())
    meta = json.loads((R / "decoded/meta.json").read_text())
    target = MENU_CAP * S * SS
    size = target
    for _ in range(5):
        f = ImageFont.truetype(MENU_FONT, size)
        _, t, _, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), "H", font=f)
        size = int(round(size * target / (b - t)))
    font = ImageFont.truetype(MENU_FONT, size)
    n = 0
    for h in text:
        v = meta[h]
        if (v["w"], v["h"]) != (7, 9) or not v.get("load"):
            continue
        ch = chr(48 + v["load"]["x"] // MENU_CELL)
        a = np.asarray(Image.open(R / "decoded" / f"{h}.png").convert("RGBA"))
        n += 1
        yield h, draw_glyphs((7, 9), [(ch, (0, MENU_CELL + 0.5, MENU_CAP), _menu_color(a))], font, outline=0,
                             margin=0.25)
    print(f"menu font: {n} glyph textures re-rendered")
