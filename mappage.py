#!/usr/bin/env python3
"""Pause-menu map pages: rebuilt from their strips, lettering erased, map upscaled, names redrawn clean.

The game streams a map page into one RDRAM buffer and draws it as 244x8 CI8 strips, each loaded from its own row
address, so a strip's row on the page = (load address - page base) / 244. Per page (map_pages.json):
  1. assemble the page, erase the title (white script) and the place names (red-brown serif) by filling them with
     parchment grain sampled from the same page
  2. upscale the clean page with UltraSharpV2 through ComfyUI (cached in map_cache/, so rebuilds don't need it)
  3. draw the title in the label style (white bold italic, dark outline) and the place names in Nimbus Roman Bold in
     their original colour, then cut the page back into its strips
The original title art misspells the region ("Holy Plan"); the redraw uses the right name.

map_pages.json: {"<page>": {"dump": "<decoded dir>", "title": [word, [x0,y0,x1,y1]], "names": [[word, box], ...]}}
(boxes in page pixels, generous; the exact letters are found inside them)
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(__file__).resolve().parent
S = 6
SS = 4
CACHE = R / "map_cache"
NAME_FONT = "/usr/share/fonts/urw-base35/NimbusRoman-Bold.ttf"
STRIP_W = 244


def pages():
    p = R / "map_pages.json"
    return json.loads(p.read_text()) if p.exists() else {}


def page_strips(cfg):
    m = json.loads((R / cfg["dump"] / "meta.json").read_text())
    strips = [h for h in m if m[h]["w"] == STRIP_W and m[h].get("load")]
    base = min(m[h]["load"]["addr"] for h in strips)
    return {h: ((m[h]["load"]["addr"] - base) // STRIP_W, m[h]["h"]) for h in strips}


def strip_hashes():
    return sorted({h for cfg in pages().values() for h in page_strips(cfg)})


def assemble(rows):
    H = max(r + h for r, h in rows.values())
    img = Image.new("RGBA", (STRIP_W, H), (0, 0, 0, 0))
    for h, (r, _) in rows.items():
        img.alpha_composite(Image.open(R / "decoded" / f"{h}.png").convert("RGBA"), (0, r))
    return np.asarray(img).copy()


def _dilate(mask, n):
    for _ in range(n):
        g = np.pad(mask, 1)
        mask = np.max([g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
    return mask


def clean(page, cfg):
    """Erase the lettering. Returns (clean page, [(word, kind, letter box, colour)])."""
    f = page[..., :3].astype(np.float32)
    boxes = [(cfg["title"][0], "title", cfg["title"][1])] + [(w, "name", b) for w, b in cfg["names"]]
    inbox = np.zeros(page.shape[:2], bool)
    for _, _, (x0, y0, x1, y1) in boxes:
        inbox[y0:y1, x0:x1] = True
    par = ~inbox & (page[..., 3] > 0)
    mean = np.median(f[par], 0)
    par &= np.abs(f - mean).sum(-1) < 45  # plain parchment only (no paths, icons, border)
    pool = page[par]
    out, words = page.copy(), []
    rng = np.random.default_rng(7)
    for word, kind, (x0, y0, x1, y1) in boxes:
        box = np.zeros(page.shape[:2], bool)
        box[y0:y1, x0:x1] = True
        ink = box & (np.abs(f - mean).sum(-1) > 70)
        ys, xs = np.where(ink)
        letters = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        lum = f[ink] @ np.array([0.3, 0.59, 0.11])
        col = f[ink][lum >= np.percentile(lum, 70)] if kind == "title" else f[ink][lum <= np.percentile(lum, 40)]
        erase = _dilate(ink, 2) & box
        out[erase] = pool[rng.integers(0, len(pool), erase.sum())]
        words.append((word, kind, letters, tuple(int(c) for c in col.mean(0))))
    return out, words


def upscale_page(clean_page):
    """UltraSharpV2 4x of the clean page (cached by content), returned at 6x."""
    CACHE.mkdir(exist_ok=True)
    key = hashlib.md5(clean_page.tobytes()).hexdigest()[:16]
    cached = CACHE / f"{key}.png"
    if not cached.exists():
        import upscale as up
        comfy = up.Comfy("http://127.0.0.1:8188", str(R / "workflow_api.json"), "4x-UltraSharpV2.safetensors")
        rgb, alpha, pad = up.prepare(clean_page, {"cms": 2, "cmt": 2})
        res = comfy.result(comfy.submit(f"map_{key}.png", rgb))
        up.finish(res, alpha, pad, clean_page[..., 3]).save(cached)
    big = Image.open(cached).convert("RGBA")
    h, w = clean_page.shape[:2]
    return big.resize((w * S, h * S), Image.LANCZOS)


def _fit_font(path, text, box_w, box_h, style=None):
    if style:
        import labeltext
        make = lambda s: labeltext._font(style, s)  # noqa: E731
    else:
        make = lambda s: ImageFont.truetype(path, s)  # noqa: E731
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    probe = make(200)
    l, t, r, b = d.textbbox((0, 0), text, font=probe)
    return make(max(8, int(200 * min(box_w / (r - l), box_h / (b - t)))))


def draw_words(big, words):
    k = S * SS
    w, h = big.size
    layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for word, kind, (x0, y0, x1, y1), col in words:
        if kind == "title":  # white bold italic with a dark outline, like the menu labels
            sw = int(0.9 * k)
            font = _fit_font(None, word, (x1 - x0) * k - 2 * sw, (y1 - y0) * k - 2 * sw, style="italic")
            fill, stroke = col + (255,), (24, 18, 14, 255)
        else:  # serif place names in their original ink colour
            sw = 0
            font = _fit_font(NAME_FONT, word, (x1 - x0) * 1.04 * k, (y1 - y0) * k)
            fill, stroke = col + (255,), None
        l, t, r, b = d.textbbox((0, 0), word, font=font, stroke_width=sw)
        cx, cy = (x0 + x1) / 2 * k, (y0 + y1) / 2 * k
        if kind == "title":
            cx = x0 * k + (r - l) / 2  # keep the title's left edge where it was
        d.text((cx - (l + r) / 2, cy - (t + b) / 2), word, font=font, fill=fill, stroke_width=sw,
               stroke_fill=stroke)
    layer = layer.convert("RGBa").resize((w, h), Image.BOX).convert("RGBA")
    out = big.copy()
    out.alpha_composite(layer)
    return out


def render_pages():
    for name, cfg in pages().items():
        rows = page_strips(cfg)
        page = assemble(rows)
        cleaned, words = clean(page, cfg)
        big = draw_words(upscale_page(cleaned), words)
        a = np.asarray(big)
        for h, (r, hh) in rows.items():
            yield h, np.ascontiguousarray(a[r * S:(r + hh) * S])
        print(f"map page {name}: {len(rows)} strips, redrew " + ", ".join(w for w, *_ in words))


if __name__ == "__main__":
    for name, cfg in pages().items():
        rows = page_strips(cfg)
        page = assemble(rows)
        cleaned, words = clean(page, cfg)
        Image.fromarray(cleaned).resize((cleaned.shape[1] * 4, cleaned.shape[0] * 4), Image.NEAREST).save(
            f"/tmp/q64sheets/map_clean_{name.replace(' ', '_')}.png")
        big = draw_words(upscale_page(cleaned), words)
        big.save(f"/tmp/q64sheets/map_final_{name.replace(' ', '_')}.png")
        print(name, words)
