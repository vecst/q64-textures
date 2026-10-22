#!/usr/bin/env python3
"""Stylized-realism faces as flat painted surfaces.

Face textures are UV layouts (an eye here, mouth scraps there) wrapped onto low-poly heads, so letting a model "see a
face" makes it invent noses and eyes. Instead: generate face-free material swatches (painted skin, hair strands), find
the skin and hair regions in each original texture, and add only the swatch's fine detail to those regions. Eyes,
brows, mouths and any scraps stay exactly as HDCube drew them; no shading is added.

usage: face_material.py swatches      generate the material swatches
       face_material.py proto         comparison sheet for a few faces
       face_material.py pack          build Stylized Faces (Subtle) and (Bold)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import realify as rf

ROOT = Path.home() / "games/q64-textures"
SWATCHES = ROOT / "stylize" / "swatches"
OUT = ROOT / "stylize" / "faces_material"
STRENGTH = {"subtle": 0.8, "bold": 2.0}
SWATCH_PROMPTS = {
    "skin": ("seamless close-up texture of stylized hand-painted human skin, fine pores, soft peach fuzz, a few faint "
             "freckles, subtle painterly brush grain, even flat albedo, no features, no lighting, no shadows, "
             "modern stylized 3D game asset, uniform pale peach colour"),
    "hair": ("seamless close-up texture of stylized hand-painted hair, many fine parallel vertical hair strands, soft "
             "painterly brush strokes, flat albedo, no lighting, no shadows, uniform medium brown colour"),
}
SWATCH_NEG = "face, eye, nose, mouth, person, portrait, object, text, watermark, dramatic lighting, shadow, vignette, blurry"


def wf_txt2img(prompt, seed):
    return {
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Juggernaut-XI-byRunDiffusion.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": SWATCH_NEG, "clip": ["3", 1]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m",
                                                   "scheduler": "karras", "denoise": 1.0, "model": ["3", 0],
                                                   "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "q64real/swatch"}},
    }, "10"


def make_swatches():
    SWATCHES.mkdir(parents=True, exist_ok=True)
    for name, prompt in SWATCH_PROMPTS.items():
        wf, save = wf_txt2img(prompt, 11)
        rf.fetch(rf.submit(wf), save).save(SWATCHES / f"{name}.png")
        print(f"swatch {name} done", flush=True)


# --- regions -------------------------------------------------------------------------------------------------------

def _lab(rgb):
    """sRGB (uint8) -> CIE Lab with L in 0..100."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]], np.float32)
    xyz /= np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def regions(orig_rgba):
    """Skin = the dominant warm, light colour; hair = large darker/browner areas; everything else is a feature."""
    lab = _lab(np.asarray(orig_rgba.convert("RGB")))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    warm_light = (L > 50) & (a > 2) & (b > 2)
    if warm_light.sum() < 8:
        warm_light = np.ones_like(L, bool)
    skin_ref = np.median(lab[warm_light], axis=0)
    skin = np.linalg.norm(lab - skin_ref, axis=-1) < 12
    # Hair: clearly darker than skin, not a thin line (opened mask), and not saturated blue/purple eye/brow paint.
    darker = (L < skin_ref[0] - 15) & (np.abs(a) < 40) & (b > -5)
    hair = np.asarray(Image.fromarray((darker * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
                      .filter(ImageFilter.MaxFilter(3))) > 0
    # Keep a 1-texel margin around everything that is neither skin nor hair (features stay crisp).
    feature = ~(skin | hair)
    grow = np.asarray(Image.fromarray((feature * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))) > 0
    return skin & ~grow, hair & ~grow


def _soft(mask, size, texel):
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    m = m.filter(ImageFilter.GaussianBlur(texel * 0.5))
    return (np.asarray(m).astype(np.float32) / 255.0)[..., None]


def _detail(swatch, size, seed, radius):
    rng = np.random.default_rng(seed)
    sw = swatch.resize((max(size[0], 512), max(size[1], 512)), Image.LANCZOS) if swatch.width < size[0] else swatch
    x = int(rng.integers(0, sw.width - size[0] + 1))
    y = int(rng.integers(0, sw.height - size[1] + 1))
    crop = np.asarray(sw.crop((x, y, x + size[0], y + size[1])).convert("RGB")).astype(np.float32)
    low = np.asarray(Image.fromarray(crop.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))).astype(np.float32)
    d = crop - low
    lum = d.mean(axis=-1, keepdims=True)
    return 0.75 * lum + 0.25 * d  # mostly brightness grain, a little colour (freckles)


def _blurred_noise(rng, size, sigma):
    n = rng.normal(0.0, 1.0, (size[1], size[0])).astype(np.float32)
    img = Image.fromarray(((n * 40) + 128).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    a = np.asarray(img).astype(np.float32) - 128.0
    return a / (a.std() + 1e-6)


def skin_detail(size, seed, texel):
    """Procedural stylized skin: soft warm/cool mottling, sparse soft pores and a velvety grain (RGB offsets)."""
    rng = np.random.default_rng(seed)
    mottle = _blurred_noise(rng, size, sigma=max(3.0, texel * 1.6))
    warm = _blurred_noise(rng, size, sigma=max(4.0, texel * 2.2))
    grain = _blurred_noise(rng, size, sigma=0.7)
    pores = np.zeros((size[1], size[0]), np.float32)
    count = int(size[0] * size[1] / 260)
    ys, xs = rng.integers(0, size[1], count), rng.integers(0, size[0], count)
    pores[ys, xs] = 1.0
    pores = np.asarray(Image.fromarray((pores * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.1))).astype(np.float32)
    pores /= pores.max() + 1e-6
    lum = 3.0 * mottle + 1.6 * grain - 9.0 * pores
    tint = np.stack([2.0 * warm, 0.4 * warm, -1.6 * warm], -1)  # warm patches lean red/orange, cool lean blue
    return lum[..., None] + tint


def stylize_face(h, strength):
    hd = Image.open(rf.HD_PACK / f"{h}.png").convert("RGBA")
    orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
    scale = min(8.0, rf.MAX_SIDE / max(orig.width, orig.height))
    size = (round(orig.width * scale), round(orig.height * scale))
    texel = size[0] / orig.width
    base = np.asarray(hd.resize(size, Image.LANCZOS)).astype(np.float32)
    skin, hair = regions(orig)
    seed = int(h[:8], 16)
    out = base[..., :3].copy()
    skin_d = skin_detail(size, seed, texel)
    hair_d = _detail(Image.open(SWATCHES / "hair.png"), size, seed + 1, radius=max(2.0, texel * 0.4))
    out += strength * _soft(skin, size, texel) * np.clip(skin_d, -20, 20)
    out += strength * _soft(hair, size, texel) * np.clip(hair_d, -26, 26)
    return Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), base[..., 3].astype(np.uint8)]), "RGBA"), skin, hair


def proto():
    from stylize import SOURCES, PROTO_FACES
    idxs = PROTO_FACES
    meta = json.loads((rf.DECODED / "meta.json").read_text())
    cols = ["Original", "Regions", "HDCube", "Subtle", "Bold", "Klein (previous bold)"]
    cell, gap = 210, 6
    sheet = Image.new("RGB", (len(cols) * (cell + gap) + gap, 26 + len(idxs) * (cell + 8)), (26, 26, 32))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((gap + c * (cell + gap), 6), name, fill=(255, 220, 120))
    y = 24
    for i in idxs:
        h = rf.CANDIDATES[i]
        orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
        subtle, skin, hair = stylize_face(h, STRENGTH["subtle"])
        bold, _, _ = stylize_face(h, STRENGTH["bold"])
        reg = np.zeros(skin.shape + (3,), np.uint8)
        reg[skin] = (240, 190, 160)
        reg[hair] = (120, 70, 40)
        klein = SOURCES["klein"] / f"{h}.png"
        row = [orig, Image.fromarray(reg), Image.open(rf.HD_PACK / f"{h}.png"), subtle, bold,
               Image.open(klein) if klein.exists() else None]
        for c, im in enumerate(row):
            if im is None:
                continue
            im = im.convert("RGBA")
            k = min(cell / im.width, cell / im.height)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.NEAREST if c < 2 else Image.LANCZOS)
            bg = Image.new("RGBA", im.size, (26, 26, 32, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert("RGB"), (gap + c * (cell + gap), y))
        y += cell + 8
    sheet.save(ROOT / "stylize" / "faces_material.png")
    print("sheet done")


def build_packs():
    for variant, strength in STRENGTH.items():
        files = {}
        (OUT / variant).mkdir(parents=True, exist_ok=True)
        for h, cat, desc, info in rf.entries({"F"}):
            p = OUT / variant / f"{h}.png"
            stylize_face(h, strength)[0].save(p)
            files[h] = p
        rf.write_pack(f"Quest64Stylized-Faces{variant.capitalize()}", f"Faces: Stylized ({variant.capitalize()})",
                      "HDCube faces with hand-painted skin and hair material added only to skin and hair regions; "
                      "eyes, brows and mouths unchanged.", files)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "proto"
    {"swatches": make_swatches, "proto": proto, "pack": build_packs}.get(cmd, lambda: sys.exit(__doc__))()
