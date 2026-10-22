#!/usr/bin/env python3
"""Stylized-realism via detail transfer.

1. Juggernaut XI img2img at low strength turns the HDCube texture into an aligned, hand-painted stylized-3D albedo
   (features stay where they were).
2. Frequency separation: keep HDCube's colours and shapes (low frequencies), add only the fine detail (skin grain,
   hair strands, stone pores) from the stylized version, so no lighting/shadows get baked in.
3. Optionally protect facial features (eyes, brows, mouth) by keeping HDCube wherever the original has hard lines.

usage: stylize.py proto              build the comparison sheet for a few faces + world textures
       stylize.py gen [F|S|O ...]    generate stylized sources for every labelled texture of those categories
       stylize.py pack <variant>     build a pack from the sources (variant: light, strong, protected)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import realify as rf
from labels import LABELS

ROOT = Path.home() / "games/q64-textures"
SRC_DIR = ROOT / "stylize" / "source"
RAW_DIR = ROOT / "stylize" / "raw"
NEG = ("photo, photograph, harsh shadows, dramatic lighting, ambient occlusion, specular highlights, glossy, "
       "deep eye sockets, lowres, blurry, text, watermark, frame, border")


def prompt(cat, desc):
    if cat == "F":
        return (f"stylized realistic {desc} texture, hand-painted character skin albedo like a modern stylized 3D game, "
                f"Arcane and Pixar style, soft skin pores, painted hair strands, clear expressive eyes, "
                f"flat diffuse map, no lighting, no shadows, game asset")
    return (f"stylized realistic {desc} albedo texture, hand-painted PBR game asset like a modern stylized 3D game, "
            f"rich natural surface detail, flat diffuse map, no lighting, no shadows, seamless")


# --- generation ----------------------------------------------------------------------------------------------------

def generate(entries):
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries if not (SRC_DIR / f"{e[0]}.png").exists()]
    print(f"[stylize] {len(todo)} to generate", flush=True)
    for i in range(0, len(todo), rf.BATCH):
        jobs = []
        for h, cat, desc, info in todo[i:i + rf.BATCH]:
            img, geo = rf.hd_padded(h, info, cat == "F")
            ref = rf.upload(f"{h}.png", img)
            wf, save = rf.wf_sdxl("Juggernaut-XI-byRunDiffusion.safetensors", ref, ref, prompt(cat, desc), NEG, 0.4,
                                  int(h[:8], 16))
            jobs.append((h, rf.submit(wf), save, geo))
        for h, pid, save, geo in jobs:
            try:
                raw = rf.fetch(pid, save)
                raw.save(RAW_DIR / f"{h}.png")
                rf.finish(raw, geo).save(SRC_DIR / f"{h}.png")
            except Exception as e:
                print(f"  FAILED {h}: {str(e)[:200]}", flush=True)
        print(f"  [stylize] {min(i + rf.BATCH, len(todo))}/{len(todo)}", flush=True)


# --- detail transfer -----------------------------------------------------------------------------------------------

def _arr(img):
    return np.asarray(img.convert("RGB")).astype(np.float32)


def _blur(a, radius):
    return _arr(Image.fromarray(a.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)))


def feature_mask(orig, size):
    """Hard lines in the original N64 texture (eyes, brows, mouth outlines): high local contrast."""
    lum = np.asarray(orig.convert("L")).astype(np.float32)
    gy, gx = np.gradient(lum)
    edge = (np.hypot(gx, gy) > 38).astype(np.float32)
    m = Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))  # grow by 1 texel
    m = m.resize(size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(size[0] / orig.width * 0.6))
    return (np.asarray(m).astype(np.float32) / 255.0)[..., None]


def variants(h, cat):
    hd_full = Image.open(rf.HD_PACK / f"{h}.png").convert("RGBA")
    src = Image.open(SRC_DIR / f"{h}.png").convert("RGB")
    orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
    size = src.size
    texel = size[0] / orig.width
    hd = _arr(hd_full.resize(size, Image.LANCZOS))
    s = _arr(src)
    fine = s - _blur(s, 0.6 * texel)
    mid = s - _blur(s, 1.4 * texel)
    out = {
        "light": hd + 0.7 * fine,
        "strong": hd + 1.0 * mid + 0.3 * fine,
    }
    if cat == "F":
        m = feature_mask(orig, size)
        out["protected"] = out["strong"] * (1 - m) + hd * m
    alpha = np.asarray(hd_full.resize(size, Image.LANCZOS))[..., 3]
    return {k: Image.fromarray(np.dstack([v.clip(0, 255).astype(np.uint8), alpha]), "RGBA") for k, v in out.items()}


# --- v2: richer sources + edge-aware blend -------------------------------------------------------------------------

SOURCES = {
    "jugg60": ROOT / "stylize" / "source_jugg60",
    "klein": ROOT / "stylize" / "source_klein",
}


def prompt_v2(cat, desc):
    if cat == "F":
        return (f"stylized realistic {desc}, hand-painted character skin albedo texture, visible skin pores, fine peach "
                f"fuzz, subtle freckles, individual painted hair strands, painterly brush texture, modern stylized 3D "
                f"game like Arcane, flat diffuse map, no lighting, no shadows")
    return (f"stylized realistic {desc}, hand-painted PBR albedo texture, rich tactile surface detail, painterly brush "
            f"texture, modern stylized 3D game asset, flat diffuse map, no lighting, no shadows, seamless")


def prompt_klein_texture(cat, desc):
    what = "subtle realistic skin texture and fine painted hair strands" if cat == "F" else f"realistic {desc} surface detail"
    return (f"Add {what} to this game texture in a hand-painted stylized-3D style. Do not change the drawing: keep "
            f"every shape, line, eye, mouth, color and position exactly the same. No lighting or shadows.")


def generate_v2(entries, method):
    out_dir = SOURCES[method]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries if not (out_dir / f"{e[0]}.png").exists()]
    print(f"[{method}] {len(todo)} to generate", flush=True)
    for i in range(0, len(todo), rf.BATCH):
        jobs = []
        for h, cat, desc, info in todo[i:i + rf.BATCH]:
            img, geo = rf.hd_padded(h, info, cat == "F")
            ref = rf.upload(f"{h}.png", img)
            seed = int(h[:8], 16)
            if method == "jugg60":
                wf, save = rf.wf_sdxl("Juggernaut-XI-byRunDiffusion.safetensors", ref, ref, prompt_v2(cat, desc), NEG,
                                      0.6, seed)
            else:
                wf, save = rf.wf_klein(ref, prompt_klein_texture(cat, desc), seed)
            jobs.append((h, rf.submit(wf), save, geo))
        for h, pid, save, geo in jobs:
            try:
                rf.finish(rf.fetch(pid, save), geo).save(out_dir / f"{h}.png")
            except Exception as e:
                print(f"  FAILED {h}: {str(e)[:200]}", flush=True)
        print(f"  [{method}] {min(i + rf.BATCH, len(todo))}/{len(todo)}", flush=True)


def blend_v2(h, cat, method, volume=0.4, detail=1.0):
    hd_full = Image.open(rf.HD_PACK / f"{h}.png").convert("RGBA")
    src = Image.open(SOURCES[method] / f"{h}.png").convert("RGB")
    orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
    size = src.size
    texel = size[0] / orig.width
    hd = _arr(hd_full.resize(size, Image.LANCZOS))
    s = _arr(src)
    hd_low, s_low = _blur(hd, texel), _blur(s, texel)
    base = hd + volume * (s_low - hd_low)                 # borrow soft volume, keep HDCube's colours
    fine = s - _blur(s, 0.8 * texel)
    fine = 28.0 * np.tanh(fine / 28.0)                     # soft-limit so nothing turns into a hard outline
    gy, gx = np.gradient(hd_low.mean(axis=2))
    damp = np.clip(1.0 - np.hypot(gx, gy) / 6.0, 0.15, 1.0)[..., None]  # less detail across strong edges: no halos
    out = base + detail * fine * damp
    if cat == "F":
        m = feature_mask(orig, size)
        out = out * (1 - m) + hd * m
    alpha = np.asarray(hd_full.resize(size, Image.LANCZOS))[..., 3]
    return Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), alpha]), "RGBA")


def proto2():
    picks = [entry_for(i) for i in PROTO_FACES + PROTO_WORLD]
    for method in SOURCES:
        generate_v2(picks, method)
    cols = ["Original", "HDCube", "Jugg 0.6 source", "Jugg blend", "Klein texture-edit", "Klein blend", "Cursed"]
    cell, gap = 160, 6
    W = len(cols) * (cell + gap) + gap
    sheet = Image.new("RGB", (W, 30 + len(picks) * (cell + 16)), (26, 26, 32))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((gap + c * (cell + gap), 6), name, fill=(255, 220, 120))
    y = 24
    for h, cat, desc, info in picks:
        cursed_dir = "face_real" if cat == "F" else ("klein" if cat == "S" else "jugg")
        cursed = rf.OUT / cursed_dir / f"{h}.png"
        row = [Image.open(rf.DECODED / f"{h}.png"), Image.open(rf.HD_PACK / f"{h}.png"),
               Image.open(SOURCES["jugg60"] / f"{h}.png"), blend_v2(h, cat, "jugg60"),
               Image.open(SOURCES["klein"] / f"{h}.png"), blend_v2(h, cat, "klein"),
               Image.open(cursed) if cursed.exists() else None]
        d.text((gap, y), desc, fill=(190, 190, 200))
        for c, im in enumerate(row):
            if im is None:
                continue
            im = im.convert("RGBA")
            k = min(cell / im.width, (cell - 4) / im.height)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.NEAREST if c == 0 else Image.LANCZOS)
            bg = Image.new("RGBA", im.size, (26, 26, 32, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert("RGB"), (gap + c * (cell + gap), y + 12))
        y += cell + 16
    out = ROOT / "stylize" / "comparison_v2.png"
    sheet.crop((0, 0, W, y + 4)).save(out)
    print(f"sheet: {out}")


def build_v2_packs():
    out = ROOT / "stylize" / "packs"
    bold, subtle, world = {}, {}, {}
    for h, cat, desc, info in rf.entries({"F", "S", "O"}):
        # Klein's texture edit sometimes paints whole new objects onto object textures, so objects use Juggernaut.
        method = "jugg60" if cat == "O" else "klein"
        if not (SOURCES[method] / f"{h}.png").exists():
            continue
        if cat == "F":
            bold[h] = SOURCES["klein"] / f"{h}.png"
            p = out / "faces_subtle" / f"{h}.png"
            store = subtle
        else:
            p = out / "world" / f"{h}.png"
            store = world
        p.parent.mkdir(parents=True, exist_ok=True)
        blend_v2(h, cat, method).save(p)
        store[h] = p
    rf.write_pack("Quest64Stylized-World", "World: Stylized",
                  "Hand-painted stylized-realism detail on world surfaces and objects, over HDCube's colours and shapes. "
                  "Load above HD Textures.", world)
    rf.write_pack("Quest64Stylized-FacesBold", "Faces: Stylized (Bold)",
                  "Faces repainted as modern stylized-3D characters (FLUX.2 Klein texture edit).", bold)
    rf.write_pack("Quest64Stylized-FacesSubtle", "Faces: Stylized (Subtle)",
                  "HDCube faces with painted skin texture and soft volume; features unchanged.", subtle)


# --- prototype sheet -----------------------------------------------------------------------------------------------

PROTO_FACES = [93, 53, 195, 346, 475, 174, 376, 437]
PROTO_WORLD = [9, 87, 61, 156, 21, 6]


def entry_for(idx):
    meta = json.loads((rf.DECODED / "meta.json").read_text())
    cat, desc = LABELS[idx]
    h = rf.CANDIDATES[idx]
    return h, cat, desc, meta[h]


def proto():
    picks = [entry_for(i) for i in PROTO_FACES + PROTO_WORLD]
    generate(picks)
    cell, gap = 170, 6
    face_cols = ["Original", "HDCube", "Light", "Strong", "Protected", "Cursed (photoreal)"]
    world_cols = ["Original", "HDCube", "Light", "Strong", "", "Photoreal pack"]
    rows = []
    for h, cat, desc, info in picks:
        v = variants(h, cat)
        orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
        hd = Image.open(rf.HD_PACK / f"{h}.png").convert("RGBA")
        cursed_dir = "face_real" if cat == "F" else ("klein" if cat == "S" else "jugg")
        cursed = rf.OUT / cursed_dir / f"{h}.png"
        row = [orig, hd, v["light"], v["strong"], v.get("protected"), Image.open(cursed) if cursed.exists() else None]
        rows.append((cat, desc, row))
    W = len(face_cols) * (cell + gap) + gap
    H = 24 + len(rows) * (cell + 20) + 40
    sheet = Image.new("RGB", (W, H), (26, 26, 32))
    d = ImageDraw.Draw(sheet)
    y = 4
    for c, name in enumerate(face_cols):
        d.text((gap + c * (cell + gap), y), name, fill=(255, 220, 120))
    y += 20
    for i, (cat, desc, row) in enumerate(rows):
        if i == len(PROTO_FACES):
            for c, name in enumerate(world_cols):
                d.text((gap + c * (cell + gap), y + 4), name, fill=(255, 220, 120))
            y += 24
        d.text((gap, y), desc, fill=(190, 190, 200))
        for c, im in enumerate(row):
            if im is None:
                continue
            im = im.convert("RGBA")
            k = min(cell / im.width, (cell - 14) / im.height)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.NEAREST if c == 0 else Image.LANCZOS)
            bg = Image.new("RGBA", im.size, (26, 26, 32, 255))
            bg.alpha_composite(im)
            sheet.paste(bg.convert("RGB"), (gap + c * (cell + gap), y + 14))
        y += cell + 14
    sheet = sheet.crop((0, 0, W, y + 6))
    out = ROOT / "stylize" / "comparison.png"
    sheet.save(out)
    print(f"sheet: {out}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "proto"
    if cmd == "proto":
        proto()
    elif cmd == "proto2":
        proto2()
    elif cmd == "full2":
        generate_v2(list(rf.entries({"F", "S", "O"})), "klein")
        generate_v2(list(rf.entries({"O"})), "jugg60")
        build_v2_packs()
    elif cmd == "packs2":
        build_v2_packs()
    elif cmd == "gen":
        cats = set(sys.argv[2:]) or {"F", "S", "O"}
        generate(list(rf.entries(cats)))
    elif cmd == "pack":
        variant = sys.argv[2]
        built = {}
        tmp = ROOT / "stylize" / variant
        tmp.mkdir(parents=True, exist_ok=True)
        for h, cat, desc, info in rf.entries({"F", "S", "O"}):
            if not (SRC_DIR / f"{h}.png").exists():
                continue
            v = variants(h, cat)
            img = v.get(variant) or v["strong"]
            img.save(tmp / f"{h}.png")
            built[h] = tmp / f"{h}.png"
        rf.write_pack(f"Quest64Fun-Stylized-{variant.capitalize()}", f"Stylized Realism ({variant})",
                      "HDCube colours and shapes with hand-painted realistic detail transferred on top.", built)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
