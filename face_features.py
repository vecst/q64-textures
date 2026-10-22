#!/usr/bin/env python3
"""Face variants that read at play distance: change the features, not just the skin.

A ("klein"):  FLUX.2 Klein's stylized eyes/brows/lips, used only inside the original feature areas (slightly grown),
              over the painted-skin face. Nothing outside the original features can change.
B ("eyes"):   procedural eye paint on the painted-skin face: iris gradient, catchlight, soft upper-lid shade.

usage: face_features.py proto            comparison at in-game size and large
       face_features.py pack <variant>   build a Faces pack (variant: klein, eyes, both)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import face_material as fm
import realify as rf

ROOT = Path.home() / "games/q64-textures"
KLEIN = ROOT / "stylize" / "source_klein"


def _feature_mask(orig):
    """Eyes, brows, lashes and mouth in the original, detected directly: dark lines, eye whites, and colours much more
    saturated than the skin. Blush, soft shading and hair are not features, so nothing gets painted there."""
    lab = fm._lab(np.asarray(orig.convert("RGB")))
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)
    skin, hair = fm.regions(orig)
    ref = lab[skin].mean(axis=0) if skin.any() else np.median(lab.reshape(-1, 3), axis=0)
    ref_chroma = float(np.hypot(ref[1], ref[2]))
    dark = L < ref[0] - 28
    white = (L > max(85.0, ref[0] + 5)) & (chroma < 12)
    hue_far = np.hypot(a - ref[1], b - ref[2]) > 30  # blue/green/purple irises, red lips; pink blush stays close
    saturated = (chroma > ref_chroma + 18) & hue_far
    feat = (dark | white | saturated) & ~hair
    # Drop single stray texels (noise), keep everything else.
    from scipy import ndimage
    labels, n = ndimage.label(feat)
    sizes = ndimage.sum(feat, labels, range(1, n + 1))
    keep = np.isin(labels, [i + 1 for i, s in enumerate(sizes) if s >= 2])
    return keep


def _grow(mask, texels):
    m = Image.fromarray((mask * 255).astype(np.uint8))
    for _ in range(texels):
        m = m.filter(ImageFilter.MaxFilter(3))
    return np.asarray(m) > 0


def _up(mask, size, feather):
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return (np.asarray(m).astype(np.float32) / 255.0)[..., None]


def variant_klein(h, base):
    orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
    k_path = KLEIN / f"{h}.png"
    if not k_path.exists():
        return base
    size = base.size
    texel = size[0] / orig.width
    klein = np.asarray(Image.open(k_path).convert("RGB").resize(size, Image.LANCZOS)).astype(np.float32)
    feat = _feature_mask(orig)
    m = _up(_grow(feat, 1), size, texel * 0.4)
    b = np.asarray(base).astype(np.float32)
    out = b[..., :3] * (1 - m) + klein * m
    return Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), b[..., 3].astype(np.uint8)]), "RGBA")


def _iris_mask(orig):
    """Saturated, mid-dark colour blobs inside feature areas that sit next to bright eye-white texels."""
    rgb = np.asarray(orig.convert("RGB")).astype(np.float32)
    lab = fm._lab(np.asarray(orig.convert("RGB")))
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    feat = _feature_mask(orig)
    white = (lab[..., 0] > 80) & (chroma < 15)
    near_white = _grow(white, 2)
    iris = feat & (chroma > 25) & (lab[..., 0] > 15) & (lab[..., 0] < 70) & near_white
    return iris


def variant_eyes(h, base):
    orig = Image.open(rf.DECODED / f"{h}.png").convert("RGBA")
    iris = _iris_mask(orig)
    if iris.sum() == 0:
        return base
    size = base.size
    texel = size[0] / orig.width
    b = np.asarray(base).astype(np.float32)
    out = b[..., :3].copy()
    # Label iris blobs and paint each one.
    from scipy import ndimage  # noqa: WPS433 (ComfyUI's venv ships scipy)
    labels, n = ndimage.label(iris)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if len(ys) < 2:
            continue
        cy, cx = (ys.mean() + 0.5) * texel, (xs.mean() + 0.5) * texel
        ry = max(1.0, (ys.max() - ys.min() + 1) / 2) * texel
        rx = max(1.0, (xs.max() - xs.min() + 1) / 2) * texel
        blob = _up(labels == i, size, texel * 0.3)[..., 0]
        d = np.clip(np.hypot((yy - cy) / ry, (xx - cx) / rx), 0, 1)
        # Iris: darker rim, brighter lower centre (painted look); stays within the original iris shape.
        lift = (1.0 - d) * 28.0 + np.clip((yy - cy) / ry, 0, 1) * 18.0 - d ** 3 * 30.0
        out += (blob * lift)[..., None]
        # Catchlight: small bright dot upper-left of the iris centre.
        hx, hy = cx - rx * 0.35, cy - ry * 0.4
        r = max(1.2, min(rx, ry) * 0.28)
        dot = np.clip(1.0 - np.hypot(xx - hx, yy - hy) / r, 0, 1) ** 1.5
        out = out * (1 - (dot * blob)[..., None]) + 255.0 * (dot * blob)[..., None]
        # Soft upper-lid shade across the top of the iris.
        lid = np.clip(1.0 - (yy - (cy - ry)) / (ry * 0.7), 0, 1) * blob
        out -= (lid * 35.0)[..., None]
    return Image.fromarray(np.dstack([out.clip(0, 255).astype(np.uint8), b[..., 3].astype(np.uint8)]), "RGBA")


def make(h, variant):
    base = fm.stylize_face(h, fm.STRENGTH["bold"])[0]
    if variant in ("klein", "both"):
        base = variant_klein(h, base)
    if variant in ("eyes", "both"):
        base = variant_eyes(h, base)
    return base


def proto():
    idxs = [93, 346, 174, 195, 475, 437, 53, 376]
    cols = ["HD Base", "Faces Bold (now)", "A: Klein features", "B: painted eyes", "A + B"]
    big, small = 230, 70
    cell_w = big + small + 14
    sheet = Image.new("RGB", (len(cols) * cell_w + 10, 22 + len(idxs) * (big + 8)), (26, 26, 32))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((10 + c * cell_w, 4), name + "  (big + in-game size)", fill=(255, 220, 120))
    for r, i in enumerate(idxs):
        h = rf.CANDIDATES[i]
        bold = fm.stylize_face(h, fm.STRENGTH["bold"])[0]
        ims = [Image.open(rf.HD_PACK / f"{h}.png").convert("RGBA"), bold, variant_klein(h, bold),
               variant_eyes(h, bold), variant_eyes(h, variant_klein(h, bold))]
        for c, im in enumerate(ims):
            k = big / max(im.size)
            b = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
            s = im.resize((max(1, int(im.width * small / max(im.size))), max(1, int(im.height * small / max(im.size)))),
                          Image.LANCZOS)
            x, y = 10 + c * cell_w, 22 + r * (big + 8)
            for im2, pos in ((b, (x, y)), (s, (x + big + 6, y + big - s.height))):
                bg = Image.new("RGBA", im2.size, (26, 26, 32, 255))
                bg.alpha_composite(im2)
                sheet.paste(bg.convert("RGB"), pos)
    out = ROOT / "stylize" / "face_features.png"
    sheet.save(out)
    print(f"sheet: {out}")


def pack(variant):
    out_dir = ROOT / "stylize" / f"faces_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for h, cat, desc, info in rf.entries({"F"}):
        p = out_dir / f"{h}.png"
        make(h, variant).save(p)
        files[h] = p
    names = {"klein": "Painted Features", "eyes": "Painted Eyes", "both": "Painted Features + Eyes"}
    rf.write_pack(f"Quest64Stylized-Faces-{variant}", f"Faces: Stylized ({names[variant]})",
                  "Painted skin plus stylized features kept inside the original eye/brow/mouth shapes. "
                  "Use ONE Faces pack at a time.", files)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "proto"
    if cmd == "proto":
        proto()
    elif cmd == "pack":
        pack(sys.argv[2])
    else:
        sys.exit(__doc__)
