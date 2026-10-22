#!/usr/bin/env python3
"""Convert a PNG texture-pack folder into a mipmapped DDS pack (RGBA8, full mip chain) for RT64.

Mipmaps remove distant-texture shimmer and let RT64 use anisotropic filtering on replacements. Mips are averaged in
linear light (so bright detail isn't darkened at distance) with premultiplied alpha, and cut-out textures keep their
alpha-test coverage at every level (so foliage/fences don't thin out and vanish far away).

usage: make_dds.py <src_pack_dir> <dst_pack_dir>
Copies mod.json / rt64.json; writes <name>.dds next to nothing else (RT64 prefers .dds over .png for the same path).
"""
import json
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA_REF = 0.5


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def downsample(a):
    """2x box filter (odd edges duplicated) on a float HxWxC array."""
    h, w = a.shape[:2]
    if h % 2 and h > 1:
        a = np.concatenate([a, a[-1:]], 0)
    if w % 2 and w > 1:
        a = np.concatenate([a, a[:, -1:]], 1)
    if a.shape[0] > 1:
        a = 0.5 * (a[0::2] + a[1::2])
    if a.shape[1] > 1:
        a = 0.5 * (a[:, 0::2] + a[:, 1::2])
    return a


def coverage(alpha, ref):
    return float((alpha > ref).mean())


def preserve_coverage(alpha, target):
    """Scale alpha so the fraction above ALPHA_REF matches the top level (binary search on the scale)."""
    lo, hi = 0.0, 8.0
    for _ in range(16):
        mid = (lo + hi) / 2
        if coverage(np.clip(alpha * mid, 0, 1), ALPHA_REF) < target:
            lo = mid
        else:
            hi = mid
    return np.clip(alpha * hi, 0, 1)


def fill_transparent(rgba8, steps=24):
    """Colour for fully transparent texels: spread the neighbouring opaque colours outward (then the mean colour).
    Never black (the game draws some textures ignoring alpha) and never the artists' key colour (bright pink under
    cut-outs, which bilinear filtering would blend into the edges)."""
    known = rgba8[..., 3] > 0
    if known.all() or not known.any():
        return rgba8
    rgb = rgba8[..., :3].astype(np.float32)
    rgb[~known] = 0
    k = known.copy()
    for _ in range(steps):
        acc, cnt = np.zeros_like(rgb), np.zeros(k.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            kk = np.roll(k, (dy, dx), (0, 1))
            acc += np.roll(rgb, (dy, dx), (0, 1)) * kk[..., None]
            cnt += kk
        grow = ~k & (cnt > 0)
        if not grow.any():
            break
        rgb[grow] = acc[grow] / cnt[grow, None]
        k |= grow
    rgb[~k] = rgb[known].mean(0)
    out = rgba8.copy()
    out[..., :3] = np.clip(rgb, 0, 255).round().astype(np.uint8)
    return out


def build_mips(img):
    rgba = fill_transparent(np.asarray(img.convert("RGBA"))).astype(np.float64) / 255.0
    alpha = rgba[..., 3:4]
    cut = bool((alpha < 1).any())
    binary = cut and bool(np.isin(np.round(alpha * 255), (0, 255)).all())
    target = coverage(alpha[..., 0], ALPHA_REF) if binary else None
    plain = srgb_to_linear(rgba[..., :3])
    lin = plain * alpha  # premultiplied linear colour
    levels = []
    cur_lin, cur_a, cur_plain = lin, alpha, plain
    while True:
        a = cur_a
        # Fully transparent texels keep the filled colour (plain average), never black: the game sometimes draws a
        # texture opaque and ignores its alpha, and black there shows up as black patches (Grand Abbott's robe).
        col = np.where(a > 1e-6, cur_lin / np.maximum(a, 1e-6), cur_plain)
        out_a = a[..., 0]
        if binary and len(levels) > 0:
            out_a = preserve_coverage(out_a, target)
        px = np.dstack([linear_to_srgb(col), out_a[..., None]])
        levels.append(np.round(px * 255).astype(np.uint8))
        if cur_lin.shape[0] == 1 and cur_lin.shape[1] == 1:
            break
        cur_lin, cur_a, cur_plain = downsample(cur_lin), downsample(cur_a), downsample(cur_plain)
    return levels


def dds_header(w, h, mips):
    DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PITCH, DDSD_PIXELFORMAT, DDSD_MIPMAPCOUNT = 1, 2, 4, 8, 0x1000, 0x20000
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT
    DDPF_ALPHAPIXELS, DDPF_RGB = 0x1, 0x40
    pf = struct.pack("<II4sIIIII", 32, DDPF_ALPHAPIXELS | DDPF_RGB, b"\0\0\0\0", 32,
                     0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)  # R8G8B8A8_UNORM byte order
    DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
    caps = DDSCAPS_COMPLEX | DDSCAPS_TEXTURE | DDSCAPS_MIPMAP
    hdr = struct.pack("<IIIIIII", 124, flags, h, w, w * 4, 0, mips) + b"\0" * 44 + pf + struct.pack("<IIIII", caps, 0, 0, 0, 0)
    assert len(hdr) == 124
    return b"DDS " + hdr


def convert(args):
    src, dst = args
    levels = build_mips(Image.open(src))
    h, w = levels[0].shape[:2]
    with open(dst, "wb") as f:
        f.write(dds_header(w, h, len(levels)))
        for lv in levels:
            f.write(lv.tobytes())
    return dst.stat().st_size


def main():
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    if dst_dir.parent.name != "dds" or dst_dir.name in ("", "/"):
        sys.exit(f"refusing to write outside a dds/<pack> folder: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stale in dst_dir.rglob("*.dds"):  # previous run of this pack only
        stale.unlink()
    for meta in ("mod.json", "rt64.json"):
        if (src_dir / meta).exists():
            shutil.copy2(src_dir / meta, dst_dir / meta)
    jobs = [(p, dst_dir / (p.relative_to(src_dir).with_suffix(".dds"))) for p in sorted(src_dir.rglob("*.png"))]
    for _, d in jobs:
        d.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        total = sum(ex.map(convert, jobs, chunksize=8))
    print(f"{len(jobs)} textures -> {dst_dir} ({total / 2**20:.0f} MB uncompressed DDS)")


if __name__ == "__main__":
    main()
