#!/usr/bin/env python3
"""Decode RT64 texture dumps (<hash>.v5.tmem + <hash>.v5.tile.json) into PNGs.

A straight port of RT64's sampleTMEM() (src/shaders/TextureDecoder.hlsli), so the
decoded image is exactly what the game samples. Output files are named by RT64 hash.

usage: decode.py <dump_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TMEM_BYTES, TMEM_PALETTE, MASK8, MASK16 = 0x1000, 0x800, 0xFFF, 0x7FF
FMT_RGBA, FMT_YUV, FMT_CI, FMT_IA, FMT_I = range(5)
SIZ_4, SIZ_8, SIZ_16, SIZ_32 = range(4)
TLUT_NONE, TLUT_RGBA16, TLUT_IA16 = range(3)  # RT64's LoadTLUT enum
TLUT_NAMES = {"None": TLUT_NONE, "RGBA16": TLUT_RGBA16, "IA16": TLUT_IA16}  # tile.json stores it by name


def rgba16(v):
    r, g, b = (v >> 11) & 0x1F, (v >> 6) & 0x1F, (v >> 1) & 0x1F
    return np.stack([(r << 3) | (r >> 2), (g << 3) | (g >> 2), (b << 3) | (b >> 2), (v & 1) * 255], -1)


def ia16(v):
    i, a = (v >> 8) & 0xFF, v & 0xFF
    return np.stack([i, i, i, a], -1)


def decode(tmem, tile, width, height, tlut):
    fmt, siz, pal = tile["fmt"], tile["siz"], tile["palette"]
    address, stride = tile["tmem"] << 3, tile["line"] << 3
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    odd_row, odd_col = (y & 1).astype(bool), (x & 1).astype(bool)
    is_rgba32 = fmt == FMT_RGBA and siz == SIZ_32
    uses_tlut = tlut > 0
    shift = 2 if is_rgba32 else siz
    mask = MASK16 if (is_rgba32 or uses_tlut) else MASK8

    def load(rel, or_addr=0):
        if stride:
            row_start = (rel // stride) * stride
            swapped = address + row_start + ((((rel - row_start) // 4) ^ 1) * 4) + (rel & 3)
        else:
            swapped = address + rel
        final = np.where(odd_row, swapped, address + rel)
        return tmem[((final & mask) | or_addr) & MASK8].astype(np.int64)

    pa = y * stride + ((x << shift) >> 1)
    p0, p1 = load(pa), load(pa + 1)
    p4 = (p0 >> np.where(odd_col, 0, 4)) & 0xF

    if uses_tlut:
        paddr = TMEM_PALETTE + ((pal << 7) + (p4 << 3) if siz == SIZ_4 else (p0 << 3))
        pv = (tmem[paddr & MASK8].astype(np.int64) << 8) | tmem[(paddr + 1) & MASK8]
        if tlut == TLUT_RGBA16:
            return rgba16(pv)
        if tlut == TLUT_IA16:
            return ia16(pv)
        return None

    pa2 = pa if is_rgba32 else pa + 2
    or2 = TMEM_BYTES >> 1 if is_rgba32 else 0
    p2, p3 = load(pa2, or2), load(pa2 + 1, or2)

    def gray(i, a=None):
        return np.stack([i, i, i, i if a is None else a], -1)

    if siz == SIZ_4:
        if fmt == FMT_CI:  # CI without TLUT: palette index as intensity
            return gray((pal << 4) | p4)
        if fmt == FMT_IA:
            i = p4 & 0b1110
            i = (i << 4) | (i << 1) | (i >> 2)
            return gray(i, (p4 & 1) * 255)
        if fmt in (FMT_I, FMT_RGBA):
            return gray((p4 << 4) | p4)
    elif siz == SIZ_8:
        if fmt == FMT_IA:
            i, a = (p0 >> 4) & 0xF, p0 & 0xF
            return gray((i << 4) | i, (a << 4) | a)
        if fmt in (FMT_I, FMT_RGBA, FMT_CI):
            return gray(p0)
    elif siz == SIZ_16:
        v = p1 | (p0 << 8)
        if fmt == FMT_RGBA:
            return rgba16(v)
        if fmt == FMT_IA:
            return ia16(v)
        if fmt in (FMT_CI, FMT_I):
            return np.stack([p0, p1, p0, p1], -1)
    elif siz == SIZ_32:
        if fmt == FMT_RGBA:
            return np.stack([p0, p1, p2, p3], -1)
        if fmt in (FMT_CI, FMT_IA, FMT_I):
            return np.where(odd_col[..., None], np.stack([p0, p1, p0, p1], -1), np.stack([p2, p3, p2, p3], -1))
    return None  # YUV / unsupported


def load_source(dump_dir, base):
    """Where in RDRAM this texture was loaded from (RT64's .rice.json), so strips of one image can be regrouped."""
    rice = dump_dir / f"{base}.rice.json"
    if not rice.exists():
        return None
    r = json.loads(rice.read_text())
    if r.get("type") != "Tile":
        return None
    pal_path = dump_dir / f"{base}.rice.palette.json"
    pal = json.loads(pal_path.read_text())["texture"]["address"] if pal_path.exists() else None
    tex, tile = r["texture"], r["tile"]
    return {"addr": tex["address"], "width": tex["width"], "siz": tex["siz"], "pal": pal,
            "x": tile["uls"] >> 2, "y": tile["ult"] >> 2}


def main(dump_dir, out_dir):
    dump_dir, out_dir = Path(dump_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, done, skipped = {}, 0, 0
    for tile_path in sorted(dump_dir.glob("*.tile.json")):
        base = tile_path.name[: -len(".tile.json")]  # "<hash>.v5"
        tmem_path = dump_dir / f"{base}.tmem"
        if not tmem_path.exists():
            continue
        info = json.loads(tile_path.read_text())
        tmem = np.frombuffer(tmem_path.read_bytes(), dtype=np.uint8)
        if tmem.size != TMEM_BYTES:
            skipped += 1
            continue
        w, h = info["width"], info["height"]
        tlut = TLUT_NAMES.get(info["tlut"], info["tlut"])
        img = decode(tmem, info["tile"], w, h, tlut)
        if img is None:
            skipped += 1
            continue
        hash_hex = base.split(".")[0]
        Image.fromarray(img.astype(np.uint8), "RGBA").save(out_dir / f"{hash_hex}.png")
        t = info["tile"]
        meta[hash_hex] = {"w": w, "h": h, "fmt": t["fmt"], "siz": t["siz"], "tlut": tlut,
                          "cms": t["cms"], "cmt": t["cmt"], "hash_version": base.split(".v")[-1]}
        load = load_source(dump_dir, base)
        if load:
            meta[hash_hex]["load"] = load
        done += 1
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"decoded {done} textures, skipped {skipped} -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
