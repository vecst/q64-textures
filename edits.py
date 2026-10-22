#!/usr/bin/env python3
"""Hand-edit workflow for the HD Text & UI pack (edit in GIMP, rebuild, done).

  edits.py export   copy every Text & UI texture into edit/text-ui/ as plain PNGs (6x), plus a matching
                    _reference/ copy of the original pixel art at the same size. Strip-drawn art (title logo, menus)
                    is exported as ONE assembled image and sliced back automatically. Files you have edited are
                    never overwritten; unedited ones are refreshed from the latest build.
  edits.py apply    (called by build_textui.py) put every edited file into the staging pack.

A file counts as edited when a <name>_edit.png sits next to it (that copy always wins), or when the file itself
differs from what export wrote (md5 in edit/text-ui/manifest.json). To throw an edit away, delete the _edit file
(or delete an overwritten file and run export again).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import upscale as up  # noqa: E402

S = 6
EDIT = R / "edit/text-ui"
STAGING = R / "staging/Quest64HD-TextUI"
MANIFEST = EDIT / "manifest.json"
FONT_NAMES = ["font-1-digits-symbols", "font-2-lowercase", "font-3-uppercase"]


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def sets():
    meta = json.loads((R / "decoded/meta.json").read_text())
    text = json.loads((R / "text_textures.json").read_text())
    cand = sorted((set(json.loads((R / "ui_textures.json").read_text()))
                   | set(json.loads((R / "art_textures.json").read_text()))) - set(text))
    textui = set(json.loads((R / "textui_textures.json").read_text()))
    ui = sorted(textui - set(text))
    return meta, text, cand, ui


def entries():
    """name -> list of (hash, x, y) in source pixels."""
    meta, text, cand, ui = sets()
    out = {}
    for i, h in enumerate(text):
        m = meta[h]
        name = f"font/{FONT_NAMES[i]}" if i < 3 else f"text/text-{i:03d}-{m['w']}x{m['h']}"
        out[f"{name}__{h}.png"] = [(h, 0, 0)]
    n = 0
    for u in up.group_strips(ui, meta):
        if len(u) > 1:
            w = max(x + meta[h]["w"] for h, x, _ in u)
            hh = max(y + meta[h]["h"] for h, _, y in u)
            out[f"art/art-{n:02d}-{w}x{hh}__{u[0][0]}.png"] = u
            n += 1
        else:
            h = u[0][0]
            num = f"{cand.index(h):03d}" if h in cand else "x"  # x = added from a later dump
            out[f"ui/ui-{num}-{meta[h]['w']}x{meta[h]['h']}__{h}.png"] = u
    return meta, text, out


def assemble(unit, meta, get):
    w = max(x + meta[h]["w"] for h, x, _ in unit) * S
    hh = max(y + meta[h]["h"] for h, _, y in unit) * S
    c = Image.new("RGBA", (w, hh), (0, 0, 0, 0))
    for h, x, y in unit:
        im = get(h)
        size = (meta[h]["w"] * S, meta[h]["h"] * S)
        if im.size != size:
            im = im.resize(size, Image.BICUBIC)
        c.alpha_composite(im, (x * S, y * S))
    return c


def export():
    meta, text, ents = entries()
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    textset = set(text)

    def hd(h):
        return Image.open(STAGING / f"{h}.png").convert("RGBA")

    def ref(h):
        im = Image.open(R / ("decoded_text" if h in textset else "decoded") / f"{h}.png").convert("RGBA")
        return im.resize((im.width * S, im.height * S), Image.NEAREST)

    kept = written = 0
    try:
        _export_all(ents, man, meta, hd, ref)
    finally:  # always record what was written, so a crash never makes fresh exports look like hand edits
        MANIFEST.write_text(json.dumps(man, indent=1))


def _export_all(ents, man, meta, hd, ref):
    kept = written = 0
    for name, unit in ents.items():
        p = EDIT / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and name in man and md5(p) != man[name]["md5"]:
            kept += 1
            continue
        assemble(unit, meta, hd).save(p)
        r = EDIT / "_reference" / name
        r.parent.mkdir(parents=True, exist_ok=True)
        assemble(unit, meta, ref).save(r)
        man[name] = {"md5": md5(p), "slices": [list(s) for s in unit]}
        written += 1
    print(f"exported {written} files to {EDIT} ({kept} edited files left untouched)")


def apply(pack=STAGING):
    if not MANIFEST.exists():
        return []
    meta = json.loads((R / "decoded/meta.json").read_text())
    man = json.loads(MANIFEST.read_text())
    done = []
    for name, e in man.items():
        p = EDIT / name
        alt = p.with_name(p.stem + "_edit.png")  # saved next to the original as <name>_edit.png: always wins
        if alt.exists():
            p = alt
        elif not p.exists() or md5(p) == e["md5"]:
            continue
        im = Image.open(p).convert("RGBA")
        unit = e["slices"]
        w = max(x + meta[h]["w"] for h, x, _ in unit) * S
        hh = max(y + meta[h]["h"] for h, _, y in unit) * S
        if im.size != (w, hh):  # drawn at another size: scale to the export size (keep the aspect ratio!)
            im = im.resize((w, hh), Image.LANCZOS)
        a = np.asarray(im)
        for h, x, y in unit:
            piece = a[y * S:(y + meta[h]["h"]) * S, x * S:(x + meta[h]["w"]) * S]
            Image.fromarray(np.ascontiguousarray(piece), "RGBA").save(pack / f"{h}.png")
        done.append(p.name)
    print(f"applied {len(done)} hand edits" + (": " + ", ".join(Path(n).name for n in done) if done else ""))
    return done


if __name__ == "__main__":
    {"export": export, "apply": apply}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
