#!/usr/bin/env python3
"""Decide which replacements are drawn as 2D (screen-space rects) and must not get RT64's half-texel shift.

Measured on matched pack-off/pack-on screenshots: unshifted UI lines up with the original within 0.05 px, while a
half-shifted 2D element (the element wheel) lands 0.5 texel off and exposes the leftover row/column at its edge.
The opaque 3D world lines up with the default half shift. Rule: transparency, UI or title/menu art -> "none".

usage: shift_rule.py [pack_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(__file__).resolve().parent
PACK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".config/Quest64Recompiled/mods/Quest64HD-HDCube3"


def main():
    meta = json.loads((R / "decoded/meta.json").read_text())
    flat = set()
    for name in ("ui_textures.json", "art_textures.json"):
        if (R / name).exists():
            flat |= set(json.loads((R / name).read_text()))
    cut = 0
    for h in meta:
        a = np.asarray(Image.open(R / "decoded" / f"{h}.png"))[..., 3]
        if a.max() > 0 and a.min() < 255:
            flat.add(h)
            cut += 1
    (R / "ui_textures.json").write_text(json.dumps(sorted(flat), indent=0))
    p = PACK / "rt64.json"
    db = json.loads(p.read_text())
    n = 0
    for t in db["textures"]:
        if t["hashes"]["rt64"] in flat:
            t["shift"] = "none"
            n += 1
        else:
            t.pop("shift", None)
    p.write_text(json.dumps(db, indent=1))
    print(f"{cut} textures with transparency; {n} of {len(db['textures'])} entries unshifted (2D); "
          f"the rest keep the default half shift (3D world)")


if __name__ == "__main__":
    main()
