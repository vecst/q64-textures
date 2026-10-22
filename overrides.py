#!/usr/bin/env python3
"""Per-category model overrides for the HD Base pack (applied after the main HDCube pass).

  text   -> 4x-NXbrz       (shape-preserving: no redrawn letters)
  art    -> 4x-UltraSharpV2 (title/menu art reassembled from strips, large UI panels: keeps canvas/fabric as fabric)

usage: overrides.py [pack_dir]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import upscale as up

R = Path(__file__).resolve().parent
PY = sys.executable
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
PACK = Path(_args[0]) if _args else Path.home() / ".config/Quest64Recompiled/mods/Quest64HD-HDCube3"


def art_list(meta, text):
    """Images the game draws in strips (title/menu art) plus large opaque-ish UI panels."""
    picks = set()
    for unit in up.group_strips(sorted(meta), meta):
        if len(unit) < 2 or any(h in text for h, _, _ in unit):
            continue
        width = max(x + meta[h]["w"] for h, x, _ in unit)
        if width >= 128:
            picks.update(h for h, _, _ in unit)
    ui = set(json.loads((R / "ui_textures.json").read_text())) if (R / "ui_textures.json").exists() else set()
    for h in ui - text:
        v = meta[h]
        if max(v["w"], v["h"]) >= 64 and v["fmt"] not in (3, 4):  # not soft intensity effects
            picks.add(h)
    return sorted(picks)


def cleaned_text_dir(meta, text):
    """Copy of the decoded text glyphs with leftovers erased. Glyphs of one font (same box size and palette) share
    a glyph height; rows below it hold the next glyph or stale TMEM, so they are cleared for the whole font."""
    import numpy as np
    from PIL import Image
    from collections import Counter

    out = R / "decoded_text"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    fonts = {}
    for h in text:
        v = meta[h]
        if v["w"] <= up.GLYPH_MAX[0] and v["h"] <= up.GLYPH_MAX[1]:
            fonts.setdefault((v["w"], v["h"], (v.get("load") or {}).get("pal")), []).append(h)
    for h in text:
        rgba = np.asarray(Image.open(R / "decoded" / f"{h}.png").convert("RGBA")).copy()
        v = meta[h]
        key = (v["w"], v["h"], (v.get("load") or {}).get("pal"))
        if key in fonts:
            # The same font loaded into a shorter box (same width and palette) gives the true glyph height.
            shorter = [k[1] for k in fonts if k[0] == key[0] and k[2] == key[2] and k[2] is not None and k[1] < key[1]]
            cut = min(shorter) if shorter else None
            if cut is None and len(fonts[key]) >= 5:
                cuts = []
                for g in fonts[key]:
                    rows = (np.asarray(Image.open(R / "decoded" / f"{g}.png"))[..., 3] > 0).sum(1)
                    empty = [y for y in range(2, len(rows)) if rows[y] == 0]
                    if empty:
                        cuts.append(empty[0])
                common, n = Counter(cuts).most_common(1)[0] if cuts else (None, 0)
                cut = common if n >= len(fonts[key]) // 2 else None
            if cut is not None:
                rgba[cut:, :, 3] = 0
            rgba = up.clean_glyph_edges(rgba)
        Image.fromarray(rgba).save(out / f"{h}.png")
    (out / "meta.json").write_text(json.dumps(meta))
    return out


def run(name, model, hashes, extra=(), src="decoded"):
    out = Path("/tmp/q64-override") / name
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([PY, "upscale.py", str(src), name, "--model", model, "--mods-dir", str(out), "--only",
                    ",".join(hashes), *extra], cwd=R, check=True)
    n = 0
    for p in (out / f"Quest64HD-{name}").glob("*.png"):
        shutil.copy2(p, PACK / p.name)
        n += 1
    print(f"{name}: {n} textures from {model} -> {PACK}")


def main():
    meta = json.loads((R / "decoded/meta.json").read_text())
    text = set(json.loads((R / "text_textures.json").read_text()))
    art = art_list(meta, text)
    (R / "art_textures.json").write_text(json.dumps(art, indent=0))
    if "--text-only" not in sys.argv:
        run("OverrideArt", "4x-UltraSharpV2.safetensors", art)
    run("OverrideText", "4x-NXbrz.safetensors", sorted(text), ["--no-group"], src=cleaned_text_dir(meta, text))


if __name__ == "__main__":
    main()
