#!/usr/bin/env python3
"""Split HD Base into two packs so text/UI and the world can be treated differently.

  Quest64HD-TextUI : text glyphs, lettering, HUD, menus, icons, title/menu art -> textui.py "hybrid" at 6x
                     (deterministic: xBRZ edges + smooth gradients, no ML, so no stray noise or redrawn letters)
  Quest64HD-World  : every other HDCube3 texture (the existing DDS, minus the TextUI set)

Classification: text_textures.json, plus every 2D/transparent candidate (ui_textures + art_textures) except the
world sprites listed in WORLD below (trees, fire, smoke, waterfalls, scenery cutouts), picked by eye from the
contact sheets made by sheets.py (indices into /tmp/q64sheets/ui_index.json order = sorted candidate hashes).

usage: build_textui.py            (writes PNG staging packs; build_split.sh turns them into .rtz)
"""
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import digits  # noqa: E402
import textui  # noqa: E402
import upscale as up  # noqa: E402

SCALE = 6
FONT_COLOR = (26, 20, 16)   # rich dark brown-black
STAGING = R / "staging"
HDCUBE_DDS = R / "dds" / "Quest64HD-HDCube3"
# Painted full-screen art (title/menu backdrops: woven canvas, 255 colours) turns into xBRZ "worms"; it keeps the
# UltraSharpV2 result from the HD Base build instead.
PAINTED_SRC = Path.home() / "games/q64-mods/archive/png-packs/Quest64HD-HDCube3"
META = json.loads((R / "decoded/meta.json").read_text())

WORLD = {
    5, 6, 8, 9, 15, 18, 23, 25, 28, 30, 31, 40, 43, 44, 45, 52, 54, 55, 60, 61, 63, 70, 76, 77, 78, 80, 83, 84, 86, 90,
    99, 101, 102, 112, 118, 123, 124, 125, 133, 134, 136, 143, 144, 146, 150, 152, 155, 157, 158, 161, 164, 165, 166,
    173, 175, 178, 179, 191,
    194, 199, 200, 201, 202, 208, 210, 218, 222, 226, 232, 233, 235, 239, 244, 250, 251, 252, 253, 257, 258, 259, 262,
    263, 266, 276, 278, 281, 286, 287,
    297, 300, 302, 305, 308, 319, 321, 323, 336, 339, 344, 346, 348, 352, 353, 354, 360, 362, 365, 371, 372, 373, 375,
    378, 383,
    391, 392, 393, 395, 396, 405, 406, 408, 411, 420, 421, 422, 425, 426, 428, 429, 430, 434, 444, 445, 451, 452, 453,
    454, 455, 456, 457, 460, 462, 464, 465, 469, 478,
}


def candidates():
    text = set(json.loads((R / "text_textures.json").read_text()))
    cand = sorted((set(json.loads((R / "ui_textures.json").read_text()))
                   | set(json.loads((R / "art_textures.json").read_text()))) - text)
    return text, cand


def safe_reset(d):
    d = d.resolve()
    if d.parent != STAGING.resolve() and d.parent != (R / "dds").resolve():
        sys.exit(f"refusing to clear {d}")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def painted(canvas):
    op = canvas[..., 3] > 0
    return (op.mean() >= 0.95 and canvas.shape[0] * canvas.shape[1] >= 128 * 64
            and len(np.unique(canvas[op].view(np.uint32))) >= 128)


def hard_alpha(big, src):
    """HUD/menu art with 1-bit alpha in the original keeps 1-bit alpha: the game draws it with a dithered alpha
    test, which turns the soft (partly transparent) edges of an upscale into scattered black dots."""
    if np.isin(src[..., 3], (0, 255)).all():
        big = big.copy()
        big[..., 3] = np.where(big[..., 3] >= 128, 255, 0).astype(np.uint8)
    return big


def render_unit(job):
    unit, src, out, hard = job
    meta = META
    if len(unit) == 1:
        h = unit[0][0]
        rgba = np.asarray(Image.open(Path(src) / f"{h}.png").convert("RGBA"))
        big = textui.upscale(rgba, "hybrid", SCALE)
        Image.fromarray(hard_alpha(big, rgba) if hard else big, "RGBA").save(Path(out) / f"{h}.png")
        return 1
    canvas, _ = up.assemble(unit, Path(src), meta)
    panel = digits.PANEL_TOP in [h for h, _, _ in unit]
    if panel:  # "Select an element": erase the pixel lettering before upscaling, redraw it clean afterwards
        canvas, boxes, col = digits.clean_panel(canvas)
    use_painted = painted(canvas)
    big = textui.upscale(canvas, "hybrid", SCALE)
    if hard:
        big = hard_alpha(big, canvas)
    if panel:
        big = digits.draw_panel_text(big, boxes, col)
    for h, x, y in unit:
        if use_painted and (PAINTED_SRC / f"{h}.png").exists():  # strips under 4 px were never upscaled: hybrid
            shutil.copy2(PAINTED_SRC / f"{h}.png", Path(out) / f"{h}.png")
            continue
        w, hh = meta[h]["w"], meta[h]["h"]
        piece = big[y * SCALE:(y + hh) * SCALE, x * SCALE:(x + w) * SCALE]
        Image.fromarray(np.ascontiguousarray(piece), "RGBA").save(Path(out) / f"{h}.png")
    return len(unit)


def manifest(pack, ident, name, desc):
    (pack / "mod.json").write_text(json.dumps({
        "game_id": "qs64", "id": ident, "display_name": name, "description": desc, "version": "0.2.0",
        "authors": ["Lukas"], "minimum_recomp_version": "0.1.0"}, indent=1))


def main():
    meta = META
    text, cand = candidates()
    ui = [h for i, h in enumerate(cand) if i not in WORLD]
    # later dumps: caption/narration strips (streamed through one buffer, so each strip is scaled on its own)
    extra = json.loads((R / "textui_extra.json").read_text()) if (R / "textui_extra.json").exists() else []
    import mappage  # pause-menu map pages (strips rebuilt, names redrawn)
    extra = sorted(set(extra) | set(mappage.strip_hashes()))
    extra = [h for h in extra if h not in text and h not in ui]
    textui_set = sorted(text | set(ui) | set(extra))
    (R / "textui_textures.json").write_text(json.dumps(textui_set, indent=0))
    print(f"text {len(text)}, ui {len(ui)} (of {len(cand)} candidates, {len(WORLD)} kept as world), "
          f"extra {len(extra)}")

    # --- Text & UI pack (PNG staging; build_split.sh converts to DDS/.rtz)
    pack = STAGING / "Quest64HD-TextUI"
    safe_reset(pack)
    jobs = [([(h, 0, 0)], str(R / "decoded_text"), str(pack), False) for h in sorted(text)]
    jobs += [(u, str(R / "decoded"), str(pack), True) for u in up.group_strips(sorted(ui), meta)]
    jobs += [([(h, 0, 0)], str(R / "decoded"), str(pack), False) for h in extra]
    with ProcessPoolExecutor() as ex:
        n = sum(ex.map(render_unit, jobs, chunksize=4))
    # dialog font: rendered from a real font (clean dark letters, no drop shadow) instead of upscaled
    import fontsheet
    for h, sheet in zip(*fontsheet.render_sheets(color=FONT_COLOR)):
        Image.fromarray(sheet, "RGBA").save(pack / f"{h}.png")
    print(f"dialog font: {Path(fontsheet.FONT).stem} (monospaced, 8-px slots), colour {FONT_COLOR}")
    import labeltext  # stylized labels (Name, Agi, LOAD, NEW ...): clean text with a dark outline
    n_labels = 0
    for h, rgba in labeltext.render_labels():
        Image.fromarray(rgba, "RGBA").save(pack / f"{h}.png")
        n_labels += 1
    print(f"labels: {n_labels} re-rendered as clean text")
    n_digits = 0  # HUD/battle numbers: every 8x16 digit variant identified and redrawn clean
    for h, rgba in digits.render_digits():
        Image.fromarray(rgba, "RGBA").save(pack / f"{h}.png")
        n_digits += 1
    print(f"numbers: {n_digits} textures re-rendered")
    for h, rgba in digits.render_menu_font():  # 7x9 menu font: spell names, EMPTY ...
        Image.fromarray(rgba, "RGBA").save(pack / f"{h}.png")
    for h, rgba in mappage.render_pages():
        Image.fromarray(rgba, "RGBA").save(pack / f"{h}.png")
    import hudfont  # Tale overlay font atlas (Experience panel etc.), generated texture, not dumped
    h, rgba = hudfont.render()
    Image.fromarray(rgba, "RGBA").save(pack / f"{h}.png")
    textui_set = sorted(set(textui_set) | {h})
    print(f"overlay font: {h}")
    import edits  # hand edits from edit/text-ui (GIMP) replace the generated textures
    edits.apply(pack)
    db = {"configuration": {"autoPath": "rt64", "configurationVersion": 3, "hashVersion": 5,
                            "defaultOperation": "stream", "defaultShift": "half"},
          "textures": [{"hashes": {"rt64": h}, "path": h, "shift": "none"} for h in textui_set]}
    (pack / "rt64.json").write_text(json.dumps(db, indent=1))
    manifest(pack, "q64tex_textui", "Textures: HD Text & UI",
             "Sharp, noise-free text, HUD, menus and icons (xBRZ edges + smooth gradients, no AI redraw). "
             "Pair with Textures: HD World.")
    print(f"TextUI: {n} textures at {SCALE}x -> {pack}")

    # --- World pack: existing HDCube3 DDS minus the Text & UI set
    world = R / "dds" / "Quest64HD-World"
    safe_reset(world)
    skip = set(textui_set)
    db = json.loads((HDCUBE_DDS / "rt64.json").read_text())
    db["textures"] = [t for t in db["textures"] if t["hashes"]["rt64"] not in skip]
    keep = {t["path"] for t in db["textures"]}
    copied = 0
    for p in HDCUBE_DDS.glob("*.dds"):
        if p.stem in keep:
            shutil.copy2(p, world / p.name)
            copied += 1
    (world / "rt64.json").write_text(json.dumps(db, indent=1))
    manifest(world, "q64tex_hdworld", "Textures: HD World (HDCube3)",
             "HDCube3 upscale of the 3D world, effects and sprites; leaves text and UI to Textures: HD Text & UI. "
             "Keep this on as the base layer under any World or Faces pack.")
    print(f"World: {copied} DDS ({len(db['textures'])} entries) -> {world}")


if __name__ == "__main__":
    main()
