#!/usr/bin/env python3
"""Give the Quest 64 mods category-first names and sort the mod list into groups.

usage: organize_mods.py names     rename packs (safe while the game runs; applied on next launch)
       organize_mods.py order     group mods.json order + enable the recommended set (run with the game closed)
"""
import json
import sys
from pathlib import Path

CFG = Path.home() / ".config/Quest64Recompiled"
MODS = CFG / "mods"
ONE_WORLD = "Use ONE World pack at a time, over Textures: HD Base."
ONE_FACES = "Use ONE Faces pack at a time."

# folder -> (display name, description). Listed in the order they should appear: higher = wins over lower.
PACKS = {
    "Quest64Stylized-Faces-klein": ("Faces: Stylized (Painted Features)",
                                    f"Painted stylized eyes, brows and lips kept inside the original feature shapes, plus painted skin. {ONE_FACES}"),
    "Quest64Stylized-FacesBold": ("Faces: Stylized (Bold)",
                                  f"Painted skin and hair material on every face; eyes and mouths unchanged. {ONE_FACES}"),
    "Quest64Fun-RealFaces": ("Faces: Photoreal (cursed)", f"Every face as a photo-real person. Uncanny. {ONE_FACES}"),
    "Quest64Fun-RoundTripFaces": ("Faces: Real-and-Back (cursed)",
                                  f"Faces turned photoreal, then back into cartoons. {ONE_FACES}"),
    "Quest64Stylized-World": ("World: Stylized",
                              f"Hand-painted stylized-realism detail on ground, walls, wood and objects. {ONE_WORLD}"),
    "Quest64Fun-RealWorld": ("World: Photoreal (cursed)", f"Photorealistic world surfaces. {ONE_WORLD}"),
    "Quest64HD-HDCube3": ("Textures: HD Base (HDCube3)",
                          "Clean 4x upscale of every texture. Keep this on as the base layer under any World or Faces pack."),
}
REBALANCE_ID = "q64_rebalance"
RECOMMENDED = [REBALANCE_ID, "quest64stylized_faces_klein", "quest64stylized_world", "q64tex_hdcube3"]


def names():
    for folder, (name, desc) in PACKS.items():
        p = MODS / folder / "mod.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        m["display_name"], m["description"] = name, desc
        p.write_text(json.dumps(m, indent=1))
        print(f"{m['id']:30s} -> {name}")


def order():
    p = CFG / "mods.json"
    cfg = json.loads(p.read_text())
    ids = [REBALANCE_ID] + [json.loads((MODS / f / "mod.json").read_text())["id"]
                            for f in PACKS if (MODS / f / "mod.json").exists()]
    rest = [m for m in cfg.get("mod_order", []) if m not in ids]
    cfg["mod_order"] = ids + rest
    cfg["enabled_mods"] = [m for m in RECOMMENDED if m in ids]
    (CFG / "mods.json.before-organize").write_text(p.read_text())
    p.write_text(json.dumps(cfg, indent=4))
    print("order:", *cfg["mod_order"], sep="\n  ")
    print("enabled:", ", ".join(cfg["enabled_mods"]))


if __name__ == "__main__":
    {"names": names, "order": order}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
