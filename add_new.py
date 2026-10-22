"""Sort newly dumped textures (new_textures.json) into the Text & UI pack without renumbering existing lists:
digit variants and menu-font glyphs are appended to text_textures.json (fontsheet/digits pick them up by shape),
narration/caption text strips go to textui_extra.json (hybrid upscale). Everything else stays world."""
import json
import numpy as np
from PIL import Image
from pathlib import Path
R = Path(".")
m = json.loads((R / "decoded/meta.json").read_text())
fresh = json.loads((R / "new_textures.json").read_text())
text = json.loads((R / "text_textures.json").read_text())
extra = json.loads((R / "textui_extra.json").read_text()) if (R / "textui_extra.json").exists() else []
add_text, add_extra = [], []
for h in fresh:
    v = m[h]; l = v.get("load") or {}
    if h in text or h in extra:
        continue
    if (v["w"], v["h"]) == (8, 16) and 0x60000 <= l.get("addr", 0) < 0x61000:
        add_text.append(h); continue
    if (v["w"], v["h"]) == (7, 9) and l.get("addr") == 0x3A2110:
        add_text.append(h); continue
    a = np.asarray(Image.open(R / "decoded" / f"{h}.png").convert("RGBA")).astype(int)
    op = a[..., 3] > 0
    if v["h"] <= 10 and v["w"] >= 100 and 0.02 < op.mean() < 0.7:
        lum = (a[..., :3] @ [.3, .59, .11])[op]
        if ((lum > 200) | (lum < 50)).mean() > 0.6:  # white letters with dark outline
            add_extra.append(h)
(R / "text_textures.json").write_text(json.dumps(text + add_text, indent=0))
(R / "textui_extra.json").write_text(json.dumps(sorted(set(extra + add_extra)), indent=0))
print(f"text +{len(add_text)} (digits/menu glyphs), extra +{len(add_extra)} (text strips)")
