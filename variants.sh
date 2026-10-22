#!/usr/bin/env bash
# Font weight comparison: dialog previews with the rendered font in several weights.
set -euo pipefail
cd "$(dirname "$0")"
PY=$HOME/games/q64-mods/.venv/bin/python
for w in Medium SemiBold Bold ExtraBold; do
  d=/tmp/q64var/$w
  mkdir -p "$d"
  cp staging/Quest64HD-TextUI/*.png "$d/"
  $PY - "$w" "$d" <<'PYEOF'
import sys
from PIL import Image
sys.path.insert(0, ".")
import fontsheet
for h, s in zip(*fontsheet.render_sheets(sys.argv[1])):
    Image.fromarray(s, "RGBA").save(f"{sys.argv[2]}/{h}.png")
PYEOF
  $PY preview.py /tmp/q64sheets/var_$w.png "$d" > /dev/null
done
$PY preview.py /tmp/q64sheets/var_current.png > /dev/null
$PY - <<'PYEOF'
from PIL import Image, ImageDraw
ims = []
for w in ["current", "Medium", "SemiBold", "Bold", "ExtraBold"]:
    im = Image.open(f"/tmp/q64sheets/var_{w}_dialog.png")
    im = im.crop((0, im.height // 2 + 6, im.width, im.height - 50))  # HD half: dialog box only
    d = ImageDraw.Draw(im); d.text((im.width - 330, 10), w, fill=(255, 255, 255), font_size=40)
    ims.append(im)
W = ims[0].width; H = sum(i.height + 8 for i in ims)
out = Image.new("RGB", (W, H)); y = 0
for i in ims:
    out.paste(i, (0, y)); y += i.height + 8
out.save("/tmp/q64sheets/font_weights.png")
PYEOF
echo done
