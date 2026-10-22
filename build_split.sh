#!/usr/bin/env bash
# Build the split HD packs: Text & UI (hybrid + hand edits from edit/text-ui) and World (HDCube3 DDS minus
# text/UI) -> .rtz.  usage: build_split.sh [--install]   (--install copies both .rtz into the game's mods folder)
set -euo pipefail
cd "$(dirname "$0")"
TP=$HOME/games/Quest64Tale-src/build/game/texture_packer
PY=$HOME/games/q64-mods/.venv/bin/python
OUT=$HOME/games/q64-textures/dds
MODS=$HOME/games/Quest64Tale/mods
$PY build_textui.py
$PY make_dds.py staging/Quest64HD-TextUI "$OUT/Quest64HD-TextUI" > /dev/null
for pack in Quest64HD-TextUI Quest64HD-World; do
  $TP "$OUT/$pack" --create-low-mip-cache > "$OUT/$pack.log" 2>&1 || { tail -5 "$OUT/$pack.log"; exit 1; }
  $TP "$OUT/$pack" --create-pack >> "$OUT/$pack.log" 2>&1 || { tail -5 "$OUT/$pack.log"; exit 1; }
  $PY add_manifest.py "$OUT/$pack/$pack.rtz" "$OUT/$pack/mod.json"
done
$PY preview.py "$HOME/games/q64-textures/edit/preview.png" > /dev/null && echo "preview: ~/games/q64-textures/edit/preview.png"
if [ "${1:-}" = "--install" ]; then
  if pgrep -x Quest64Recompil > /dev/null; then echo "game is running: close it, then run again with --install"; exit 1; fi
  cp "$OUT/Quest64HD-TextUI/Quest64HD-TextUI.rtz" "$OUT/Quest64HD-World/Quest64HD-World.rtz" "$MODS/"
  echo "installed into $MODS"
fi
