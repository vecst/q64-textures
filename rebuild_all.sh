#!/usr/bin/env bash
# Rebuild every texture pack's DDS/.rtz from its PNG source (after a make_dds.py fix) and install the enabled ones.
set -euo pipefail
cd "$(dirname "$0")"
TP=$HOME/games/Quest64Tale-src/build/game/texture_packer
PY=$HOME/games/q64-mods/.venv/bin/python
ARC=$HOME/games/q64-mods/archive/png-packs
OUT=$HOME/games/q64-textures/dds
MODS=$HOME/games/Quest64Tale/mods
$PY make_dds.py "$ARC/Quest64HD-HDCube3" "$OUT/Quest64HD-HDCube3" | tail -1   # base for HD World
for pack in Quest64Stylized-World Quest64Stylized-Faces-klein Quest64Models-HDTextures; do
  $PY make_dds.py "$ARC/$pack" "$OUT/$pack" | tail -1
  $TP "$OUT/$pack" --create-low-mip-cache > "$OUT/$pack.log" 2>&1 || { tail -5 "$OUT/$pack.log"; exit 1; }
  $TP "$OUT/$pack" --create-pack >> "$OUT/$pack.log" 2>&1 || { tail -5 "$OUT/$pack.log"; exit 1; }
  $PY add_manifest.py "$OUT/$pack/$pack.rtz" "$OUT/$pack/mod.json"
done
./build_split.sh | grep -E "numbers|menu font|labels|applied" || true
if pgrep -x Quest64Recompil > /dev/null; then echo "GAME RUNNING: packs built but not installed"; exit 0; fi
for pack in Quest64HD-World Quest64HD-TextUI Quest64Stylized-World Quest64Stylized-Faces-klein Quest64Models-HDTextures; do
  cp "$OUT/$pack/$pack.rtz" "$MODS/"
done
echo "installed all packs"
