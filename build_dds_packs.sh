#!/usr/bin/env bash
# PNG texture packs -> mipmapped DDS + RT64 low-mip cache -> .rtz (in ~/games/q64-textures/dds/<pack>/<pack>.rtz).
# usage: build_dds_packs.sh <pack folder name in ~/games/Quest64Tale/mods> ...
set -euo pipefail
cd "$(dirname "$0")"
TP=$HOME/games/Quest64Tale-src/build/game/texture_packer
PY=$HOME/games/q64-mods/.venv/bin/python
SRC=$HOME/games/Quest64Tale/mods
OUT=$HOME/games/q64-textures/dds
mkdir -p "$OUT"
for pack in "$@"; do
  echo "== $pack"
  $PY make_dds.py "$SRC/$pack" "$OUT/$pack"
  $TP "$OUT/$pack" --create-low-mip-cache > "$OUT/$pack.log" 2>&1 || { tail -5 "$OUT/$pack.log"; exit 1; }
  $TP "$OUT/$pack" --create-pack >> "$OUT/$pack.log" 2>&1 || { tail -5 "$OUT/$pack.log"; exit 1; }
  # texture_packer leaves mod.json out; add it so the pack keeps its name and id.
  $PY add_manifest.py "$OUT/$pack/$pack.rtz" "$OUT/$pack/mod.json"
  du -h "$OUT/$pack/$pack.rtz"
done
