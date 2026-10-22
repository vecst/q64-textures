#!/usr/bin/env bash
# usage: ./build.sh [model] [pack-name]     e.g.  ./build.sh 4x-UltraSharpV2.safetensors UltraSharp
# Decodes RT64 dumps in ./dumps, upscales them through ComfyUI, and installs the pack as a game mod.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/comfy/ComfyUI/.venv/bin/python
MODEL=${1:-4xHDcube3_500k.safetensors}
NAME=${2:-$(basename "${MODEL%.*}")}
source ~/comfy/comfy.sh && comfy start > /dev/null
$PY decode.py dumps decoded
$PY upscale.py decoded "$NAME" --model "$MODEL"

# Per-category models: NXbrz for text, UltraSharp V2 for title/menu art and large UI panels
$PY overrides.py "$HOME/.config/Quest64Recompiled/mods/Quest64HD-$NAME"

# 2D elements (UI, transparency, title/menu art) are mapped without the half-texel shift
$PY shift_rule.py "$HOME/.config/Quest64Recompiled/mods/Quest64HD-$NAME"
