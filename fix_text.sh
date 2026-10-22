#!/usr/bin/env bash
# Re-upscale the text/UI-lettering textures (text_textures.json) with 4x-NXbrz and drop them into the HD Base pack.
# ESRGAN "detail" models like HDCube can redraw letters (MONSTER -> MOYSTER); NXbrz mimics xBRZ and keeps shapes.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/comfy/ComfyUI/.venv/bin/python
PACK=${1:-$HOME/.config/Quest64Recompiled/mods/Quest64HD-HDCube3}
source ~/comfy/comfy.sh && comfy start > /dev/null
rm -rf /tmp/q64-text
$PY upscale.py decoded Text --model 4x-NXbrz.safetensors --mods-dir /tmp/q64-text \
    --only "$($PY -c "import json;print(\",\".join(json.load(open(\"text_textures.json\"))))")"
cp /tmp/q64-text/Quest64HD-Text/*.png "$PACK/"
echo "text fix applied to $PACK"
