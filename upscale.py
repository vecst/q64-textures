#!/usr/bin/env python3
"""Upscale decoded Quest 64 textures through a ComfyUI workflow and build an RT64 texture pack mod.

The workflow is any ComfyUI graph exported in API format: the script feeds each texture into its
LoadImage node, reads back its SaveImage node, and (optionally) swaps the UpscaleModelLoader model.

usage: upscale.py <decoded_dir> <pack_name> [--model NAME] [--workflow FILE] [--server URL]
                  [--mods-dir DIR] [--limit N] [--only HASH,HASH]
"""
import argparse
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image

HERE = Path(__file__).resolve().parent


def bleed(rgba, steps=8):
    """Spread opaque colours into fully transparent pixels so the model doesn't drag black halos in."""
    rgb, known = rgba[..., :3].astype(np.float32), rgba[..., 3] > 0
    if known.all() or not known.any():
        return rgba[..., :3]
    for _ in range(steps):
        acc, cnt = np.zeros_like(rgb), np.zeros(known.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            k = np.roll(known, (dy, dx), (0, 1))
            acc += np.roll(rgb, (dy, dx), (0, 1)) * k[..., None]
            cnt += k
        grow = ~known & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow, None]
        known = known | grow
        if known.all():
            break
    return rgb.clip(0, 255).astype(np.uint8)


class Comfy:
    def __init__(self, server, workflow, model):
        self.server, self.client_id = server.rstrip("/"), uuid.uuid4().hex
        self.workflow = json.loads(Path(workflow).read_text())
        self.load_id = self._node("LoadImage")
        self.save_id = self._node("SaveImage")
        if model:
            self.workflow[self._node("UpscaleModelLoader")]["inputs"]["model_name"] = model

    def _node(self, cls):
        ids = [k for k, v in self.workflow.items() if v.get("class_type") == cls]
        if len(ids) != 1:
            raise SystemExit(f"workflow needs exactly one {cls} node (found {len(ids)})")
        return ids[0]

    def submit(self, name, img):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        r = requests.post(f"{self.server}/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                          data={"subfolder": "q64", "overwrite": "true"})
        r.raise_for_status()
        wf = json.loads(json.dumps(self.workflow))
        wf[self.load_id]["inputs"]["image"] = f"q64/{name}"
        wf[self.save_id]["inputs"]["filename_prefix"] = f"q64/{Path(name).stem}"
        r = requests.post(f"{self.server}/prompt", json={"prompt": wf, "client_id": self.client_id})
        if r.status_code != 200:
            raise SystemExit(f"ComfyUI rejected the workflow: {r.text}")
        return r.json()["prompt_id"]

    def result(self, prompt_id, timeout=600):
        end = time.time() + timeout
        while time.time() < end:
            hist = requests.get(f"{self.server}/history/{prompt_id}").json()
            if prompt_id in hist:
                entry = hist[prompt_id]
                if entry.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(json.dumps(entry["status"].get("messages", []))[:500])
                imgs = entry["outputs"].get(self.save_id, {}).get("images", [])
                if imgs:
                    r = requests.get(f"{self.server}/view", params=imgs[0])
                    r.raise_for_status()
                    return Image.open(io.BytesIO(r.content)).convert("RGB")
            time.sleep(0.05)
        raise TimeoutError(prompt_id)


def pad_mode(info, axis_flag, rgba):
    """Wrap only textures that really tile: opaque, not a long strip (text/UI), not tiny. Everything else is
    mirrored, which gives the model natural context without dragging in content from the far edge."""
    h, w = rgba.shape[:2]
    tiles = (rgba[..., 3] == 255).all() and max(w, h) <= 4 * min(w, h) and min(w, h) >= 16
    if axis_flag == 0 and tiles:
        return "wrap"
    return "symmetric"


def prepare(rgba, info):
    h, w = rgba.shape[:2]
    px, py = max(4, min(16, w // 2)), max(4, min(16, h // 2))
    rgb = bleed(rgba)
    mode_y, mode_x = pad_mode(info, info["cmt"], rgba), pad_mode(info, info["cms"], rgba)
    rgb = np.pad(rgb, ((py, py), (0, 0), (0, 0)), mode=mode_y)
    rgb = np.pad(rgb, ((0, 0), (px, px), (0, 0)), mode=mode_x)
    alpha = rgba[..., 3]
    alpha = np.pad(np.pad(alpha, ((py, py), (0, 0)), mode=mode_y), ((0, 0), (px, px)), mode=mode_x)
    return Image.fromarray(rgb, "RGB"), alpha, (px, py)


def finish(up_rgb, alpha_padded, pad, orig_alpha):
    px, py = pad
    scale = up_rgb.width / alpha_padded.shape[1]
    sx, sy = round(px * scale), round(py * scale)
    out_w, out_h = up_rgb.width - 2 * sx, up_rgb.height - 2 * sy
    rgb = np.asarray(up_rgb)[sy:sy + out_h, sx:sx + out_w]
    if (orig_alpha == 255).all():
        a = np.full(rgb.shape[:2], 255, np.uint8)
    else:
        a_img = Image.fromarray(alpha_padded).resize((up_rgb.width, up_rgb.height), Image.LANCZOS)
        a = np.asarray(a_img)[sy:sy + out_h, sx:sx + out_w]
        if np.isin(orig_alpha, (0, 255)).all():  # 1-bit alpha (cutouts): keep edges crisp
            a = np.where(a >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


MAX_CANVAS = 1024


def group_strips(todo, meta):
    """The game draws big images (title art, menus, fancy text) as stacks of thin strips and loads UI pieces from
    shared atlases. Upscaling those pieces alone gives the model no context at the cut lines, so strips loaded from
    the same RDRAM image are reassembled into one canvas. Pieces that overlap (animation frames reusing one spot)
    stay separate. Returns a list of units, each a list of (hash, x, y)."""
    by_source = {}
    for h in todo:
        load = meta[h].get("load")
        if load:
            by_source.setdefault((load["addr"], load["width"], load["siz"], load["pal"]), []).append(h)

    units, grouped = [], set()
    for members in by_source.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda h: (meta[h]["load"]["y"], meta[h]["load"]["x"]))
        x0 = min(meta[h]["load"]["x"] for h in members)
        y0 = min(meta[h]["load"]["y"] for h in members)
        placed = []
        for h in members:
            x, y = meta[h]["load"]["x"] - x0, meta[h]["load"]["y"] - y0
            w, hgt = meta[h]["w"], meta[h]["h"]
            # Neighbouring strips may share one row/column for filtering; anything more is a different frame.
            if any(min(x + w, px + pw) - max(x, px) > 1 and min(y + hgt, py + ph) - max(y, py) > 1
                   for _, px, py, pw, ph in placed):
                continue
            placed.append((h, x, y, w, hgt))
        if len(placed) < 2:
            continue
        if max(x + w for _, x, _, w, _ in placed) > MAX_CANVAS or max(y + hh for _, _, y, _, hh in placed) > MAX_CANVAS:
            continue
        units.append([(h, x, y) for h, x, y, _, _ in placed])
        grouped.update(h for h, *_ in placed)
    units += [[(h, 0, 0)] for h in todo if h not in grouped]
    return units


def clean_glyph_edges(rgba):
    """Glyph boxes carry slivers of the neighbouring glyph (last column, past a near-empty separator column) and
    leftover TMEM rows at the bottom. The game point-samples text so it never shows them, but filtered HD
    replacements bleed them in as thin bars ("DROP|", dashes under numbers). Erase them."""
    a = rgba[..., 3] > 0
    out = rgba.copy()
    h, w = a.shape
    cols, rows = a.sum(0), a.sum(1)
    for band in (1, 2):  # right-edge sliver: 1-2 columns after a column with at most one pixel
        sep = w - band - 1
        if sep >= 2 and cols[sep] <= 1 and cols[w - band:].sum() > 0 and cols[:sep].sum() > cols[w - band:].sum():
            out[:, w - band:, 3] = 0
            break
    for band in (1, 2):  # bottom leftovers: 1-2 rows below an empty gap
        sep = h - band - 1
        if sep >= 2 and rows[sep] == 0 and rows[h - band:].sum() > 0 and rows[:sep].sum() > 0:
            out[h - band:, :, 3] = 0
            break
    else:
        if rows[-1] >= w - 1 and rows[-2] < w - 1:  # full-width underline row directly under the glyph
            out[-1, :, 3] = 0
    return out


GLYPH_MAX = (16, 32)
CLEAN_EDGES = False


def assemble(unit, src, meta):
    if len(unit) == 1:
        h = unit[0][0]
        rgba = np.asarray(Image.open(src / f"{h}.png").convert("RGBA"))
        if CLEAN_EDGES and meta[h]["w"] <= GLYPH_MAX[0] and meta[h]["h"] <= GLYPH_MAX[1]:
            rgba = clean_glyph_edges(rgba)
        return rgba, meta[h]
    w = max(x + meta[h]["w"] for h, x, _ in unit)
    hgt = max(y + meta[h]["h"] for h, _, y in unit)
    canvas = Image.new("RGBA", (w, hgt), (0, 0, 0, 0))
    for h, x, y in unit:
        canvas.paste(Image.open(src / f"{h}.png").convert("RGBA"), (x, y))
    # A reassembled image doesn't tile, so pad it by mirroring.
    return np.asarray(canvas), {"cms": 2, "cmt": 2}


def write_pack(pack_dir, name, hashes, hash_version, model):
    db = {
        "configuration": {"autoPath": "rt64", "configurationVersion": 3, "hashVersion": hash_version,
                          "defaultOperation": "stream", "defaultShift": "half"},
        "textures": [{"hashes": {"rt64": h}, "path": h} for h in sorted(hashes)],
    }
    # Text and UI pieces carry slivers of neighbouring glyphs at their edges; RT64's half-texel shift exposes them
    # as thin bars ("DROP|"), so those are mapped without it.
    ui_list = HERE / "ui_textures.json"
    if ui_list.exists():
        ui = set(json.loads(ui_list.read_text()))
        for t in db["textures"]:
            if t["hashes"]["rt64"] in ui:
                t["shift"] = "none"
    (pack_dir / "rt64.json").write_text(json.dumps(db, indent=1))
    mod = {"game_id": "qs64", "id": f"q64tex_{name.lower()}", "display_name": f"Textures: HD Base ({name})",
           "description": f"Quest 64 textures upscaled with {model or 'custom workflow'} via ComfyUI.",
           "version": "0.1.0", "authors": ["Lukas"], "minimum_recomp_version": "0.1.0"}
    (pack_dir / "mod.json").write_text(json.dumps(mod, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decoded_dir")
    ap.add_argument("pack_name")
    ap.add_argument("--model", default="4xHDcube3_500k.safetensors")
    ap.add_argument("--workflow", default=str(HERE / "workflow_api.json"))
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--mods-dir", default=str(Path.home() / ".config/Quest64Recompiled/mods"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", type=int, default=24, help="prompts queued at once")
    ap.add_argument("--clean-edges", action="store_true", help="erase neighbour-glyph slivers from glyph-sized textures")
    ap.add_argument("--no-group", action="store_true",
                    help="upscale every texture on its own (font sheets overlap by a column, so regrouping glyphs "
                         "would put the neighbour slivers back)")
    args = ap.parse_args()
    global CLEAN_EDGES
    CLEAN_EDGES = args.clean_edges

    src = Path(args.decoded_dir)
    meta = json.loads((src / "meta.json").read_text())
    todo = sorted(meta)
    if args.only:
        todo = [h for h in todo if h in set(args.only.split(","))]
    todo = [h for h in todo if min(meta[h]["w"], meta[h]["h"]) >= 4]  # leave 1-3px textures alone
    if args.limit:
        todo = todo[: args.limit]

    pack_dir = Path(args.mods_dir) / f"Quest64HD-{args.pack_name}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    comfy = Comfy(args.server, args.workflow, args.model)
    done = set(p.stem for p in pack_dir.glob("*.png"))
    todo_new = [h for h in todo if h not in done]
    print(f"{len(todo)} textures ({len(todo) - len(todo_new)} already in pack) -> {pack_dir}")

    units = [[(h, 0, 0)] for h in todo_new] if args.no_group else group_strips(todo_new, meta)
    grouped = sum(len(u) for u in units if len(u) > 1)
    print(f"  {len(units)} upscale jobs ({grouped} strips/atlas pieces regrouped into "
          f"{sum(1 for u in units if len(u) > 1)} full images)")

    t0, failed = time.time(), []
    for i in range(0, len(units), args.batch):
        jobs = []
        for unit in units[i:i + args.batch]:
            rgba, info = assemble(unit, src, meta)
            img, alpha, pad = prepare(rgba, info)
            jobs.append((unit, comfy.submit(f"{unit[0][0]}.png", img), alpha, pad, rgba[..., 3]))
        for unit, pid, alpha, pad, orig_alpha in jobs:
            try:
                full = finish(comfy.result(pid), alpha, pad, orig_alpha)
                if len(unit) == 1:
                    full.save(pack_dir / f"{unit[0][0]}.png")
                else:
                    scale = full.width / orig_alpha.shape[1]
                    for h, x, y in unit:
                        w, hgt = meta[h]["w"], meta[h]["h"]
                        full.crop((round(x * scale), round(y * scale), round((x + w) * scale),
                                   round((y + hgt) * scale))).save(pack_dir / f"{h}.png")
                done.update(h for h, _, _ in unit)
            except Exception as e:  # keep going; report at the end
                failed.append((unit[0][0], str(e)[:200]))
        n = min(i + args.batch, len(units))
        print(f"  {n}/{len(units)}  ({time.time() - t0:.0f}s)", flush=True)

    hash_version = int(next(iter(meta.values()))["hash_version"]) if meta else 5
    write_pack(pack_dir, args.pack_name, done, hash_version, args.model)
    print(f"pack ready: {pack_dir} ({len(done)} textures)")
    for h, e in failed:
        print(f"  FAILED {h}: {e}")


if __name__ == "__main__":
    main()
