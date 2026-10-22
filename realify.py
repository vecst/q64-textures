#!/usr/bin/env python3
"""Make realistic (and round-trip) variants of hand-labelled Quest 64 textures, then build fun packs.

Passes (one model each, so ComfyUI never swaps models mid-run; every pass resumes where it left off):
  jugg       Juggernaut XI img2img            surfaces + objects
  klein      FLUX.2 Klein reference edit      surfaces + objects
  face_real  FLUX.2 Klein reference edit      faces -> photoreal
  face_toon  Illustrious XL img2img           photoreal faces -> back to cartoon
Then `pack` writes RT64 texture pack mods from the results.

usage: realify.py [jugg|klein|face_real|face_toon|pack|all]
"""
import io
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from labels import LABELS

SERVER = "http://127.0.0.1:8188"
ROOT = Path.home() / "games/q64-textures"
DECODED = ROOT / "decoded"
HD_PACK = Path.home() / ".config/Quest64Recompiled/mods/Quest64HD-HDCube3"
MODS = Path.home() / ".config/Quest64Recompiled/mods"
OUT = ROOT / "realify"
CANDIDATES = json.loads((ROOT / "candidates.json").read_text()) if (ROOT / "candidates.json").exists() else None
MEGAPIXELS = 0.5
MAX_SIDE = 512
BATCH = 6

PAD_MODES = {0: "wrap", 1: "symmetric", 2: "edge", 3: "edge"}
NEG_REAL = "cartoon, anime, illustration, painting, drawing, lowres, blurry, text, watermark, frame, border, vignette, perspective"
NEG_TOON = "photo, photorealistic, realistic, 3d render, lowres, blurry, text, watermark, frame, border"


def prompt_jugg(desc):
    return (f"photorealistic {desc}, seamless tileable texture, flat orthographic view, even diffuse lighting, "
            f"natural detail, highly detailed photo scan")


def prompt_klein(desc):
    return (f"Turn this into a photorealistic {desc}. Keep the exact layout, shapes, proportions and colors of the "
            f"original. Flat, evenly lit, seamless, highly detailed photo scan.")


def prompt_face_real(desc):
    return (f"Turn this cartoon game texture of a {desc} into a photorealistic {desc} with real skin, eyes and hair. "
            f"Keep the exact layout, position, proportions, expression and colors. Front view, evenly lit, photo.")


def prompt_face_toon(desc):
    return (f"anime style {desc}, cel shading, clean lineart, flat colors, retro 1990s video game character art, "
            f"front view, masterpiece, best quality")


# --- workflows ---------------------------------------------------------------------------------------------------

def _scaled(node_id, image_node):
    return {node_id: {"class_type": "ImageScaleToTotalPixels",
                      "inputs": {"image": [image_node, 0], "upscale_method": "lanczos", "megapixels": MEGAPIXELS,
                                 "resolution_steps": 64}}}


def wf_sdxl(ckpt, image, ref_image, prompt, neg, denoise, seed):
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "11": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 1]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["3", 2]}},
        "7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 26, "cfg": 5.5, "sampler_name": "dpmpp_2m",
                                                   "scheduler": "karras", "denoise": denoise, "model": ["3", 0],
                                                   "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 2]}},
        "9": {"class_type": "ColorMatchLab", "inputs": {"image": ["8", 0], "reference": ["12", 0], "strength": 0.6}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "q64real/out"}},
    }
    wf.update(_scaled("2", "1"))
    wf.update(_scaled("12", "11"))
    return wf, "10"


def wf_klein(image, prompt, seed):
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 0]}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["5", 0]}},
        "8": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["7", 0]}},
        "9": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "10": {"class_type": "GetImageSize", "inputs": {"image": ["2", 0]}},
        "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["10", 0], "height": ["10", 1], "batch_size": 1}},
        "12": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": ["10", 0], "height": ["10", 1]}},
        "13": {"class_type": "CFGGuider", "inputs": {"cfg": 1.0, "model": ["3", 0], "positive": ["8", 0], "negative": ["9", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "16": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["13", 0], "sampler": ["15", 0],
                                                                 "sigmas": ["12", 0], "latent_image": ["11", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["5", 0]}},
        "18": {"class_type": "ColorMatchLab", "inputs": {"image": ["17", 0], "reference": ["2", 0], "strength": 0.6}},
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "q64real/out"}},
    }
    wf.update(_scaled("2", "1"))
    return wf, "19"


# --- comfy plumbing ----------------------------------------------------------------------------------------------

def upload(name, img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = requests.post(f"{SERVER}/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                      data={"subfolder": "q64real", "overwrite": "true"})
    r.raise_for_status()
    return f"q64real/{name}"


def submit(wf):
    r = requests.post(f"{SERVER}/prompt", json={"prompt": wf, "client_id": uuid.uuid4().hex})
    if r.status_code != 200:
        raise RuntimeError(r.text[:800])
    return r.json()["prompt_id"]


def fetch(pid, save_node, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        h = requests.get(f"{SERVER}/history/{pid}").json()
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(e["status"]["messages"])[-600:])
            img = e["outputs"][save_node]["images"][0]
            return Image.open(io.BytesIO(requests.get(f"{SERVER}/view", params=img).content)).convert("RGB")
        time.sleep(0.25)
    raise TimeoutError(pid)


# --- geometry: pad before generation, crop + seam-blend after ----------------------------------------------------

def make_padded(img, info, is_face):
    a = np.asarray(img)
    h, w = a.shape[:2]
    py = max(2, h // 4)
    a = np.pad(a, ((py, py), (0, 0), (0, 0)), mode=PAD_MODES.get(info["cmt"], "edge"))
    if is_face and info["cms"] == 1:
        # Half-face mirrored across its right edge in game: show the model the whole face, keep the left half.
        a = np.concatenate([a, a[:, ::-1]], axis=1)
        px_l, px_r = 0, w
    else:
        px_l = px_r = max(2, w // 4)
        a = np.pad(a, ((0, 0), (px_l, px_r), (0, 0)), mode=PAD_MODES.get(info["cms"], "edge"))
    return Image.fromarray(a), {"w": w, "h": h, "px_l": px_l, "px_r": px_r, "py": py,
                                "cms": info["cms"], "cmt": info["cmt"]}


def _blend_axis(g, start, length, axis):
    """Crop [start, start+length) along axis; if wrapping, crossfade the leading band with the generated content
    just past the far edge (the same texels, drawn in context), so the two edges meet seamlessly."""
    g = np.moveaxis(g, axis, 0)
    tile = g[start:start + length].copy()
    band = min(start, length // 4, g.shape[0] - (start + length))
    if band > 1:
        ext = g[start + length:start + length + band]
        a = np.linspace(0.0, 1.0, band)[:, None, None]
        tile[:band] = (1 - a) * ext + a * tile[:band]
    return np.moveaxis(tile, 0, axis)


def finish(raw, geo):
    pw, ph = geo["w"] + geo["px_l"] + geo["px_r"], geo["h"] + 2 * geo["py"]
    sx, sy = raw.width / pw, raw.height / ph
    g = np.asarray(raw).astype(np.float32)
    x0, wl = round(geo["px_l"] * sx), round(geo["w"] * sx)
    y0, hl = round(geo["py"] * sy), round(geo["h"] * sy)
    if geo["cms"] == 0:
        g = _blend_axis(g, x0, wl, 1)
    else:
        g = g[:, x0:x0 + wl]
    if geo["cmt"] == 0:
        g = _blend_axis(g, y0, hl, 0)
    else:
        g = g[y0:y0 + hl]
    out = Image.fromarray(g.clip(0, 255).astype(np.uint8))
    scale = min(8.0, MAX_SIDE / max(geo["w"], geo["h"]))
    return out.resize((max(1, round(geo["w"] * scale)), max(1, round(geo["h"] * scale))), Image.LANCZOS)


# --- passes --------------------------------------------------------------------------------------------------------

def entries(cats):
    meta = json.loads((DECODED / "meta.json").read_text())
    for idx, (cat, desc) in sorted(LABELS.items()):
        if cat in cats:
            h = CANDIDATES[idx]
            yield h, cat, desc, meta[h]


def run_pass(name, cats, build):
    raw_dir, fin_dir = OUT / "raw" / name, OUT / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    fin_dir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries(cats) if not (fin_dir / f"{e[0]}.png").exists()]
    print(f"[{name}] {len(todo)} to do", flush=True)
    t0, failed = time.time(), []
    for i in range(0, len(todo), BATCH):
        jobs = []
        for h, cat, desc, info in todo[i:i + BATCH]:
            try:
                wf, save, geo = build(h, cat, desc, info)
                jobs.append((h, submit(wf), save, geo))
            except Exception as e:
                failed.append((h, str(e)[:200]))
        for h, pid, save, geo in jobs:
            try:
                raw = fetch(pid, save)
                raw.save(raw_dir / f"{h}.png")
                (raw_dir / f"{h}.json").write_text(json.dumps(geo))
                finish(raw, geo).save(fin_dir / f"{h}.png")
            except Exception as e:
                failed.append((h, str(e)[:200]))
        done = min(i + BATCH, len(todo))
        rate = (time.time() - t0) / max(1, done)
        print(f"  [{name}] {done}/{len(todo)}  {rate:.1f}s each, ~{rate * (len(todo) - done) / 60:.0f} min left", flush=True)
    for h, e in failed:
        print(f"  FAILED {h}: {e}", flush=True)


def hd_padded(h, info, is_face):
    return make_padded(Image.open(HD_PACK / f"{h}.png").convert("RGB"), info, is_face)


def build_jugg(h, cat, desc, info):
    img, geo = hd_padded(h, info, False)
    ref = upload(f"{h}.png", img)
    wf, save = wf_sdxl("Juggernaut-XI-byRunDiffusion.safetensors", ref, ref, prompt_jugg(desc), NEG_REAL, 0.55, 7)
    return wf, save, geo


def build_klein(h, cat, desc, info):
    img, geo = hd_padded(h, info, cat == "F")
    ref = upload(f"{h}.png", img)
    prompt = prompt_face_real(desc) if cat == "F" else prompt_klein(desc)
    wf, save = wf_klein(ref, prompt, int(h[:8], 16) if cat == "F" else 7)
    return wf, save, geo


def build_toon(h, cat, desc, info):
    raw_real = OUT / "raw" / "face_real" / f"{h}.png"
    geo = json.loads((OUT / "raw" / "face_real" / f"{h}.json").read_text())
    src = upload(f"{h}.real.png", Image.open(raw_real).convert("RGB"))
    hd_img, _ = hd_padded(h, info, True)
    ref = upload(f"{h}.png", hd_img)
    wf, save = wf_sdxl("Illustrious-XL-v2.0.safetensors", src, ref, prompt_face_toon(desc), NEG_TOON, 0.6, int(h[:8], 16))
    return wf, save, geo


# --- packs ---------------------------------------------------------------------------------------------------------

def write_pack(folder, display, description, files):
    d = MODS / folder
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    for h, src in files.items():
        (d / f"{h}.png").write_bytes(src.read_bytes())
    db = {"configuration": {"autoPath": "rt64", "configurationVersion": 3, "hashVersion": 5,
                            "defaultOperation": "stream", "defaultShift": "half"},
          "textures": [{"hashes": {"rt64": h}, "path": h} for h in sorted(files)]}
    (d / "rt64.json").write_text(json.dumps(db, indent=1))
    mod = {"game_id": "qs64", "id": folder.lower().replace("-", "_"), "display_name": display,
           "description": description, "version": "0.1.0", "authors": ["Lukas"], "minimum_recomp_version": "0.1.0"}
    (d / "mod.json").write_text(json.dumps(mod, indent=1))
    print(f"pack {folder}: {len(files)} textures")


def build_packs():
    world, real_faces, toon_faces = {}, {}, {}
    for h, cat, desc, info in entries({"S", "O", "F"}):
        if cat == "F":
            for store, variant in ((real_faces, "face_real"), (toon_faces, "face_toon")):
                p = OUT / variant / f"{h}.png"
                if p.exists():
                    store[h] = p
            continue
        preferred = ("klein", "jugg") if cat == "S" else ("jugg", "klein")
        for variant in preferred:
            p = OUT / variant / f"{h}.png"
            if p.exists():
                world[h] = p
                break
    write_pack("Quest64Fun-RealWorld", "World: Photoreal (cursed)",
               "World surfaces made photorealistic (FLUX.2 Klein for natural materials, Juggernaut XI for structures). "
               "Load above HD Textures.", world)
    write_pack("Quest64Fun-RealFaces", "Faces: Photoreal (cursed)", "Every face turned photorealistic. Uncanny.", real_faces)
    write_pack("Quest64Fun-RoundTripFaces", "Faces: Real-and-Back (cursed)",
               "Faces turned photoreal, then back into cartoons with Illustrious XL.", toon_faces)


def main(which):
    if CANDIDATES is None:
        raise SystemExit("missing candidates.json (run the candidate listing first)")
    steps = {
        "jugg": lambda: run_pass("jugg", {"S", "O"}, build_jugg),
        "klein": lambda: run_pass("klein", {"S", "O"}, build_klein),
        "face_real": lambda: run_pass("face_real", {"F"}, build_klein),
        "face_toon": lambda: run_pass("face_toon", {"F"}, build_toon),
        "pack": build_packs,
    }
    order = ["face_real", "face_toon", "klein", "jugg", "pack"] if which == "all" else [which]
    for s in order:
        steps[s]()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
