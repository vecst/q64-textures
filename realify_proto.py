#!/usr/bin/env python3
"""Prototype: turn a few Quest 64 surface textures photorealistic with Juggernaut XI (img2img) and
FLUX.2 Klein (reference edit), then build a side-by-side comparison sheet.

usage: realify_proto.py <out_dir>
"""
import io
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

SERVER = "http://127.0.0.1:8188"
DECODED = Path.home() / "games/q64-textures/decoded"
HD_PACK = Path.home() / ".config/Quest64Recompiled/mods/Quest64HD-HDCube3"
NEG = "cartoon, illustration, painting, drawing, lowres, blurry, text, watermark, frame, border, vignette, shadow, perspective"

# (hash prefix, material) picked by eye from the dump.
PICKS = [
    ("5b50fdc40a31e247", "lush green grass lawn"),
    ("e5539d", "weathered wooden log planks"),
    ("090b0a", "golden desert sand"),
    ("341638", "brown dirt footpath with grass edges"),
    ("ad5b1d", "sandstone block wall"),
    ("033010", "terracotta clay roof tiles"),
    ("3e3324", "pale cobblestones with moss in the gaps"),
    ("d1be1d", "grey stone brick wall"),
]


def realistic_prompt(material):
    return (f"photorealistic {material} texture, seamless tileable game texture, flat top-down orthographic view, "
            f"even diffuse lighting, natural detail, 4k photo scan")


def klein_prompt(material):
    return (f"Turn this into a photorealistic {material} texture. Keep the exact layout, shapes, proportions and "
            f"colors of the original. Flat, evenly lit, seamless, highly detailed photo scan.")


def wf_juggernaut(image, prompt, seed):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["1", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 64}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "Juggernaut-XI-byRunDiffusion.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["3", 1]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["3", 2]}},
        "7": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 30, "cfg": 5.5, "sampler_name": "dpmpp_2m", "scheduler": "karras",
                                                   "denoise": 0.55, "model": ["3", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 2]}},
        "9": {"class_type": "ColorMatchLab", "inputs": {"image": ["8", 0], "reference": ["2", 0], "strength": 0.6}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "q64real/jugg"}},
    }


def wf_klein(image, prompt, seed):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["1", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 64}},
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
        "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "q64real/klein"}},
    }


def run(workflow, save_node):
    r = requests.post(f"{SERVER}/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex})
    if r.status_code != 200:
        raise SystemExit(f"ComfyUI rejected workflow: {r.text[:1500]}")
    pid = r.json()["prompt_id"]
    while True:
        h = requests.get(f"{SERVER}/history/{pid}").json()
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("status_str") == "error":
                raise SystemExit(json.dumps(e["status"]["messages"])[-1500:])
            img = e["outputs"][save_node]["images"][0]
            return Image.open(io.BytesIO(requests.get(f"{SERVER}/view", params=img).content)).convert("RGB")
        time.sleep(0.3)


def upload(name, img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    requests.post(f"{SERVER}/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                  data={"subfolder": "q64real", "overwrite": "true"}).raise_for_status()
    return f"q64real/{name}"


def wrap_pad(img, frac=0.25):
    a = np.asarray(img)
    py, px = int(a.shape[0] * frac), int(a.shape[1] * frac)
    return Image.fromarray(np.pad(a, ((py, py), (px, px), (0, 0)), mode="wrap")), frac


def unpad(img, frac, size):
    w, h = img.size
    tw, th = w / (1 + 2 * frac), h / (1 + 2 * frac)
    left, top = (w - tw) / 2, (h - th) / 2
    return img.crop((round(left), round(top), round(left + tw), round(top + th))).resize(size, Image.LANCZOS)


def tile2(img):
    w, h = img.size
    out = Image.new("RGB", (w * 2, h * 2))
    for y in range(2):
        for x in range(2):
            out.paste(img, (x * w, y * h))
    return out.resize((w, h), Image.LANCZOS)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DECODED / "meta.json").read_text())
    rows = []
    for prefix, material in PICKS:
        h = next(k for k in meta if k.startswith(prefix))
        hd = Image.open(HD_PACK / f"{h}.png").convert("RGB")
        orig = Image.open(DECODED / f"{h}.png").convert("RGB")
        padded, frac = wrap_pad(hd)
        ref = upload(f"{h}.png", padded)
        final = (512, round(512 * hd.height / hd.width))
        t0 = time.time()
        jugg = unpad(run(wf_juggernaut(ref, realistic_prompt(material), 7), "10"), frac, final)
        t1 = time.time()
        klein = unpad(run(wf_klein(ref, klein_prompt(material), 7), "19"), frac, final)
        t2 = time.time()
        jugg.save(out / f"{h}.jugg.png")
        klein.save(out / f"{h}.klein.png")
        print(f"{material:45s} juggernaut {t1 - t0:5.1f}s  klein {t2 - t1:5.1f}s", flush=True)
        rows.append((material, orig.resize(final, Image.NEAREST), hd.resize(final, Image.LANCZOS), jugg, klein))

    cell, gap, label = 256, 8, 22
    cols = ["Original", "HDCube 3", "Juggernaut XI", "FLUX.2 Klein", "Juggernaut 2x2", "Klein 2x2"]
    W = len(cols) * (cell + gap) + gap
    H = label + len(rows) * (cell // 1 + gap + label) + gap
    sheet = Image.new("RGB", (W, H), (28, 28, 32))
    d = ImageDraw.Draw(sheet)
    for c, name in enumerate(cols):
        d.text((gap + c * (cell + gap), 6), name, fill=(255, 220, 120))
    y = label
    for material, o, hd, j, k in rows:
        d.text((gap, y + 2), material, fill=(200, 200, 210))
        y += label - 6
        for c, im in enumerate((o, hd, j, k, tile2(j), tile2(k))):
            s = cell / max(im.size)
            sheet.paste(im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS), (gap + c * (cell + gap), y))
        y += cell + gap
    sheet.save(out / "comparison.png")
    print(f"sheet: {out / 'comparison.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "games/q64-textures/realify-proto"))
