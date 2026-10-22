#!/usr/bin/env python3
"""Deterministic upscalers for text and UI textures (no ML, so no invented detail or noise).

  xbrz   - xBRZ pixel-art scaler at 6x, then exact 8/6 resample to 8x
  hybrid - xBRZ wherever the source has a real edge (local contrast high), smooth premultiplied bicubic inside soft
           gradients where xBRZ would leave flat bands; the mix weight comes from the source pixels, so it is exact
           and repeatable
  trace  - (rejected: potrace turns 1-px pixel art into blobs) layered vector trace: every palette colour is traced with potrace in stacked order (largest first, each
           layer = union of itself and everything drawn above it, so neighbours never leave gaps), then rasterised
           with 4x4 supersampling. Interiors keep the exact palette colour; only edges are antialiased.

usage as a library: from textui import upscale; upscale(rgba_uint8_array, "trace", scale=8)
"""
import numpy as np
import potrace
import xbrz
from PIL import Image, ImageDraw

SS = 4  # supersampling for trace rasterisation


def _xbrz(rgba, scale):
    big = np.asarray(xbrz.scale_pillow(Image.fromarray(rgba, "RGBA"), 6))
    if scale == 6:
        return big
    h, w = rgba.shape[:2]
    return np.asarray(Image.fromarray(big, "RGBA").resize((w * scale, h * scale), Image.LANCZOS))


def _flatten(curve, k):
    """potrace curve -> polygon points scaled by k (bezier segments subdivided)."""
    pts = []
    p0 = curve.start_point
    pts.append((p0.x * k, p0.y * k))
    for seg in curve.segments:
        if seg.is_corner:
            pts.append((seg.c.x * k, seg.c.y * k))
            pts.append((seg.end_point.x * k, seg.end_point.y * k))
        else:
            a, b, c, d = p0, seg.c1, seg.c2, seg.end_point
            n = 12
            for i in range(1, n + 1):
                t = i / n
                u = 1 - t
                x = u * u * u * a.x + 3 * u * u * t * b.x + 3 * u * t * t * c.x + t * t * t * d.x
                y = u * u * u * a.y + 3 * u * u * t * b.y + 3 * u * t * t * c.y + t * t * t * d.y
                pts.append((x * k, y * k))
        p0 = seg.end_point
    return pts


def _trace_mask(mask, scale, alphamax):
    """Binary mask -> antialiased coverage at `scale`, float32 0..1."""
    h, w = mask.shape
    k = scale * SS
    # pad by 1 so shapes touching the border trace with straight edges, then shift back
    pad = np.zeros((h + 2, w + 2), bool)
    pad[1:-1, 1:-1] = mask
    path = potrace.Bitmap(~pad).trace(  # potracer fills the False pixels
        turdsize=0, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY,
                                     alphamax=alphamax, opticurve=False)
    canvas = np.zeros(((h + 2) * k, (w + 2) * k), np.uint8)
    for curve in path.curves:
        layer = Image.new("1", (canvas.shape[1], canvas.shape[0]), 0)
        ImageDraw.Draw(layer).polygon(_flatten(curve, k), fill=1)
        canvas ^= np.asarray(layer, np.uint8)
    canvas = canvas[k:-k, k:-k].astype(np.float32)
    return canvas.reshape(h * scale, SS, w * scale, SS).mean((1, 3))


def _trace(rgba, scale, alphamax=1.0):
    h, w = rgba.shape[:2]
    flat = rgba.reshape(-1, 4)
    keys = flat.view(np.uint32).reshape(h, w)
    opaque = rgba[..., 3] > 0
    cols, counts = np.unique(keys[opaque], return_counts=True)
    order = cols[np.argsort(-counts, kind="stable")]
    out = np.zeros((h * scale, w * scale, 4), np.float32)  # premultiplied
    for i, c in enumerate(order):
        mask = np.isin(keys, order[i:]) & opaque  # stacked: this colour plus everything drawn after it
        cov = _trace_mask(mask, scale, alphamax)
        if not cov.any():
            continue
        col = np.frombuffer(np.uint32(c).tobytes(), np.uint8).astype(np.float32)
        a = col[3] / 255.0
        src = np.array([col[0] * a, col[1] * a, col[2] * a, col[3]], np.float32)
        cov = cov[..., None]
        out = out * (1 - cov) + src * cov  # replace, not blend: stacked layers must not mix colours
    alpha = out[..., 3:4]
    rgb = np.where(alpha > 0, out[..., :3] * 255.0 / np.maximum(alpha, 1e-6), 0)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 255)]).round().astype(np.uint8)


def _smooth(rgba, scale):
    """Premultiplied bicubic upscale (no dark fringes at transparent edges)."""
    f = rgba.astype(np.float32)
    a = f[..., 3:4] / 255.0
    pm = np.dstack([f[..., :3] * a, f[..., 3:4]])
    h, w = rgba.shape[:2]
    up = np.dstack([np.asarray(Image.fromarray(pm[..., i]).resize((w * scale, h * scale), Image.BICUBIC))
                    for i in range(4)])
    up = np.clip(up, 0, 255)
    al = up[..., 3:4] / 255.0
    rgb = np.where(al > 1e-3, up[..., :3] / np.maximum(al, 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, 255), up[..., 3:4]])


def edge_weight(rgba, lo=20.0, hi=44.0):
    """Per source pixel: 1 where it differs strongly from a neighbour (an edge xBRZ should shape), 0 in gradients."""
    f = rgba.astype(np.float32)
    a = f[..., 3:4] / 255.0
    y = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]) * a[..., 0]
    cb = (f[..., 2] - y) * 0.5 * a[..., 0]
    cr = (f[..., 0] - y) * 0.5 * a[..., 0]
    feat = np.dstack([y, cb, cr, f[..., 3]])
    p = np.pad(feat, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = rgba.shape[:2]
    c = np.zeros((h, w), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                d = p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] - feat
                c = np.maximum(c, np.sqrt((d * d).sum(-1)))
    return np.clip((c - lo) / (hi - lo), 0, 1)


def _hybrid(rgba, scale):
    xb = _xbrz(rgba, scale).astype(np.float32)
    sm = _smooth(rgba, scale)
    h, w = rgba.shape[:2]
    wt = np.asarray(Image.fromarray(edge_weight(rgba)).resize((w * scale, h * scale), Image.BILINEAR))[..., None]
    # mix in premultiplied space so edges against transparency stay clean
    def pm(x):
        return np.dstack([x[..., :3] * x[..., 3:4] / 255.0, x[..., 3:4]])
    out = pm(xb) * wt + pm(sm) * (1 - wt)
    al = out[..., 3:4]
    # transparent texels get the neighbouring opaque colour (never black, never the pink key colour)
    from make_dds import fill_transparent
    near = np.asarray(Image.fromarray(fill_transparent(rgba), "RGBA").resize((w * scale, h * scale),
                                                                           Image.NEAREST))[..., :3]
    rgb = np.where(al > 0.5, out[..., :3] * 255.0 / np.maximum(al, 1e-3), near)
    return np.dstack([np.clip(rgb, 0, 255), np.clip(al, 0, 255)]).round().astype(np.uint8)


def upscale(rgba, method, scale=8, **kw):
    rgba = np.ascontiguousarray(rgba, np.uint8)
    if method == "xbrz":
        return _xbrz(rgba, scale)
    if method == "hybrid":
        return _hybrid(rgba, scale)
    if method == "trace":
        return _trace(rgba, scale, **kw)
    if method == "nearest":
        return np.asarray(Image.fromarray(rgba, "RGBA").resize((rgba.shape[1] * scale, rgba.shape[0] * scale),
                                                               Image.NEAREST))
    raise ValueError(method)
