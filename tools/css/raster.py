"""
Textured-brush renderer: stamps a jot stroke texture along each stroke and
writes a PNG.

jot's stroke textures are one period of the brush, W samples along the stroke
by H across it, stored as an alpha mask.  We build a quad strip along the
stroke -- offset left and right by half the brush width -- and texture it with
u = frac(t) along and v across.  Because u comes from the coherent parameter
rather than screen arclength, the stamps stay pinned to the surface between
frames.

SVG cannot warp a bitmap along a curve without an unreasonable number of
elements, so textured brushes raster directly.
"""

import numpy as np

from brushes import (PRESS_PROFILES, WIGGLE_PROFILES, Envelope,
                     end_taper, polyline_frame)


def _sample(tex, u, v):
    """Bilinear lookup. u wraps (it is a period of t), v clamps (across)."""
    h, w = tex.shape
    x = (u % 1.0) * w - 0.5
    y = np.clip(v, 0.0, 1.0) * (h - 1)

    x0 = np.floor(x).astype(np.int64)
    fx = x - x0
    x0 %= w
    x1 = (x0 + 1) % w

    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fy = y - y0

    a = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    b = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return a * (1 - fy) + b * fy


def _tri(cov, tex, p, attr):
    """Rasterize one triangle, compositing coverage with max.

    p is (3,2) pixel coords; attr is (3,2) holding (t, v) per corner."""
    H, W = cov.shape
    x0 = max(int(np.floor(p[:, 0].min())), 0)
    x1 = min(int(np.ceil(p[:, 0].max())) + 1, W)
    y0 = max(int(np.floor(p[:, 1].min())), 0)
    y1 = min(int(np.ceil(p[:, 1].max())) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return

    ax, ay = p[0]
    bx, by = p[1]
    cx, cy = p[2]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) < 1e-12:
        return

    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    gx, gy = np.meshgrid(xs, ys)

    l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
    l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
    l2 = 1.0 - l0 - l1

    e = -1e-6
    inside = (l0 >= e) & (l1 >= e) & (l2 >= e)
    if not inside.any():
        return

    l0, l1, l2 = l0[inside], l1[inside], l2[inside]
    t = l0 * attr[0, 0] + l1 * attr[1, 0] + l2 * attr[2, 0]
    v = l0 * attr[0, 1] + l1 * attr[1, 1] + l2 * attr[2, 1]

    a = _sample(tex, t, v)
    sub = cov[y0:y1, x0:x1]
    cur = sub[inside]
    sub[inside] = np.maximum(cur, a)


def render(strokes_px, size, tex, width=6.0, taper_px=0.0, wiggle_px=0.0,
           wiggle_profile="hand", press_profile="flat", ss=2, envelope=None):
    """Rasterize a frame.  `strokes_px` is a list of (t array, Nx2 pixel
    array, |kappa_r| array or None).  Returns a float coverage image of shape
    (size, size)."""
    n = int(size * ss)
    cov = np.zeros((n, n), dtype=np.float32)
    wig = WIGGLE_PROFILES[wiggle_profile]
    prs = PRESS_PROFILES[press_profile]
    env = envelope if envelope is not None else Envelope("end", taper_px)

    for ts, px, kap in strokes_px:
        if len(px) < 2:
            continue
        ts = np.asarray(ts, dtype=float)
        p = np.asarray(px, dtype=float) * ss

        if wiggle_px:
            _, nrm, _ = polyline_frame(p)
            p = p + nrm * (wiggle_px * ss * wig(ts % 1.0))[:, None]

        _, nrm, arc = polyline_frame(p)
        total = arc[-1]
        # `arc` is in supersampled pixels, so the end taper's length has to
        # be scaled to match; the curvature term is scale-free.
        half = 0.5 * width * ss * prs(ts % 1.0) * env(arc, total, kap, ss)
        L = p + nrm * half[:, None]
        R = p - nrm * half[:, None]

        for i in range(len(p) - 1):
            a = np.array([[ts[i], 0.0], [ts[i], 1.0], [ts[i + 1], 1.0]])
            _tri(cov, tex, np.array([L[i], R[i], R[i + 1]]), a)
            a = np.array([[ts[i], 0.0], [ts[i + 1], 1.0], [ts[i + 1], 0.0]])
            _tri(cov, tex, np.array([L[i], R[i + 1], L[i + 1]]), a)

    if ss > 1:
        cov = cov.reshape(size, ss, size, ss).mean(axis=(1, 3))
    return cov


def write_png(path, cov, color=(0, 0, 0), alpha=1.0, background=None):
    from PIL import Image

    a = np.clip(cov * alpha, 0.0, 1.0)
    rgb = np.array(color, dtype=np.float32).reshape(1, 1, 3)

    if background is None:
        out = np.concatenate(
            [np.repeat(rgb, a.shape[0], 0).repeat(a.shape[1], 1), a[..., None]],
            axis=2)
    else:
        bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
        img = bg * (1 - a[..., None]) + rgb * a[..., None]
        out = np.concatenate([img, np.ones_like(a)[..., None]], axis=2)

    Image.fromarray((out * 255).round().astype(np.uint8), "RGBA").save(path)
