"""
Brush models.

A brush is any stylization expressed as a function of the coherent stroke
parameter `t`.  That is the whole contract with the CSS machinery: section 5
hands us (t, position) pairs whose `t` is stable across frames, so anything
driven by `t` stays anchored to the surface instead of sliding along the line.
Anything driven by screen arclength instead -- SVG's own `stroke-dasharray`,
for instance -- swims, which is exactly the artifact the paper is about.

Two families are supported.

Procedural profiles
    Periodic functions of u = frac(t), used for dashes, stipples, variable
    width and lateral wiggle.  The wiggle is jot's own brush model: a
    `BaseStrokeOffsetLIST` (stroke/base_stroke.H:113) is a recorded gesture
    stored as lateral displacement plus pressure against position within the
    pattern, rubber-stamped every `_pix_len` pixels.  Our `offset_pix_len` is
    that period, so frac(t) is position within one stamp.

jot stroke textures
    nprdata/stroke_textures/*.png are pure alpha masks, W across one period
    of t by H across the stroke width; the colour comes from the preset, not
    the texture.  nprdata/stroke_presets/*.pre carry colour, alpha, width and
    taper alongside the texture name.
"""

import math
import os
import re
import sys

import numpy as np

JOT_ROOT = os.environ.get(
    "JOT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
TEXTURE_DIR = os.path.join(JOT_ROOT, "nprdata", "stroke_textures")
PRESET_DIR = os.path.join(JOT_ROOT, "nprdata", "stroke_presets")


# ------------------------------------------------------- periodic profiles
#
# Every profile takes u in [0,1) and returns a value.  They must agree at
# u = 0 and u = 1 or the stroke kinks at each period boundary.

def _sine(u):
    return np.sin(2.0 * np.pi * u)


def _hand(u):
    """A few harmonics that read as an unsteady hand, and still close up."""
    return (0.62 * np.sin(2 * np.pi * u)
            + 0.26 * np.sin(4 * np.pi * u + 1.1)
            + 0.12 * np.sin(6 * np.pi * u + 2.3))


WIGGLE_PROFILES = {"sine": _sine, "hand": _hand}


def _p_flat(u):
    return np.ones_like(u)


def _p_swell(u):
    """Fat in the middle of each stamp, thin at the joins."""
    return 0.25 + 0.75 * np.sin(np.pi * u) ** 1.5


def _p_pencil(u):
    """Quick press, slow release: a dragged pencil stroke."""
    return 0.3 + 0.7 * np.clip(np.sin(np.pi * u ** 0.6), 0.0, 1.0)


def _p_blob(u):
    return 0.12 + 0.88 * np.exp(-((u - 0.5) ** 2) / (2 * 0.16 ** 2))


PRESS_PROFILES = {"flat": _p_flat, "swell": _p_swell,
                  "pencil": _p_pencil, "blob": _p_blob}


def end_taper(arc_px, total_px, taper_px):
    """Width envelope that ramps in and out over `taper_px` at each end --
    jot's `taper` preset field."""
    if taper_px <= 0.0 or total_px <= 0.0:
        return np.ones_like(arc_px)
    k = min(taper_px, 0.5 * total_px)
    a = np.clip(arc_px / k, 0.0, 1.0)
    b = np.clip((total_px - arc_px) / k, 0.0, 1.0)
    return np.minimum(a, b) ** 0.6


def curv_envelope(kappa, ref, gamma=0.5, floor=0.0):
    """Width envelope driven by |radial curvature| instead of by position.

        w = floor + (1 - floor) * clamp(|kappa_r| / ref, 0, 1) ** gamma

    `ref` is the curvature at which the stroke reaches full width.  It MUST be
    a constant for the whole sequence: normalizing per frame (by that frame's
    own max, say) would make the width of a stationary point depend on what
    else is on screen, which is the swimming this is meant to remove.

    NaN -- curvature not evaluated at that node -- reads as full width, so an
    incomplete curvature field degrades to the plain stroke rather than to a
    hole.
    """
    k = np.asarray(kappa, dtype=float)
    if ref <= 0.0:
        return np.ones_like(k)
    w = np.clip(np.abs(k) / ref, 0.0, 1.0) ** gamma
    return floor + (1.0 - floor) * np.where(np.isfinite(w), w, 1.0)


class Envelope:
    """How a stroke's width varies along it.

    mode "end"       jot's taper: distance from the stroke's own endpoints.
    mode "curvature" |kappa_r|, a property of the surface point.
    mode "both"      the product -- geometry decides, and the stroke still
                     closes cleanly where it was cut by an occluder rather
                     than by a cusp.
    mode "none"      constant.
    """

    def __init__(self, mode="end", taper_px=0.0, ref=0.0, gamma=0.5,
                 floor=0.0):
        self.mode = mode
        self.taper_px = taper_px
        self.ref = ref
        self.gamma = gamma
        self.floor = floor

    @property
    def needs_curvature(self):
        return self.mode in ("curvature", "both")

    def __call__(self, arc, total, kappa=None, scale=1.0):
        """`scale` converts taper_px into the units `arc` is measured in
        (the raster renderer works in supersampled pixels)."""
        arc = np.asarray(arc, dtype=float)
        if self.mode == "none":
            return np.ones_like(arc)
        if self.mode == "end" or kappa is None:
            return end_taper(arc, total, self.taper_px * scale)
        c = curv_envelope(kappa, self.ref, self.gamma, self.floor)
        if self.mode == "both":
            return c * end_taper(arc, total, self.taper_px * scale)
        return c


# ----------------------------------------------------------- jot presets

_NUM = r"[-+0-9.eE]+"


class Preset:
    """A jot BaseStroke preset (nprdata/stroke_presets/*.pre)."""

    def __init__(self, color=(0, 0, 0), alpha=1.0, width=4.0, taper=0.0,
                 texture=None, name="default"):
        self.color = color
        self.alpha = alpha
        self.width = width
        self.taper = taper
        self.texture = texture
        self.name = name

    @classmethod
    def load(cls, spec):
        path = spec
        if not os.path.isfile(path):
            path = os.path.join(PRESET_DIR, spec)
            if not os.path.isfile(path) and not spec.endswith(".pre"):
                path += ".pre"
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "no such preset: %s (looked in %s)" % (spec, PRESET_DIR))
        txt = open(path).read()

        def num(key, default):
            m = re.search(r"\b%s\s+(%s)\s" % (key, _NUM), txt)
            return float(m.group(1)) if m else default

        m = re.search(r"\bcolor\s*\{\s*(%s)\s+(%s)\s+(%s)\s*\}" % (_NUM, _NUM, _NUM), txt)
        color = tuple(float(m.group(i)) for i in (1, 2, 3)) if m else (0, 0, 0)

        m = re.search(r"stroke_texture_file\s*\{\s*(\S+)\s*\}", txt)
        tex = m.group(1) if m and m.group(1) != "NULL_STR" else None

        return cls(color=color, alpha=num("alpha", 1.0),
                   width=num("width", 4.0), taper=num("taper", 0.0),
                   texture=tex, name=os.path.basename(path))

    def rgb255(self):
        return tuple(int(round(255 * min(max(c, 0.0), 1.0))) for c in self.color)

    def hexcolor(self):
        return "#%02x%02x%02x" % self.rgb255()


def list_presets():
    if not os.path.isdir(PRESET_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(PRESET_DIR) if f.endswith(".pre"))


def list_textures():
    if not os.path.isdir(TEXTURE_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(TEXTURE_DIR) if f.endswith(".png"))


# jot draws an untextured BaseStroke as a plain quad strip, so the stand-in
# for "no texture" is a mask that is solid everywhere.
SOLID = np.ones((1, 1), dtype=np.float32)


def preset_texture(pre, warn=True):
    """(mask, resolved name) for a preset, tolerating the two ways one can
    lack a usable texture.

    Some presets name no stroke texture at all (wash_blue), and one names a
    file jot does not ship (balloon -> noisy1.png).  Both should still draw,
    using their own colour, width and alpha over a solid mask, rather than
    failing.  The returned name is None when the mask is that stand-in."""
    if not pre.texture:
        return SOLID.copy(), None
    try:
        return load_texture(pre.texture), pre.texture
    except FileNotFoundError:
        if warn:
            sys.stderr.write(
                "warning: preset %s names a texture jot does not ship (%s); "
                "drawing it solid\n" % (pre.name, pre.texture))
        return SOLID.copy(), None


def load_texture(spec):
    """Load a jot stroke texture as a float alpha mask, shape (across, along).

    These PNGs carry the ink in their alpha channel; the grey channel is a
    constant 255 because the colour lives in the preset."""
    from PIL import Image

    path = spec
    for cand in (spec,
                 os.path.join(JOT_ROOT, spec),
                 os.path.join(TEXTURE_DIR, spec),
                 os.path.join(TEXTURE_DIR, spec + ".png")):
        if os.path.isfile(cand):
            path = cand
            break
    else:
        raise FileNotFoundError(
            "no such stroke texture: %s (looked in %s)" % (spec, TEXTURE_DIR))

    im = Image.open(path).convert("LA")
    a = np.asarray(im, dtype=np.float32) / 255.0
    alpha = a[..., 1]
    if alpha.max() <= 0.0:          # a mask-less texture: use the grey
        alpha = 1.0 - a[..., 0]
    return alpha                    # rows = across the stroke, cols = along t


# ------------------------------------------------------------- path frames

def polyline_frame(px):
    """Unit tangents and left normals at each vertex of a pixel polyline,
    plus cumulative arclength."""
    p = np.asarray(px, dtype=float)
    d = np.diff(p, axis=0)
    seg = np.linalg.norm(d, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])

    tan = np.zeros_like(p)
    good = seg > 1e-12
    unit = np.zeros_like(d)
    unit[good] = d[good] / seg[good, None]
    tan[:-1] += unit
    tan[1:] += unit
    ln = np.linalg.norm(tan, axis=1)
    ln[ln < 1e-12] = 1.0
    tan /= ln[:, None]

    nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    return tan, nrm, arc
