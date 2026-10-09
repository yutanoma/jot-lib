"""
Vector brushes: rendering stylized strokes to SVG.

Every mark here is driven by the coherent stroke parameter `t`, which is the
point of the exercise -- a dash pattern laid out in screen arclength (what
SVG's own `stroke-dasharray` would give you) swims along the line as the
model turns, while one laid out in `t` stays pinned to the surface.

Textured brushes live in raster.py; SVG cannot warp a bitmap along a curve
without an unreasonable number of elements.
"""

import colorsys
import math

import numpy as np

from brushes import (PRESS_PROFILES, WIGGLE_PROFILES, Envelope, end_taper,
                     polyline_frame)

STYLES = ["plain", "dash", "stipple", "ribbon", "phase"]


class Brush:
    """Everything the vector renderers need, in one place."""

    def __init__(self, style="dash", width=2.0, color="#000000", duty=0.6,
                 wiggle=0.0, wiggle_profile="hand", taper=0.0,
                 press_profile="flat", opacity=1.0):
        self.style = style
        self.width = width
        self.color = color
        self.duty = duty
        self.wiggle = wiggle
        self.wiggle_profile = wiggle_profile
        self.taper = taper
        # Set by stylize.py; `end` reproduces the previous behaviour exactly.
        self.envelope = Envelope("end", taper_px=taper)
        self.press_profile = press_profile
        self.opacity = opacity


def _lerp(a, b, w):
    return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)


def split_by_param(ts, pts, duty):
    """Cut a stroke into the pieces whose fractional parameter is < duty.

    Returns a list of (points, period index)."""
    if duty >= 1.0:
        return [(list(pts), 0)]

    pieces = []
    run, run_k = [], None

    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        p0, p1 = pts[i], pts[i + 1]
        if t1 == t0:
            continue
        lo, hi = (t0, t1) if t1 > t0 else (t1, t0)

        cuts = set()
        k = math.floor(lo)
        while k <= math.ceil(hi):
            for b in (float(k), k + duty):
                if lo < b < hi:
                    cuts.add(b)
            k += 1
        bounds = [lo] + sorted(cuts) + [hi]

        for a, b in zip(bounds, bounds[1:]):
            if b <= a:
                continue
            mid = 0.5 * (a + b)
            on = (mid - math.floor(mid)) < duty
            ua = (a - t0) / (t1 - t0)
            ub = (b - t0) / (t1 - t0)
            if t1 < t0:
                ua, ub = ub, ua
            pa, pb = _lerp(p0, p1, ua), _lerp(p0, p1, ub)
            kk = int(math.floor(mid))
            if on:
                if run and run_k == kk:
                    run.append(pb)
                else:
                    if len(run) > 1:
                        pieces.append((run, run_k))
                    run, run_k = [pa, pb], kk
            else:
                if len(run) > 1:
                    pieces.append((run, run_k))
                run, run_k = [], None

    if len(run) > 1:
        pieces.append((run, run_k))
    return pieces


def apply_wiggle(ts, px, amp, profile):
    """Displace the stroke sideways by a profile repeating once per period --
    jot's BaseStrokeOffset rubber-stamping, in miniature."""
    if not amp:
        return px
    p = np.asarray(px, dtype=float)
    _, nrm, _ = polyline_frame(p)
    f = WIGGLE_PROFILES[profile](np.asarray(ts, dtype=float) % 1.0)
    return p + nrm * (amp * f)[:, None]


def _hue(x):
    r, g, b = colorsys.hsv_to_rgb(x % 1.0, 0.85, 0.85)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _poly(pts, color, width, opacity=1.0):
    d = " ".join("%.3f,%.3f" % (x, y) for x, y in pts)
    op = "" if opacity >= 1.0 else ' stroke-opacity="%.3f"' % opacity
    return ('  <polyline points="%s" fill="none" stroke="%s" '
            'stroke-width="%.3f" stroke-linecap="round" '
            'stroke-linejoin="round"%s/>\n' % (d, color, width, op))


def _ribbon(ts, px, brush, kappa=None):
    """A filled outline whose half-width follows a pressure profile and the
    width envelope -- a stroke with weight, rather than a constant hairline."""
    p = np.asarray(px, dtype=float)
    if len(p) < 2:
        return ""
    _, nrm, arc = polyline_frame(p)
    t = np.asarray(ts, dtype=float)
    half = (0.5 * brush.width
            * PRESS_PROFILES[brush.press_profile](t % 1.0)
            * brush.envelope(arc, arc[-1], kappa))
    L = p + nrm * half[:, None]
    R = p - nrm * half[:, None]
    ring = list(L) + list(R[::-1])
    d = "M " + " L ".join("%.3f,%.3f" % (q[0], q[1]) for q in ring) + " Z"
    op = "" if brush.opacity >= 1.0 else ' fill-opacity="%.3f"' % brush.opacity
    return '  <path d="%s" fill="%s" stroke="none"%s/>\n' % (d, brush.color, op)


def _stipple(ts, px, brush, kappa=None):
    """One dot per period, placed at the centre of the inked span.  The
    envelope scales each dot, so a curvature-driven run fades out in dot size
    the way a ribbon fades in width."""
    out = []
    r0 = 0.5 * brush.width
    # The envelope is per sample of the stroke, while the dots sit wherever a
    # period lands, so each dot reads the value of the sample nearest to it.
    samples = np.asarray(px, dtype=float)
    env = (None if kappa is None
           else brush.envelope(np.zeros(len(samples)), 1.0, kappa))
    op = "" if brush.opacity >= 1.0 else ' fill-opacity="%.3f"' % brush.opacity
    for pts, _period in split_by_param(ts, px, brush.duty):
        a = np.asarray(pts, dtype=float)
        seg = np.linalg.norm(np.diff(a, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] <= 0:
            continue
        i = int(np.searchsorted(arc, 0.5 * arc[-1]))
        i = min(max(i, 1), len(a) - 1)
        w = (0.5 * arc[-1] - arc[i - 1]) / max(seg[i - 1], 1e-9)
        c = a[i - 1] + (a[i] - a[i - 1]) * w
        r = r0
        if env is not None:
            j = int(((samples - c) ** 2).sum(axis=1).argmin())
            r = r0 * float(env[j])
        if r <= 0.0:
            continue
        out.append('  <circle cx="%.3f" cy="%.3f" r="%.3f" fill="%s"%s/>\n'
                   % (c[0], c[1], r, brush.color, op))
    return "".join(out)


def write_svg(path, strokes, cam, brush, background=None, canvas=0):
    """Write one frame.

    With canvas == 0 the output mirrors this frame's own crop, so it overlays
    the exporter's SVGs exactly.  With canvas == N every frame is drawn on the
    same N x N square covering the full NDC range, which is what you want for
    an animation when the exporter's crop varies frame to frame."""
    if canvas:
        w = h = float(canvas)
    else:
        w, h = cam.svg_w, cam.svg_h

    out = ['<?xml version="1.0" encoding="UTF-8"?>\n',
           '<svg xmlns="http://www.w3.org/2000/svg" width="%g" height="%g" '
           'viewBox="0 0 %g %g">\n' % (w, h, w, h)]
    if background:
        out.append('  <rect width="%g" height="%g" fill="%s"/>\n'
                   % (w, h, background))

    for st in strokes:
        mapped = (cam.ndc_to_canvas(st.ndc, canvas) if canvas
                  else cam.ndc_to_svg(st.ndc))
        px = apply_wiggle(st.t, mapped, brush.wiggle, brush.wiggle_profile)
        px = [tuple(q) for q in px]

        if brush.style == "plain":
            out.append(_poly(px, brush.color, brush.width, brush.opacity))
        elif brush.style == "ribbon":
            out.append(_ribbon(st.t, px, brush, st.k))
        elif brush.style == "stipple":
            out.append(_stipple(st.t, px, brush, st.k))
        elif brush.style == "phase":
            for i in range(len(px) - 1):
                tm = 0.5 * (st.t[i] + st.t[i + 1])
                out.append(_poly([px[i], px[i + 1]], _hue(tm), brush.width,
                                 brush.opacity))
        else:   # dash
            for pts, _k in split_by_param(st.t, px, brush.duty):
                out.append(_poly(pts, brush.color, brush.width, brush.opacity))

    out.append("</svg>\n")
    with open(path, "w") as f:
        f.writelines(out)


def strokes_to_pixels(strokes, cam, canvas):
    """(t, pixel-polyline, |kappa_r|) triples, for the raster renderer."""
    return [(list(st.t),
             cam.ndc_to_canvas(st.ndc, canvas) if canvas
             else cam.ndc_to_svg(st.ndc),
             st.k)
            for st in strokes]
