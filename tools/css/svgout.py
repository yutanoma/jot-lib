"""
Rendering stylized strokes to SVG.

The point of CSS is that the stroke-space parameter `t` is temporally
coherent, so any stylization expressed as a function of `t` stays put on the
line instead of swimming.  These renderers all drive off `t` for exactly that
reason -- the dash pattern is in parameter space, not screen arclength, so
`stroke-dasharray` would not do.
"""

import colorsys
import math


def _lerp(a, b, w):
    return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)


def split_by_param(ts, pts, duty):
    """Cut a stroke into the pieces whose fractional parameter is < duty.

    Returns a list of (points, k) where k is the integer period index."""
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

        # Every period boundary and every duty boundary inside (lo, hi).
        cuts = set()
        k0 = math.floor(lo)
        k = k0
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


def _hue(x):
    r, g, b = colorsys.hsv_to_rgb(x % 1.0, 0.85, 0.85)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _poly(pts, color, width, extra=""):
    d = " ".join("%.3f,%.3f" % (x, y) for x, y in pts)
    return ('  <polyline points="%s" fill="none" stroke="%s" '
            'stroke-width="%.3f" stroke-linecap="round" '
            'stroke-linejoin="round"%s/>\n' % (d, color, width, extra))


def write_svg(path, strokes, cam, style="dash", duty=0.6, width=2.0,
              color="#000000", background=None, canvas=0):
    """Write one frame.  `strokes` carry NDC points; they are mapped into the
    SVG pixel frame here.

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
        px = [tuple(q) for q in mapped]

        if style == "plain":
            out.append(_poly(px, color, width))
            continue

        if style == "phase":
            # One short segment per sample, tinted by fractional parameter.
            for i in range(len(px) - 1):
                tm = 0.5 * (st.t[i] + st.t[i + 1])
                out.append(_poly([px[i], px[i + 1]], _hue(tm), width))
            continue

        # style == "dash": a dash pattern laid out in parameter space.
        for pts, k in split_by_param(st.t, px, duty):
            out.append(_poly(pts, color, width))

    out.append("</svg>\n")
    with open(path, "w") as f:
        f.writelines(out)
