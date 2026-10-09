#!/usr/bin/env python3
"""
Run Coherent Stylized Silhouettes (Kalnins et al. 2003) over a directory of
exported 3D contour frames.

  python3 stylize.py <results-dir> [--out DIR] [options]

<results-dir> is either one frame directory or a parent holding many; each
frame directory must contain contour_export_3d.obj,
contour_export_3d_edge_visibility.txt and contour_export_3d_camera.json.

Frames are processed in sorted order and the parameterization is carried
forward between them -- that is the whole point, so do not reorder or
parallelize them.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brushes                                          # noqa: E402
import curvature                                        # noqa: E402
from css import COVER_NAMES, FIT_NAMES, Params          # noqa: E402
from export_io import find_frames, load_frame           # noqa: E402
from stylizer import Stylizer                           # noqa: E402
from svgout import (PRESS_PROFILES, STYLES, WIGGLE_PROFILES, Brush,  # noqa: E402
                    strokes_to_pixels, write_svg)

DEFAULT_TEXTURE = "2D--gauss-med-32"


def build_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", help="frame directory, or a directory "
                                           "of them")
    p.add_argument("--out", help="output directory.  Default for SVG is "
                                 "<root>_css beside the input; for PNG it is "
                                 "the sibling results/png/<name>_css, which "
                                 "is where make_video.py looks")
    p.add_argument("--start", type=int, default=0, help="first frame index")
    p.add_argument("--end", type=int, help="stop before this frame index")

    g = p.add_argument_group("coherence (section 5)")
    g.add_argument("--fit", choices=sorted(FIT_NAMES), default="optimize",
                   help="fit policy (jot default: optimize)")
    g.add_argument("--cover", choices=sorted(COVER_NAMES), default="hybrid",
                   help="coverage policy (jot default: hybrid)")
    g.add_argument("--fit-pix", type=float, default=48.0,
                   help="knot spacing for the optimizing fit, in pixels")
    g.add_argument("--wf", type=float, default=1.0, help="fit weight")
    g.add_argument("--ws", type=float, default=1.0, help="scale weight")
    g.add_argument("--wb", type=float, default=1.0, help="distortion weight")
    g.add_argument("--wh", type=float, default=1.0, help="healing weight")
    g.add_argument("--no-heal", action="store_true", help="disable healing")

    g = p.add_argument_group("propagation (section 4)")
    g.add_argument("--max-steps", type=int, default=6,
                   help="MAX_LUBO_STEPS: search distance, in pixels")
    g.add_argument("--sample-step", type=int, default=4,
                   help="LUBO_SAMPLE_STEP: sample spacing multiplier")
    g.add_argument("--one-sided", action="store_true",
                   help="search only along +normal, as jot does")
    g.add_argument("--vote-from-hidden", action="store_true",
                   help="let samples hidden last frame vote too; jot does not")
    g.add_argument("--incoherent", action="store_true",
                   help="drop propagation entirely: each frame parameterized "
                        "from scratch.  Use this to see what CSS is fixing")

    g = p.add_argument_group("brush")
    g.add_argument("--style", choices=STYLES + ["texture"], default="dash",
                   help="mark to lay down along the stroke.  'texture' stamps "
                        "a jot stroke texture and writes PNG; the rest are "
                        "vector and write SVG")
    g.add_argument("--preset", help="jot stroke preset (name or path to a "
                                    ".pre); supplies colour, alpha, width, "
                                    "taper and texture.  Implies --style "
                                    "texture unless you say otherwise")
    g.add_argument("--texture", help="jot stroke texture, name or path "
                                     "(default: %s)" % DEFAULT_TEXTURE)
    g.add_argument("--period-pix", type=float,
                   help="pixels per unit of stroke parameter t: the "
                        "stylization period.  Default 30 for vector brushes; "
                        "for a texture it defaults to width * (W/H) of the "
                        "texture, so the stamp is not squashed")
    g.add_argument("--duty", type=float, default=0.6,
                   help="inked fraction of each period (dash, stipple)")
    g.add_argument("--width", type=float, help="brush width in pixels")
    g.add_argument("--color", help="e.g. #000000")
    g.add_argument("--opacity", type=float, help="0..1")
    g.add_argument("--taper", type=float,
                   help="pixels over which the brush narrows at each stroke "
                        "end (ribbon and texture)")
    g.add_argument("--taper-mode", choices=["end", "curvature", "both",
                                            "none"], default="curvature",
                   help="what drives the width envelope (default "
                        "curvature).  'end' is jot's: "
                        "distance from the stroke's own ends, which move "
                        "whenever a chain splits or an occluder sweeps past, "
                        "so the envelope swims even though the "
                        "parameterization under it does not.  'curvature' "
                        "drives it from the per-node radial curvature "
                        "instead -- a property of the surface point, hence "
                        "as coherent as the geometry, and near zero exactly "
                        "at the cusps where the contour really does end.  "
                        "'both' multiplies them.  A frame set with no "
                        "curvature in it falls back to 'end' with a warning, "
                        "unless you asked for curvature explicitly")
    g.add_argument("--curv-ref", type=float, default=0.0, metavar="K",
                   help="|radial curvature| at which the stroke reaches full "
                        "width.  0 (default) picks the median over the first "
                        "frame's visible nodes and then HOLDS it: "
                        "re-normalizing per frame would make a stationary "
                        "point's width depend on what else is on screen")
    g.add_argument("--curv-gamma", type=float, default=1.0, metavar="G",
                   help="shaping exponent on the normalized curvature; "
                        "below 1 widens the midtones, above 1 thins them")
    g.add_argument("--curv-floor", type=float, default=0.0, metavar="F",
                   help="narrowest the envelope goes, as a fraction of full "
                        "width.  0 lets a stroke taper away to nothing")
    g.add_argument("--press", choices=sorted(PRESS_PROFILES), default="flat",
                   help="width profile within each period (ribbon, texture)")
    g.add_argument("--wiggle", type=float, default=0.0,
                   help="lateral displacement amplitude in pixels, one "
                        "wobble per period -- jot's BaseStrokeOffset model")
    g.add_argument("--wiggle-profile", choices=sorted(WIGGLE_PROFILES),
                   default="hand")
    g.add_argument("--background", help="e.g. #ffffff; default transparent")
    g.add_argument("--canvas", type=int, default=0, metavar="N",
                   help="render every frame on a fixed N x N canvas covering "
                        "the whole NDC square.  Default 0 mirrors each "
                        "frame's own crop")
    g.add_argument("--ss", type=int, default=2,
                   help="supersampling for textured brushes")

    p.add_argument("--list-brushes", action="store_true",
                   help="list available jot presets and textures, then exit")
    p.add_argument("--report", action="store_true",
                   help="print propagation and swimming diagnostics")
    p.add_argument("--json", action="store_true",
                   help="also write the raw (t, position) stroke samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def default_out_dir(root, brush_tag=None):
    """Where results go when --out is not given.

    SVG lands beside the input as <root>_css.  PNG goes to the sibling
    results/png/ instead, matching the <png-dir>/<object> layout
    make_video.py already uses, so a textured run drops straight into the
    video pipeline with no SVG-to-PNG step.

    The PNG folder is tagged with the brush, e.g. bunny_hr_css_pencil.  Plain
    <name>_css would collide with the folder make_video.py fills when it
    rasterizes the SVG run of the same name, silently destroying it."""
    root = root.rstrip(os.sep)
    base = os.path.basename(root) + "_css"
    if brush_tag:
        parent, tail = os.path.split(os.path.dirname(root))
        name = base + "_" + re.sub(r"[^A-Za-z0-9._-]+", "-", brush_tag)
        if tail == "rendering":
            return os.path.join(parent, "png", name)
        return os.path.join(os.path.dirname(root), name)
    return root + "_css"


def _hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def resolve_brush(args):
    """Merge the preset (if any) with the explicit flags; flags win."""
    pre = brushes.Preset.load(args.preset) if args.preset else None

    style = args.style
    if pre is not None and "--style" not in sys.argv:
        # jot draws a preset as a textured quad strip, solid when the preset
        # names no texture.  Honour that whether or not it has one.
        style = "texture"

    width = args.width
    if width is None:
        width = pre.width if pre else (2.0 if style != "texture" else 6.0)

    color = args.color or (pre.hexcolor() if pre else "#000000")
    opacity = args.opacity
    if opacity is None:
        opacity = pre.alpha if pre else 1.0
    taper = args.taper
    if taper is None:
        taper = pre.taper if pre else 0.0

    texture = args.texture or (pre.texture if pre else None)
    if texture is None and pre is None:
        texture = DEFAULT_TEXTURE

    brush = Brush(style=style, width=width, color=color, duty=args.duty,
                  wiggle=args.wiggle, wiggle_profile=args.wiggle_profile,
                  taper=taper, press_profile=args.press, opacity=opacity)
    brush.envelope = brushes.Envelope(args.taper_mode, taper_px=taper,
                                      ref=args.curv_ref,
                                      gamma=args.curv_gamma,
                                      floor=args.curv_floor)
    return brush, texture, pre


def _auto_curv_ref(frame):
    """Median |kappa_r| over this frame's VISIBLE nodes.

    Visible only: the far side of the surface carries the opposite sign and a
    different magnitude (on the bunny, 58% of hidden nodes are negative
    against 0.3% of visible ones), so including it would move the reference
    for reasons the viewer never sees.
    """
    vals = []
    for c in frame.chains:
        if c.kappa is None:
            continue
        vis = np.zeros(len(c.wpts), dtype=bool)
        vis[:-1] |= c.edge_vis
        vis[1:] |= c.edge_vis
        k = np.abs(c.kappa[vis])
        vals.append(k[np.isfinite(k)])
    if not vals:
        return 0.0
    allk = np.concatenate(vals)
    return float(np.median(allk)) if len(allk) else 0.0


def main(argv=None):
    args = build_args().parse_args(argv)

    if args.list_brushes:
        print("presets (nprdata/stroke_presets) -- use with --preset:")
        for n in brushes.list_presets():
            print("   ", n)
        print("\ntextures (nprdata/stroke_textures) -- use with --texture:")
        for n in brushes.list_textures():
            print("   ", n)
        return
    if not args.root:
        build_args().error("the following arguments are required: root")

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit("no such directory: %s" % root)
    frames = find_frames(root)
    if not frames:
        sys.exit("no frame directories with contour_export_3d_camera.json "
                 "under %s" % root)
    frames = frames[args.start:args.end]

    try:
        brush, texture_name, preset = resolve_brush(args)
    except FileNotFoundError as e:
        sys.exit("%s\n(run --list-brushes to see what is available)" % e)

    tex = None
    period = args.period_pix
    if brush.style == "texture":
        import raster
        if preset and not args.texture:
            tex, texture_name = brushes.preset_texture(preset)
        else:
            try:
                tex = brushes.load_texture(texture_name)
            except FileNotFoundError as e:
                sys.exit("%s\n(run --list-brushes to see what is available)"
                         % e)
        if not args.canvas:
            sys.exit("--style texture needs a fixed canvas; pass --canvas 840")
        if period is None:
            # One stamp per period, at the texture's own aspect ratio, so the
            # mark is not squashed.  A 1-column texture is a pure
            # cross-section with no structure along the stroke, so its period
            # is arbitrary; extreme aspects are clamped to something a stroke
            # can actually fit.
            across, along = tex.shape
            period = (30.0 if along <= 1
                      else min(max(brush.width * along / float(across),
                                   8.0), 240.0))
    if period is None:
        period = 30.0

    tag = None
    if tex is not None:
        tag = (os.path.splitext(preset.name)[0] if preset
               else os.path.splitext(os.path.basename(texture_name))[0]
               if texture_name else "solid")
    out_dir = args.out or default_out_dir(root, tag)
    os.makedirs(out_dir, exist_ok=True)

    params = Params(
        fit_type=FIT_NAMES[args.fit],
        cover_type=COVER_NAMES[args.cover],
        fit_pix=args.fit_pix,
        weight_fit=args.wf,
        weight_scale=args.ws,
        weight_distort=args.wb,
        weight_heal=0.0 if args.no_heal else args.wh,
        max_lubo_steps=args.max_steps,
        lubo_sample_step=args.sample_step,
        search_both=not args.one_sided,
        vote_requires_visible=not args.vote_from_hidden,
        offset_pix_len=period,
        seed=args.seed,
    )

    def describe_brush():
        """Printed after the first frame, not before: the envelope is only
        settled once we know whether that frame carries curvature."""
        env = brush.envelope
        taper = (env.mode if not env.taper_px or env.mode == "curvature"
                 else "%s/%.0fpx" % (env.mode, env.taper_px))
        print("brush: %s  width %.1f  period %.0fpx%s%s  taper %s"
              % (brush.style, brush.width, period,
                 "  texture " + (texture_name or "solid")
                 if tex is not None else "",
                 "  preset " + preset.name if preset else "", taper))

    sty = Stylizer(params)
    t0 = time.time()
    totals = {"votes": 0, "strokes": 0, "groups": 0}
    resid, sizes = [], set()

    curv_src = None
    asked_for_curvature = "--taper-mode" in sys.argv
    for i, d in enumerate(frames):
        name = os.path.basename(d.rstrip(os.sep)) or "frame"
        frame = load_frame(d)

        if brush.envelope.needs_curvature:
            src = curvature.load(d, frame.cam, frame.chains)
            if src is None:
                msg = ("%s carries no per-node radial curvature: neither %s "
                       "nor a %s with data-radial-curvature.  See README "
                       "\"Radial curvature\"."
                       % (d, curvature.KAPPA, curvature.QI_SVG))
                if asked_for_curvature:
                    sys.exit("--taper-mode %s needs it.  %s"
                             % (brush.envelope.mode, msg))
                # Curvature is only the DEFAULT here, not a request, so fall
                # back to the taper jot itself would have used rather than
                # refusing to draw an older export.
                sys.stderr.write("warning: %s\n         falling back to "
                                 "--taper-mode end\n" % msg)
                # needs_curvature is False from here on, so later frames
                # skip this block entirely.
                brush.envelope = brushes.Envelope("end",
                                                  taper_px=brush.taper)
            elif curv_src is None:
                curv_src = src
                if brush.envelope.ref <= 0.0:
                    brush.envelope.ref = _auto_curv_ref(frame)
                if brush.envelope.ref <= 0.0:
                    sys.exit("curvature field is empty or all NaN in %s" % d)
                if not args.quiet:
                    print("curvature: %s, reference |kappa_r| = %.4f "
                          "(gamma %.2f, floor %.2f)"
                          % (src, brush.envelope.ref, brush.envelope.gamma,
                             brush.envelope.floor))

        if i == 0 and not args.quiet:
            describe_brush()

        if args.incoherent:
            sty.samples = []

        res = sty.run_frame(frame)

        if tex is not None:
            import raster
            cov = raster.render(
                strokes_to_pixels(res.strokes, frame.cam, args.canvas),
                args.canvas, tex, width=brush.width, taper_px=brush.taper,
                wiggle_px=brush.wiggle, wiggle_profile=brush.wiggle_profile,
                press_profile=brush.press_profile, ss=args.ss,
                envelope=brush.envelope)
            raster.write_png(
                os.path.join(out_dir, name + ".png"), cov,
                color=_hex_to_rgb(brush.color), alpha=brush.opacity,
                background=_hex_to_rgb(args.background) if args.background
                else None)
        else:
            write_svg(os.path.join(out_dir, name + ".svg"), res.strokes,
                      frame.cam, brush, background=args.background,
                      canvas=args.canvas)

        if args.json:
            data = [{"path": s.path_index, "group": s.group_id,
                     "t": [round(v, 6) for v in s.t],
                     "ndc": [[round(float(q[0]), 6), round(float(q[1]), 6)]
                             for q in s.ndc]}
                    for s in res.strokes]
            with open(os.path.join(out_dir, name + ".json"), "w") as f:
                json.dump({"frame": name, "strokes": data}, f)

        sizes.add((round(frame.cam.svg_w, 2), round(frame.cam.svg_h, 2)))
        for k in totals:
            totals[k] += res.stats[k]
        if res.stats.get("resid_n"):
            resid.append(res.stats["resid_mean_px"])

        if not args.quiet:
            s = res.stats
            line = ("[%3d/%3d] %-8s paths %-3d samples %-5d votes %-5d "
                    "groups %-4d strokes %-4d"
                    % (i + 1, len(frames), name, s["paths"], s["samples_in"],
                       s["votes"], s["groups"], s["strokes"]))
            if args.report:
                line += ("  swim %5.2fpx (med %5.2f)  miss: frustum %d "
                         "normal %d search %d far %d hidden %d"
                         % (s["resid_mean_px"], s["resid_med_px"],
                            s["miss_frustum"], s["miss_normal"],
                            s["miss_search"], s["miss_far"],
                            s["miss_hidden"]))
            print(line)

    dt = time.time() - t0
    print("\n%d frames in %.1fs (%.2fs/frame) -> %s"
          % (len(frames), dt, dt / max(len(frames), 1), out_dir))
    print("totals: votes %d  groups %d  strokes %d"
          % (totals["votes"], totals["groups"], totals["strokes"]))
    if len(sizes) > 1 and not args.canvas:
        print("note: the export uses %d different crop sizes across these "
              "frames, so the SVGs vary in size.\n      Pass --canvas 840 "
              "for a fixed canvas if you are making an animation."
              % len(sizes))
    if resid:
        print("mean swimming across frames: %.3f px of a %.0f px period"
              % (statistics.mean(resid), period))


if __name__ == "__main__":
    main()
