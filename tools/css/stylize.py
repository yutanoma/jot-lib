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
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from css import COVER_NAMES, FIT_NAMES, Params          # noqa: E402
from export_io import find_frames, load_frame           # noqa: E402
from stylizer import Stylizer                           # noqa: E402
from svgout import write_svg                            # noqa: E402


def build_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="frame directory, or a directory of them")
    p.add_argument("--out", help="output directory "
                                 "(default: <root>_css next to the input)")
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
                   help="search only along +normal, as jot does.  Default is "
                        "to try both, since an external normal's outward sign "
                        "is not guaranteed to match jot's")
    g.add_argument("--vote-from-hidden", action="store_true",
                   help="let samples that were hidden last frame vote too.  "
                        "jot does not: it matches sample visibility to path "
                        "visibility")
    g.add_argument("--incoherent", action="store_true",
                   help="drop propagation entirely: each frame parameterized "
                        "from scratch.  Use this to see what CSS is fixing")

    g = p.add_argument_group("stylization")
    g.add_argument("--period-pix", type=float, default=30.0,
                   help="pixels per unit of stroke parameter t")
    g.add_argument("--style", choices=["dash", "plain", "phase"],
                   default="dash")
    g.add_argument("--duty", type=float, default=0.6,
                   help="fraction of each period that is inked (dash style)")
    g.add_argument("--width", type=float, default=2.0, help="stroke width, px")
    g.add_argument("--color", default="#000000")
    g.add_argument("--background", help="e.g. #ffffff; default transparent")
    g.add_argument("--canvas", type=int, default=0, metavar="N",
                   help="render every frame on a fixed N x N canvas covering "
                        "the whole NDC square.  Default 0 mirrors each "
                        "frame's own crop, which overlays the exporter's own "
                        "SVGs but varies in size between frames")

    p.add_argument("--report", action="store_true",
                   help="print propagation and swimming diagnostics")
    p.add_argument("--json", action="store_true",
                   help="also write the raw (t, position) stroke samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_args().parse_args(argv)

    root = os.path.abspath(args.root)
    frames = find_frames(root)
    if not frames:
        sys.exit("no frame directories with contour_export_3d_camera.json "
                 "under %s" % root)
    frames = frames[args.start:args.end]

    out_dir = args.out or (root.rstrip(os.sep) + "_css")
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
        offset_pix_len=args.period_pix,
        seed=args.seed,
    )

    sty = Stylizer(params)
    t0 = time.time()
    sizes = set()
    totals = {"votes": 0, "strokes": 0, "groups": 0}
    resid = []

    for i, d in enumerate(frames):
        name = os.path.basename(d.rstrip(os.sep)) or "frame"
        frame = load_frame(d)

        if args.incoherent:
            sty.samples = []

        res = sty.run_frame(frame)

        write_svg(os.path.join(out_dir, name + ".svg"), res.strokes,
                  frame.cam, style=args.style, duty=args.duty,
                  width=args.width, color=args.color,
                  background=args.background, canvas=args.canvas)

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
        import statistics
        print("mean swimming across frames: %.3f px of a %.0f px period"
              % (statistics.mean(resid), args.period_pix))


if __name__ == "__main__":
    main()
