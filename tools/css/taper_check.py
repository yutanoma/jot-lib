#!/usr/bin/env python3
"""
Temporal stability of the width envelope: end taper vs radial curvature.

  python3 taper_check.py <results-dir> [--frames N] [--taper PX]

The taper affects neither the stroke geometry nor the parameterization, so a
single run of the stylizer supplies both envelopes on identical samples.
Each sample of frame i is then matched to the nearest point of frame i+1's
curves and the metric is how much the envelope at a fixed place on screen
changed -- which is what the eye does.  Samples with no partner are contour
that genuinely appeared or vanished and are dropped.

`spread` is in the output because a constant envelope would score a perfect
zero.  Compare rows with similar spread, or read the mean/spread column.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brushes                                          # noqa: E402
import curvature                                        # noqa: E402
from brushes import polyline_frame                      # noqa: E402
from css import Params                                  # noqa: E402
from export_io import find_frames, load_frame           # noqa: E402
from stylizer import Stylizer                           # noqa: E402

GAMMAS = (0.35, 0.5, 0.75, 1.0, 1.5)
CANVAS = 840
MATCH_PX = 1.5
DENSIFY_PX = 0.4


def _visible_kappa(frame):
    out = []
    for c in frame.chains:
        if c.kappa is None:
            continue
        vis = np.zeros(len(c.wpts), dtype=bool)
        vis[:-1] |= c.edge_vis
        vis[1:] |= c.edge_vis
        k = np.abs(c.kappa[vis])
        out.append(k[np.isfinite(k)])
    return np.concatenate(out) if out else np.empty(0)


def _densify(P, E, C, step=DENSIFY_PX):
    """Resample each stroke, carrying every envelope.

    Matching sample to sample undercounts badly -- the stylizer places
    samples every 6 px and the camera moves between frames -- but the CURVE
    is in nearly the same place even when the samples are not."""
    Q, QE, QC = [], [], []
    for px, e, c in zip(P, E, C):
        d = np.linalg.norm(np.diff(px, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(d)])
        if arc[-1] <= step:
            Q.append(px), QE.append(e), QC.append(c)
            continue
        u = np.arange(0.0, arc[-1], step)
        Q.append(np.stack([np.interp(u, arc, px[:, 0]),
                           np.interp(u, arc, px[:, 1])], axis=1))
        QE.append(np.interp(u, arc, e))
        QC.append(np.stack([np.interp(u, arc, ci) for ci in c]))
    return (np.concatenate(Q), np.concatenate(QE),
            np.concatenate(QC, axis=1))


def _match(p0, p1, r):
    """Nearest neighbour within r, via a uniform cell hash."""
    cell = {}
    for i, (a, b) in enumerate(np.floor(p1 / r).astype(np.int64)):
        cell.setdefault((a, b), []).append(i)
    cell = {k: np.asarray(v) for k, v in cell.items()}
    idx = np.full(len(p0), -1, dtype=np.int64)
    for i, (a, b) in enumerate(np.floor(p0 / r).astype(np.int64)):
        cand = [cell[key] for key in
                ((a + da, b + db) for da in (-1, 0, 1) for db in (-1, 0, 1))
                if key in cell]
        if not cand:
            continue
        cand = np.concatenate(cand)
        d = np.linalg.norm(p1[cand] - p0[i], axis=1)
        j = int(d.argmin())
        if d[j] <= r:
            idx[i] = cand[j]
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--taper", type=float, default=20.0, metavar="PX",
                    help="end-taper length to compare against")
    args = ap.parse_args(argv)

    frames = find_frames(args.root)[:args.frames]
    if len(frames) < 2:
        sys.exit("need at least two frames under %s" % args.root)

    sty = Stylizer(Params())
    env_end = brushes.Envelope("end", taper_px=args.taper)
    env_cur = [brushes.Envelope("curvature", gamma=g) for g in GAMMAS]

    per_frame = []
    for d in frames:
        fr = load_frame(d)
        if curvature.load(d, fr.cam, fr.chains) is None:
            sys.exit("no radial curvature in %s -- see README "
                     "\"Radial curvature\"" % d)
        if env_cur[0].ref <= 0.0:
            ref = float(np.median(_visible_kappa(fr)))
            for e in env_cur:
                e.ref = ref
            print("reference |kappa_r| = %.4f (median over the first frame's "
                  "visible nodes)" % ref)

        res = sty.run_frame(fr)
        P, E, C = [], [], []
        for st in res.strokes:
            px = fr.cam.ndc_to_canvas(st.ndc, CANVAS)
            if len(px) < 2:
                continue
            _, _, arc = polyline_frame(px)
            P.append(px)
            E.append(env_end(arc, arc[-1]))
            C.append(np.stack([e(arc, arc[-1], st.k) for e in env_cur]))
        if P:
            per_frame.append((np.concatenate(P), np.concatenate(E),
                              np.concatenate(C, axis=1), _densify(P, E, C)))

    de, dc, ntot = [], [[] for _ in GAMMAS], 0
    for (p0, e0, c0, _), (_, _, _, (p1, e1, c1)) in zip(per_frame,
                                                        per_frame[1:]):
        idx = _match(p0, p1, MATCH_PX)
        ok = idx >= 0
        ntot += len(p0)
        de.append(np.abs(e0[ok] - e1[idx[ok]]))
        for gi in range(len(GAMMAS)):
            dc[gi].append(np.abs(c0[gi][ok] - c1[gi][idx[ok]]))
    de = np.concatenate(de)
    dc = [np.concatenate(x) for x in dc]
    alle = np.concatenate([f[1] for f in per_frame])
    allc = np.concatenate([f[2] for f in per_frame], axis=1)

    print("\n%d frames, %d of %d samples matched within %.1f px (%.0f%%)\n"
          % (len(per_frame), len(de), ntot, MATCH_PX,
             100.0 * len(de) / max(ntot, 1)))
    hdr = ("envelope", "spread", "mean|d|", "p95|d|", "max|d|", ">0.25",
           "mean/spread")
    print("%-22s %8s %8s %8s %8s %8s %12s" % hdr)
    rows = [("end taper (%g px)" % args.taper, de, alle)]
    rows += [("curvature gamma %.2f" % g, dc[i], allc[i])
             for i, g in enumerate(GAMMAS)]
    for nm, d, a in rows:
        print("%-22s %8.3f %8.4f %8.4f %8.4f %7.2f%% %12.3f"
              % (nm, a.std(), d.mean(), np.percentile(d, 95), d.max(),
                 100 * (d > 0.25).mean(), d.mean() / max(a.std(), 1e-9)))


if __name__ == "__main__":
    main()
