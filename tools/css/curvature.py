"""
Per-node radial curvature for the CSS port.

WHY THIS EXISTS
---------------
`end_taper` thins a stroke by how far a point is from the stroke's own ENDS,
measured in screen arclength.  Those ends are not a property of the surface:
they move when a chain splits or merges, when an occlusion boundary sweeps
across, and when a group's span is re-cut.  So the width envelope swims even
though the parameterization underneath it does not -- the one part of the
drawing CSS's vote propagation never reached.

Radial curvature

    kappa_r(x) = (v^T H v) / (|v|^2 |grad f|)       v = x - eye (or the view
                                                    direction, orthographic)

is the normal curvature of the surface in the viewing direction.  It is a
property of the surface point, so it moves smoothly with the camera and knows
nothing about chain topology.  Driving the envelope from it makes the envelope
as coherent as the geometry.

It also happens to taper in the right PLACES.  v^T H v = 0 is the third
equation of the cusp test, so kappa_r -> 0 exactly where the contour
terminates or doubles back (Koenderink: K = kappa_app * kappa_r, so a cusp's
blowing-up apparent curvature forces kappa_r to zero at finite K).  A stroke
whose width follows |kappa_r| therefore fades out at the cusps on its own,
which is what the end taper was imitating by hand.

TWO SOURCES, same values
------------------------
  1. `contour_export_3d_curvature.txt` -- one float per OBJ vertex, in the
     OBJ's own `v` order, exactly like the sibling *_visibility.txt files.
     This is what `grid_primitive --emit-curvature` writes.
  2. Failing that, the `data-radial-curvature` attributes already present in
     `contour_export_bsp_qi.svg`, matched back onto the OBJ by projecting the
     vertices with this frame's camera.  A stopgap so the idea can be measured
     on exports that predate (1), not a long-term path: the QI SVG splits
     chains at crossings and carries the nodes in pixel precision only.

NaN means "not evaluated here" and is preserved as NaN all the way to the
brush, which treats it as "no opinion" rather than as zero curvature -- a
silent 0 is indistinguishable from a genuine cusp.
"""

import os
import re

import numpy as np

# What krawczyk-contours' --emit-curvature writes, newest name first.  Both
# are one float per OBJ vertex in `v` order; the alias is the name this port
# asked for before the exporter settled on the shorter one.
KAPPA_FILES = ("contour_export_3d_curvature.txt",
               "contour_export_3d_radial_curvature.txt")
KAPPA = KAPPA_FILES[0]
QI_SVG = "contour_export_bsp_qi.svg"

_POLYLINE = re.compile(r"<polyline\b[^>]*>")
_POINTS = re.compile(r'\bpoints="([^"]*)"')
_KAPPA_ATTR = re.compile(r'\bdata-radial-curvature="([^"]*)"')


def _read_txt(path):
    with open(path) as f:
        return np.asarray([float(t) for t in f.read().split()], dtype=float)


def _harvest_svg(path):
    """(points, kappa) pairs for every polyline in the SVG that carries both."""
    txt = open(path).read()
    out = []
    for tag in _POLYLINE.finditer(txt):
        s = tag.group(0)
        mp, mk = _POINTS.search(s), _KAPPA_ATTR.search(s)
        if not (mp and mk):
            continue
        xy = np.fromstring(mp.group(1).replace(",", " "), sep=" ")
        k = np.fromstring(mk.group(1), sep=" ")
        xy = xy[: 2 * (len(xy) // 2)].reshape(-1, 2)
        if len(xy) and len(xy) == len(k):
            out.append((xy, k))
    return out


def _snap(runs, px):
    """Scatter each SVG run's curvature onto the nearest projected vertex.

    The runs are sub-paths of the very chains `px` came from, written by the
    same transform, so this is a lookup rather than a fit -- but vertices
    project a median 0.1 px apart, so neighbouring nodes are interchangeable
    at pixel precision.  They also carry near-identical curvature, which is
    why that ambiguity does not matter here.
    """
    kap = np.full(len(px), np.nan)
    if not len(px):
        return kap

    # A uniform cell hash rather than a KD-tree: scipy is an awkward
    # dependency to require for one nearest-neighbour query, and the points
    # are a dense curve, so the cells stay small.  Cell size is the search
    # radius; a hit must be in the 3x3 neighbourhood.
    r = 1.0
    cell = {}
    for i, (a, b) in enumerate(np.floor(px / r).astype(np.int64)):
        cell.setdefault((a, b), []).append(i)
    cell = {k: np.asarray(v) for k, v in cell.items()}

    for xy, k in runs:
        for q, kq in zip(xy, k):
            a, b = int(q[0] // r), int(q[1] // r)
            cand = [cell[key] for key in
                    ((a + da, b + db) for da in (-1, 0, 1) for db in (-1, 0, 1))
                    if key in cell]
            if not cand:
                continue
            cand = np.concatenate(cand)
            kap[cand[((px[cand] - q) ** 2).sum(axis=1).argmin()]] = kq
    return kap


def _fill_gaps(kap):
    """Linear interpolation across unmatched nodes, NaN only at the ends."""
    good = np.isfinite(kap)
    if not good.any() or good.all():
        return kap
    i = np.arange(len(kap))
    out = kap.copy()
    inner = (i >= i[good][0]) & (i <= i[good][-1])
    out[inner] = np.interp(i[inner], i[good], kap[good])
    return out


def load(frame_dir, cam, chains, verbose=False):
    """Attach `kappa` (parallel to `wpts`) to every chain of a loaded frame.

    Returns the name of the source used, or None if neither was available.
    """
    counts = [len(c.wpts) for c in chains]
    total = sum(counts)

    path = next((q for q in (os.path.join(frame_dir, f) for f in KAPPA_FILES)
                 if os.path.isfile(q)), None)
    if path is not None:
        # The file is one value per OBJ vertex; a closed chain repeats its
        # first vertex in `wpts`, so the per-chain slices are not contiguous
        # in general.  export_io keeps the vertex sequence it walked, so ask
        # the chains for it rather than re-deriving the split here.
        if any(getattr(c, "vseq", None) is None for c in chains):
            raise ValueError("%s needs export_io to record vertex indices"
                             % os.path.basename(path))
        vals = _read_txt(path)
        # Exact length is not checked against the OBJ's `v` count -- the
        # chains only reference the vertices they walked -- but an index past
        # the end means the two files are not describing the same frame.
        need = max(int(c.vseq.max()) for c in chains) + 1
        if len(vals) < need:
            raise ValueError("%s: %d values, but the OBJ references vertex %d"
                             % (path, len(vals), need - 1))
        for c in chains:
            c.kappa = vals[c.vseq]
        return os.path.basename(path)

    svg = os.path.join(frame_dir, QI_SVG)
    if not os.path.isfile(svg):
        return None
    runs = _harvest_svg(svg)
    if not runs:
        return None

    px = cam.ndc_to_svg(np.concatenate([cam.ndc(c.wpts) for c in chains]))
    kap = _snap(runs, px)

    at = 0
    hit = 0
    for c, n in zip(chains, counts):
        k = _fill_gaps(kap[at:at + n])
        # A closed chain repeats its first vertex last; the QI SVG writes that
        # node once, so the copy comes back unmatched.
        if c.closed and np.isnan(k[-1]):
            k[-1] = k[0]
        c.kappa = k
        hit += int(np.isfinite(kap[at:at + n]).sum())
        at += n
    if verbose:
        print("    curvature: %d/%d nodes matched from %s"
              % (hit, total, QI_SVG))
    return QI_SVG
