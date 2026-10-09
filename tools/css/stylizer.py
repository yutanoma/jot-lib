"""
Section 4 of the CSS paper: propagating the parameterization from one frame's
brush paths to the next, and turning fitted groups into strokes.

Ports npr/zxedge_stroke_texture.C's propagate_sil_parameterization(),
regen_group_samples() / LuboPath::gen_group_samples(), and
npr/sil_and_crease_texture.C's generate_strokes_from_groups().
"""

import math
import random

import numpy as np

from css import (FIT_GOOD, VG_GOOD, VOTE_HEALER, IdRefImage, Params,
                 Path, Sample)
from fitting import generate_sil_groups

# propagate_sil_parameterization()'s 3x3 neighbourhood, probed at step j == 0.
OFFSETS = [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, 0),
           (1, 0), (-1, 1), (0, 1), (1, 1)]


class Stroke:
    __slots__ = ("path_index", "group_id", "t", "ndc", "k")

    def __init__(self, path_index, group_id, t, ndc, k=None):
        self.path_index = path_index
        self.group_id = group_id
        self.t = t          # list of stroke-space parameters
        self.ndc = ndc      # list of (x, y) NDC positions, same length
        self.k = k          # |radial curvature| per sample, or None


class FrameResult:
    __slots__ = ("strokes", "stats")

    def __init__(self, strokes, stats):
        self.strokes = strokes
        self.stats = stats


class Stylizer:
    """Runs the CSS loop over a sequence of frames, carrying samples forward.

    jot regenerates its brush paths from scratch every frame and relies
    entirely on the vote propagation to carry the parameterization across --
    there are no persistent chain IDs.  Same here."""

    def __init__(self, params=None):
        self.params = params or Params()
        self.rng = random.Random(self.params.seed)
        self.samples = []       # LuboSamples carried from the previous frame
        self._stroke_id = 0

    def gen_stroke_id(self):
        self._stroke_id += 1
        return self._stroke_id

    # ------------------------------------------------------------- one frame

    def run_frame(self, frame):
        params = self.params
        cam = frame.cam

        paths = []
        for chain in frame.chains:
            if len(chain.edge_vis) == 0 or not chain.edge_vis.any():
                continue        # nothing of this chain is visible this frame
            p = Path(len(paths), chain, cam, params)
            if p.length <= 0.0:
                continue
            p.gen_stroke_id = self.gen_stroke_id
            paths.append(p)

        idref = IdRefImage(cam, params)
        next_id = 1
        for p in paths:
            next_id = p.build_ffsegs(next_id)
            idref.draw_path(p)

        prop = self.propagate(paths, idref, cam)

        sampling_dist_pix = params.vis_sample_spacing * params.lubo_sample_step
        generate_sil_groups(paths, params, self.rng, sampling_dist_pix)

        strokes = self.generate_strokes(paths)

        stats = dict(prop)
        stats.update({
            "paths": len(paths),
            "groups": sum(1 for p in paths for g in p.groups
                          if g.status == VG_GOOD),
            "strokes": len(strokes),
        })
        stats.update(_fit_residual(paths, params))
        self.samples = self.regen_samples(paths)
        stats["samples_out"] = len(self.samples)
        return FrameResult(strokes, stats)

    # ----------------------------------------------------------- propagation

    def propagate(self, paths, idref, cam):
        """propagate_sil_parameterization() -- the "Lubo" algorithm.

        Reproject each sample from the previous frame, crawl a few pixels
        along its projected surface normal through the ID reference image,
        and hand its parameter value to whichever brush path it lands on."""
        params = self.params
        st = {"samples_in": len(self.samples), "votes": 0, "miss_frustum": 0,
              "miss_normal": 0, "miss_search": 0, "miss_far": 0,
              "miss_hidden": 0}
        if not self.samples or not paths:
            return st

        step = cam.pix_to_ndc                 # one reference-image pixel
        max_steps = params.max_lubo_steps
        ndc2pix = 1.0 / cam.pix_to_ndc

        need_vis = params.vote_requires_visible
        for smp in self.samples:
            if need_vis and not smp.vis:
                st["miss_hidden"] += 1
                continue
            p_ndc = cam.ndc(smp.wpt)
            if not cam.in_frustum(p_ndc, smp.wpt):
                st["miss_frustum"] += 1
                continue

            n = self._ndc_normal(cam, smp)
            if n is None:
                st["miss_normal"] += 1
                continue

            dirs = [n, -n] if params.search_both else [n]
            best = None
            for dv in dirs:
                best = self._crawl(paths, idref, p_ndc, dv * step, max_steps)
                if best is not None:
                    break
            if best is None:
                st["miss_search"] += 1
                continue

            dist, hit_pt, hit_s, path = best
            ndc_dist_pix = float(np.linalg.norm(hit_pt - p_ndc)) * ndc2pix
            if path.register_vote(smp, hit_pt, hit_s, ndc_dist_pix):
                st["votes"] += 1
            else:
                st["miss_far"] += 1
        return st

    def _ndc_normal(self, cam, smp):
        """NDCZvec(wn, obj.derivative(wp)) -- the surface normal pushed into
        NDC, by finite difference.  This is the search direction: at a contour
        the surface normal is perpendicular to the view, so its projection
        points across the line."""
        d = float(cam.depth(smp.wpt))
        if not np.isfinite(d) or d <= 0.0:
            return None
        eps = 1e-3 * d
        a = cam.ndc(smp.wpt)
        b = cam.ndc(smp.wpt + eps * smp.wnrm)
        v = b - a
        ln = float(np.linalg.norm(v))
        if not np.isfinite(ln) or ln < 1e-15:
            return None
        return v / ln

    def _crawl(self, paths, idref, p_ndc, delt, max_steps):
        """Walk the reference image from p_ndc along delt, collecting ids."""
        ids = []
        seen = set()
        for j in range(max_steps):
            cur = p_ndc + delt * j
            px = idref.ndc_to_pix(cur)
            if not np.isfinite(px).all():
                break
            cx, cy = int(round(px[0])), int(round(px[1]))

            found = 0
            probes = OFFSETS if j == 0 else [(0, 0)]
            for dx, dy in probes:
                sid, lb = idref.sample(cx + dx, cy + dy)
                if sid:
                    found = sid
                    if sid not in seen:
                        seen.add(sid)
                        ids.append((sid, lb))

            matches = []
            for sid, lb in ids:
                owner = idref.owner.get(sid)
                if owner is None:
                    continue
                path, seg = owner
                if path.in_range(seg, lb):
                    matches.append((path, seg, lb, cur))

            if matches:
                best = None
                for path, seg, lb, c in matches:
                    r = path.get_closest_point_at(seg, lb, c)
                    if r is None:
                        continue
                    if best is None or r[0] < best[0]:
                        best = (r[0], r[1], r[2], path)
                if best is not None:
                    return best

            # Hit background well away from the start: nothing more to find.
            if j > 2 and not found:
                break
        return None

    # ------------------------------------------------------ sample regen

    def regen_samples(self, paths):
        """LuboPath::gen_group_samples -- lay samples along each surviving
        group at LUBO_SAMPLE_STEP pixel spacing, tagged with the group's id
        and its fitted parameter, to be propagated into the next frame."""
        params = self.params
        out = []
        for p in paths:
            vis_spans = _visible_spans(p)
            spacing = (params.vis_sample_spacing * p.pix_to_ndc
                       * params.lubo_sample_step)
            for g in p.groups:
                if g.status != VG_GOOD or g.fstatus != FIT_GOOD:
                    continue
                span = g.end - g.begin
                if span < spacing * 0.1:
                    continue
                nsegs = int(math.ceil(span / spacing))
                nspacing = span / nsegs
                for k in range(nsegs + 1):
                    target_s = g.begin + nspacing * k
                    wpt, wnrm = p.world_at_s(target_s)
                    vis = any(a <= target_s <= b for a, b in vis_spans)
                    out.append(Sample(wpt, wnrm, g.get_t(target_s),
                                      g.base_id, p.index, vis))
        return out

    # --------------------------------------------------- stroke generation

    def generate_strokes(self, paths):
        """generate_strokes_from_groups().

        Deviation from jot: the emitted geometry is clipped to the visible
        edge runs of the path.  jot keeps separate LuboPaths per visibility
        and lets the stroke pools decide; we keep the whole chain as one path
        so the parameterization survives short occlusions, then trim here."""
        params = self.params
        if not paths:
            return []
        step_size = paths[0].pix_to_ndc * params.stroke_pix_sampling
        if step_size <= 0.0:
            return []

        strokes = []
        for p in paths:
            vis_spans = _visible_spans(p)
            if not vis_spans:
                continue
            for g in p.groups:
                if g.status != VG_GOOD or g.fstatus != FIT_GOOD:
                    continue
                if not g.fits:
                    continue
                for a, b in _clip(g.begin, g.end, vis_spans):
                    s = self._emit(p, g, a, b, step_size)
                    if s is not None:
                        strokes.append(s)
        return strokes

    def _emit(self, p, g, sbegin, send, step_size):
        sdelta = send - sbegin
        if sdelta <= 0.0:
            return None
        num = int(max(2.0, math.ceil(sdelta / step_size)))
        sd = sdelta / (num - 1)

        ts, pts = [], []
        for j in range(num):
            s = sbegin + j * sd
            ts.append(g.get_t(s))
            pts.append(p.at_s(s))
        if num == 2 and np.linalg.norm(pts[1] - pts[0]) < 1e-12:
            return None
        ss = sbegin + np.arange(num) * sd
        k = None if p.kappa is None else p.kappa_at_s(ss)
        return Stroke(p.index, g.base_id, ts, pts, k)


def _visible_spans(p):
    """Global NDC-arclength intervals covered by runs of visible edges."""
    vis = p.edge_vis
    spans, m = [], len(vis)
    i = 0
    while i < m:
        if not vis[i]:
            i += 1
            continue
        j = i
        while j < m and vis[j]:
            j += 1
        spans.append((float(p.s[i]), float(p.s[j])))
        i = j
    return spans


def _clip(a, b, spans):
    out = []
    for lo, hi in spans:
        x, y = max(a, lo), min(b, hi)
        if y > x:
            out.append((x, y))
    return out


def _fit_residual(paths, params):
    """How far each propagated vote ended up from the parameterization the
    frame actually settled on, in pixels of stroke pattern.  This is the
    swimming the algorithm exists to suppress: small means the previous
    frame's phase survived into this one."""
    errs = []
    for p in paths:
        for g in p.groups:
            if g.status != VG_GOOD or not g.fits:
                continue
            for v in g.votes:
                if v.status == VOTE_HEALER:
                    continue
                errs.append(abs(v.t - g.get_t(v.s)))
    if not errs:
        return {"resid_mean_px": 0.0, "resid_med_px": 0.0, "resid_n": 0}
    a = np.sort(np.asarray(errs)) * params.offset_pix_len
    return {"resid_mean_px": float(a.mean()),
            "resid_med_px": float(a[len(a) // 2]),
            "resid_n": len(a)}
