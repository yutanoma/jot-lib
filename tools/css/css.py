"""
Coherent Stylized Silhouettes (Kalnins et al., SIGGRAPH 2003), ported out of
jot so it can run on externally supplied contours.

This is a "lift, don't link" port of two pieces of jot:

  Section 4, brush path generation / the "Lubo" propagation
      npr/zxedge_stroke_texture.C : draw_id_ref_param_vis_pass(),
      propagate_sil_parameterization(), LuboPath::register_vote(),
      LuboPath::in_range(), LuboPath::get_closest_point_at(),
      LuboPath::gen_group_samples()

  Section 5, parameterization
      npr/sil_and_crease_texture.C : generate_sil_groups() and everything it
      calls -- build_groups, the culls and splits, the four fits, the three
      coverage policies, heal_groups, generate_strokes_from_groups()

What changed, and why
---------------------
* The GL ID reference image is rasterized in software (IdRefImage).  jot gets
  it from glLineWidth(3) + GL_SMOOTH + GL_DEPTH_TEST; we rasterize 3px-wide
  lines with an interpolated arclength byte and a z-buffer.  Visibility comes
  from the caller's per-edge flag instead of a depth test against the surface.
* jot packs (path id | visibility | arclength byte) into one 32-bit pixel.  We
  keep the segment id and the arclength byte in two parallel buffers.  The
  8-bit quantization of the byte is preserved, since it sets the search window
  width in get_closest_point_at().
* Samples carry an explicit 3D point and surface normal.  jot recovers both
  from a Bsimplex + barycentric coords, which needs a mesh; we have neither.
* Arclength is planar (x,y) NDC length.  jot's NDCZpt_list length() includes
  the depth coordinate, which is meaningless here.
* optimizing_fit solves with numpy instead of jot's ludcmp/lubksb, and its
  backwards-fit test compares against the previous knot.  jot's copy compares
  against a constant 1.0 -- see the "Karol->" comment at
  sil_and_crease_texture.C:2480, which re-declares fj_1 inside the loop.
"""

import math
from bisect import bisect_left, bisect_right

import numpy as np

# ---------------------------------------------------------------- constants

FIT_RANDOM, FIT_SIGMA, FIT_PHASE, FIT_INTERPOLATE, FIT_OPTIMIZE = range(5)
COVER_MAJORITY, COVER_ONE_TO_ONE, COVER_TRIMMED = range(3)

FIT_NAMES = {"random": FIT_RANDOM, "sigma": FIT_SIGMA, "phase": FIT_PHASE,
             "interpolate": FIT_INTERPOLATE, "optimize": FIT_OPTIMIZE}
COVER_NAMES = {"majority": COVER_MAJORITY, "one-to-one": COVER_ONE_TO_ONE,
               "hybrid": COVER_TRIMMED}

# VoteGroup::vg_status_t
VG_GOOD = 0
VG_LOW_LENGTH, VG_LOW_VOTES, VG_BAD_DENSITY = 1, 2, 3
VG_SPLIT_LOOP, VG_SPLIT_LARGE_DELTA, VG_SPLIT_GAP = 4, 5, 6
VG_CULL_BACKWARDS, VG_SPLIT_ALL_BACKTRACK = 7, 8
VG_FIT_BACKWARDS, VG_FINAL_FIT_BACKWARDS = 9, 10
VG_NOT_MAJORITY, VG_NOT_ONE_TO_ONE, VG_NOT_HYBRID, VG_HEALED = 11, 12, 13, 14

# VoteGroup::fit_status_t
FIT_GOOD, FIT_NONE, FIT_BACKWARDS, FIT_OLD = range(4)

# LuboVote::lv_status_t
VOTE_GOOD, VOTE_OUTLIER, VOTE_HEALER = range(3)


class Params:
    """Every jot Config/SilUI knob the port reads, with jot's own defaults."""

    def __init__(self, **kw):
        # Section 4
        self.max_lubo_steps = 6          # MAX_LUBO_STEPS
        self.vis_path_width = 3          # SIL_VIS_PATH_WIDTH
        self.vis_sample_spacing = 2.0    # SIL_VIS_SAMPLE_SPACING
        self.lubo_sample_step = 4        # LUBO_SAMPLE_STEP
        self.search_both = True          # see Stylizer.propagate()
        self.vote_requires_visible = True  # sample_matches_path's vis test
        # Section 5
        self.fit_type = FIT_OPTIMIZE     # SilUI default
        self.cover_type = COVER_TRIMMED  # SilUI default
        self.min_path_pix = 2.0          # MIN_PATH_PIX
        self.min_votes_per_group = 2     # MIN_VOTES_PER_GROUP
        self.min_pix_per_group = 5.0     # MIN_PIX_PER_GROUP
        self.min_frac_per_group = 0.05   # MIN_FRAC_PER_GROUP
        self.sparse_factor = 3.0         # SPARSE_FACTOR
        self.fit_pix = 48.0              # FIT_PIX
        self.weight_fit = 1.0            # WEIGHT_FIT
        self.weight_scale = 1.0          # WEIGHT_SCALE
        self.weight_distort = 1.0        # WEIGHT_DISTORT
        self.weight_heal = 1.0           # WEIGHT_HEAL
        self.heal_join_pix = 3.0         # HEAL_JOIN_PIX_THRESH
        self.heal_drag_pix = 15.0        # HEAL_DRAG_PIX_THRESH
        # Stroke generation
        self.stroke_pix_sampling = 6.0   # SIL_TO_STROKE_PIX_SAMPLING
        self.offset_pix_len = 30.0       # stylization period, in pixels
        self.stretch = 1.0               # 1.0 == SilUI "NoSig" / sigma-one
        self.idref_margin = 32           # px of slack around the SVG frame
        self.seed = 0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown parameter %r" % k)
            setattr(self, k, v)

    @property
    def do_heal(self):
        return self.cover_type == COVER_TRIMMED and self.weight_heal > 0.0


# ------------------------------------------------------------- data classes

class Vote:
    """LuboVote (zxedge_stroke_texture.H:117)."""

    __slots__ = ("s", "t", "conf", "status", "stroke_id", "path_id")

    def __init__(self, s=0.0, t=0.0, conf=1.0, stroke_id=0, path_id=-1):
        self.s = s
        self.t = t
        self.conf = conf
        self.status = VOTE_GOOD
        self.stroke_id = stroke_id
        self.path_id = path_id


class Sample:
    """LuboSample, but carrying a world point + normal instead of a simplex.

    `vis` records whether the point lay on a visible run when it was laid
    down; jot keeps the same distinction on the sample and refuses to match a
    sample to a path of the other visibility (sample_matches_path,
    zxedge_stroke_texture.H:1123)."""

    __slots__ = ("wpt", "wnrm", "t", "stroke_id", "path_id", "vis")

    def __init__(self, wpt, wnrm, t, stroke_id, path_id, vis=True):
        self.wpt = wpt
        self.wnrm = wnrm
        self.t = t
        self.stroke_id = stroke_id
        self.path_id = path_id
        self.vis = vis


class Group:
    """VoteGroup (zxedge_stroke_texture.H:221)."""

    __slots__ = ("path", "status", "fstatus", "confidence",
                 "begin", "end", "base_id", "votes", "fits")

    def __init__(self, base_id=0, path=None):
        self.path = path
        self.status = VG_GOOD
        self.fstatus = FIT_NONE
        self.confidence = 0.0
        self.begin = 0.0
        self.end = 0.0
        self.base_id = base_id
        self.votes = []
        self.fits = []          # list of (s, t) knots

    @property
    def num(self):
        return len(self.votes)

    def add(self, v):
        self.votes.append(v)

    def sort(self):
        self.votes.sort(key=lambda v: v.s)
        if self.votes:
            self.begin = self.votes[0].s
            self.end = self.votes[-1].s
        else:
            self.begin = self.end = 0.0

    def get_t(self, s):
        """VoteGroup::get_t -- clamped piecewise-linear lookup over `fits`."""
        f = self.fits
        n = len(f)
        if n == 0:
            return 0.0
        if n == 1 or s <= f[0][0]:
            return f[0][1]
        if s >= f[-1][0]:
            return f[-1][1]
        r = bisect_right(f, (s,)) # first knot strictly past s
        l = r - 1
        ds = f[r][0] - f[l][0]
        if ds <= 0.0:
            return f[l][1]
        w = (s - f[l][0]) / ds
        return f[l][1] * (1.0 - w) + f[r][1] * w


class FFSeg:
    """One maximal run of visible edges along a path -- jot's "front-facing
    segment", the unit that gets its own id in the reference image."""

    __slots__ = ("seg_id", "verts", "ff_s", "length")

    def __init__(self, seg_id, verts, ff_s):
        self.seg_id = seg_id
        self.verts = verts      # global vertex indices, in order
        self.ff_s = ff_s        # arclength within this segment, same order
        self.length = ff_s[-1] if len(ff_s) else 0.0


class Path:
    """LuboPath (zxedge_stroke_texture.H:480), 2D + world, no mesh."""

    def __init__(self, index, chain, cam, params):
        self.index = index
        self.params = params
        self.wpts = chain.wpts
        self.wnrm = chain.wnrm
        self.edge_vis = chain.edge_vis
        self.closed = chain.closed
        # Per-node radial curvature, or None when the frame did not carry it.
        self.kappa = getattr(chain, "kappa", None)

        self.ndc = cam.ndc(chain.wpts)
        self.depth = cam.depth(chain.wpts)

        # Planar NDC arclength, jot's `s`.
        d = np.linalg.norm(np.diff(self.ndc, axis=0), axis=1)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0)
        self.seg_len = d
        self.s = np.concatenate([[0.0], np.cumsum(d)])
        self.length = float(self.s[-1])

        self.pix_to_ndc = cam.pix_to_ndc
        self.stretch = params.stretch
        self.offset_pix_len = params.offset_pix_len

        self.ffsegs = []
        self.ff_of_vert = {}
        self.votes = []
        self.groups = []

    @property
    def freq(self):
        """fit_initial_groups(): t advances `freq` per unit of NDC arclength."""
        return self.stretch / (self.pix_to_ndc * self.offset_pix_len)

    def pt(self, u):
        """LuboPath::pt(u) -- NDCZpt_list::interpolate over total length."""
        return self.at_s(u * self.length)

    def at_s(self, s):
        if self.length <= 0.0:
            return self.ndc[0]
        s = min(max(s, 0.0), self.length)
        i = min(bisect_right(self.s, s) - 1, len(self.seg_len) - 1)
        i = max(i, 0)
        if self.seg_len[i] <= 0.0:
            return self.ndc[i]
        w = (s - self.s[i]) / self.seg_len[i]
        return self.ndc[i] * (1.0 - w) + self.ndc[i + 1] * w

    def world_at_s(self, s):
        """3D point and surface normal at NDC arclength `s`."""
        if self.length <= 0.0:
            return self.wpts[0], self.wnrm[0]
        s = min(max(s, 0.0), self.length)
        i = min(bisect_right(self.s, s) - 1, len(self.seg_len) - 1)
        i = max(i, 0)
        w = 0.0 if self.seg_len[i] <= 0.0 else (s - self.s[i]) / self.seg_len[i]
        p = self.wpts[i] * (1.0 - w) + self.wpts[i + 1] * w
        nv = self.wnrm[i] * (1.0 - w) + self.wnrm[i + 1] * w
        ln = np.linalg.norm(nv)
        return p, (nv / ln if ln > 1e-12 else self.wnrm[i])

    def kappa_at_s(self, s):
        """|radial curvature| at NDC arclengths `s` (array in, array out).

        NaN -- a node whose curvature could not be evaluated -- propagates,
        and the brush reads it as "no opinion" rather than as a cusp.
        """
        s = np.asarray(s, dtype=float)
        if self.kappa is None:
            return np.full(s.shape, np.nan)
        if self.length <= 0.0:
            return np.full(s.shape, abs(self.kappa[0]))
        return np.interp(np.clip(s, 0.0, self.length), self.s,
                         np.abs(self.kappa))

    # ------------------------------------------------- front-facing segments

    def build_ffsegs(self, next_id):
        """Split the path into maximal runs of visible edges.

        A closed path whose first and last edges are both visible gets ONE
        wrapped segment, matching jot's `_id_set[0] == _id_set[n-1]` case in
        LuboPath::in_range()."""
        vis = self.edge_vis
        m = len(vis)
        if m == 0 or not vis.any():
            return next_id

        runs = []
        i = 0
        while i < m:
            if not vis[i]:
                i += 1
                continue
            j = i
            while j < m and vis[j]:
                j += 1
            runs.append((i, j))     # edges [i, j)
            i = j

        # Wrap the tail run onto the head run for a closed loop.
        if self.closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == m:
            a, b = runs.pop()
            runs[0] = (a, runs[0][1] + m)   # edge indices continue past m

        for a, b in runs:
            verts, ff_s, acc = [a % m], [0.0], 0.0
            for e in range(a, b):
                acc += float(self.seg_len[e % m])
                verts.append((e + 1) % m if e + 1 >= m else e + 1)
                ff_s.append(acc)
            seg = FFSeg(next_id, verts, ff_s)
            next_id += 1
            self.ffsegs.append(seg)
            for v in verts:
                self.ff_of_vert.setdefault(v, seg)
        return next_id

    # -------------------------------------------------------- vote plumbing

    def in_range(self, seg, lbyte):
        """LuboPath::in_range -- is this decoded arclength inside the segment?"""
        length = lbyte * seg.length / 256.0
        return 0.0 < length < seg.length

    def get_closest_point_at(self, seg, lbyte, p):
        """LuboPath::get_closest_point_at.

        Restrict the closest-point search to the window of the ff segment that
        the quantized arclength byte points at, then return
        (distance, closest NDC point, global arclength of that point)."""
        if seg.length <= 0.0:
            return None
        target = lbyte * seg.length / 255.0
        margin = max(seg.length / 255.0, 2.0 * self.pix_to_ndc)
        delta = 3.0 * margin

        lo = bisect_left(seg.ff_s, target - delta) - 1
        hi = bisect_right(seg.ff_s, target + delta)
        lo = max(lo, 0)
        hi = min(hi, len(seg.verts) - 1)

        best = None
        for k in range(lo, hi):
            a, b = seg.verts[k], seg.verts[k + 1]
            pa, pb = self.ndc[a], self.ndc[b]
            ab = pb - pa
            den = float(ab @ ab)
            w = 0.0 if den <= 0.0 else min(max(float((p - pa) @ ab) / den, 0.0), 1.0)
            q = pa + w * ab
            dist = float(np.linalg.norm(p - q))
            if best is None or dist < best[0]:
                # Global arclength: jot's get_s(i) + q.dist(pt(i)).  Use the
                # edge's own index so wrapped segments stay monotone.
                e = a if b == a + 1 else (b if a == b + 1 else a)
                sg = float(self.s[e] + np.linalg.norm(q - self.ndc[e]))
                best = (dist, q, sg)
        return best

    def register_vote(self, sample, hit_pt, hit_s, ndc_dist_pix):
        """LuboPath::register_vote.  Drops votes that travelled further than
        the propagation search could legitimately reach."""
        if ndc_dist_pix >= (self.params.max_lubo_steps + 1):
            return False
        v = Vote(hit_s, sample.t, 1.0, sample.stroke_id, sample.path_id)
        self.votes.append(v)
        return True


# ----------------------------------------------------------- ID ref. image

class IdRefImage:
    """Software stand-in for jot's GL ID reference image.

    draw_id_ref_param_vis_pass() rasterizes each front-facing run as a
    GL_LINE_STRIP of width SIL_VIS_PATH_WIDTH under GL_SMOOTH, so each pixel
    carries the run's id plus a byte of arclength interpolated along it, and
    under GL_DEPTH_TEST so nearer runs win.  Same here, except visibility is
    the caller's per-edge flag rather than a depth test against the surface."""

    def __init__(self, cam, params):
        m = params.idref_margin
        self.w = int(math.ceil(cam.svg_w)) + 2 * m
        self.h = int(math.ceil(cam.svg_h)) + 2 * m
        self.margin = m
        self.cam = cam
        self.seg = np.zeros((self.h, self.w), dtype=np.int32)    # 0 == empty
        self.lb = np.zeros((self.h, self.w), dtype=np.int32)
        self.z = np.full((self.h, self.w), np.inf)
        self.owner = {}                                          # seg key -> (path, FFSeg)
        self.half = params.vis_path_width // 2

    def ndc_to_pix(self, ndc):
        p = self.cam.ndc_to_svg(ndc)
        return p + self.margin

    def pix_to_ndc(self, px):
        return self.cam.svg_to_ndc(np.asarray(px, dtype=float) - self.margin)

    def draw_path(self, path):
        for seg in path.ffsegs:
            self.owner[seg.seg_id] = (path, seg)
            if seg.length <= 0.0:
                continue
            pts = self.ndc_to_pix(path.ndc[seg.verts])
            dep = path.depth[seg.verts]
            # jot: len = (uint)(255.0 * pl / ffseg_length)
            lb = np.minimum(255.0 * np.asarray(seg.ff_s) / seg.length, 255.0)
            for k in range(len(seg.verts) - 1):
                self._line(pts[k], pts[k + 1], dep[k], dep[k + 1],
                           lb[k], lb[k + 1], seg.seg_id)

    def _line(self, p0, p1, z0, z1, l0, l1, seg_id):
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            return
        d = p1 - p0
        n = int(max(abs(d[0]), abs(d[1]))) + 1
        ts = np.linspace(0.0, 1.0, n + 1)
        xs = np.rint(p0[0] + ts * d[0]).astype(np.int64)
        ys = np.rint(p0[1] + ts * d[1]).astype(np.int64)
        zs = z0 + ts * (z1 - z0)
        ls = np.rint(l0 + ts * (l1 - l0)).astype(np.int32)
        hw = self.half
        for dy in range(-hw, hw + 1):
            yy = ys + dy
            for dx in range(-hw, hw + 1):
                xx = xs + dx
                ok = (xx >= 0) & (xx < self.w) & (yy >= 0) & (yy < self.h)
                if not ok.any():
                    continue
                ax, ay, az, al = xx[ok], yy[ok], zs[ok], ls[ok]
                win = az < self.z[ay, ax]
                if not win.any():
                    continue
                ax, ay, az, al = ax[win], ay[win], az[win], al[win]
                self.z[ay, ax] = az
                self.seg[ay, ax] = seg_id
                self.lb[ay, ax] = al

    def sample(self, px, py):
        if 0 <= px < self.w and 0 <= py < self.h:
            s = int(self.seg[py, px])
            if s:
                return s, int(self.lb[py, px])
        return 0, 0
