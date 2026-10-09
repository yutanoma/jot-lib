"""
Section 5 of the CSS paper: grouping, culling, coverage, fitting, healing.

A transcription of npr/sil_and_crease_texture.C's generate_sil_groups() and
everything it calls.  The pipeline order below is that function's, verbatim.
"""

import math

import numpy as np

from css import (COVER_MAJORITY, COVER_ONE_TO_ONE, COVER_TRIMMED,
                 FIT_BACKWARDS, FIT_GOOD, FIT_INTERPOLATE, FIT_NONE, FIT_OLD,
                 FIT_OPTIMIZE, FIT_PHASE, FIT_RANDOM, FIT_SIGMA, Group, Vote,
                 VG_BAD_DENSITY, VG_CULL_BACKWARDS, VG_FINAL_FIT_BACKWARDS,
                 VG_FIT_BACKWARDS, VG_GOOD, VG_HEALED, VG_LOW_LENGTH,
                 VG_LOW_VOTES, VG_NOT_HYBRID, VG_NOT_MAJORITY,
                 VG_NOT_ONE_TO_ONE, VG_SPLIT_ALL_BACKTRACK, VG_SPLIT_GAP,
                 VG_SPLIT_LARGE_DELTA, VOTE_HEALER, VOTE_OUTLIER)

TWO_PI = 2.0 * math.pi
COVERAGE_START, COVERAGE_END, COVERAGE_BAD = 0, 1, 2


def generate_sil_groups(paths, params, rng, sampling_dist_pix):
    """SilAndCreaseTexture::generate_sil_groups (sil_and_crease_texture.C:1113)."""
    if not paths:
        return
    min_ndc = paths[0].pix_to_ndc * params.min_path_pix
    fit_func = _FITS[params.fit_type]
    cover_func = _COVERS[params.cover_type]

    for p in paths:
        if p.length < min_ndc:
            continue

        build_groups(p)

        cull_small_groups(p, params)
        cull_short_groups(p, params)

        split_gapped_groups(p)
        split_large_delta_groups(p)

        cull_backwards_groups(p)

        if params.fit_type == FIT_INTERPOLATE:
            split_all_backtracking_groups(p)

        cull_small_groups(p, params)
        cull_short_groups(p, params)
        cull_sparse_groups(p, params, sampling_dist_pix)

        if params.fit_type != FIT_INTERPOLATE:
            fit_initial_groups(p, fit_func, params, rng)

            if params.fit_type == FIT_OPTIMIZE:
                cull_bad_fit_groups(p)

            cover_func(p, params)
            cull_outliers_in_groups(p)
            fit_final_groups(p, fit_func, params, rng)

            if params.do_heal:
                heal_groups(p, fit_func, params, rng)

            refit_backward_fit_groups(p, params, rng)
        else:
            cover_func(p, params)
            fit_final_groups(p, fit_func, params, rng)


# ------------------------------------------------------------ build / cull

def build_groups(p):
    votes = sorted(p.votes, key=lambda v: v.s)
    p.groups = []
    by_id = {}
    for v in votes:
        g = by_id.get(v.stroke_id)
        if g is None:
            g = Group(v.stroke_id, p)
            by_id[v.stroke_id] = g
            p.groups.append(g)
        g.add(v)
    for g in p.groups:
        g.sort()
        g.base_id = p.gen_stroke_id()


def cull_small_groups(p, params):
    for g in p.groups:
        if g.status == VG_GOOD and g.num < params.min_votes_per_group:
            g.status = VG_LOW_VOTES


def cull_short_groups(p, params):
    min_length = min(params.min_pix_per_group * p.pix_to_ndc,
                     params.min_frac_per_group * p.length)
    for g in p.groups:
        if g.status == VG_GOOD and (g.end - g.begin) < min_length:
            g.status = VG_LOW_LENGTH


def cull_sparse_groups(p, params, sampling_dist_pix):
    spacing = sampling_dist_pix * p.pix_to_ndc
    for g in p.groups:
        if g.status != VG_GOOD or g.num < 2:
            continue
        if (g.end - g.begin) / (g.num - 1) > params.sparse_factor * spacing:
            g.status = VG_BAD_DENSITY


def cull_backwards_groups(p):
    for g in p.groups:
        if g.status != VG_GOOD or g.num < 2:
            continue
        cnt = sum(-1.0 if g.votes[j + 1].t < g.votes[j].t else 1.0
                  for j in range(g.num - 1))
        if cnt / (g.num - 1.0) <= 0.0:
            g.status = VG_CULL_BACKWARDS


# ------------------------------------------------------------------ splits

def _emit_split(p, g, j0, j):
    ng = Group(p.gen_stroke_id(), p)
    for k in range(j0, j + 1):
        ng.add(g.votes[k])
    ng.begin = g.votes[j0].s
    ng.end = g.votes[j].s
    p.groups.append(ng)


def split_gapped_groups(p):
    THRESH_FRACTION, THRESH_FACTOR = 0.75, 4.0
    for i in range(len(p.groups)):
        g = p.groups[i]
        nv = g.num
        if g.status != VG_GOOD or nv < 5:
            continue
        gaps = [g.votes[j + 1].s - g.votes[j].s for j in range(nv - 1)]
        thresh = THRESH_FACTOR * sorted(gaps)[int(THRESH_FRACTION * (nv - 2.0))]

        j0 = 0
        for j in range(nv - 1):
            if gaps[j] > thresh:
                _emit_split(p, g, j0, j)
                j0 = j + 1
        if j0 != 0:
            g.status = VG_SPLIT_GAP
            _emit_split(p, g, j0, nv - 2)


def split_large_delta_groups(p):
    DELTA_FRACTION = 0.75
    DELTA_FORWARD_FACTOR, DELTA_REVERSE_FACTOR = 6.0, 3.0
    for i in range(len(p.groups)):
        g = p.groups[i]
        nv = g.num
        if g.status != VG_GOOD or nv < 5:
            continue
        deltas = [g.votes[j + 1].t - g.votes[j].t for j in range(nv - 1)]
        cnt = sum(-1.0 if d < 0.0 else 1.0 for d in deltas)
        # Bail out if the deltas don't mostly agree in sign.
        if abs(cnt) / (nv - 1.0) < 0.5:
            continue
        sd = sorted(deltas, reverse=(cnt < 0.0))
        pivot = sd[int(DELTA_FRACTION * (nv - 2.0))]
        rev_thresh = -DELTA_REVERSE_FACTOR * pivot
        fwd_thresh = DELTA_FORWARD_FACTOR * pivot

        j0 = 0
        for j in range(nv - 1):
            if ((rev_thresh and deltas[j] / rev_thresh > 1.0) or
                    (fwd_thresh and deltas[j] / fwd_thresh > 1.0)):
                _emit_split(p, g, j0, j)
                j0 = j + 1
        if j0 != 0:
            g.status = VG_SPLIT_LARGE_DELTA
            _emit_split(p, g, j0, nv - 2)


def split_all_backtracking_groups(p):
    for i in range(len(p.groups)):
        g = p.groups[i]
        nv = g.num
        if g.status != VG_GOOD or nv < 2:
            continue
        deltas = [g.votes[j + 1].t - g.votes[j].t for j in range(nv - 1)]
        j0 = 0
        for j in range(nv - 1):
            if deltas[j] < 0.0:
                _emit_split(p, g, j0, j)
                j0 = j + 1
        if j0 != 0:
            g.status = VG_SPLIT_ALL_BACKTRACK
            _emit_split(p, g, j0, nv - 2)


# -------------------------------------------------------------------- fits

def fit_initial_groups(p, fit_func, params, rng):
    freq = p.freq
    for g in p.groups:
        if g.status != VG_GOOD:
            continue
        if g.num == 0:
            arclength_fit(g, freq, params, rng)
        else:
            fit_func(g, freq, params, rng)


def cull_bad_fit_groups(p):
    for g in p.groups:
        if g.status == VG_GOOD and g.fstatus == FIT_BACKWARDS:
            g.status = VG_FIT_BACKWARDS


def cull_outliers_in_groups(p):
    THRESH_FRACTION, THRESH_FACTOR = 0.75, 20.0
    for g in p.groups:
        nv = g.num
        if g.status != VG_GOOD or nv < 5:
            continue
        errs = [abs(v.t - g.get_t(v.s)) for v in g.votes]
        thresh = THRESH_FACTOR * sorted(errs)[int(THRESH_FRACTION * (nv - 2.0))]
        cnt = 0
        for j, e in enumerate(errs):
            if e > thresh:
                g.votes[j].status = VOTE_OUTLIER
                cnt += 1
        if cnt:
            g.fstatus = FIT_OLD


def fit_final_groups(p, fit_func, params, rng):
    freq = p.freq
    for g in p.groups:
        if g.status != VG_GOOD or g.fstatus == FIT_GOOD:
            continue
        g.fstatus = FIT_NONE
        g.fits = []
        if g.num == 0:
            arclength_fit(g, freq, params, rng)
        else:
            fit_func(g, freq, params, rng)


def refit_backward_fit_groups(p, params, rng):
    freq = p.freq
    for i in range(len(p.groups)):
        g = p.groups[i]
        if g.status != VG_GOOD or g.fstatus == FIT_GOOD:
            continue
        ng = Group(p.gen_stroke_id(), p)
        ng.votes = list(g.votes)
        ng.begin, ng.end = g.begin, g.end
        p.groups.append(ng)
        g.status = VG_FINAL_FIT_BACKWARDS
        if ng.num == 0:
            arclength_fit(ng, freq, params, rng)
        else:
            phasing_fit(ng, freq, params, rng)


def random_fit(g, freq, params, rng):
    phase = rng.random()
    g.fits = [(g.begin, phase), (g.end, phase + (g.end - g.begin) * freq)]
    g.fstatus = FIT_GOOD


def sigma_fit(g, freq, params, rng):
    g.fits = [(g.begin, 0.0), (g.end, (g.end - g.begin) * freq)]
    g.fstatus = FIT_GOOD


def arclength_fit(g, freq, params=None, rng=None):
    g.fits = [(g.begin, g.begin * freq), (g.end, g.end * freq)]
    g.fstatus = FIT_GOOD


def phasing_fit(g, freq, params, rng):
    sx = cx = phase_ave = 0.0
    count = 0
    for v in g.votes:
        if v.status == VOTE_OUTLIER:
            continue
        pi = v.t - v.s * freq
        sx += math.sin(pi * TWO_PI)
        cx += math.cos(pi * TWO_PI)
        phase_ave += pi
        count += 1
    if count == 0:
        arclength_fit(g, freq)
        return
    phase = math.atan2(sx / count, cx / count)
    if phase < 0.0:
        phase += TWO_PI
    phase /= TWO_PI
    phase += math.floor(phase_ave / count)

    s_begin = min(g.begin, g.votes[0].s)
    s_end = max(g.end, g.votes[-1].s)
    g.fits = [(s_begin, s_begin * freq + phase), (s_end, s_end * freq + phase)]
    g.fstatus = FIT_GOOD


def interpolating_fit(g, freq, params, rng):
    bad = False
    t_begin = g.votes[0].t + freq * (g.begin - g.votes[0].s)
    t_end = g.votes[-1].t + freq * (g.end - g.votes[-1].s)

    fits = []
    if g.begin < g.votes[0].s:
        fits.append((g.begin, t_begin))
    t_last = -float("inf")
    for v in g.votes:
        fits.append((v.s, v.t))
        if v.t < t_last:
            bad = True
        t_last = v.t
    if g.end > g.votes[-1].s:
        fits.append((g.end, t_end))

    g.fits = fits
    g.fstatus = FIT_BACKWARDS if bad else FIT_GOOD


def optimizing_fit(g, freq, params, rng):
    """SilAndCreaseTexture::optimizing_fit -- the energy minimization of
    section 5.3: a fit term against the votes, plus scale, distortion and
    healing terms, solved on a uniform knot sequence."""
    w_fit = params.weight_fit
    w_scale = params.weight_scale
    w_distort = params.weight_distort
    w_heal = params.weight_heal

    begin = min(g.begin, g.votes[0].s)
    end = max(g.end, g.votes[-1].s)

    span = end - begin
    n = max(2, int(math.ceil(span / g.path.pix_to_ndc / params.fit_pix)))
    delta = span / (n - 1)
    if delta <= 0.0:
        arclength_fit(g, freq)
        return

    A = np.zeros((n, n))
    d = np.zeros(n)
    nv = g.num

    i = 0
    while i < nv and g.votes[i].s < begin:
        i += 1

    xj = xjd = xj_1 = xj_1d = 0.0
    for j in range(n):
        # Fit, first hat-function term: votes in (x_{j-1}, x_j].
        if j > 0:
            while i < nv and g.votes[i].s <= xj_1d:
                v = g.votes[i]
                if v.status != VOTE_OUTLIER:
                    y = v.t - v.s * freq
                    tij = (v.s - xj_1) / delta
                    if v.status == VOTE_HEALER:
                        factor = w_heal * 2.0 * 1.0 * tij
                    else:
                        factor = w_fit * 2.0 * (1.0 / nv) * tij
                    A[j, j - 1] += factor * (1.0 - tij)
                    A[j, j] += factor * tij
                    d[j] += factor * y
                i += 1

        # Fit, second hat-function term: votes in (x_j, x_{j+1}].
        if j < n - 1:
            i0 = i
            xj = begin + j * delta
            xjd = xj + delta
            while i < nv and g.votes[i].s <= xjd:
                v = g.votes[i]
                if v.status != VOTE_OUTLIER:
                    y = v.t - v.s * freq
                    tij = (v.s - xj) / delta
                    if v.status == VOTE_HEALER:
                        factor = w_heal * 2.0 * 1.0 * (1.0 - tij)
                    else:
                        factor = w_fit * 2.0 * (1.0 / nv) * (1.0 - tij)
                    A[j, j] += factor * (1.0 - tij)
                    A[j, j + 1] += factor * tij
                    d[j] += factor * y
                i += 1
            i = i0

        xj_1, xj_1d = xj, xjd

        # Scale.
        factor = 2.0 * w_scale / float(n * n)
        A[j, :] -= factor
        A[j, j] += n * factor

        # Distortion (second difference, three overlapping stencils).
        factor = 2.0 * w_distort / float(n)
        if j > 1:
            A[j, j - 2] += factor
            A[j, j - 1] += -2.0 * factor
            A[j, j] += factor
        if 0 < j < n - 1:
            A[j, j - 1] += -2.0 * factor
            A[j, j] += 4.0 * factor
            A[j, j + 1] += -2.0 * factor
        if j < n - 2:
            A[j, j] += factor
            A[j, j + 1] += -2.0 * factor
            A[j, j + 2] += factor

    try:
        sol = np.linalg.solve(A, d)
    except np.linalg.LinAlgError:
        arclength_fit(g, freq)
        return
    if not np.isfinite(sol).all():
        arclength_fit(g, freq)
        return

    bad = False
    fits = []
    fj_1 = None
    for j in range(n):
        xj = begin + j * delta
        fj = sol[j] + xj * freq
        if fj_1 is not None and fj < fj_1:
            bad = True
        fits.append((xj, float(fj)))
        fj_1 = fj

    g.fits = fits
    g.fstatus = FIT_BACKWARDS if bad else FIT_GOOD


_FITS = {FIT_RANDOM: random_fit, FIT_SIGMA: sigma_fit, FIT_PHASE: phasing_fit,
         FIT_INTERPOLATE: interpolating_fit, FIT_OPTIMIZE: optimizing_fit}


# -------------------------------------------------------- coverage policies
#
# Each policy decides which groups survive and stretches their [begin,end]
# windows so the surviving groups tile the whole path exactly once.

class _CB:
    """CoverageBoundary."""
    __slots__ = ("vg", "s", "type")

    def __init__(self, vg, s, type_):
        self.vg, self.s, self.type = vg, s, type_


def _cb_key(cb):
    # Ties resolve START before END, so groups that abut exactly leave no hole.
    return (cb.s, cb.type)


def majority_cover(p, params):
    """SIL_COVER_MAJORITY -- the single group with the most votes takes the
    whole path."""
    groups = p.groups
    max_ind, max_votes = -1, 0
    for i, g in enumerate(groups):
        if g.status != VG_GOOD:
            continue
        g.status = VG_NOT_MAJORITY
        if g.num > max_votes:
            max_ind, max_votes = i, g.num

    if max_ind != -1:
        g = groups[max_ind]
        g.status = VG_GOOD
        if 0.0 < g.begin and 0.0 < g.votes[0].s and g.fstatus != FIT_NONE:
            g.fstatus = FIT_OLD
        g.begin = 0.0
        if p.length > g.end and p.length > g.votes[-1].s and g.fstatus != FIT_NONE:
            g.fstatus = FIT_OLD
        g.end = p.length
    else:
        ng = Group(p.gen_stroke_id(), p)
        ng.begin, ng.end = 0.0, p.length
        groups.append(ng)


def one_to_one_cover(p, params):
    """SIL_COVER_ONE_TO_ONE -- keep every long-enough group and split the
    holes between neighbours in proportion to their vote counts."""
    min_length = min(params.min_pix_per_group * p.pix_to_ndc,
                     params.min_frac_per_group * p.length)
    groups = p.groups
    fb = []
    for i, g in enumerate(groups):
        if g.status != VG_GOOD:
            continue
        if (g.end - g.begin) < min_length:
            g.status = VG_NOT_ONE_TO_ONE
        else:
            fb.append(_CB(i, g.begin, COVERAGE_START))
            fb.append(_CB(i, g.end, COVERAGE_END))

    fb.sort(key=_cb_key)
    n = len(fb)

    cnt = 1
    for i in range(1, n):
        if cnt == 0:
            del_ = fb[i].s - fb[i - 1].s
            if del_ > 0.0:
                gi_1, gi = groups[fb[i - 1].vg], groups[fb[i].vg]
                a, b = gi_1.num, gi.num
                s = gi_1.end + del_ * a / float(a + b) if (a + b) else gi_1.end
                if (s < gi.begin and gi.votes and s < gi.votes[0].s
                        and gi.fstatus != FIT_NONE):
                    gi.fstatus = FIT_OLD
                gi.begin = s
                if (gi_1.end < s and gi_1.votes and gi_1.votes[-1].s < s
                        and gi_1.fstatus != FIT_NONE):
                    gi_1.fstatus = FIT_OLD
                gi_1.end = s
            cnt += 1
        else:
            cnt += 1 if fb[i].type == COVERAGE_START else -1

    if n > 0 and cnt == 0:
        gf, gl = groups[fb[0].vg], groups[fb[-1].vg]
        if 0.0 < gf.begin and gf.votes and 0.0 < gf.votes[0].s and gf.fstatus != FIT_NONE:
            gf.fstatus = FIT_OLD
        gf.begin = 0.0
        if (gl.end < p.length and gl.votes and gl.votes[-1].s < p.length
                and gl.fstatus != FIT_NONE):
            gl.fstatus = FIT_OLD
        gl.end = p.length
    else:
        ng = Group(p.gen_stroke_id(), p)
        ng.begin, ng.end = 0.0, p.length
        groups.append(ng)


def hybrid_cover(p, params):
    """SIL_COVER_TRIMMED, the GUI's "Hybrid" -- where groups overlap, the more
    confident one wins and the other is trimmed back."""
    min_length = min(params.min_pix_per_group * p.pix_to_ndc,
                     params.min_frac_per_group * p.length)
    groups = p.groups

    boundary = []
    for i, g in enumerate(groups):
        if g.status != VG_GOOD:
            continue
        boundary.append(_CB(i, g.begin, COVERAGE_START))
        boundary.append(_CB(i, g.end, COVERAGE_END))
        g.status = VG_NOT_HYBRID

    final = []
    if boundary:
        for i in range(len(boundary) // 2):
            g = groups[boundary[2 * i].vg]
            g.confidence = (g.end - g.begin) / p.length if p.length else 0.0

        boundary.sort(key=_cb_key)
        final.append(boundary[0])
        current = [boundary[0].vg]

        for cb in boundary[1:]:
            if final[-1].type == COVERAGE_END:
                final.append(cb)
                current = [cb.vg]
            elif cb.type == COVERAGE_END:
                if cb.vg != current[0]:
                    if cb.vg in current:
                        current.remove(cb.vg)
                else:
                    final.append(cb)
                    current.remove(cb.vg)
                    if current:
                        current.sort(key=lambda v: -groups[v].confidence)
                        final.append(_CB(current[0], cb.s, COVERAGE_START))
            else:
                if groups[cb.vg].confidence <= groups[current[0]].confidence:
                    current.append(cb.vg)
                else:
                    final.append(_CB(current[0], cb.s, COVERAGE_END))
                    current[0] = cb.vg
                    final.append(cb)

        # Kill off coverages that ended up too narrow.
        for i in range(len(final) // 2):
            if (final[2 * i + 1].s - final[2 * i].s) < min_length:
                final[2 * i].type = final[2 * i + 1].type = COVERAGE_BAD

    # Fill the holes: each gap is split in proportion to neighbour length.
    i0, i0_len = -1, 0.0
    n = len(final) // 2
    for i in range(n):
        if final[2 * i].type == COVERAGE_BAD:
            continue
        i_len = final[2 * i + 1].s - final[2 * i].s
        if i0 == -1:
            final[2 * i].s = 0.0
        else:
            del_ = final[2 * i].s - final[2 * i0 + 1].s
            if del_ > 0.0:
                denom = i0_len + i_len
                shift = del_ * i0_len / denom if denom else del_ * 0.5
                final[2 * i0 + 1].s += shift
                final[2 * i].s = final[2 * i0 + 1].s
        i0, i0_len = i, i_len

    if i0 != -1:
        final[2 * i0 + 1].s = p.length
        for i in range(n):
            if final[2 * i].type == COVERAGE_BAD:
                continue
            vg = groups[final[2 * i].vg]
            if vg.status == VG_NOT_HYBRID:
                s = final[2 * i].s
                if (s < vg.begin and vg.votes and s < vg.votes[0].s
                        and vg.fstatus != FIT_NONE):
                    vg.fstatus = FIT_OLD
                vg.begin = s
                s = final[2 * i + 1].s
                if (vg.end < s and vg.votes and vg.votes[-1].s < s
                        and vg.fstatus != FIT_NONE):
                    vg.fstatus = FIT_OLD
                vg.end = s
                vg.status = VG_GOOD
            else:
                # Already used: this group got segmented, so clone it.
                ng = Group(p.gen_stroke_id(), p)
                ng.votes = list(vg.votes)
                ng.begin = final[2 * i].s
                ng.end = final[2 * i + 1].s
                groups.append(ng)
    else:
        ng = Group(p.gen_stroke_id(), p)
        ng.begin, ng.end = 0.0, p.length
        groups.append(ng)


# ----------------------------------------------------------------- healing

def heal_groups(p, fit_func, params, rng):
    """SilAndCreaseTexture::heal_groups -- section 5.4.

    Walk the groups in path order.  Where two neighbours already agree in
    phase to within HEAL_JOIN_PIX, merge them into one stroke; where they
    disagree by less than HEAL_DRAG_PIX, plant a pair of VOTE_HEALER votes
    that drag the two fits toward each other on the next refit."""
    join_thresh = params.heal_join_pix
    drag_thresh = params.heal_drag_pix
    pix_to_ndc = p.pix_to_ndc
    freq = p.freq
    groups = p.groups

    final = [i for i, g in enumerate(groups) if g.status == VG_GOOD]
    if len(final) < 2:
        return
    final.sort(key=lambda i: groups[i].begin)

    i0, pi, pi_1 = -1, 0, 0
    i = 0
    while i < len(final):
        gi = groups[final[i]]
        attach = False

        if i < len(final) - 1:
            gi_1 = groups[final[i + 1]]
            ti = gi.get_t(gi.end)
            ti_1 = gi_1.get_t(gi_1.begin)
            floori = math.floor(ti)
            floori_1 = math.floor(ti_1)
            d = (ti_1 - floori_1) - (ti - floori)
            if d > 0.5:
                floori_1 += 1
                d -= 1.0
            elif d < -0.5:
                floori_1 -= 1
                d += 1.0

            dpix = abs((d / freq) / pix_to_ndc) if freq else 0.0

            if dpix < join_thresh:
                attach = True
            elif dpix < drag_thresh:
                nvi = Vote()
                nvi.path_id = gi.votes[0].path_id if gi.votes else -1
                nvi.stroke_id = gi.votes[0].stroke_id if gi.votes else gi.base_id
                nvi.s = gi.end
                nvi.t = ti + d / 2.0
                nvi.status = VOTE_HEALER
                gi.votes.append(nvi)

                nvi_1 = Vote()
                nvi_1.path_id = gi_1.votes[0].path_id if gi_1.votes else -1
                nvi_1.stroke_id = (gi_1.votes[0].stroke_id
                                   if gi_1.votes else gi_1.base_id)
                nvi_1.s = gi_1.begin
                nvi_1.t = ti_1 - d / 2.0
                nvi_1.status = VOTE_HEALER
                gi_1.votes.append(nvi_1)

                gi.fstatus = FIT_OLD
                gi.votes.sort(key=lambda v: v.s)
                gi_1.fstatus = FIT_OLD
                gi_1.votes.sort(key=lambda v: v.s)

            pi_1 = pi + int(floori - floori_1)

        if attach:
            if i0 == -1:
                i0 = i
                ng = Group(p.gen_stroke_id(), p)
                ng.votes = list(gi.votes)
                ng.begin = gi.begin
                groups.append(ng)
            else:
                ng = groups[-1]
                for v in list(gi.votes):
                    w = Vote(v.s, v.t + pi, v.conf, v.stroke_id, v.path_id)
                    w.status = v.status
                    ng.add(w)
            pi = pi_1
        else:
            if i0 != -1:
                ng = groups[-1]
                for v in list(gi.votes):
                    w = Vote(v.s, v.t + pi, v.conf, v.stroke_id, v.path_id)
                    w.status = v.status
                    ng.add(w)
                ng.end = gi.end

                while i > i0:
                    groups[final[i]].status = VG_HEALED
                    del final[i]
                    i -= 1
                groups[final[i]].status = VG_HEALED

                final[i] = len(groups) - 1
                final.sort(key=lambda k: groups[k].begin)

                ng.votes.sort(key=lambda v: v.s)
                if ng.num == 0:
                    arclength_fit(ng, freq)
                else:
                    fit_func(ng, freq, params, rng)

                i0, pi = -1, 0
        i += 1

    # Refit anything that picked up healer votes.
    for k in final:
        g = groups[k]
        if g.fstatus == FIT_OLD:
            g.fits = []
            if g.num == 0:
                arclength_fit(g, freq)
            else:
                fit_func(g, freq, params, rng)


_COVERS = {COVER_MAJORITY: majority_cover,
           COVER_ONE_TO_ONE: one_to_one_cover,
           COVER_TRIMMED: hybrid_cover}
