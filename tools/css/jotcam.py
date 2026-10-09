"""
jot's camera convention, reimplemented outside jot.

Mirrors CAMdata (disp/cam.H:65-73), CAMdata::xform() / projection_xform()
(disp/cam.C:168-360) and CAM::ndc_projection() (disp/cam.C:671).

Conventions, all verified against jot's source:

  * from / at / up are world POINTS.  The view direction is at - from and the
    up direction is up - from (NOT up itself).
  * Eye space is right-handed: the camera looks down -Z with +Y up, so the
    view transform equals gluLookAt(from, at, up - from).
  * The eye-space origin sits at the film-plane center, from + focal*dir --
    jot translates `center`, not the eye, to the origin.  The eye is then at
    z = +focal.
  * projection_xform maps the world-space film rectangle (width x height) to
    [-1,1]^2, then applies the perspective matrix
        [f 0  0 0; 0 f 0 0; 0 0 1 0; 0 0 -1 f]
    so ndc = f*X / (f - Z).
  * ndc_projection then multiplies by (w/h,1,1) or (1,h/w,1): the SHORTER
    image axis spans [-1,1] and the longer spans [-L,L], L = long/short.
  * NDC and pixel coordinates are Y-UP (mlib/points.C:312): pixel row 0 is
    NDC y = -1.  SVG is Y-down from the top-left, hence the flip in
    ndc_to_svg / svg_to_ndc.
"""

import json
import math

import numpy as np


def _unit(v):
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return v / n


class JotCamera:
    """A jot CAMdata plus the image framing needed to land on the SVG."""

    def __init__(self, d):
        self.frm = np.asarray(d["from"], dtype=float)
        self.at = np.asarray(d["at"], dtype=float)
        self.up = np.asarray(d["up"], dtype=float)
        self.focal = float(d["focal"])
        self.film_w = float(d["width"])
        self.film_h = float(d["height"])
        self.perspective = bool(d.get("perspective", True))

        self.image_w = float(d["image_width"])
        self.image_h = float(d["image_height"])

        frame = d.get("svg_frame") or {}
        self.svg_w = float(frame.get("width", self.image_w))
        self.svg_h = float(frame.get("height", self.image_h))
        self.off_x = float(frame.get("offset_x", 0.0))
        self.off_y = float(frame.get("offset_y", 0.0))

        # Eye basis.  jot: "Rotate so that normal lies on negative z-axis,
        # and proj_up is on y-axis."
        self.dir = _unit(self.at - self.frm)          # view direction
        up_dir = _unit(self.up - self.frm)            # up is a POINT
        self.ez = -self.dir
        self.ey = _unit(up_dir - np.dot(up_dir, self.ez) * self.ez)
        self.ex = np.cross(self.ey, self.ez)
        self.center = self.frm + self.focal * self.dir

        # Aspect scaling from CAM::ndc_projection().
        if self.image_w > self.image_h:
            self.Lx, self.Ly = self.image_w / self.image_h, 1.0
        else:
            self.Lx, self.Ly = 1.0, self.image_h / self.image_w

        # VIEW::pix_to_ndc_scale() == 2/min(w,h)  (disp/view.C:1420)
        self.pix_to_ndc = 2.0 / min(self.image_w, self.image_h)

    # ---------------------------------------------------------------- load

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    # ---------------------------------------------------------- projection

    def eye(self, pts):
        """World points (...,3) -> eye coords, origin at the film-plane center."""
        d = np.asarray(pts, dtype=float) - self.center
        return np.stack([d @ self.ex, d @ self.ey, d @ self.ez], axis=-1)

    def depth(self, pts):
        """Distance along the view direction from the eye.  Smaller == nearer."""
        return (np.asarray(pts, dtype=float) - self.frm) @ self.dir

    def ndc(self, pts):
        """World points (...,3) -> NDC xy (...,2).  Points at/behind the eye
        come back as NaN."""
        e = self.eye(pts)
        X = 2.0 * e[..., 0] / self.film_w
        Y = 2.0 * e[..., 1] / self.film_h
        if self.perspective:
            w = self.focal - e[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(w > 1e-12, self.focal * X / w, np.nan)
                y = np.where(w > 1e-12, self.focal * Y / w, np.nan)
        else:
            x, y = X, Y
        return np.stack([x * self.Lx, y * self.Ly], axis=-1)

    def in_frustum(self, ndc, pts=None):
        """NDCZpt::in_frustum (mlib/points.C:194), plus an in-front-of-eye test."""
        ndc = np.asarray(ndc, dtype=float)
        ok = (np.abs(ndc[..., 0]) <= self.Lx) & (np.abs(ndc[..., 1]) <= self.Ly)
        ok &= np.isfinite(ndc).all(axis=-1)
        if pts is not None:
            ok &= self.depth(pts) > 0.0
        return ok

    # ------------------------------------------------------- ndc <-> pixel

    def ndc_to_svg(self, ndc):
        """NDC xy -> SVG pixel xy (y-down, origin at the SVG frame's top-left)."""
        ndc = np.asarray(ndc, dtype=float)
        x = (ndc[..., 0] / self.Lx + 1.0) * 0.5 * self.image_w - self.off_x
        y = (1.0 - ndc[..., 1] / self.Ly) * 0.5 * self.image_h - self.off_y
        return np.stack([x, y], axis=-1)

    def ndc_to_canvas(self, ndc, size):
        """NDC xy -> pixels on a fixed `size` x `size` canvas spanning the
        full NDC square, ignoring this frame's own crop.  Use this when the
        exporter's per-frame crop varies and you want a stable animation."""
        ndc = np.asarray(ndc, dtype=float)
        x = (ndc[..., 0] / self.Lx + 1.0) * 0.5 * size
        y = (1.0 - ndc[..., 1] / self.Ly) * 0.5 * size
        return np.stack([x, y], axis=-1)

    def svg_to_ndc(self, px):
        px = np.asarray(px, dtype=float)
        x = (2.0 * (px[..., 0] + self.off_x) / self.image_w - 1.0) * self.Lx
        y = (1.0 - 2.0 * (px[..., 1] + self.off_y) / self.image_h) * self.Ly
        return np.stack([x, y], axis=-1)
