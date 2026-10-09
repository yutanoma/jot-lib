"""
Reader for krawczyk-contours' `--output-3d-polylines` export.

Per frame directory:
  contour_export_3d.obj                   per chain: `o chain_i`, its `v`, its
                                          `vn`, then its `l a b` (global 1-based)
  contour_export_3d_visibility.txt        one flag per vertex, in `v` order
  contour_export_3d_edge_visibility.txt   one flag per edge,   in `l` order
  contour_export_3d_camera.json           jot-convention camera (see jotcam.py)
"""

import os

import numpy as np

from jotcam import JotCamera

OBJ = "contour_export_3d.obj"
VIS_V = "contour_export_3d_visibility.txt"
VIS_E = "contour_export_3d_edge_visibility.txt"
CAM = "contour_export_3d_camera.json"


class Chain:
    """One exported polyline: M vertices, M-1 edges.

    A closed chain repeats its first vertex at the end, which is also how jot
    marks a closed LuboPath (`_pts[0] == _pts.last()`, zxedge_stroke_texture.H:491).
    """

    __slots__ = ("name", "wpts", "wnrm", "edge_vis", "closed", "vseq",
                 "kappa")

    def __init__(self, name, wpts, wnrm, edge_vis, vseq=None):
        self.name = name
        self.wpts = wpts
        self.wnrm = wnrm
        self.edge_vis = edge_vis
        # The OBJ `v` indices this chain walked, parallel to wpts.  Sibling
        # per-vertex files (visibility, radial curvature) are in `v` order,
        # so this is what maps them onto the chain.
        self.vseq = None if vseq is None else np.asarray(vseq, dtype=np.intp)
        self.kappa = None
        self.closed = len(wpts) > 2 and np.array_equal(wpts[0], wpts[-1])


class Frame:
    __slots__ = ("path", "cam", "chains")

    def __init__(self, path, cam, chains):
        self.path = path
        self.cam = cam
        self.chains = chains


def _read_flags(path, n, what):
    with open(path) as f:
        vals = [int(float(t)) for t in f.read().split()]
    if len(vals) != n:
        raise ValueError("%s: expected %d %s flags, got %d"
                         % (path, n, what, len(vals)))
    return np.asarray(vals, dtype=bool)


def load_frame(d):
    """Load one frame directory into a Frame."""
    cam = JotCamera.from_json(os.path.join(d, CAM))

    verts, norms = [], []
    edges = []            # (a, b) 0-based, in file order
    edge_obj = []         # index into `objs` for each edge
    objs = []             # object names, in file order
    cur = -1

    with open(os.path.join(d, OBJ)) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            tag, _, rest = line.partition(" ")
            if tag == "v":
                verts.append([float(x) for x in rest.split()[:3]])
            elif tag == "vn":
                norms.append([float(x) for x in rest.split()[:3]])
            elif tag == "l":
                idx = [int(x) for x in rest.split()]
                for a, b in zip(idx, idx[1:]):
                    edges.append((a - 1, b - 1))
                    edge_obj.append(cur)
            elif tag == "o":
                objs.append(rest.strip())
                cur = len(objs) - 1

    verts = np.asarray(verts, dtype=float)
    norms = np.asarray(norms, dtype=float) if norms else np.zeros_like(verts)
    if len(norms) != len(verts):
        raise ValueError("%s: %d vertices but %d normals"
                         % (d, len(verts), len(norms)))

    evis = _read_flags(os.path.join(d, VIS_E), len(edges), "edge")

    # Group edges by object and walk each run into an ordered vertex list.  A
    # chain whose edges are not head-to-tail is split at the break.
    chains = []
    i, n = 0, len(edges)
    while i < n:
        o = edge_obj[i]
        j = i + 1
        while j < n and edge_obj[j] == o and edges[j][0] == edges[j - 1][1]:
            j += 1
        seq = [edges[i][0]] + [edges[k][1] for k in range(i, j)]
        if len(seq) >= 2:
            name = objs[o] if 0 <= o < len(objs) else "chain_?"
            chains.append(Chain(name, verts[seq], norms[seq], evis[i:j],
                                seq))
        i = j

    return Frame(d, cam, chains)


def find_frames(root):
    """Frame directories under `root`, sorted by name.  `root` itself counts
    if it is one."""
    if os.path.isfile(os.path.join(root, CAM)):
        return [root]
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, CAM)):
            out.append(p)
    return out
