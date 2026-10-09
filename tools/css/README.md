# Coherent Stylized Silhouettes, outside jot

A standalone port of the temporal-coherence machinery from
*Coherent Stylized Silhouettes* (Kalnins, Davidson, Markosian, Finkelstein,
SIGGRAPH 2003), lifted out of jot so it can run on contours produced by
something other than jot's own silhouette extractor.

It consumes the 3D polyline export from `krawczyk-contours`
(`grid_primitive --output-3d-polylines`) and writes one stylized SVG per
frame, with the stroke parameter carried forward from frame to frame so a
dash pattern stays anchored to the surface instead of swimming along the
line.

## Running

    python3 stylize.py <results-dir> [--out DIR] [options]

`<results-dir>` is one frame directory, or a parent holding many. Each frame
directory needs `contour_export_3d.obj`,
`contour_export_3d_edge_visibility.txt` and `contour_export_3d_camera.json`.
Frames run in sorted order and the parameterization flows between them, so
they cannot be reordered or run in parallel.

    # all 200 bunny frames, fixed canvas, with diagnostics
    python3 stylize.py ~/Projects/research/krawczyk-contours/results/rendering/bunny_hr \
        --canvas 840 --report

Output lands in `<results-dir>_css` unless you pass `--out`. Roughly
0.4 s/frame; the full 200 take about 70 s.

Only numpy is required.

### Options worth knowing

| flag | what it does |
|---|---|
| `--style` | which mark to lay down: see **Brushes** below. |
| `--period-pix N` | pixels per unit of stroke parameter: the stylization period. 30 by default for vector brushes; textures derive their own. |
| `--out DIR` | where results go. Default for SVG is `<root>_css` beside the input; PNG goes to the sibling `results/png/<name>_css_<brush>/` instead — see **Where output lands** below. |
| `--canvas N` | render every frame on a fixed N×N canvas. The exporter crops each frame differently (70 distinct sizes over the bunny sequence), so without this the output varies in size. Default 0 mirrors each frame's own crop, which overlays the exporter's own SVGs exactly. Required for textured brushes. |
| `--fit`, `--cover` | jot's fit and coverage policies. Defaults match jot: `optimize` / `hybrid`. |
| `--taper-mode` | what drives a stroke's width envelope. Defaults to `curvature`, which is **not** what jot does — see **Radial curvature** below. `end` restores jot's behaviour. |
| `--report` | per-frame propagation stats plus the swimming measure described below. |
| `--incoherent` | drop propagation entirely, for comparison. |

## Brushes

A brush is anything expressed as a function of the coherent parameter `t`.
That is the entire contract: section 5 hands back (t, position) pairs whose
`t` is stable between frames, so a mark driven by `t` stays pinned to the
surface. A mark driven by screen arclength instead — SVG's own
`stroke-dasharray`, for one — swims, which is the artifact the paper exists
to fix.

### Seeing them

`--list-brushes` prints the names. To actually look at them, `brush_sheet.py`
renders one frame with every brush and tiles the results:

    python3 brush_sheet.py <results-dir> --what all --out sheets/

That writes `brushes_styles.png` (the vector brushes and their modifiers),
`brushes_presets.png` (all 29 jot presets) and `brushes_textures.png` (all
114 jot textures). Tiles are a 1:1 crop centred on the drawing so the brush
character is visible; `--whole` fits the whole figure instead, and
`--tile WxH` / `--cols N` control the grid. Budget about a minute for the
texture sheet.

### Vector brushes (write SVG)

| `--style` | mark |
|---|---|
| `plain` | the bare contour, constant width |
| `dash` | on/off in parameter space — the default |
| `stipple` | one dot per period |
| `ribbon` | filled outline whose width follows `--press` and `--taper` |
| `phase` | every segment tinted by fractional `t`; a coherence debug view |

Modifiers, usable with any of them:

    --width PX            brush width
    --color '#000000'     --opacity 0..1
    --duty 0.6            inked fraction of each period (dash, stipple)
    --press flat|swell|pencil|blob     width profile within a period
    --taper PX            narrowing at each stroke end (ribbon)
    --wiggle PX           lateral wobble, one per period
    --wiggle-profile hand|sine

`--wiggle` is jot's own brush model in miniature: a `BaseStrokeOffsetLIST`
(`stroke/base_stroke.H:113`) is a recorded gesture stored as lateral
displacement against position within the pattern, rubber-stamped every
`_pix_len` pixels. Our period is that `_pix_len`, so `frac(t)` is position
within one stamp.

    python3 stylize.py <dir> --canvas 840 --style ribbon --width 8 \
        --press pencil --wiggle 3 --taper 60

### Textured brushes (write PNG)

jot ships 114 brush textures in `nprdata/stroke_textures` and 29 presets in
`nprdata/stroke_presets`. A texture is a pure alpha mask, W samples along one
period of `t` by H across the stroke; the colour comes from the preset, not
the texture. These stamp along the stroke with `u = frac(t)`, so they are
coherent for the same reason the dashes are.

    python3 stylize.py <dir> --canvas 840 --preset pencil --width 16
    python3 stylize.py <dir> --canvas 840 --style texture \
        --texture 2D--dot-dash-64 --width 14

`--preset` supplies colour, alpha, width, taper and texture, and implies
`--style texture`; any explicit flag overrides it. Two presets have no usable
texture -- `wash_blue` names none, and `balloon` names a `noisy1.png` that
jot does not ship -- and both draw over a solid mask instead, which is what
jot does for an untextured BaseStroke.

The period defaults to `width * (W/H)` so the stamp is not squashed, clamped
to [8, 240] px; a 1-column texture is a pure cross-section with no structure
along the stroke, so it just takes the 30 px default. `--period-pix`
overrides it, and is the main dial for how a texture reads: the `2D--gauss-*`
family is a blob centred in its tile, so at the default period the stamps sit
end to end and bead, while a shorter period overlaps them into a continuous
ribbed line. `--ss` controls supersampling (2 by default).

SVG cannot warp a bitmap along a curve without an unreasonable number of
elements, which is why these raster. Expect about 0.5 s/frame.

### Where output lands

Vector runs write SVG to `<root>_css` beside the input, e.g.
`results/rendering/bunny_hr_css/`. Textured runs write PNG to the sibling
`results/png/` instead, matching the `<png-dir>/<object>` layout
`make_video.py` already uses, so they drop straight into the video pipeline
with no SVG-to-PNG step:

    results/rendering/bunny_hr        ->  results/png/bunny_hr_css_pencil

The folder is tagged with the preset or texture name. That matters: a plain
`bunny_hr_css` would collide with the folder `make_video.py` fills when it
rasterizes the SVG run of the same name, and would silently destroy it.
Tagging also lets several brushes coexist. `--out` overrides all of this.

Since the PNGs are already rasterized, build a video straight from them
rather than going through `make_video.py`:

    ffmpeg -framerate 24 -pattern_type glob -i 'results/png/bunny_hr_css_pencil/*.png' \
        -c:v libx264 -pix_fmt yuv420p results/videos/bunny_hr_css_pencil.mp4

### Adding your own

Everything funnels through two places. For a vector mark, add a branch in
`svgout.write_svg` and a profile in `brushes.py` — profiles are plain
functions of `u = frac(t)` and must agree at `u = 0` and `u = 1` or the
stroke kinks at each period boundary. For a textured one, drop a grayscale +
alpha PNG anywhere and pass its path to `--texture`. If you want to drive
something else entirely, `--json` writes the raw (t, position) samples per
stroke and you can render them however you like.

## Radial curvature

`--taper-mode` decides what drives a stroke's width envelope.  It defaults to
`curvature`; `end` is what jot does.

`end` narrows the stroke over `--taper` pixels at each of its own ends. Those
ends are not a property of the surface. They are wherever this frame happened
to cut the stroke — a chain split, a group
boundary, an occluder sweeping past — so the envelope moves even when the
parameterization under it does not. CSS removes the swim from *where the
marks sit*; the taper puts some of it back in *how wide they are*.

`curvature` drives the envelope from the radial curvature instead:

    kappa_r(x) = (v^T H v) / (|v|^2 |grad f|)        v = x - eye

the normal curvature of the surface in the viewing direction. It belongs to
the surface point, not to the stroke, so it moves smoothly with the camera
and knows nothing about chain topology. It also tapers in the right places
on its own: `v^T H v = 0` is the cusp condition, so `kappa_r -> 0` exactly
where the contour terminates or doubles back.

On the bunny that is not a hope, it is measurable. Over 40 frames, taking the
ends of every visible run against their interiors:

| | median `|kappa_r|` | fraction below 0.2 |
|---|---|---|
| ends of visible runs (n=946) | 0.0000 | 52% |
| interiors (n=95429) | 1.6040 | 1.5% |

The ends of the visible contour here really are cusps, and the curvature
field finds them without being told where a stroke stops.

### Is it actually steadier?

The taper does not touch stroke geometry, so one run supplies both envelopes
and they can be compared on identical samples. Match each sample of frame *i*
to the nearest point of frame *i+1*'s curves (within 1.5 px — 39% match, the
rest being contour that genuinely appeared or vanished) and ask how much the
envelope at a fixed place on screen changed. 60 frames, 12194 matched pairs:

| envelope | spread | mean &#124;d&#124; | p95 &#124;d&#124; | jumps > 0.25 | mean/spread |
|---|---|---|---|---|---|
| end taper, 20 px | 0.232 | 0.056 | 0.420 | 7.6% | 0.241 |
| end taper, 40 px | 0.257 | 0.061 | 0.395 | 7.7% | 0.237 |
| curvature, gamma 0.5 | 0.158 | 0.030 | 0.146 | 2.9% | 0.189 |
| curvature, gamma 1.0 (default) | 0.192 | 0.038 | 0.213 | 4.1% | 0.198 |
| curvature, gamma 1.5 | 0.222 | 0.043 | 0.254 | 5.1% | 0.195 |

`python3 taper_check.py <results-dir> --frames 60` reproduces that table;
re-run it once the exporter writes the real curvature file.

`spread` is the standard deviation of the envelope over the whole sequence,
and it is in the table because a constant envelope would score a perfect
zero: the curvature rows look better partly because at these settings they
vary less. The honest comparison is the row with matching spread — gamma 1.5
against the 20 px end taper — and there it is still 23% steadier on the mean,
40% on the p95, and a third fewer of the jumps large enough to see.
Lengthening the end taper does not close the gap (0.237 vs 0.241 normalized),
which is the point: its instability is structural, not a tuning artifact.

What remains is real. The 0.016% of visible nodes with no curvature value are
far too few to explain the tail, so the surviving jumps are cusps genuinely
sweeping across the drawing — an event, not an artifact.

`--taper-mode both` multiplies the two, which is what you want if strokes get
cut by occluders often: geometry sets the width, and the stroke still closes
cleanly where an occluder, rather than a cusp, ended it.

### Normalization

`--curv-ref` is the curvature at which a stroke reaches full width. Left at 0
it is the median `|kappa_r|` over the **first** frame's visible nodes and is
then held fixed. Do not make this per-frame: normalizing by each frame's own
range would make a stationary point's width depend on what else is on screen,
which is precisely the swimming being removed.

`--curv-gamma` shapes the ramp (below 1 widens the midtones), `--curv-floor`
sets how thin it is allowed to get. The field spans a wide range — on the
bunny's visible nodes, p01 0.07 to p99 16 around a median of 1.6 — so some
shaping is not optional; a raw proportionality leaves almost everything at
hairline.

Two chains on this model sit near `|kappa_r|` 0.12 and 0.34 throughout. They
are genuinely flat and will draw uniformly thin. That is the feature working,
but raise `--curv-floor` if you would rather they stayed visible.

### The file

Preferred, and what the exporter should write:

    contour_export_3d_radial_curvature.txt

one float per OBJ vertex, in the OBJ's own `v` order, one per line — exactly
the convention of the sibling `_visibility.txt` files. `nan` for a node whose
curvature could not be evaluated; never a silent 0, which is indistinguishable
from a genuine cusp. The reader treats `nan` as "no opinion" and draws full
width there.

In krawczyk-contours this is a few lines next to `writeVisibilityFile`.
`computeChainNodeRadialCurvature` and `buildContour3dPolylines` already run on
the same `g_refinedContourBeadChains`, node for node, so for emitted chain `c`
the value for OBJ vertex `chainRanges[c][0] + i` is
`kappaR[chainSourceIndex[c]][i]`. Gate it on the existing `--emit-curvature`.

Until then there is a fallback, used automatically: the
`data-radial-curvature` attributes already present in
`contour_export_bsp_qi.svg` are matched back onto the OBJ by projecting the
vertices with this frame's camera. It recovers 99.7% of nodes and costs about
0.15 s per frame. It is a stopgap, not a plan — the QI SVG splits chains at
crossings and carries positions at pixel precision — but it is why the numbers
above exist without touching the exporter.

## Checking that it works

`--report` prints, per frame, the mean and median distance between what the
previous frame's samples asked for and the parameterization this frame
settled on — the swimming the algorithm exists to suppress, in pixels of
stroke pattern.

Over all 200 bunny frames with the defaults:

    median swim    0.95 px   (p95 2.07, max 3.16) of a 30 px period

For a baseline, `--fit sigma` keeps propagation running (so the measurement
is still made) but fits each group from scratch, ignoring the votes:

    median swim    7–46 px

Note that `--fit interpolate` always reports zero swim: that fit passes
exactly through every vote, so the residual is zero by construction and the
number means nothing for that policy.

The baseline is deceptively stable while the chain topology holds and pops by
up to half a period when a chain splits or merges, which is exactly the
artifact the paper is about. To see it:

    for m in optimize sigma; do
      python3 stylize.py <results-dir> --out /tmp/$m --canvas 840 --end 8 --fit $m --width 3 -q
    done
    # then overlay /tmp/optimize/0006.svg vs 0007.svg, and the same for sigma

## What is ported, and from where

| CSS section | jot source | here |
|---|---|---|
| §4 brush path generation, ID reference image | `npr/zxedge_stroke_texture.C`: `draw_id_ref_param_vis_pass`, `propagate_sil_parameterization`, `LuboPath::register_vote` / `in_range` / `get_closest_point_at` / `gen_group_samples` | `css.py` (`Path`, `IdRefImage`), `stylizer.py` |
| §5 parameterization | `npr/sil_and_crease_texture.C`: `generate_sil_groups` and everything it calls | `fitting.py` |
| stroke emission | `generate_strokes_from_groups` | `stylizer.py` |
| camera convention | `disp/cam.{H,C}`, `mlib/points.C` | `jotcam.py` |

Every jot `Config` / `SilUI` knob the port reads is in `css.Params` with
jot's own default next to it.

## Deviations from jot, and why

* **The ID reference image is rasterized in software.** jot gets it from
  `glLineWidth(3)` + `GL_SMOOTH` + `GL_DEPTH_TEST`. We rasterize 3 px lines
  with an interpolated arclength byte and a z-buffer, and take visibility
  from the caller's per-edge flag rather than a depth test against a surface
  we do not have. The 8-bit quantization of the arclength byte is kept,
  because it sets the search window width in `get_closest_point_at`.
* **Ids are not bit-packed.** jot packs path id, visibility and arclength
  into one 32-bit pixel; we keep the segment id and the byte in two parallel
  buffers. Semantically identical, and it avoids reproducing the bit layout.
* **Samples carry a 3D point and surface normal directly.** jot recovers both
  from a `Bsimplex` plus barycentric coordinates, which requires a mesh.
  `LuboSample` has no `Wpt` field at all and `get_wpt()` silently no-ops on a
  null simplex, so this had to change.
* **Arclength is planar.** jot's `NDCZpt_list::length()` includes the depth
  coordinate; here depth is a world distance, so including it would be
  meaningless.
* **Emitted geometry is clipped to visible edge runs.** jot splits its
  silhouette strips into separate `LuboPath`s per visibility and lets the
  stroke pools decide what to draw. We keep each exported chain whole, so the
  parameterization survives short occlusions, and trim at output time.
* **Both normal directions are searched** (`--one-sided` restores jot's
  behavior). jot crawls outward along the surface normal; an externally
  supplied normal's outward sign is not guaranteed to match, so trying both
  is the safe default. On the bunny sequence it makes no difference at all --
  the `+normal` pass already finds everything, so the exported normals agree
  with jot's convention — but it costs nothing and protects another dataset.
* **`optimizing_fit` solves with numpy** instead of jot's `ludcmp`/`lubksb`,
  and its backwards-fit test compares against the previous knot. jot's copy
  compares against a constant — see the `Karol->` comment at
  `sil_and_crease_texture.C:2480`, which re-declares `fj_1` inside the loop.
* **`split_looped_groups` is not ported.** It is commented out in jot's own
  pipeline.
* **Stroke width tapers on radial curvature by default**, where jot tapers on
  distance from the stroke's ends. This is the one deliberate departure from
  jot's output rather than from its implementation, and it exists because the
  end taper is the last part of the drawing CSS's propagation never reached:
  it keys off stroke ends, which move. `--taper-mode end` restores jot's
  behaviour exactly. See **Radial curvature** for the measurements.

## Camera convention

`jotcam.py` implements jot's convention exactly; the export's own acceptance
check reproduces to under 0.001 px. The parts that bite:

* `from`, `at` and `up` are all world **points** — the up direction is
  `up - from`, not `up`.
* Eye space is right-handed, looking down −Z with +Y up, with the origin at
  the film-plane centre (`from + focal*dir`) and the eye at `z = +focal`.
* The shorter image axis spans NDC [−1,1]; the longer spans [−L,L],
  L = long/short.
* NDC and pixel coordinates are **y-up**; SVG is y-down. Getting this
  backwards produces a vertically mirrored result that still looks plausible.
