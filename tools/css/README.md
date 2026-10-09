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
| `--canvas N` | render every frame on a fixed N×N canvas. The exporter crops each frame differently (70 distinct sizes over the bunny sequence), so without this the output varies in size. Default 0 mirrors each frame's own crop, which overlays the exporter's own SVGs exactly. Required for textured brushes. |
| `--fit`, `--cover` | jot's fit and coverage policies. Defaults match jot: `optimize` / `hybrid`. |
| `--report` | per-frame propagation stats plus the swimming measure described below. |
| `--incoherent` | drop propagation entirely, for comparison. |

## Brushes

A brush is anything expressed as a function of the coherent parameter `t`.
That is the entire contract: section 5 hands back (t, position) pairs whose
`t` is stable between frames, so a mark driven by `t` stays pinned to the
surface. A mark driven by screen arclength instead — SVG's own
`stroke-dasharray`, for one — swims, which is the artifact the paper exists
to fix.

`--list-brushes` prints every preset and texture available.

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
`--style texture`; any explicit flag overrides it. The period defaults to
`width * (W/H)` so the stamp is not squashed, clamped to [8, 240] px; a
1-column texture is a pure cross-section with no structure along the stroke,
so it just takes the 30 px default. `--ss` controls supersampling (2 by
default).

SVG cannot warp a bitmap along a curve without an unreasonable number of
elements, which is why these raster. Expect about 0.5 s/frame.

### Adding your own

Everything funnels through two places. For a vector mark, add a branch in
`svgout.write_svg` and a profile in `brushes.py` — profiles are plain
functions of `u = frac(t)` and must agree at `u = 0` and `u = 1` or the
stroke kinks at each period boundary. For a textured one, drop a grayscale +
alpha PNG anywhere and pass its path to `--texture`. If you want to drive
something else entirely, `--json` writes the raw (t, position) samples per
stroke and you can render them however you like.

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
