#!/usr/bin/env python3
"""
Render one frame with every available brush and tile the results into a
contact sheet, so you can look at the brush library instead of reading a list
of names.

  python3 brush_sheet.py <results-dir> [--what all] [--out sheet.png]

`--what styles` covers the vector brushes and their modifiers, `presets` the
29 jot stroke presets, `textures` all 114 jot stroke textures, `all` does
each as its own sheet.  Tiles are a 1:1 crop centred on the drawing so the
brush character is actually visible; pass --whole to fit the figure instead.
"""

import argparse
import io
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import brushes                                      # noqa: E402
import raster                                       # noqa: E402
from css import Params                              # noqa: E402
from export_io import find_frames, load_frame       # noqa: E402
from stylizer import Stylizer                       # noqa: E402
from svgout import Brush, strokes_to_pixels, write_svg   # noqa: E402

CANVAS = 840


# -------------------------------------------------------------- the recipes

def style_variants():
    return [
        ("plain", dict(style="plain", width=2)),
        ("dash", dict(style="dash", width=2)),
        ("dash duty .3", dict(style="dash", width=2, duty=0.3)),
        ("stipple", dict(style="stipple", width=5, duty=0.25)),
        ("ribbon", dict(style="ribbon", width=9)),
        ("ribbon swell", dict(style="ribbon", width=9, press="swell")),
        ("ribbon pencil", dict(style="ribbon", width=9, press="pencil")),
        ("ribbon blob", dict(style="ribbon", width=10, press="blob")),
        ("ribbon taper", dict(style="ribbon", width=9, press="swell",
                              taper=60)),
        ("wiggle 4 hand", dict(style="dash", width=2, wiggle=4)),
        ("wiggle 6 sine", dict(style="dash", width=2, wiggle=6,
                               wiggle_profile="sine")),
        ("ribbon + wiggle", dict(style="ribbon", width=8, press="pencil",
                                 wiggle=3, taper=60)),
        ("phase", dict(style="phase", width=3)),
    ]


def preset_variants():
    out = []
    for name in brushes.list_presets():
        try:
            pre = brushes.Preset.load(name)
        except Exception:
            continue
        out.append((name, dict(style="texture", preset=pre)))
    return out


def texture_variants():
    return [(n, dict(style="texture", texture=n, width=12))
            for n in brushes.list_textures()]


# ------------------------------------------------------------- one rendering

def _svg_to_image(path):
    try:
        import cairosvg
        png = cairosvg.svg2png(url=path, background_color="white")
        return Image.open(io.BytesIO(png)).convert("RGB")
    except ImportError:
        pass
    import shutil
    import subprocess
    exe = shutil.which("rsvg-convert")
    if not exe:
        raise RuntimeError("need cairosvg or rsvg-convert to draw vector "
                           "brushes on the sheet")
    out = path + ".png"
    subprocess.run([exe, "-b", "white", "-o", out, path], check=True)
    return Image.open(out).convert("RGB")


def render_one(frame, spec, tmpdir):
    """Render the frame with one brush; returns an RGB image or None."""
    pre = spec.get("preset")
    style = spec.get("style", "dash")

    texture_name = spec.get("texture") or (pre.texture if pre else None)
    width = spec.get("width", pre.width if pre else 2.0)
    color = spec.get("color", pre.hexcolor() if pre else "#000000")
    opacity = spec.get("opacity", pre.alpha if pre else 1.0)
    taper = spec.get("taper", pre.taper if pre else 0.0)

    tex = None
    period = spec.get("period")
    if style == "texture":
        tex = (brushes.preset_texture(pre, warn=False)[0] if pre
               else brushes.load_texture(texture_name))
        if period is None:
            across, along = tex.shape
            period = (30.0 if along <= 1
                      else min(max(width * along / float(across), 8.0), 240.0))
    if period is None:
        period = 30.0

    sty = Stylizer(Params(offset_pix_len=period))
    res = sty.run_frame(frame)

    brush = Brush(style=style, width=width, color=color,
                  duty=spec.get("duty", 0.6), wiggle=spec.get("wiggle", 0.0),
                  wiggle_profile=spec.get("wiggle_profile", "hand"),
                  taper=taper, press_profile=spec.get("press", "flat"),
                  opacity=opacity)

    if tex is not None:
        cov = raster.render(strokes_to_pixels(res.strokes, frame.cam, CANVAS),
                            CANVAS, tex, width=brush.width,
                            taper_px=brush.taper, wiggle_px=brush.wiggle,
                            wiggle_profile=brush.wiggle_profile,
                            press_profile=brush.press_profile, ss=2)
        a = np.clip(cov * brush.opacity, 0.0, 1.0)[..., None]
        rgb = np.array([int(color[i:i + 2], 16) / 255.0
                        for i in (1, 3, 5)]).reshape(1, 1, 3)
        img = (1.0 - a) + rgb * a
        return Image.fromarray((img * 255).round().astype(np.uint8), "RGB")

    path = os.path.join(tmpdir, "f.svg")
    write_svg(path, res.strokes, frame.cam, brush, background="#ffffff",
              canvas=CANVAS)
    return _svg_to_image(path)


# ------------------------------------------------------------------- tiling

def ink_box(img):
    a = np.asarray(img.convert("L"))
    ys, xs = np.where(a < 245)
    if len(xs) == 0:
        return (0, 0, img.width, img.height)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def crop_tile(img, tw, th, box, whole):
    if whole:
        x0, y0, x1, y1 = box
        m = 12
        c = img.crop((max(x0 - m, 0), max(y0 - m, 0),
                      min(x1 + m, img.width), min(y1 + m, img.height)))
        c.thumbnail((tw, th), Image.LANCZOS)
        out = Image.new("RGB", (tw, th), "white")
        out.paste(c, ((tw - c.width) // 2, (th - c.height) // 2))
        return out
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    x = min(max(cx - tw // 2, 0), max(img.width - tw, 0))
    y = min(max(cy - th // 2, 0), max(img.height - th, 0))
    return img.crop((x, y, x + tw, y + th))


def compose(tiles, cols, tw, th, title):
    font = ImageFont.load_default(size=15)
    tfont = ImageFont.load_default(size=22)
    bar, pad, top = 22, 6, 34
    rows = (len(tiles) + cols - 1) // cols
    W = cols * (tw + pad) + pad
    H = top + rows * (th + bar + pad) + pad
    sheet = Image.new("RGB", (W, H), "#ffffff")
    d = ImageDraw.Draw(sheet)
    d.text((pad + 2, 8), title, fill="#000000", font=tfont)

    for i, (label, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = top + r * (th + bar + pad)
        d.rectangle([x, y, x + tw - 1, y + bar - 1], fill="#eeeeee")
        d.text((x + 4, y + 3), label[:46], fill="#111111", font=font)
        sheet.paste(img, (x, y + bar))
        d.rectangle([x, y, x + tw - 1, y + bar + th - 1], outline="#999999")
    return sheet


# --------------------------------------------------------------------- main

def build(frame, variants, title, out_path, cols, tw, th, whole, quiet):
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, spec) in enumerate(variants):
            try:
                img = render_one(frame, spec, tmp)
            except Exception as e:
                if not quiet:
                    print("  %-28s skipped: %s" % (label, e))
                continue
            tiles.append((label, crop_tile(img, tw, th, ink_box(img), whole)))
            if not quiet:
                print("  [%3d/%3d] %s" % (i + 1, len(variants), label))
    if not tiles:
        print("nothing rendered for %s" % title)
        return
    compose(tiles, cols, tw, th, title).save(out_path)
    print("wrote %s  (%d brushes)" % (out_path, len(tiles)))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="frame directory, or a directory of them")
    p.add_argument("--what", choices=["styles", "presets", "textures", "all"],
                   default="all")
    p.add_argument("--out", help="output PNG (or, with --what all, the "
                                 "directory to write the three sheets into)")
    p.add_argument("--frame", type=int, default=0,
                   help="index of the frame to draw")
    p.add_argument("--cols", type=int, help="tiles per row")
    p.add_argument("--tile", default="360x280", help="tile size, WxH")
    p.add_argument("--whole", action="store_true",
                   help="fit the whole figure in each tile instead of a 1:1 "
                        "detail crop")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        sys.exit("no such directory: %s" % root)
    frames = find_frames(root)
    if not frames:
        sys.exit("no frame directories under %s" % root)
    frame = load_frame(frames[min(a.frame, len(frames) - 1)])

    tw, th = (int(v) for v in a.tile.lower().split("x"))

    jobs = {"styles": (style_variants, "Vector brushes (--style and modifiers)", 4),
            "presets": (preset_variants, "jot stroke presets (--preset)", 5),
            "textures": (texture_variants, "jot stroke textures (--texture)", 8)}
    which = list(jobs) if a.what == "all" else [a.what]

    out_dir = a.out if (a.what == "all" and a.out) else os.getcwd()
    if a.what == "all":
        os.makedirs(out_dir, exist_ok=True)

    for k in which:
        make, title, defcols = jobs[k]
        out = (a.out if a.what != "all" and a.out
               else os.path.join(out_dir, "brushes_%s.png" % k))
        if not a.quiet:
            print(title)
        build(frame, make(), title, out, a.cols or defcols, tw, th, a.whole,
              a.quiet)


if __name__ == "__main__":
    main()
