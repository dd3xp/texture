"""Half-page teaser for paper2: (a) task + one-line method, (b) Artist / TRD / Render."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "model"), str(ROOT / "tools")]
from colour_task import targets  # noqa: E402

BASE = ROOT / "experiments/baselines"
OUT = ROOT / "figures/fig_teaser.png"
PAD = 8
LAB = 88
S = 124
COLS = [
    ("default_cobble", "cobble"),
    ("mcl_core_log_birch", "log birch"),
    ("mcl_core_gold_ore", "gold ore"),
    ("mcl_core_emerald_ore", "emerald ore"),
    ("default_cactus_side", "cactus side"),
    ("mcl_deepslate_diamond_ore", "deepslate"),
]
TASK = "mcl_core_log_spruce"
T = {t["slug"]: t for t in targets("E_mat", 16) if t["j"] == 0}

try:
    FONT = ImageFont.truetype("arial.ttf", 14)
    BOLD = ImageFont.truetype("arialbd.ttf", 14)
    BIG = ImageFont.truetype("arialbd.ttf", 16)
    TINY = ImageFont.truetype("arial.ttf", 12)
except OSError:
    FONT = BOLD = BIG = TINY = ImageFont.load_default()


def load(method, slug):
    if method is None:
        return np.asarray(T[slug]["ref"], np.uint8)
    d = BASE / method / "16"
    for p in (d / f"{slug}_0.png", d / f"{slug}.png"):
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    raise FileNotFoundError(f"{method} {slug}")


def nn(arr, size):
    return Image.fromarray(np.asarray(arr, np.uint8)).resize((size, size), Image.NEAREST)


def tile2x2(arr, size):
    return nn(np.tile(np.asarray(arr, np.uint8), (2, 2, 1)), size)


def wall(arr, n=3, cell=26):
    t = np.asarray(nn(arr, cell))
    im = Image.fromarray(np.tile(t, (n, n, 1)))
    d = ImageDraw.Draw(im)
    for i in range(n + 1):
        p = min(i * cell, im.width - 1)
        d.line([(p, 0), (p, im.height - 1)], fill=(40, 40, 40), width=1)
        d.line([(0, p), (im.width - 1, p)], fill=(40, 40, 40), width=1)
    return im


def palette_ramp(arr, n=8, w=12, h=16):
    pix = np.asarray(arr, np.uint8).reshape(-1, 3)
    uniq = np.unique(pix, axis=0)
    order = np.argsort(uniq.astype(np.float32) @ [0.2126, 0.7152, 0.0722])
    cols = uniq[order][:n]
    im = Image.new("RGB", (n * w, h), (245, 250, 255))
    d = ImageDraw.Draw(im)
    for i, c in enumerate(cols):
        d.rectangle([i * w, 0, (i + 1) * w - 1, h - 1], fill=tuple(int(x) for x in c))
    return im


def ranks(arr, size):
    g = np.asarray(arr, np.float32) @ [0.2126, 0.7152, 0.0722]
    lo, hi = g.min(), g.max()
    n = np.zeros_like(g, np.uint8) if hi <= lo else np.round(255 * (g - lo) / (hi - lo)).astype(np.uint8)
    return nn(np.stack([n, n, n], -1), size)


def wrap_icon(size=48):
    im = Image.new("RGB", (size, size), (245, 250, 255))
    d = ImageDraw.Draw(im)
    cs = size // 4
    for i in range(4):
        for j in range(4):
            fill = (70, 130, 210) if i in (0, 3) and j == 1 else (210, 210, 210)
            d.rectangle([i * cs, j * cs, i * cs + cs - 2, j * cs + cs - 2], fill=fill)
    d.arc([-6, cs - 8, size + 5, cs + 18], 200, 340, fill=(40, 90, 180), width=2)
    return im


def labelled(im, text, font=TINY):
    """Stack a caption centred under an image. Empty text keeps the original size."""
    if not text:
        return im
    w, h = im.size
    out = Image.new("RGB", (max(w, 72), h + 18), "white")
    out.paste(im, ((out.width - w) // 2, 0))
    ImageDraw.Draw(out).text((out.width // 2, h + 2), text, fill=(80, 80, 80), font=font, anchor="mt")
    return out


def plus_or_arrow(ch, h):
    im = Image.new("RGB", (28, h), "white")
    ImageDraw.Draw(im).text((14, h // 2), ch, fill="black", font=BIG, anchor="mm")
    return im


def gap(w, h):
    return Image.new("RGB", (w, h), "white")


def panel_task(width):
    tile = load("C_TRD16_E_mat", TASK)
    rgb = tuple(int(round(x)) for x in T[TASK]["rgb"])
    solid = np.full((16, 16, 3), rgb, np.uint8)

    name = Image.new("RGB", (118, 86), (255, 248, 230))
    nd = ImageDraw.Draw(name)
    nd.rounded_rectangle([0, 0, 117, 85], radius=5, outline=(40, 40, 40), width=1)
    nd.text((10, 10), "log spruce", fill="black", font=BOLD)
    nd.rectangle([10, 36, 50, 70], fill=rgb, outline=(40, 40, 40))
    nd.text((58, 46), "colour", fill=(80, 80, 80), font=TINY)

    trd = Image.new("RGB", (280, 118), (245, 250, 255))
    td = ImageDraw.Draw(trd)
    td.rounded_rectangle([0, 0, 279, 117], radius=5, outline=(40, 40, 40), width=1)
    td.text((10, 6), "TRD", fill="black", font=BOLD)
    td.text((48, 8), "palette + ranks + torus", fill=(60, 60, 60), font=TINY)
    pal = palette_ramp(tile)
    rk = ranks(tile, 48)
    wr = wrap_icon(48)
    for x0, piece, cap in ((12, pal, "palette"), (12 + pal.width + 16, rk, "ranks"),
                           (12 + pal.width + 16 + 48 + 16, wr, "wrap")):
        trd.paste(piece, (x0, 32))
        td.text((x0 + piece.width // 2, 86), cap, fill=(80, 80, 80), font=TINY, anchor="mt")

    mid_h = 118
    row = [
        labelled(wall(solid), "solid region"),
        plus_or_arrow("+", mid_h),
        labelled(name, ""),
        plus_or_arrow("\u2192", mid_h),
        labelled(trd, ""),
        plus_or_arrow("\u2192", mid_h),
        labelled(nn(tile, 72), "16px"),
        gap(12, mid_h),
        labelled(tile2x2(tile, 88), "2\u00d72 wrap"),
        gap(12, mid_h),
        labelled(wall(tile), "painted region"),
    ]
    mid_h = max(p.height for p in row)

    H = 36 + mid_h + 8
    im = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(im)
    d.text((PAD, 6), "(a) Task: material name + region colour  \u2192  native 16px tile",
           fill="black", font=BIG)
    used = sum(p.width for p in row)
    x = max(PAD, (width - used) // 2)
    y0 = 32
    for p in row:
        im.paste(p, (x, y0 + (mid_h - p.height) // 2))
        x += p.width
    return im


def panel_grid(width):
    rows = [("Artist", None), ("TRD+rr", "TRD16c_rr4"), ("Render", "B2")]
    head = 52
    H = head + len(rows) * (S + PAD) + PAD
    im = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(im)
    d.text((PAD, 4), "(b) Held-out 16px tiles, 2\u00d72 so wrap seams show", fill="black", font=BIG)

    x0 = LAB
    # keep the six columns left-aligned with the row labels, not centred in leftover space
    for c, (_, name) in enumerate(COLS):
        cx = x0 + c * (S + PAD) + S // 2
        lines = name.split()
        top = head - 6 - 16 * (len(lines) - 1)
        for i, line in enumerate(lines):
            d.text((cx, top + 16 * i), line, fill="black", font=FONT, anchor="ms")
    for r, (label, method) in enumerate(rows):
        y = head + r * (S + PAD)
        d.text((LAB - 10, y + S // 2), label, fill="black",
               font=BOLD if "TRD" in label else FONT, anchor="rm")
        for c, (slug, _) in enumerate(COLS):
            x = x0 + c * (S + PAD)
            im.paste(tile2x2(load(method, slug), S), (x, y))
            if method == "TRD16c_rr4":
                d.rectangle([x - 2, y - 2, x + S + 1, y + S + 1], outline=(40, 90, 200), width=2)
    return im


def main():
    width = LAB + len(COLS) * (S + PAD) + PAD
    a, b = panel_task(width), panel_grid(width)
    out = Image.new("RGB", (width, a.height + b.height), "white")
    out.paste(a, (0, 0))
    out.paste(b, (0, a.height))
    out.save(OUT, dpi=(200, 200))
    print(OUT, out.size)


if __name__ == "__main__":
    main()
