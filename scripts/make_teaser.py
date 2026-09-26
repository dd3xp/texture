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
S, PAD = 112, 6
COLS = [
    ("default_cobble", "cobble"),
    ("mcl_core_log_birch", "log birch"),
    ("mcl_core_gold_ore", "gold ore"),
    ("mcl_core_emerald_ore", "emerald ore"),
    ("default_cactus_side", "cactus side"),
    ("mcl_deepslate_diamond_ore", "deepslate diamond"),
]
TASK = "mcl_core_log_spruce"
T = {t["slug"]: t for t in targets("E_mat", 16) if t["j"] == 0}

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
    BOLD = ImageFont.truetype("arialbd.ttf", 15)
    BIG = ImageFont.truetype("arialbd.ttf", 17)
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


def wall(arr, cols=3, rows=3, cell=28):
    t = np.asarray(nn(arr, cell))
    big = np.tile(t, (rows, cols, 1))
    im = Image.fromarray(big)
    d = ImageDraw.Draw(im)
    for i in range(cols + 1):
        x = min(i * cell, im.width - 1)
        d.line([(x, 0), (x, im.height - 1)], fill=(30, 30, 30), width=1)
    for j in range(rows + 1):
        y = min(j * cell, im.height - 1)
        d.line([(0, y), (im.width - 1, y)], fill=(30, 30, 30), width=1)
    return np.asarray(im)


def palette_ramp(arr, n=8, w=10, h=18):
    pix = np.asarray(arr, np.uint8).reshape(-1, 3)
    uniq = np.unique(pix, axis=0)
    order = np.argsort(uniq.astype(np.float32) @ [0.2126, 0.7152, 0.0722])
    cols = uniq[order][:n]
    im = Image.new("RGB", (n * w, h), "white")
    d = ImageDraw.Draw(im)
    for i, c in enumerate(cols):
        d.rectangle([i * w, 0, (i + 1) * w - 1, h - 1], fill=tuple(int(x) for x in c))
    return im


def ranks(arr, size):
    g = np.asarray(arr, np.float32) @ [0.2126, 0.7152, 0.0722]
    lo, hi = g.min(), g.max()
    n = np.zeros_like(g, np.uint8) if hi <= lo else np.round(255 * (g - lo) / (hi - lo)).astype(np.uint8)
    return nn(np.stack([n, n, n], -1), size)


def box(draw, xy, fill, outline=(40, 40, 40)):
    draw.rounded_rectangle(xy, radius=5, fill=fill, outline=outline, width=1)


def panel_task():
    tile = load("C_TRD16_E_mat", TASK)
    rgb = tuple(int(round(x)) for x in T[TASK]["rgb"])
    solid = np.full((16, 16, 3), rgb, np.uint8)
    W, H = 1280, 248
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((8, 4), "(a) Task: material name + region colour  \u2192  native tileable 16px texture",
           fill="black", font=BIG)

    x, y = 16, 40
    w0 = Image.fromarray(wall(solid))
    im.paste(w0, (x, y + 16))
    d.text((x + w0.width // 2, y), "solid region", fill=(80, 80, 80), font=TINY, anchor="ms")
    x += w0.width + 8
    d.text((x + 6, y + 70), "+", fill="black", font=BIG)
    x += 24
    box(d, [x, y + 36, x + 124, y + 118], (255, 248, 230))
    d.text((x + 10, y + 44), "log spruce", fill="black", font=BOLD)
    d.rectangle([x + 10, y + 68, x + 52, y + 100], fill=rgb, outline=(40, 40, 40))
    d.text((x + 60, y + 76), "colour", fill=(80, 80, 80), font=TINY)
    x += 136
    d.text((x, y + 70), "\u2192", fill="black", font=BIG)
    x += 28
    box(d, [x, y + 8, x + 300, y + 196], (245, 250, 255))
    d.text((x + 12, y + 14), "TRD", fill="black", font=BOLD)
    d.text((x + 52, y + 16), "palette + ranks + torus", fill=(60, 60, 60), font=TINY)
    im.paste(palette_ramp(tile, n=8, w=14, h=22), (x + 14, y + 42))
    d.text((x + 14, y + 68), "retrieved palette", fill=(90, 90, 90), font=TINY)
    im.paste(ranks(tile, 64), (x + 14, y + 88))
    d.text((x + 14, y + 156), "luminance ranks", fill=(90, 90, 90), font=TINY)
    gx, gy, cs = x + 168, y + 88, 14
    for i in range(4):
        for j in range(4):
            fill = (70, 130, 210) if i in (0, 3) and j == 1 else (215, 215, 215)
            d.rectangle([gx + i * cs, gy + j * cs, gx + i * cs + cs - 2, gy + j * cs + cs - 2], fill=fill)
    d.arc([gx - 14, gy - 2, gx + 4 * cs + 12, gy + 36], 200, 340, fill=(40, 90, 180), width=2)
    d.text((x + 150, y + 156), "wrap neighbours", fill=(90, 90, 90), font=TINY)
    x += 314
    d.text((x, y + 70), "\u2192", fill="black", font=BIG)
    x += 28
    im.paste(nn(tile, 72), (x, y + 40))
    d.text((x + 36, y + 20), "16px", fill=(80, 80, 80), font=TINY, anchor="ms")
    x += 84
    im.paste(tile2x2(tile, 96), (x, y + 28))
    d.text((x + 48, y + 8), "2\u00d72 wrap", fill=(80, 80, 80), font=TINY, anchor="ms")
    x += 110
    w1 = Image.fromarray(wall(tile))
    im.paste(w1, (x, y + 16))
    d.text((x + w1.width // 2, y), "painted region", fill=(80, 80, 80), font=TINY, anchor="ms")
    return im


def panel_grid():
    rows = [("Artist", None), ("TRD+rr", "TRD16c_rr4"), ("Render", "B2")]
    lab_w, head = 78, 44
    W = lab_w + len(COLS) * (S + PAD) + PAD
    H = head + len(rows) * (S + PAD) + PAD + 8
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((8, 4), "(b) Held-out 16px tiles, 2\u00d72 so wrap seams show", fill="black", font=BIG)
    for c, (_, name) in enumerate(COLS):
        x = lab_w + PAD + c * (S + PAD)
        d.text((x + S // 2, head - 8), name, fill="black", font=FONT, anchor="ms")
    for r, (label, method) in enumerate(rows):
        y = head + PAD + r * (S + PAD)
        d.text((lab_w - 8, y + S // 2), label, fill="black", font=BOLD if "TRD" in label else FONT, anchor="rm")
        for c, (slug, _) in enumerate(COLS):
            x = lab_w + PAD + c * (S + PAD)
            im.paste(tile2x2(load(method, slug), S), (x, y))
            if method == "TRD16c_rr4":
                d.rectangle([x - 2, y - 2, x + S + 1, y + S + 1], outline=(40, 90, 200), width=2)
    return im


def main():
    a, b = panel_task(), panel_grid()
    out = Image.new("RGB", (max(a.width, b.width), a.height + b.height + 8), "white")
    out.paste(a, (0, 0))
    out.paste(b, ((out.width - b.width) // 2, a.height + 8))
    out.save(OUT, dpi=(200, 200))
    print(OUT, out.size)


if __name__ == "__main__":
    main()
