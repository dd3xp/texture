"""Supplementary 16px qualitative figures from the final test-set outputs (experiments/baselines).

Every cell is a 2x2 repetition of the tile, so seams are visible.
  fig_supp_qual_name.png    name-only task: (a) materials the judge gave to TRD+rr over both B2 and B5,
                            (b) 6 materials drawn with random.Random(0)
  fig_supp_colour.png       colour task, materials the judge gave to TRD over B2, B5 and B7
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "model"), str(ROOT / "tools")]
from colour_task import targets  # noqa: E402

BASE = ROOT / "experiments/baselines"
FIG = ROOT / "figures"
S, PAD, HEAD, SIDE = 112, 6, 34, 210

NAME_COLS = [("Artist (ref.)", None), ("TRD+rr (ours)", "TRD16c_rr4"), ("B2 render+crop", "B2"),
             ("B7 pixel diff.", "B7"), ("B5 retrieval", "B5"), ("B1 SDXL+down", "B1"), ("B4 LoRA", "B4")]
COLOUR_COLS = [("Target colour", "swatch"), ("Artist (ref.)", None), ("TRD (ours)", "C_TRD16_E_mat"),
               ("B2 + Lab shift", "C_B2_E_mat"), ("B7 + Lab shift", "C_B7_E_mat"),
               ("B5 + Lab shift", "C_B5_E_mat")]
BEST = ["default_cobble", "mcl_core_log_birch", "mcl_core_gold_ore", "mcl_core_emerald_ore",
        "default_cactus_side", "mcl_deepslate_diamond_ore"]
COLOUR_BEST = ["default_dirt", "mcl_core_log_spruce", "default_stone_block", "farming_straw",
               "default_snow_side", "nc_tree_tree_side"]

try:
    font = ImageFont.truetype("arial.ttf", 14)
    bold = ImageFont.truetype("arialbd.ttf", 14)
    big = ImageFont.truetype("arialbd.ttf", 17)
except OSError:
    font = bold = big = ImageFont.load_default()

T = {t["slug"]: t for t in targets("E_mat", 16) if t["j"] == 0}


def tile(method, slug):
    if method is None:
        return T[slug]["ref"]
    if method == "swatch":
        return np.tile(np.round(T[slug]["rgb"]).astype(np.uint8), (16, 16, 1))
    d = BASE / method / "16"
    for p in (d / f"{slug}_0.png", d / f"{slug}.png"):
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    raise FileNotFoundError(f"{method} {slug}")


def cell(arr, method):
    im = Image.fromarray(np.asarray(arr, np.uint8))
    if method != "swatch":
        im = Image.fromarray(np.tile(np.asarray(im), (2, 2, 1)))
    return im.resize((S, S), Image.NEAREST)


def sheet(cols, slugs, title=None):
    top = 30 if title else 0
    W = SIDE + len(cols) * (S + PAD) + PAD
    H = top + HEAD + len(slugs) * (S + PAD) + PAD
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    if title:
        d.text((PAD, 8), title, fill="black", font=big)
    HEAD_ = HEAD + top
    for c, (label, m) in enumerate(cols):
        x = SIDE + PAD + c * (S + PAD)
        d.text((x + S // 2, top + HEAD // 2), label, fill="black", font=bold if "ours" in label else font,
               anchor="mm")
    for r, slug in enumerate(slugs):
        y = HEAD_ + PAD + r * (S + PAD)
        d.text((SIDE - 8, y + S // 2), T[slug]["prompt"], fill="black", font=font, anchor="rm")
        for c, (_, m) in enumerate(cols):
            x = SIDE + PAD + c * (S + PAD)
            im.paste(cell(tile(m, slug), m), (x, y))
            if cols[c][0].endswith("(ours)"):
                d.rectangle([x - 2, y - 2, x + S + 1, y + S + 1], outline=(40, 90, 200), width=2)
    return im


def tsheet(cols, groups, gap=40):
    """Methods as rows, materials as columns; groups = [(title, slugs), ...] placed side by side."""
    f = ImageFont.truetype("arial.ttf", 26)
    fb = ImageFont.truetype("arialbd.ttf", 26)
    ft = ImageFont.truetype("arialbd.ttf", 30)
    lab_w, title_h, head_h = 250, 48, 98
    n = sum(len(s) for _, s in groups)
    W = lab_w + n * (S + PAD) + gap * (len(groups) - 1)
    H = title_h + head_h + len(cols) * (S + PAD)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for r, (label, _) in enumerate(cols):
        y = title_h + head_h + r * (S + PAD)
        d.text((lab_w - 12, y + S // 2), label.replace(" (ours)", ""), fill="black",
               font=fb if "ours" in label else f, anchor="rm")
    x0 = lab_w
    for title, slugs in groups:
        d.text((x0, 6), title, fill="black", font=ft)
        for c, slug in enumerate(slugs):
            x = x0 + c * (S + PAD)
            words, lines = T[slug]["prompt"].split(), [""]
            for w in words:
                if d.textlength((lines[-1] + " " + w).strip(), font=f) > S + PAD - 2 and lines[-1]:
                    lines.append(w)
                else:
                    lines[-1] = (lines[-1] + " " + w).strip()
            for k, line in enumerate(lines):
                d.text((x + S // 2, title_h + head_h - 6 - (len(lines) - k - 1) * 28), line,
                       fill="black", font=f, anchor="ms")
            for r, (label, m) in enumerate(cols):
                y = title_h + head_h + r * (S + PAD)
                im.paste(cell(tile(m, slug), m), (x, y))
                if label.endswith("(ours)"):
                    d.rectangle([x - 3, y - 3, x + S + 2, y + S + 2], outline=(40, 90, 200), width=3)
        x0 += len(slugs) * (S + PAD) + gap
    return im


def save(im, out):
    im.save(FIG / out)
    print(out, im.size)


def vstack(*ims):
    out = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height
    return out


def judged(fname, a, b):
    """Map the judge's pair index back to a slug the way eval/judge_pairs.py builds pairs."""
    from prompts import load_set
    have = lambda m, s: (BASE / m / "16" / f"{s}_0.png").exists() or (BASE / m / "16" / f"{s}.png").exists()
    pairs = [(e["material"].rsplit(".", 1)[0], e["prompt"]) for e in load_set("E_mat")[0]]
    pairs = [(s, p) for s, p in pairs if have(a, s) and have(b, s)]
    out = {}
    for r in json.load(open(ROOT / "experiments" / fname, encoding="utf-8"))["records"]:
        slug, prompt = pairs[r["pair"]]
        assert prompt == r["material"], (prompt, r["material"])
        out[slug] = r["verdict"]
    return out


if __name__ == "__main__":
    a = judged("judge_full_TRD16c_rr4_vs_B2_16.json", "TRD16c_rr4", "B2")
    b = judged("judge_full_TRD16c_rr4_vs_B5_16.json", "TRD16c_rr4", "B5")
    print("name-only pool:", sum(a[s] == "A" and b.get(s) == "A" for s in a))
    assert all(a[s] == "A" and b[s] == "A" for s in BEST)
    ca = [judged(f"judge_full_C_TRD16_E_mat_vs_C_{m}_E_mat_16.json", "C_TRD16_E_mat", f"C_{m}_E_mat")
          for m in ("B2", "B5", "B7")]
    print("colour pool:", sum(all(v.get(s) == "A" for v in ca) for s in ca[0]))
    assert all(v[s] == "A" for v in ca for s in COLOUR_BEST)
    common = sorted(s for s in T if all((BASE / m / "16" / f"{s}_0.png").exists() or
                                        (BASE / m / "16" / f"{s}.png").exists()
                                        for _, m in NAME_COLS if m))
    rnd = random.Random(0).sample(common, 6)
    print("random:", rnd)
    save(tsheet(NAME_COLS, [("(a) Selected", BEST), ("(b) Unselected, random.Random(0)", rnd)]),
         "fig_supp_qual_name.png")
    save(tsheet(COLOUR_COLS, [("Region-colouring task, selected", COLOUR_BEST)]), "fig_supp_colour.png")
