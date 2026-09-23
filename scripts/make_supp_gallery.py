"""Supplementary gallery: B2 best-of-4 16px tiles, shown single and 2x2-tiled (seam check)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments/bestof"
OUT = ROOT / "figures/fig_supp_b2_gallery.png"
PICK = ["brick_wall", "stone_brick_wall", "wooden_planks_floor", "clay_roof_tiles",
        "honeycomb_wax", "hexagonal_floor_tiles", "woven_basket_surface", "slate_roof_shingles",
        "mossy_cracked_stone_bricks", "coal_ore_vein", "fish_scales", "parquet_wood_floor"]
COLS, S, PAD, LAB = 4, 128, 10, 22

try:
    font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()

cell_w, cell_h = 2 * S + PAD, S + LAB
rows = (len(PICK) + COLS - 1) // COLS
W = COLS * cell_w + (COLS + 1) * PAD
H = rows * cell_h + (rows + 1) * PAD
sheet = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(sheet)
for i, m in enumerate(PICK):
    t = Image.open(SRC / f"{m}_best.png").convert("RGB")
    single = t.resize((S, S), Image.NEAREST)
    tiled = Image.new("RGB", (2 * t.width, 2 * t.height))
    for dx in (0, 1):
        for dy in (0, 1):
            tiled.paste(t, (dx * t.width, dy * t.height))
    tiled = tiled.resize((S, S), Image.NEAREST)
    x = PAD + (i % COLS) * (cell_w + PAD)
    y = PAD + (i // COLS) * (cell_h + PAD)
    sheet.paste(single, (x, y))
    sheet.paste(tiled, (x + S + PAD, y))
    d.text((x, y + S + 4), m.replace("_", " "), fill="black", font=font)
OUT.parent.mkdir(exist_ok=True)
sheet.save(OUT)
print(OUT, sheet.size)
