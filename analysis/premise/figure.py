"""把 A/B/C 三组并排画出来，用于判断低层统计量是否漏掉了母题结构差异。

每行一个材质：左边若干张真人 16x16，右边若干张高分辨率降采样+量化的结果。
如果肉眼能一眼分开而统计量分不开，说明差异在结构层面，需要换特征。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compare import load_rgb, quantize, features

CELL = 96
PAD = 4
LABEL_W = 150
HDR = 34


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--out", type=Path, default=Path("experiments/premise/figure.png"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n-each", type=int, default=6)
    ap.add_argument("--materials", nargs="+", default=[
        "default_wood.png", "default_stone.png", "default_cobble.png",
        "default_brick.png", "default_sand.png", "default_gravel.png",
        "default_tree.png", "default_stone_brick.png"])
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text())
    mats = [m for m in args.materials if m in pairs]

    ncol = args.n_each * 2
    W = LABEL_W + ncol * (CELL + PAD) + PAD + 20
    H = HDR + len(mats) * (CELL + PAD) + PAD
    cv = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(cv)

    split_x = LABEL_W + args.n_each * (CELL + PAD) + PAD // 2
    dr.text((LABEL_W + 4, 6), f"A  真人原生 {args.size}x{args.size}（不同作者）", fill="black")
    dr.text((split_x + 14, 6), "C  高分辨率 → 降采样 → 量化到同色数", fill="black")
    dr.line([(split_x + 6, HDR - 4), (split_x + 6, H)], fill="#888", width=2)

    for ri, m in enumerate(mats):
        y = HDR + ri * (CELL + PAD)
        dr.text((6, y + CELL // 2 - 6), m.replace("default_", "").replace(".png", "")[:18], fill="black")

        lows = list(pairs[m]["low"].values())[: args.n_each]
        # 量化目标色数取该材质真人版的中位色数
        ks = [features(a)["n_colors"] for a in
              (load_rgb(p, args.size) for p in pairs[m]["low"].values()) if a is not None]
        k = int(np.median(ks)) if ks else 8

        for ci, p in enumerate(lows):
            a = load_rgb(p, args.size)
            if a is None:
                continue
            x = LABEL_W + ci * (CELL + PAD)
            cv.paste(Image.fromarray(a.astype(np.uint8)).resize((CELL, CELL), Image.NEAREST), (x, y))
            dr.rectangle([x, y, x + CELL, y + CELL], outline="#ccc")

        highs = list(pairs[m]["high"].values())[: args.n_each]
        for ci, p in enumerate(highs):
            a = load_rgb(p, args.size)
            if a is None:
                continue
            a = quantize(a, k)
            x = split_x + 14 + ci * (CELL + PAD)
            cv.paste(Image.fromarray(a.astype(np.uint8)).resize((CELL, CELL), Image.NEAREST), (x, y))
            dr.rectangle([x, y, x + CELL, y + CELL], outline="#ccc")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv.save(args.out)
    print(f"wrote {args.out}  {cv.size}  materials={len(mats)}")


if __name__ == "__main__":
    main()
