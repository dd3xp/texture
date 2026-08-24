"""用 M2 学到的材质分类器当尺子，量 M1 测试集与真人原生像素画。

训得出分类器不等于它能当尺子。这里检验它是否给出**有意义的读数**：
真人原生像素画（黄金标准）应当得分最高，
流水线输出若确实更差，应当得分更低。
若三者读数相同，则这把尺子和 CLIP 一样不可用。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learn_material import SmallCNN, HELD_OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/metric/material_cnn.pt"))
    ap.add_argument("--testset", type=Path, default=Path("experiments/metric/testset"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cls = ck["classes"]
    gray = bool(ck.get("gray", False))
    print(("灰度模型（结构，无颜色）" if gray else "彩色模型") + f"  ckpt={args.ckpt.name}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = SmallCNN(len(cls)).to(dev).eval()
    net.load_state_dict(ck["state"])

    def score(img: Image.Image, material: str):
        if material not in cls:
            return None
        a = np.asarray(img.convert("RGB").resize((args.size,) * 2, Image.NEAREST),
                       np.float32) / 255.0
        if gray:
            # 必须和训练时完全一致，否则读数无意义
            g = a @ np.array([0.299, 0.587, 0.114], np.float32)
            g = (g - g.mean()) / (g.std() + 1e-6)
            a = np.repeat(g[..., None], 3, -1)
        x = torch.tensor(a).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            p = net(x).softmax(-1)[0].cpu().numpy()
        gt = cls[material]
        return int(p.argmax() == gt), float(p[gt])

    rows = []

    # 流水线输出（M1 测试集）
    for f in sorted((args.testset / "img").glob("*.png")):
        stem = f.stem
        tag = stem.rsplit("_", 1)[1]
        size = int(stem.rsplit("_", 2)[1])
        mat = stem.rsplit("_", 2)[0] + ".png"
        r = score(Image.open(f), mat)
        if r:
            rows.append({"group": f"流水线-{tag}", "size": size,
                         "correct": r[0], "p_true": r[1]})

    # 真人原生像素画。分成"训练见过的作者"和"留出作者"，后者才是公平读数
    pairs = json.loads(args.pairs.read_text())
    for m, v in pairs.items():
        if m not in cls:
            continue
        for pack, path in v["low"].items():
            try:
                im = Image.open(path)
            except Exception:
                continue
            if im.size != (args.size, args.size):
                continue
            r = score(im, m)
            if r:
                g = "真人原生-留出作者" if pack in HELD_OUT else "真人原生-训练作者"
                rows.append({"group": g, "size": args.size,
                             "correct": r[0], "p_true": r[1]})

    agg = defaultdict(list)
    for r in rows:
        agg[(r["group"], r["size"])].append(r)

    print(f"材质分类器当尺子（{len(cls)} 类，随机基线 {1/len(cls):.2%}）\n")
    print(f"{'组':<22}{'尺寸':<8}{'n':>6}{'top-1':>9}{'真类概率中位':>14}")
    print("-" * 62)
    for k in sorted(agg, key=lambda x: (x[0], x[1])):
        v = agg[k]
        print(f"{k[0]:<22}{str(k[1])+'x'+str(k[1]):<8}{len(v):>6}"
              f"{np.mean([x['correct'] for x in v]):>8.1%}"
              f"{np.median([x['p_true'] for x in v]):>14.4f}")


if __name__ == "__main__":
    main()
