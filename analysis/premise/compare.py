"""前提验证：真人画的低分辨率纹理，和把高分辨率降采样到同尺寸，是不是可区分的两个分布。

A 组 = 多位作者原生 16x16 的手绘版本
B 组 = 多位作者的高分辨率版本，机器降采样到 16x16

两组都跨多个作者，所以"作者风格"不是组间混淆变量，而是组内方差。

特征选择的理由：像素画的四条硬约束里，有三条会在降采样中被破坏——
调色板变大、硬边被抹平、平坦色块被梯度取代。这里就量这三件事。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str, size: int) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    if im.size != (size, size):
        # 面积平均降采样：这是"把高分辨率图缩小"最标准的做法
        im = im.resize((size, size), Image.BOX)
    return np.asarray(im, dtype=np.float64)


def quantize(a: np.ndarray, k: int) -> np.ndarray:
    """把图量化到 k 色。用来做那个必须堵上的反驳：
    "降采样之后再量化到同样色数，不就等于真人版了吗？"
    """
    im = Image.fromarray(a.astype(np.uint8))
    q = im.convert("P", palette=Image.ADAPTIVE, colors=max(2, k)).convert("RGB")
    return np.asarray(q, dtype=np.float64)


def features(a: np.ndarray) -> dict:
    """三个可解释特征，全部与像素画硬约束直接对应。"""
    q = (a // 8).astype(int)  # 轻微量化，避免把肉眼同色算成不同色
    flat = q.reshape(-1, 3)
    n_colors = len(np.unique(flat, axis=0))

    # 相邻像素差：像素画有大量完全相同的相邻像素（平坦块）和少量硬跳变
    dh = np.abs(np.diff(a, axis=1)).sum(-1)
    dv = np.abs(np.diff(a, axis=0)).sum(-1)
    d = np.concatenate([dh.ravel(), dv.ravel()])

    return {
        "n_colors": n_colors,
        # 完全相同的相邻像素占比：平坦度
        "flat_frac": float((d < 1e-9).mean()),
        # 边缘锐度：非平坦跳变的中位幅度。降采样会把硬边抹成中间值
        "edge_sharp": float(np.median(d[d > 1e-9])) if (d > 1e-9).any() else 0.0,
    }


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U 导出的 AUC。0.5=不可分, 1.0=完全可分。"""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 处理并列
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    r = ranks[: len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("experiments/premise"))
    ap.add_argument("--figure-materials", nargs="+",
                    default=["default_wood.png", "default_stone.png",
                             "default_cobble.png", "default_brick.png",
                             "default_sand.png", "default_grass_side.png"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text())

    # 先算出每个材质下 A 组的中位色数，作为 C 组量化的目标色数——
    # 这样 C 组拿到的是"真人在这个材质上实际用了多少色"这个信息，
    # 是给反驳方的最有利设定。
    a_colors = {}
    for name, v in pairs.items():
        cs = []
        for path in v["low"].values():
            a = load_rgb(path, args.size)
            if a is not None:
                cs.append(features(a)["n_colors"])
        if cs:
            a_colors[name] = int(np.median(cs))

    rows = []
    for name, v in pairs.items():
        for pack, path in v["low"].items():
            a = load_rgb(path, args.size)
            if a is not None:
                rows.append({"material": name, "group": "A_artist_low", "pack": pack, **features(a)})
        for pack, path in v["high"].items():
            a = load_rgb(path, args.size)
            if a is None:
                continue
            rows.append({"material": name, "group": "B_downsampled", "pack": pack, **features(a)})
            k = a_colors.get(name)
            if k:
                rows.append({"material": name, "group": "C_down_quantized", "pack": pack,
                             **features(quantize(a, k))})

    print(f"样本数: {len(rows)}  (材质 {len(pairs)})")
    A = [r for r in rows if r["group"] == "A_artist_low"]
    B = [r for r in rows if r["group"] == "B_downsampled"]
    C = [r for r in rows if r["group"] == "C_down_quantized"]
    print(f"  A 真人低分辨率: {len(A)}   B 降采样: {len(B)}   C 降采样+量化: {len(C)}")


    def col(rs, f):
        return np.array([r[f] for r in rs], float)

    hdr = f"{'特征':<12}{'A 真人':>10}{'B 降采样':>11}{'C 降+量化':>11}{'AUC(A|B)':>11}{'AUC(A|C)':>11}"
    print(hdr)
    print("-" * 66)
    for f in ["n_colors", "flat_frac", "edge_sharp"]:
        pa, pb, pc = col(A, f), col(B, f), col(C, f)
        print(f"{f:<12}{np.median(pa):>10.2f}{np.median(pb):>11.2f}{np.median(pc):>11.2f}"
              f"{auc(pa, pb):>11.3f}{auc(pa, pc):>11.3f}")

    (args.out / "features.json").write_text(json.dumps(rows, indent=1))
    print(f"\n特征写入 {args.out/'features.json'}")


if __name__ == "__main__":
    main()
