"""S1：结构特征复测。

compare.py 的三个特征（颜色数/平坦度/边缘锐度）只看局部像素统计，
测不到母题结构。这里加三组结构特征，重问同一个问题：

    真人在 16x16 下画的母题重复次数，是否显著少于降采样保留的？

直觉：真人画木板，16px 下只画 2-3 条，因为再多就糊了；
而把 128px 的 8 条木板降下来，8 条还在，只是每条变细。
如果这个直觉对，主周期和连通块数量应该能分开两组。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from compare import load_rgb, quantize, features, auc


def dominant_period(g: np.ndarray, axis: int) -> float:
    """沿指定轴的主周期。用 1D 自相关，取第一个显著峰的滞后。

    先沿另一轴取均值压成 1D 廓线——木板/砖行这类母题正是在这个廓线上重复的。
    """
    prof = g.mean(axis=1 - axis)
    prof = prof - prof.mean()
    if np.allclose(prof, 0):
        return 0.0
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-12)
    # 跳过 lag 0，找第一个局部极大且相关性为正的滞后
    for lag in range(1, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.1:
            return float(lag)
    return 0.0


def structure_features(a: np.ndarray) -> dict:
    n = a.shape[0]
    q = (a // 8).astype(int)
    g = a.mean(-1)

    ph = dominant_period(g, axis=0)   # 竖直方向重复（横向条纹，如木板）
    pv = dominant_period(g, axis=1)   # 水平方向重复

    # 连通同色块：把量化后的颜色编码成整数标签，逐色做连通域
    code = q[:, :, 0] * 65536 + q[:, :, 1] * 256 + q[:, :, 2]
    total = 0
    sizes = []
    for c in np.unique(code):
        lab, k = ndimage.label(code == c)
        total += k
        if k:
            sizes.extend(ndimage.sum(np.ones_like(lab), lab, range(1, k + 1)))
    sizes = np.array(sizes) if sizes else np.array([0.0])

    # 游程长度：行和列方向上同色连续段的平均长度
    runs = []
    for arr in (code, code.T):
        for row in arr:
            change = np.flatnonzero(np.diff(row) != 0)
            bounds = np.concatenate([[-1], change, [len(row) - 1]])
            runs.extend(np.diff(bounds))
    runs = np.array(runs, float)

    return {
        # 主周期：越大表示母题重复次数越少（周期长 = 重复少）
        "period_h": ph,
        "period_v": pv,
        "period_max": max(ph, pv),
        # 连通块密度：越小表示色块越大越整
        "comp_density": total / (n * n),
        "comp_size_mean": float(sizes.mean()),
        "run_mean": float(runs.mean()),
    }


FEATS = ["period_max", "comp_density", "comp_size_mean", "run_mean"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("experiments/premise"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text())

    # 与 compare.py 一致：C 组量化到该材质 A 组的中位色数
    a_colors = {}
    for name, v in pairs.items():
        cs = [features(a)["n_colors"] for a in
              (load_rgb(p, args.size) for p in v["low"].values()) if a is not None]
        if cs:
            a_colors[name] = int(np.median(cs))

    rows = []
    for name, v in pairs.items():
        for pack, path in v["low"].items():
            a = load_rgb(path, args.size)
            if a is not None:
                rows.append({"material": name, "group": "A", **structure_features(a)})
        k = a_colors.get(name)
        for pack, path in v["high"].items():
            a = load_rgb(path, args.size)
            if a is None or not k:
                continue
            rows.append({"material": name, "group": "C", **structure_features(quantize(a, k))})

    A = [r for r in rows if r["group"] == "A"]
    C = [r for r in rows if r["group"] == "C"]
    print(f"样本: A(真人 16px)={len(A)}  C(降采样+量化)={len(C)}  材质={len(pairs)}\n")

    hdr = f"{'结构特征':<18}{'A 中位数':>12}{'C 中位数':>12}{'AUC(A|C)':>12}"
    print(hdr)
    print("-" * 56)
    res = {}
    for f in FEATS:
        pa = np.array([r[f] for r in A], float)
        pc = np.array([r[f] for r in C], float)
        u = auc(pa, pc)
        res[f] = u
        print(f"{f:<18}{np.median(pa):>12.3f}{np.median(pc):>12.3f}{u:>12.3f}")

    best = max(res.items(), key=lambda kv: abs(kv[1] - 0.5))
    print(f"\n分离度最强: {best[0]}  AUC={best[1]:.3f}  |AUC-0.5|={abs(best[1]-0.5):.3f}")
    print("判据: 任一结构特征 AUC>0.75 或 <0.25 则立论部分成立")

    (args.out / "structure.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
