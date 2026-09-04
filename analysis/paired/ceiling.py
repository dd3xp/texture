"""逐像素监督的天花板：不同作者画同一材质，彼此有多一致？

`feasibility.py` 量出基线与真人只有 17% 的格子一致（12 档下随机约 8%），
而基线在 A4 盲比里是赢的——说明它赢在观感，与逐像素对错无关。

所以先问天花板：**同一材质、不同作者之间的逐格一致率**。
若作者之间也很低，那"预测真人的每个像素"这个目标本身不成立，
逐像素监督（无论 pix2pix 还是本项目原来的掩码预测）都到不了，
任务必须按分布来定义，不能按逐格对错。
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16:
            bymat.setdefault(s["material"], []).append(s)

    same, cross, n_mat = [], [], 0
    for m, ss in bymat.items():
        if len(ss) < 2:
            continue
        n_mat += 1
        arrs = []
        for s in ss:
            pal = np.array(s["palette"], np.uint8)
            a = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
            # 归一到 [0,1] 的亮度档位，消除各作者调色板色数不同的影响
            arrs.append(a.astype(float) / max(len(pal) - 1, 1))
        for x, y in combinations(arrs, 2):
            # 量化到 12 档再比，与基线评估同口径
            xi = np.clip((x * 11).round(), 0, 11)
            yi = np.clip((y * 11).round(), 0, 11)
            same.append(float((xi == yi).mean()))

    # 随机基准：同一分布下打乱空间位置
    rng = np.random.default_rng(0)
    for m, ss in list(bymat.items())[:200]:
        if len(ss) < 2:
            continue
        s = ss[0]
        pal = np.array(s["palette"], np.uint8)
        a = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        x = np.clip((a.astype(float) / max(len(pal)-1, 1) * 11).round(), 0, 11)
        y = rng.permutation(x.ravel()).reshape(16, 16)
        cross.append(float((x == y).mean()))

    print(f"材质 {n_mat}，作者两两配对 {len(same)} 组")
    print(f"\n  不同作者同一材质 逐格一致率  中位 {np.median(same):.3f}"
          f"  四分位 [{np.percentile(same,25):.3f}, {np.percentile(same,75):.3f}]")
    print(f"  同图打乱空间位置（随机基准）  中位 {np.median(cross):.3f}")
    print(f"  基线 vs 真人（feasibility.py）  0.170")
    print("\n判读：")
    print("  作者间一致率若也在 0.2 上下 -> 逐像素监督的天花板就在那里，")
    print("  基线的 0.17 已经接近天花板，**这条路不是模型不够好，是目标错了**。")


if __name__ == "__main__":
    main()
