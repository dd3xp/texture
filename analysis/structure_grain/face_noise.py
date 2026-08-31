"""面内局部变化：真人 vs 生成图。同一个量，两个来源。

A3o 用**全局**统计（每图色数、相邻同色比例）选了温度 1.3。
那个量区分不了"面内均匀、面间不同"和"到处都花"——两者全局统计可以相同。
而砖块看起来不像砖，可疑机制正是后者。

上一次问这个问题时我 formulate 错了（比"面内方差/面间方差"，
所有砖面同材质、均值天然接近，比值必然 >1，见 A3s）。
这次比**同一个量在两个来源上的取值**：

    面内相邻同色比例 / 面内标准差    真人 vs 生成

排除缝所在的行，只看面。离线用 `struct_metric_tiles_*.json` 的存档。
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def hx(h):
    return np.frombuffer(bytes.fromhex(h), np.uint8).reshape(16, 16).astype(float)


def face_stats(t: np.ndarray, bands) -> tuple[float, float] | None:
    """面内（去掉缝行）的相邻同色比例与标准差，按调色板档位归一。"""
    seg = []
    for y0, y1 in bands:
        if y1 - y0 >= 2:
            seg.append(t[y0:y1])
    if not seg:
        return None
    same, sds = [], []
    for a in seg:
        if a.shape[1] < 2:
            continue
        same.append(float((a[:, :-1] == a[:, 1:]).mean()))
        if a.shape[0] >= 2:
            same.append(float((a[:-1, :] == a[1:, :]).mean()))
        sds.append(float(a.std()))
    if not same:
        return None
    return float(np.mean(same)), float(np.mean(sds))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "experiments/struct_metric_tiles_preA3v.json"
    d = json.loads(Path(path).read_text())
    use = [r for r in d if r["structured"] and len(r["bands"]) >= 2]
    print(f"{path}：有结构且分层≥2 的材质 {len(use)} 个")

    A_s, A_v, G_s, G_v, N_s, N_v = [], [], [], [], [], []
    for r in use:
        bands = [tuple(b) for b in r["bands"]]
        a = face_stats(hx(r["artist"]), bands)
        if a is None:
            continue
        gs = [face_stats(hx(h), bands) for h in r["seeded"]]
        ns = [face_stats(hx(h), bands) for h in r["none"]]
        gs = [x for x in gs if x]
        ns = [x for x in ns if x]
        if not gs:
            continue
        A_s.append(a[0]); A_v.append(a[1])
        G_s.append(float(np.mean([x[0] for x in gs])))
        G_v.append(float(np.mean([x[1] for x in gs])))
        if ns:
            N_s.append(float(np.mean([x[0] for x in ns])))
            N_v.append(float(np.mean([x[1] for x in ns])))

    print(f"\n{'':<14}{'面内相邻同色':>14}{'面内标准差':>12}")
    print("-" * 42)
    print(f"{'真人':<14}{np.median(A_s):>14.3f}{np.median(A_v):>12.3f}")
    print(f"{'先验+填充':<14}{np.median(G_s):>14.3f}{np.median(G_v):>12.3f}")
    if N_s:
        print(f"{'无种子':<14}{np.median(N_s):>14.3f}{np.median(N_v):>12.3f}")
    p1 = stats.wilcoxon(A_s, G_s).pvalue
    p2 = stats.wilcoxon(A_v, G_v).pvalue
    print(f"\n真人 vs 先验+填充：相邻同色 p={p1:.3g}   标准差 p={p2:.3g}")
    print("\n判读：生成图面内相邻同色**明显更低**、标准差更高")
    print("      -> 面被填花了，1 像素的缝显不出来，A3o 的温度需要重选")
    print("      两者接近 -> 面的噪声不是问题，砖不像砖另有原因")


if __name__ == "__main__":
    main()
