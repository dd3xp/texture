"""管线交付的瓦片，离真人的单元惯例还差多远？——**前提检验**（无 GPU）。

## 为什么问这个

本项目全部主张的机制是**单元惯例失配**：SDXL 直出的瓦片每张塞 27.7 个结构单元，
真人只放 3.2 个（B9）。方法（按周期裁剪）就是为了把这个数拉回真人那一档。

但 `tools/downsample.py` 自己的注释里写着一处**从未测过的不一致**：

    UNITS_PER_TILE = 4.5

这个 4.5 是用 `dominant_period(hi_frac=0.5)` 量真人瓦片量出来的，而生产管线里
`auto_crop` 调 `dominant_period` 走的是默认 **0.625**。同一批真人瓦片在生产口径下
是 **3.20**（`analysis/paired/resolution_tiers.py`），B9 的 125 个高分源独立给 3.33。
**定常数的口径与用常数的口径不一致，常数没有跟着重调**，已发表结果全部出自 4.5 的
流水线。原注释：「改成 3.2 会不会更好未测。改之前必须重跑盲比，别直接改这个数。」

也就是说：**方法自己没有命中它所指认的那个惯例**，差 4.5/3.2 = 1.4 倍。

## 这个脚本只回答一件事（不回答"哪个更好看"）

要判"3.2 是不是比 4.5 更好"必须重新盲比，那要渲染高分源（GPU + 磁盘）。
在花那趟之前先问一个**便宜且可能直接关掉这条线索**的问题：

    这 1.4 倍的设计差，在**交付出去的那张 16x16 瓦片上**还测得到吗？

裁剪窗口里的单元数是**恒等于 4.5 的**（`side = period * UNITS_PER_TILE`，
第 9 轮已查明这是循环论证，见 `docs/loop.md`），所以**不许拿窗口来量**。
交付物要再经过降采样与量化，估计器在 16 像素的廓线上只能搜滞后 2..10
（周期 3.56px = 4.5 单元、5.0px = 3.2 单元，两者都在范围内、且落在不同的整数滞后上，
所以这个测量在原理上分得开）。差别可能在这一步就被抹掉了。

**逻辑是不对称的，必须说清楚**：

- 测不到 -> 设计上的失配没有下游签名，**重调常数不太可能改变观感，这条线索关掉**；
- 测得到 -> **只说明残差还在**，*不*等于"调成 3.2 会更好看"。那仍然要盲比。

## 预注册判据（写死在这里，跑之前提交）

- 估计器：`dominant_period(rgb, lo=2, hi_frac=0.625)`，units = 16 / period；
  返回 0 记为未检出。与 `resolution_tiers.py` 量真人时**同一套参数**。
- 交付组：`experiments/bestof/*_best.png` 里 `best_fired == true` 的那些
  （门没触发的那些根本没用过这个常数，不算数）。
- 真人组：`data/tiles/dataset_k16.json` 中全部 16x16 瓦片里检出周期的。
- 判读（`mid = (3.20 + 4.50) / 2 = 3.85`，两组用双侧 Mann-Whitney，手写正态近似含并列校正）：

  | 条件 | 结论 | 退出码 |
  | --- | --- | --- |
  | 交付组检出率 < 50% | 测不动，不下结论 | 2 |
  | 交付组中位 units >= 3.85 **且** p < 0.05 | 残差还在，值得下一轮花 GPU 做盲比 | 0 |
  | 交付组中位 units < 3.85 **且** p >= 0.05 | 无下游签名，**关掉这条线索** | 0 |
  | 其余组合 | 不一致，不下结论 | 2 |

- **无论结果如何，本轮都不改 `UNITS_PER_TILE`**：改它会让全部已发表数字失去可复现性，
  而且按代码注释的要求，改之前必须先有盲比。

字形按 GBK 安全写（第 64 轮教训：控制台 cp936，`>=` 不写成不等号）。
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import dominant_period                              # noqa: E402

HUMAN_REF = 3.20        # 生产口径下真人 16x16 的单元数（resolution_tiers.py）
DESIGN = 4.50           # UNITS_PER_TILE
MID = (HUMAN_REF + DESIGN) / 2


def units_of(rgb: np.ndarray) -> float:
    """交付/真人瓦片的每张图单元数；未检出返回 0。"""
    n = rgb.shape[0]
    p = dominant_period(rgb, lo=2, hi_frac=0.625)
    return (n / p) if p > 0 else 0.0


def mannwhitney_u_p(a, b):
    """双侧 Mann-Whitney U 的正态近似 p（含并列校正）。无 scipy 依赖。"""
    a, b = list(a), list(b)
    n1, n2 = len(a), len(b)
    allv = sorted(a + b)
    # 平均秩
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r = (i + j) / 2 + 1
        ranks[allv[i]] = r
        i = j + 1
    r1 = sum(ranks[v] for v in a)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    # 并列校正的方差
    counts = {}
    for v in allv:
        counts[v] = counts.get(v, 0) + 1
    n = n1 + n2
    tie = sum(c ** 3 - c for c in counts.values())
    var = n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return u1, 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(z / math.sqrt(2))
    return u1, min(1.0, p)


def load_delivered():
    """返回 [(prompt, units, intended_units)]，只取门触发的那些。"""
    recs = json.loads((ROOT / "experiments" / "bestof_units.json").read_text(encoding="utf-8"))
    out = []
    for r in recs:
        if not r.get("best_fired"):
            continue
        stem = r["prompt"].replace(" ", "_")
        f = ROOT / "experiments" / "bestof" / f"{stem}_best.png"
        if not f.exists():
            print(f"  [skip] {r['prompt']}: 没有 {f.name}")
            continue
        rgb = np.asarray(Image.open(f).convert("RGB"), float)
        # 设计意图上的单元数（含 side 上钳位）：side / period，side = frac * 源边长
        per = r["periods"][r["best_k"]]
        intended = r["best_frac"] * 1024.0 / per if per > 0 else float("nan")
        out.append((r["prompt"], units_of(rgb), intended))
    return out


def load_human(size=16):
    ds = json.loads((ROOT / "data" / "tiles" / "dataset_k16.json").read_text(encoding="utf-8"))
    vals = []
    for s in ds["samples"]:
        if int(s["size"]) != size:
            continue
        pal = np.array(s["palette"], np.uint8)
        n = int(s["size"])
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        u = units_of(pal[idx].astype(float))
        if u > 0:
            vals.append(u)
    return vals


def main():
    print("交付瓦片的单元数残差 —— 前提检验（判据见本文件文档串，跑前已提交）")
    print(f"参照：真人 {HUMAN_REF:.2f}（生产口径）  设计 {DESIGN:.2f}  分界 {MID:.2f}\n")

    delivered = load_delivered()
    if not delivered:
        print("没有门触发的交付瓦片，无法测量。")
        raise SystemExit(2)

    det = [(p, u, it) for p, u, it in delivered if u > 0]
    rate = len(det) / len(delivered)
    print(f"门触发的交付瓦片 {len(delivered)} 张，检出周期 {len(det)} 张"
          f"（检出率 {rate:.1%}）")

    intended = [it for _, _, it in delivered if it == it]
    print(f"裁剪窗口里的**设计**单元数中位 {np.median(intended):.2f}"
          f"（应当 = {DESIGN:.2f}，这是循环论证的那个量，只用来核对管线口径）")

    if rate < 0.5:
        print("\n判读：检出率不足一半，这个测量在 16 像素上测不动，**不下结论**。")
        raise SystemExit(2)

    du = [u for _, u, _ in det]
    hu = load_human(16)
    md, mh = float(np.median(du)), float(np.median(hu))
    _, p = mannwhitney_u_p(du, hu)

    print(f"\n交付组 n={len(du)}  中位 units = {md:.2f}   四分位 "
          f"[{np.percentile(du,25):.2f}, {np.percentile(du,75):.2f}]")
    print(f"真人组 n={len(hu)}  中位 units = {mh:.2f}   四分位 "
          f"[{np.percentile(hu,25):.2f}, {np.percentile(hu,75):.2f}]")
    print(f"Mann-Whitney 双侧 p = {p:.4g}")
    print(f"（真人中位复算 {mh:.2f} 对照 resolution_tiers.py 的 {HUMAN_REF:.2f}）")

    print("\n逐材质（交付组，按 units 降序）：")
    for pr, u, it in sorted(det, key=lambda t: -t[1]):
        print(f"  {pr:<34}{u:>6.2f}")

    print()
    if md >= MID and p < 0.05:
        print(f"判读：中位 {md:.2f} 不低于分界 {MID:.2f} 且 p={p:.4g} 小于 0.05")
        print("  -> **残差还在**：设计上的 1.4 倍失配在交付物上测得到。")
        print("     值得下一轮渲染高分源、按 3.2 与 4.5 各裁一份做盲比。")
        print("     注意：这**不**等于 3.2 更好看，只说明这条线索没被这一步掐死。")
    elif md < MID and p >= 0.05:
        print(f"判读：中位 {md:.2f} 低于分界 {MID:.2f} 且 p={p:.4g} 不小于 0.05")
        print("  -> **无下游签名**：交付瓦片在这个量上与真人分不开，")
        print("     重调 UNITS_PER_TILE 不太可能改变观感，**这条线索关掉**。")
    else:
        print(f"判读：中位 {md:.2f} 与 p={p:.4g} 指向不一致（判据要求两者同向），")
        print("  -> **不下结论**，两个数都记下来，不许挑一个当结果。")
        raise SystemExit(2)

    print("\n本轮没有改 UNITS_PER_TILE（改它需要先有盲比，见文档串）。")


if __name__ == "__main__":
    main()
