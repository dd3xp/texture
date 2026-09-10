"""单元惯例残差——**把试点扩到交付批次**（无 GPU）。写于运行之前，本提交即预注册。

## 上一轮留下的是什么

`units_residual.py`（预注册 `0d657c6`）问的是：`UNITS_PER_TILE = 4.5` 与真人的
3.20 差 1.4 倍，这个差在**交付出去的 16x16 瓦片上**还测得到吗？
结果是**退出码 2，不下结论**：交付组 n=29，中位 4.00（高于分界 3.85，方向对），
但 Mann-Whitney p=0.0688（不小于 0.05），判据要求两者同向，它们没有。

那一轮的自我诊断是：**尺子偏钝**。16 像素的廓线上 units 只能取 `16/整数滞后`，
4.00 与 3.20 是相邻的两个可取值，而单张瓦片的量化噪声就有正负半个滞后。
钝尺子上 n=29 给出 p=0.0688，更像功效不足。

## 这一轮只改一件事：样本量

尺子改不了（交付物就是 16x16，这是要测的那个对象）。能改的只有 n。
`experiments/pack78_out/`（交付批次，78 个材质）里 size=16 的瓦片**一直躺着没用过**，
门在其中 **65/78** 上触发。它与试点那 42 个提示词**只重叠 6 个**——
所以这不只是扩大，接近一次**独立复现**。

**判据、估计器、分界、真人组一个字不改**，全部沿用 `0d657c6`。
换样本与改判据必须分开（第 68/70 轮的教训）。

## 必须先承认的偏倚：这是 optional stopping

我是**在看到 p=0.0688 之后**才决定扩大 n 的。因此本轮写死四条，事后不许挑：

1. 试点那 29 张（中位 4.00、p=0.0688）**从此只作为"试点"引用**，不再单独作为结论；
2. **正式结果 = 合并**（试点去掉与 pack78 重叠的材质后并入），无论方向如何都照报；
3. 若合并后不显著，**结论就是不显著**，不许回头说"试点方向是对的"；
4. **两批方向若不一致**（中位落在分界两侧），一律**不下结论**（退 2）——
   第 72 轮刚被单批显著坑过一次：一个小子集上的显著可能纯是噪声。

## 预注册判据（跑之前提交）

- 估计器：`units_residual.units_of` = `dominant_period(rgb, lo=2, hi_frac=0.625)`，
  units = 16 / period，返回 0 记未检出。**与试点、与 `resolution_tiers.py` 同一套**。
- 交付组：`experiments/pack78_out/manifest.json` 里 `size==16 and gate_fired` 的材质，
  读 `experiments/pack78_out/base/16/<材质名下划线>.png`；
  再并入试点 `experiments/bestof_units.json` 中 `best_fired` 且**材质不在 pack78 里**的那些。
  （门没触发的从来不走这个常数，不算数。）
- 真人组：**与试点完全相同**——`data/tiles/dataset_k16.json` 全部 16x16 中检出周期的。
- 分界 `mid = (3.20 + 4.50) / 2 = 3.85`；双侧 Mann-Whitney（`units_residual` 里那份手写实现）。

| 条件 | 结论 | 退出码 |
| --- | --- | --- |
| 合并交付组检出率 < 50% | 测不动，不下结论 | 2 |
| 两批中位落在分界两侧 | 不一致，不下结论 | 2 |
| 合并中位 >= 3.85 **且** p < 0.05 | 残差还在，值得下一轮花 GPU 做盲比 | 0 |
| 合并中位 < 3.85 **且** p >= 0.05 | 交付口径上测不到下游签名，**关掉这条线索** | 0 |
| 其余组合 | 不一致，不下结论 | 2 |

- **敏感性（跑前声明，但不参与判读）**：真人组另跑一遍 `anisotropy >= 0.20` 过滤的版本。
  第 73 轮记下的设计缺陷是两组总体口径不对等（交付组过了各向异性门，真人组没有）。
  这里补上，但**只作参照**：交付组的门是在 1024 源图上触发的，
  在 16 像素瓦片上重算并不是同一件事，所以它不够格当主判据。
- **无论结果如何，本轮都不改 `UNITS_PER_TILE`**（改它要先有盲比，见 `downsample.py` 注释）。
- 阴性只能说"在交付口径、这个样本量下测不到"，**不等于失配不存在**——
  尺子的钝在第 73 轮已经量过，别把"没测到"读成"不存在"。

字形按 GBK 安全写（第 64 轮教训：控制台 cp936）。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "paired"))
sys.path.insert(0, str(ROOT / "tools"))
from downsample import anisotropy                                    # noqa: E402
from units_residual import (HUMAN_REF, DESIGN, MID, units_of,        # noqa: E402
                            mannwhitney_u_p, load_delivered)


def load_pack78():
    """交付批次里门触发的 16x16 瓦片：[(材质, units)]。"""
    man = json.loads((ROOT / "experiments" / "pack78_out" / "manifest.json")
                     .read_text(encoding="utf-8"))
    out = []
    for r in man:
        if int(r["size"]) != 16 or not r.get("gate_fired"):
            continue
        f = ROOT / "experiments" / "pack78_out" / "base" / "16" / \
            (r["material"].replace(" ", "_") + ".png")
        if not f.exists():
            print(f"  [skip] {r['material']}: 没有 {f.name}")
            continue
        rgb = np.asarray(Image.open(f).convert("RGB"), float)
        out.append((r["material"], units_of(rgb)))
    return out


def load_human_split(size=16, aniso_min=0.20):
    """真人 16x16 检出周期的 units；返回 (全部, 仅各向异性门通过的)。"""
    ds = json.loads((ROOT / "data" / "tiles" / "dataset_k16.json")
                    .read_text(encoding="utf-8"))
    allv, gated = [], []
    for s in ds["samples"]:
        if int(s["size"]) != size:
            continue
        pal = np.array(s["palette"], np.uint8)
        n = int(s["size"])
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        rgb = pal[idx].astype(float)
        u = units_of(rgb)
        if u > 0:
            allv.append(u)
            if anisotropy(rgb) >= aniso_min:
                gated.append(u)
    return allv, gated


def describe(name, vals):
    print(f"{name:<12} n={len(vals):<6} 中位 {np.median(vals):.2f}   "
          f"四分位 [{np.percentile(vals,25):.2f}, {np.percentile(vals,75):.2f}]")


def main():
    print("单元惯例残差 —— 扩到交付批次（判据见本文件文档串，跑前已提交）")
    print(f"参照：真人 {HUMAN_REF:.2f}（生产口径）  设计 {DESIGN:.2f}  分界 {MID:.2f}\n")

    pack = load_pack78()
    pack_names = {m for m, _ in pack}
    pilot_all = load_delivered()
    pilot = [(p, u) for p, u, _ in pilot_all if p not in pack_names]
    dropped = len(pilot_all) - len(pilot)
    print(f"pack78 门触发 {len(pack)} 张；"
          f"试点门触发 {len(pilot_all)} 张，去掉与 pack78 重叠的 {dropped} 个材质后并入 {len(pilot)} 张")

    pack_det = [u for _, u in pack if u > 0]
    pilot_det = [u for _, u in pilot if u > 0]
    merged = pack_det + pilot_det
    n_all = len(pack) + len(pilot)
    rate = len(merged) / n_all
    print(f"检出周期：pack78 {len(pack_det)}/{len(pack)}，试点 {len(pilot_det)}/{len(pilot)}，"
          f"合并 {len(merged)}/{n_all}（检出率 {rate:.1%}）\n")

    if rate < 0.5:
        print("判读：检出率不足一半，这个测量在 16 像素上测不动，**不下结论**。")
        raise SystemExit(2)

    human, human_gated = load_human_split(16)
    describe("交付合并", merged)
    describe("  pack78", pack_det)
    describe("  试点", pilot_det)
    describe("真人(全部)", human)
    describe("真人(过门)", human_gated)

    _, p = mannwhitney_u_p(merged, human)
    _, p_pack = mannwhitney_u_p(pack_det, human)
    _, p_gated = mannwhitney_u_p(merged, human_gated)
    md = float(np.median(merged))
    md_pack, md_pilot = float(np.median(pack_det)), float(np.median(pilot_det))

    print(f"\n主判据 Mann-Whitney 双侧 p = {p:.4g}（合并 对 真人全部）")
    print(f"  仅 pack78 对 真人全部：中位 {md_pack:.2f}，p = {p_pack:.4g}")
    print(f"  敏感性（真人过各向异性门，不参与判读）：p = {p_gated:.4g}")
    print(f"（真人中位复算 {np.median(human):.2f} 对照 resolution_tiers.py 的 {HUMAN_REF:.2f}）")

    print()
    if (md_pack >= MID) != (md_pilot >= MID):
        print(f"判读：两批中位落在分界两侧（pack78 {md_pack:.2f}、试点 {md_pilot:.2f}，"
              f"分界 {MID:.2f}）")
        print("  -> 按预注册第 4 条，**不下结论**。")
        raise SystemExit(2)

    if md >= MID and p < 0.05:
        print(f"判读：合并中位 {md:.2f} 不低于分界 {MID:.2f} 且 p={p:.4g} 小于 0.05")
        print("  -> **残差还在**：设计上的 1.4 倍失配在交付物上测得到。")
        # 判据与输出一字未改（这份是预注册的），只补一条指路的注释：
        # 下面这句"盲比"**不能读成判官盲比**——判官臂已被前置可解率试点否掉
        # （2026-09-11，`units_ab_resolvable.py`：15 对可解 53%，空对照 60%，
        # 判官在 16 像素上分不出这两种裁法）。剩下的路只有人工盲比。
        print("     值得下一轮渲染高分源、按 3.2 与 4.5 各裁一份做盲比。")
        print("     注意：这**不**等于 3.2 更好看，只说明这条线索没被这一步掐死。")
    elif md < MID and p >= 0.05:
        print(f"判读：合并中位 {md:.2f} 低于分界 {MID:.2f} 且 p={p:.4g} 不小于 0.05")
        print("  -> **交付口径上测不到下游签名**：重调 UNITS_PER_TILE 不太可能改变观感，")
        print("     这条线索关掉（只是在这个样本量与这把尺子上测不到，不是失配不存在）。")
    else:
        print(f"判读：合并中位 {md:.2f} 与 p={p:.4g} 指向不一致（判据要求两者同向），")
        print("  -> **不下结论**，两个数都记下来，不许挑一个当结果。")
        raise SystemExit(2)

    print("\n本轮没有改 UNITS_PER_TILE（改它需要先有盲比，见 downsample.py 注释）。")


if __name__ == "__main__":
    main()
