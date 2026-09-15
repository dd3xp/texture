#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M15) cell/canvas 这个二选一本身是不是假的？——**重复次数**与**短程**分开量
（预注册：本文件先提交再跑）

## 为什么问这个

`be122c4`（(M14)）第四节记了一条**描述性观察**（`c71deb9` 已更正：它不是新的、
也不是第二次独立观察）：真人 pooled Â 的隆起，16px 在 d=4、8，32px 在 d=8、16
——**位置是 2 倍关系（随画布）**；而 (M14) 的 C 判"按格子"靠的是**短程**那一段
（Â_32(2)=0.118 远低于 Â_16(1)=0.194）。
→ 事后说法：**局部平滑按格子固定，而每张瓦片里的重复次数随画布固定**，
即 (H-cell) / (H-canvas) 这个二选一可能**根本是假的**。

账本把它列为"提案①，必须换一批口径/数据来检验"。本文件就是那个检验。

**为什么值得做**：如果二选一是假的，那么"替换/削弱归一化谐波支"这个提案方向就是**错的**
——正确形式应当是**两支并存**（归一化谐波管重复次数，格子局部支管短程）。
(M14) 的 Δ、(M9) 的方向都会变成在量一个**混合物**，解释要改写。

**零 GPU、零 API、零判官、零新臂、零配置改动。** 只读训练/验证集真人瓦片。
**不碰 test split。**

## 口径独立性——先把话说清楚，不许事后美化

新的是**统计量与设计**，**不是**一台无关的仪器：
本文件的功率谱 = 一致性自相关（`scale_prior.tile_curves` 那把尺子）的**傅里叶对偶**
（Wiener–Khinchin：sum_c onehot_c[x]·onehot_c[x+d] 恰是 1{g[x]==g[x+d]}）。
真正新的三件：①**逐瓦片 argmax 的位置统计量**（不是 pooled 曲线的形状）；
②**按材质名配对**（(M14) 的混杂 3 说过 pooled 比较材质构成不同）；
③**val split** 是那条观察从未看过的数据。

## 仪器

对秩网格 g（n×n）做 one-hot → 每行/每列 rfft → 功率谱按通道与行列求和，
取频率 f=1..n//2 的 argmax = **f\*（每张瓦片的重复次数，cycles per tile）**。
对秩的任意置换不变（置换只是换通道顺序）。每个材质把该档所有瓦片的谱**相加后**再 argmax
（避免 median/mode 的并列歧义）。

- **(H-canvas-repeat)** 重复次数随画布固定 → f\*_32 = f\*_16
- **(H-cell-period)** 周期按格子固定 → f\*_32 = 2·f\*_16

## 主判据 (T1)：配对材质的**离散**归属，噪声自动落进第三桶

在两档都有瓦片的材质里数三桶：
`n_canvas = #{f*_32 == f*_16}`、`n_cell = #{f*_32 == 2·f*_16}`、其余进 `n_other`。
统计量 `w = n_cell / (n_cell + n_canvas)`，**精确二项双侧检验 vs 0.5**（`math.comb`，无 scipy）。

为什么这样设计：**非周期材质的 argmax 是噪声**，而噪声对两桶是**严格对称**的
（f\*_16 ~ U(1..8) 时 P(f\*_32=f\*_16) = P(f\*_32=2f\*_16) = 1/16，且 2·f\*_16 ≤ 16 恒不越界）
→ 噪声只会稀释、不会偏向任何一边，**不需要任何显著性/突出度阈值**（那会是自由参数）。

- **(D1)** w 显著 **< 0.5**（α=0.05）→ 重复次数**随画布** → 与短程结论合起来才谈两成分
- **(D2)** w 显著 **> 0.5** → 周期**按格子** → 二选一站得住，(M9) 得到加强
- **(D3)** 不显著 → **未判定**

## 次判据 (T2)：短程那一支（旧尺子 Â，但按材质配对）

每个材质在两档各自 pooled Â，取
`δ_m = |Â_32(2) − Â_16(1)| − |Â_32(1) − Â_16(1)|`（= 只在 d=1 上的 E_canvas − E_cell）。
统计量 = 材质间中位数，**逐材质自助** B=2000 seed=0 取 95% 百分位 CI。
δ CI 下界 > 0 → 短程**按格子**。

## 合取判决（跑前写死）

- **两成分 (H-2comp) 成立** ⟺ (D1) **且** δ CI 下界 > 0。
- 两边都指 cell → (H-cell) 简单形式成立。
- 两边都指 canvas → (H-canvas) 简单形式成立。
- 其余组合 → **未判定**，照实记账，不动任何东西。

## 复制检验 (R1)

主 split = **train**（有功率；但那条观察正是从 train 的 pooled 曲线上看来的
→ train 上的**印证是弱证据**，train 上的**反驳才是强证据**）。
`val` 是那条观察没看过的数据，但 32px val 只有 ~1 张/材质 → **只要求方向一致，不要求显著**：
val 的 `w` 与 0.5 的方向、`δ` 中位数的符号都须与 train 同向；**任一反向 → 结论降级为"未复制"**。

## 操作检验（任一不过 → 判决作废，照实写"没测到"）

- **(OP1)** 配对材质 ≥ 50（train）。
- **(OP2)** 两档 pooled Â(1) > 0（旧尺子活着）。
- **(OP3)** 有效自助样本 ≥ 1900。
- **(OP4)** **仪器锚点必须分开**，由真实 16px 瓦片构造、无自由参数：
  - `A_cell` = 16px 瓦片 **2×2 平铺**成 32px（谱恰好搬到偶数频率、奇数频率恰为 0
    → argmax 精确 ×2）→ 要求 **w 显著 > 0.5**
  - `A_canvas` = 16px 瓦片 **最近邻 ×2 放大**成 32px（谱留在原绝对频率）→ 要求 **w 显著 < 0.5**
  任一不过 → 整个 (T1) 作废。
  ⚠ 判据写的是"显著分开"而**不是**"精确命中 1.00/0.00"，理由跑前写明：最近邻放大会给谱乘上
  `2·cos(πf/2n)` 的低通锥度（f=8 处 1.41 vs f=1 处 1.99），两个峰接近时**可能挪动 argmax**
  ——那是插值核的已知性质，不是仪器坏了，不该让它作废一个本来有效的检验。
  精确命中率照样报进 JSON（`w`），只是不作为判据。
- **(OP5)** **白噪声对照**：把每张瓦片的像素整体随机置换（保色阶直方图、毁结构，seed=0），
  同流程跑一遍 → 期望 `w ≈ 0.5` 且两桶计数都小。**这不是判据**，是把"噪声对称"这条设计假设
  摆出来给人看；若它自己就显著偏向一边 → (T1) 作废。

## 跑之前就写明的混杂（结果是哪个方向都不许事后改口径）

1. **口径不独立**：见上"口径独立性"节。(T2) 更是直接用旧尺子的 d=1、d=2，
   与 (M14) 的 C **共用同一批 Â**；(T2) 的作用是**分解**，不是独立确认。
2. **材质名相同 ≠ 同一张画**：16px 来自更多包，32px train 只有 19 个包、val 只有 4 个。
3. **val 的 32px 约 1 张/材质** → 逐材质谱=单张谱，噪声大。所以 (R1) 只看方向。
4. `extra=False`（不含 64→32 派生瓦片），与 (M9)/(M14) 口径一致；(M11) 已证派生与原生同尺。
5. **k_used ≥ 3** 过滤，与 (M9)/(M14) 一致。

## 授权（跑前写死）

- 若 **(H-2comp) 成立** → 只授权**写一份**"两支并存"偏置的预注册（动可平铺性的论证要重写）；
  同时**记为 (M14)/(M9) 的解释需改写**。**不授权训练、不授权判官、不改任何默认值。**
- 若 **(D2)**（周期也按格子）→ 二选一站得住，第四节那条描述性观察**就此关闭**，不许再提。
- 其余 → **什么都不授权**。

## 本轮不做什么

**不调判官、不出胜率、不新增臂、不训练、不改 `final_test.sh` / `UNITS_PER_TILE` /
`judge_pairs.py` / 任何默认值。**
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import tiles_data  # noqa: E402
from analysis.arch.scale_prior import tile_curves  # noqa: E402  旧尺子逐字复用

B_BOOT = 2000
MIN_BOOT = 1900
MIN_MATS = 50
ALPHA = 0.05


def spectrum(g, n):
    """瓦片秩网格 -> 长度 n//2 的功率谱（f=1..n//2），行与列相加。对秩置换不变。"""
    k = int(g.max()) + 1
    oh = np.zeros((k, n, n), dtype=np.float64)
    for c in range(k):
        oh[c] = (g == c)
    pr = np.abs(np.fft.rfft(oh, axis=2)) ** 2      # 行方向
    pc = np.abs(np.fft.rfft(oh, axis=1)) ** 2      # 列方向
    return pr.sum(axis=(0, 1))[1:n // 2 + 1] + pc.sum(axis=(0, 2))[1:n // 2 + 1]


def mat_fstar(rows, n):
    """{material: f*}，把同材质同档的谱相加后再 argmax。"""
    acc = {}
    for s in rows:
        m = s["material"]
        p = spectrum(np.asarray(s["idx"]), n)
        acc[m] = acc[m] + p if m in acc else p
    return {m: int(np.argmax(p)) + 1 for m, p in acc.items()}


def binom_two_sided(k, n):
    """p = 0.5 下的精确双侧 p 值（对称分布，2*单尾，截到 1）。"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def t1(f16, f32, label, out):
    """离散归属检验。返回 (w, p, n_cell, n_canvas)。"""
    mats = sorted(set(f16) & set(f32))
    n_cell = sum(1 for m in mats if f32[m] == 2 * f16[m])
    n_canvas = sum(1 for m in mats if f32[m] == f16[m])
    tot = n_cell + n_canvas
    w = n_cell / tot if tot else float("nan")
    p = binom_two_sided(n_cell, tot)
    rec = {"n_mats": len(mats), "n_cell": n_cell, "n_canvas": n_canvas,
           "n_other": len(mats) - tot, "w": w, "p": p}
    out[label] = rec
    print(f"[T1:{label}] mats={len(mats)} cell={n_cell} canvas={n_canvas} "
          f"other={len(mats) - tot} w={w:.3f} p={p:.4g}")
    return rec


def t2(rows16, rows32, label, out):
    """短程分解：delta_m = |A32(2)-A16(1)| - |A32(1)-A16(1)|，材质间中位数 + 自助 CI。"""
    def per_mat(rows, n):
        cur = tile_curves(rows, n)
        acc = {}
        for s, c in zip(rows, cur):
            acc.setdefault(s["material"], []).append(c)
        return {m: np.mean(v, axis=0) for m, v in acc.items()}

    c16, c32 = per_mat(rows16, 16), per_mat(rows32, 32)
    mats = sorted(set(c16) & set(c32))
    d = np.array([abs(c32[m][1] - c16[m][0]) - abs(c32[m][0] - c16[m][0]) for m in mats])
    rng = np.random.default_rng(0)
    boots = [float(np.median(d[rng.integers(0, len(d), len(d))])) for _ in range(B_BOOT)]
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    rec = {"n_mats": len(mats), "median_delta": float(np.median(d)), "ci": [lo, hi],
           "n_boot": len(boots), "op3": len(boots) >= MIN_BOOT}
    out[label] = rec
    print(f"[T2:{label}] mats={len(mats)} median_delta={np.median(d):+.4f} "
          f"95%CI [{lo:+.4f}, {hi:+.4f}]")
    return rec


def load(size, split):
    return [x for x in tiles_data.load(size=size, split=split) if x["k_used"] >= 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/repeat_count.json")
    a = ap.parse_args()
    out = {}

    r16, r32 = load(16, "train"), load(32, "train")
    v16, v32 = load(16, "val"), load(32, "val")
    print(f"train 16px={len(r16)} 32px={len(r32)} | val 16px={len(v16)} 32px={len(v32)}")

    # ---- (OP4) 仪器锚点：由真实 16px 瓦片构造，真值精确已知 ----
    f16 = mat_fstar(r16, 16)
    a_cell = mat_fstar([{"material": s["material"],
                         "idx": np.tile(np.asarray(s["idx"]), (2, 2))} for s in r16], 32)
    a_canvas = mat_fstar([{"material": s["material"],
                           "idx": np.repeat(np.repeat(np.asarray(s["idx"]), 2, 0), 2, 1)}
                          for s in r16], 32)
    ac = t1(f16, a_cell, "OP4_anchor_cell", out)
    an = t1(f16, a_canvas, "OP4_anchor_canvas", out)
    op4 = (ac["p"] < ALPHA and ac["w"] > 0.5) and (an["p"] < ALPHA and an["w"] < 0.5)
    print(f"(OP4) anchors separate: {op4}")

    # ---- (OP5) 白噪声对照：整体像素置换 ----
    rng = np.random.default_rng(0)

    def shuf(rows, n):
        o = []
        for s in rows:
            g = np.asarray(s["idx"]).ravel().copy()
            rng.shuffle(g)
            o.append({"material": s["material"], "idx": g.reshape(n, n)})
        return o

    sh = t1(mat_fstar(shuf(r16, 16), 16), mat_fstar(shuf(r32, 32), 32), "OP5_shuffle", out)
    op5 = sh["p"] >= ALPHA
    print(f"(OP5) shuffle not significant: {op5}")

    # ---- 主判据 ----
    f32 = mat_fstar(r32, 32)
    tr = t1(f16, f32, "T1_train", out)
    va = t1(mat_fstar(v16, 16), mat_fstar(v32, 32), "T1_val", out)
    d_tr = t2(r16, r32, "T2_train", out)
    d_va = t2(v16, v32, "T2_val", out)

    op1 = tr["n_mats"] >= MIN_MATS
    p16, p32 = tile_curves(r16, 16).mean(0), tile_curves(r32, 32).mean(0)
    op2 = bool(p16[0] > 0 and p32[0] > 0)
    op3 = d_tr["op3"] and d_va["op3"]
    print(f"(OP1) {op1}  (OP2) {op2}  (OP3) {op3}")

    if not (op1 and op2 and op3 and op4 and op5):
        verdict, t1v, t2v = "OP_FAIL", "void", "void"
    else:
        t1v = ("D1_canvas" if (tr["p"] < ALPHA and tr["w"] < 0.5) else
               "D2_cell" if (tr["p"] < ALPHA and tr["w"] > 0.5) else "D3_undecided")
        t2v = "cell" if d_tr["ci"][0] > 0 else ("canvas" if d_tr["ci"][1] < 0 else "undecided")
        # (R1) 方向复制
        same_w = (va["w"] - 0.5) * (tr["w"] - 0.5) > 0
        same_d = d_va["median_delta"] * d_tr["median_delta"] > 0
        if t1v == "D1_canvas" and t2v == "cell":
            verdict = "H2COMP" if (same_w and same_d) else "H2COMP_not_replicated"
        elif t1v == "D2_cell" and t2v == "cell":
            verdict = "H_CELL"
        elif t1v == "D1_canvas" and t2v == "canvas":
            verdict = "H_CANVAS"
        else:
            verdict = "UNDECIDED"
        out["replication"] = {"same_w_direction": bool(same_w), "same_delta_sign": bool(same_d)}

    out["ops"] = {"op1": bool(op1), "op2": op2, "op3": bool(op3),
                  "op4": bool(op4), "op5": bool(op5)}
    out["t1_verdict"], out["t2_verdict"], out["verdict"] = t1v, t2v, verdict
    print(f"T1={t1v}  T2={t2v}  -> VERDICT = {verdict}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
