#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TRD 的位置先验是"结构随画布缩放"——真人数据同意吗？（预注册：本文件先提交再跑）

## 为什么问这个

TRD 里**没有绝对位置**，空间结构只能由 `model/trd.py::ToroidalBias` 表达，而它是
**归一化环面偏移**的函数（`trd.py:59-60`：`dy=(y_i-y_j)/n`）。同一张偏置表被 16px 与 32px
**共用**，于是架构里写死了一条先验：

> **(H-canvas) 结构随画布缩放**——16px 上相隔 4 格 == 32px 上相隔 8 格（两者 dy 都是 0.25）。

如果真人数据其实是另一种：

> **(H-cell) 结构的格子粒度固定**——32px 的砖缝也是每隔 4 格，只是画布更大、周期数更多，

那么这张共用偏置表就是**错设的**：两档要求它在同一个输入上给出不同的值。训练包里
16px 瓦片比 32px 多约 5 倍 → 16px 的先验赢 → 32px 采样拿到错误的空间频率。
这与已记录的观察对得上：**TRD 在 32px 上"满屏单像素噪点、没有大尺度结构"**
（`docs/arch_progress.md` 2026-09-12），而 16px 结构正常。

**这是零 GPU、零 API 的纯数据测量**：只看训练集真人瓦片自己，不碰模型、不碰测试集。

## 量什么

对每张瓦片的**亮度秩网格**（`tiles_data.canonicalise`），沿 x 与 y 两个方向量
"隔 d 格的两个格子同色阶"的比例：

    A_n(d) = P(g[y,x] == g[y,(x+d) mod n])      d = 1..n/2（环面上 d 与 n-d 等价）

每张瓦片的机遇水平 c = Σ_i p_i²（p 是该瓦片的色阶边缘分布）随 k_used 变化，
所以报**超出机遇的部分**：

    Â_n(d) = (A_n(d) - c) / (1 - c)     0 = 机遇，1 = 完全相同

**相关长度** L_n = 使 Â_n(d) 首次跌到 0.5·Â_n(1) 以下的 d（相邻两点线性插值）。

## 预注册判据（跑之前写死，不许改）

主统计量：**R = L_32 / L_16**，对瓦片做自助重采样（B=2000，seed=0）取 95% 百分位区间。
- (H-canvas) 预测 **R ≈ 2**（结构随画布放大一倍）
- (H-cell)   预测 **R ≈ 1**
- 分界取二者的**几何中点 R\* = √2 ≈ 1.414**

- **(D1)** CI 上界 < √2 → **(H-cell) 这一侧**：结构不随画布缩放 → **共用归一化偏置是错设的**。
  授权**一次**架构改动（给 `ToroidalBias` 补格子单位的谐波特征）+ **一次**重训。
- **(D2)** CI 下界 > √2 → **(H-canvas) 这一侧**：架构那条先验是对的 →
  **不许动 `ToroidalBias`**，32px 的毛病另找原因。
- **(D3)** CI 跨过 √2 → **未判定**，不改任何东西，照实记账。

次要（描述性，**不作判据**）：两条曲线在重叠域上的平均绝对差
E_cell = mean_d |Â_32(d) - Â_16(d)|（d=1..8）与
E_canvas = mean_d |Â_32(2d) - Â_16(d)|（d=1..8）。

操作检验（任一不过 → 判决作废，照实写"没测到"）：
- **(OP1)** 两档各 ≥300 张 k_used≥3 的训练瓦片。
- **(OP2)** 两档的 Â_n(1) 都 > 0（否则这把尺子本身坏了）。
- **(OP3)** 两档的 pooled 曲线都**没有被截断**（即在 d ≤ n/2 内确实跌破半高）。

稳健性（预注册为**次要**，不推翻主判据）：只取两档都出现过的材质名的子集重算一遍 R。

⚠ 口径：`split="train"`、`decontam=True`、`extra=False`（不含 64→32 派生瓦片）。
   **测试集一张不看。**
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import tiles_data  # noqa: E402

R_STAR = 2.0 ** 0.5
B_BOOT = 2000
MIN_TILES = 300


def tile_curves(samples, n):
    """每张瓦片一行：[Â(1)..Â(n//2)]。返回 [T, n//2]，机遇水平已除掉。"""
    half = n // 2
    rows = np.zeros((len(samples), half), dtype=np.float64)
    for t, s in enumerate(samples):
        g = np.asarray(s["idx"])
        k = int(g.max()) + 1
        p = np.bincount(g.ravel(), minlength=k) / float(g.size)
        chance = float((p ** 2).sum())
        for j, d in enumerate(range(1, half + 1)):
            agree = (g == np.roll(g, -d, axis=1)).mean() + (g == np.roll(g, -d, axis=0)).mean()
            a = agree / 2.0
            rows[t, j] = (a - chance) / (1.0 - chance) if chance < 1.0 else 0.0
    return rows


def corr_length(curve):
    """Â 首次跌破 0.5·Â(1) 的 d（线性插值）。未跌破返回 None（截断）。"""
    if curve[0] <= 0:
        return None
    thr = 0.5 * curve[0]
    for j in range(1, len(curve)):
        if curve[j] < thr:
            d0, d1 = j, j + 1                      # curve[j-1] 对应 d=j
            y0, y1 = curve[j - 1], curve[j]
            if y0 == y1:
                return float(d1)
            return float(d0 + (y0 - thr) / (y0 - y1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/scale_prior.json")
    a = ap.parse_args()

    res = {}
    for n in (16, 32):
        s = [x for x in tiles_data.load(size=n, split="train") if x["k_used"] >= 3]
        res[n] = {"samples": s, "curves": tile_curves(s, n)}
        print(f"[{n}px] 训练瓦片 k_used>=3: {len(s)}")

    ok1 = all(len(res[n]["samples"]) >= MIN_TILES for n in (16, 32))
    pooled = {n: res[n]["curves"].mean(0) for n in (16, 32)}
    ok2 = all(pooled[n][0] > 0 for n in (16, 32))
    L = {n: corr_length(pooled[n]) for n in (16, 32)}
    ok3 = all(L[n] is not None for n in (16, 32))
    print(f"(OP1) n>=300: {ok1}   (OP2) A_hat(1)>0: {ok2}   (OP3) 未截断: {ok3}")
    for n in (16, 32):
        print(f"[{n}px] Â(d) =", " ".join(f"{v:.3f}" for v in pooled[n]))
        print(f"[{n}px] L = {L[n]}")

    out = {"op1": bool(ok1), "op2": bool(ok2), "op3": bool(ok3),
           "n_tiles": {str(n): len(res[n]["samples"]) for n in (16, 32)},
           "curve": {str(n): pooled[n].tolist() for n in (16, 32)},
           "L": {str(n): L[n] for n in (16, 32)}}

    if not (ok1 and ok2 and ok3):
        out["verdict"] = "OP_FAIL"
        print("操作检验未过 -> 判决作废")
    else:
        R = L[32] / L[16]
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(B_BOOT):
            ls = {}
            for n in (16, 32):
                c = res[n]["curves"]
                i = rng.integers(0, c.shape[0], c.shape[0])
                ls[n] = corr_length(c[i].mean(0))
            if ls[16] and ls[32]:
                boots.append(ls[32] / ls[16])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "D1_cell" if hi < R_STAR else ("D2_canvas" if lo > R_STAR else "D3_undecided")
        d = np.arange(1, 9)
        e_cell = float(np.abs(pooled[32][d - 1] - pooled[16][d - 1]).mean())
        e_canvas = float(np.abs(pooled[32][2 * d - 1] - pooled[16][d - 1]).mean())
        out.update({"R": float(R), "ci": [float(lo), float(hi)], "n_boot": len(boots),
                    "R_star": R_STAR, "verdict": verdict,
                    "E_cell": e_cell, "E_canvas": e_canvas})
        print(f"R = L32/L16 = {R:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   R* = {R_STAR:.3f}")
        print(f"次要（描述性）E_cell = {e_cell:.4f}   E_canvas = {e_canvas:.4f}")
        print(f"判决：{verdict}")

        # 稳健性（次要）：只取两档共有的材质名
        m16 = {x["material"] for x in res[16]["samples"]}
        m32 = {x["material"] for x in res[32]["samples"]}
        both = m16 & m32
        sub = {}
        for n in (16, 32):
            keep = [i for i, x in enumerate(res[n]["samples"]) if x["material"] in both]
            sub[n] = (len(keep), corr_length(res[n]["curves"][keep].mean(0)) if keep else None)
        out["matched"] = {"n_materials": len(both),
                          "n_tiles": {str(n): sub[n][0] for n in (16, 32)},
                          "L": {str(n): sub[n][1] for n in (16, 32)}}
        if sub[16][1] and sub[32][1]:
            out["matched"]["R"] = sub[32][1] / sub[16][1]
            print(f"稳健性（同材质名 {len(both)} 个）：R = {out['matched']['R']:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
