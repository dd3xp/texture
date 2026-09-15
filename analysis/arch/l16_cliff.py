#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""事后诊断（**非预注册、不产生任何判决**）：半高相关长度 L16 坐在悬崖边上。

## 怎么发现的

`scale_prior_down.py` 的**预注册次要描述量** R_mix = L(MIX)/L16 报了 **0.627**，
而上一轮 (M9) 同一个比值在 `extra=False` 口径下是 **1.011**。两者差 1.6 倍，
只因为换了 `extra` 口径（分析口径 vs 模型实际训练口径）。本文件只回答"为什么"。

## 结论（照抄实测）

16px 的 Â(d) 曲线**不是单调衰减**，它被 **4 格周期**主导：Â(4)=0.139、Â(8)=0.148
都**高于** Â(2)≈0.097。而半高阈值恰好 = 0.0970。于是：

| 口径 | n | Â(2) | 半高 | L16 | 自助 500 次 |
|---|---|---|---|---|---|
| `extra=False`（上一轮 (M9) 口径） | 2732 | 0.0923 | 0.0969 | **1.955** | IQR [1.94,1.97]，仅 3% 落在 >3 |
| `extra=packs_only`（模型训练口径） | 3177 | 0.0975 | 0.0970 | **4.706** | **双峰，51% 落在 >3** |

Â(2) 只动了 **0.005**，L16 就从 1.96 跳到 4.71（**2.4 倍**）——因为一旦 d=2 没跌破半高，
判据就被推过 d=4 那个周期峰，到 d=5 才落地。**半高相关长度对"周期 + 衰减"叠加的曲线
是个悬崖统计量**，它把两支混在一起量。

## 这对 (M9)/(D1) 做了什么、没做什么

- **没有推翻方向。** (H-canvas) 预测 L32 ≈ 2·L16。悬崖的**两侧都离它很远**：
  `extra=False` 下 R=1.011；训练口径下 R=0.627（L16 往上跳只会让 R 更小、离 2 更远）。
  → **"结构不随画布缩放"这个判决在悬崖两侧都成立**，架构改动的依据不动。
- **推翻了把 R 当精确点估计来引用。** "R = 1.011，95% CI [0.960, 1.177]" 那个窄区间
  是**该口径下**诚实的，但估计量在 0.005 之外就有断崖 → **今后引用 (M9) 只许说方向
  （不随画布缩放），不许引用 R 的数值或区间**。
- 上一轮的事后读法"**周期随画布、局部粒度随格子**"与本诊断一致：曲线本来就有两支，
  而 L 把它们混在一起。

⚠ **提案，非结果**：要量"衰减那一支"应当先把周期支去掉（例如只用 d=1..3，或先扣掉
   周期峰再取 L）。**那是另一个统计量，必须另行预注册**；本轮不换尺子、不重判。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import tiles_data  # noqa: E402
from scale_prior import corr_length, tile_curves  # noqa: E402

for ex in (False, "train_extra_packs_only.json"):
    s = [x for x in tiles_data.load(size=16, split="train", extra=ex) if x["k_used"] >= 3]
    c = tile_curves(s, 16)
    p = c.mean(0)
    print(f"16px extra={str(ex)[:28]:30s} n={len(s)} L={corr_length(p):.3f} 半高={0.5 * p[0]:.4f}")
    print("   Â(d) =", " ".join(f"{v:.4f}" for v in p))
    rng = np.random.default_rng(0)
    b = np.array([x for x in (corr_length(c[rng.integers(0, len(c), len(c))].mean(0))
                              for _ in range(500)) if x])
    print(f"   自助 L16: 中位 {np.median(b):.2f}  IQR [{np.percentile(b, 25):.2f},"
          f"{np.percentile(b, 75):.2f}]  >3 的比例 {(b > 3).mean():.2f}")
