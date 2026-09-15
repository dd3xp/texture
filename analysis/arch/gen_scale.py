#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""架构那条先验，在**模型自己的产物**里看得见吗？（预注册：本文件先提交再跑）

## 为什么问这个

(M9)（`analysis/arch/scale_prior.py`，判决 `4e965b8`）量的是**真人数据**：真人瓦片的结构
**按格子对齐**，不随画布缩放。而 `model/trd.py::ToroidalBias` 只看**归一化偏移 d/n**、
16/24/32 共用一张表 → 架构写死了"结构随画布缩放"。两边对不上 = **架构嫌疑人 (M9)**。

但从"架构有这条先验"到"32px 因此变坏"中间缺一环：**模型的输出真的表现出这条先验吗？**
网络有的是别的容量（自注意力内容项、CLIP 条件、调色板），完全可能绕过偏置表。
(M13)（`4fa50da`）给偏置表**补**格子特征，三档 14 个对比全在噪声下限内 = 什么也没测到，
当时留下两种都站得住的解释：(a) 错设先验根本不是 32px 缺口的原因；(b) 是原因但加性补充救不了。
**本测量直接分开 (a) 和 (b) 的前半句**：先验有没有落到产物上。

**零 GPU、零 API、零判官、零新臂**：只读已经生成好的 PNG + 训练集真人瓦片。
不碰测试集真人瓦片（32px 测试真人只有 2 张，本来也没法用）。

## 量什么

与 (M9) **同一把尺子**（`scale_prior.tile_curves`，逐字复用）：对每张瓦片的亮度秩网格，

    Â_n(d) = (A_n(d) − c) / (1 − c),   A_n(d) = P(g[y,x] == g[y,(x+d) mod n]) 的 x/y 平均

生成图只有 PNG，先还原成秩网格：取实际出现的颜色，按亮度（`tiles_data.W_LUM`）升序编号
（与 `tiles_data.canonicalise` 同一规则）。k_used = 实际颜色数。

## 主统计量：**无阈值**的两模型对比（不是相关长度）

⚠ (M12)（`5042ca5`）已判定：半高相关长度 L 在 16px 上有悬崖（Â 被 4 格周期主导，
半高阈离 Â(2) 只有 0.005）→ **L / R 的数值与 CI 一律不许引**。所以本轮**主判据里不出现 L**。

对一对档位 m < n，在重叠域 d = 1..m//2 上比两条**预测**与实测 Â_n 的差：

    E_cell   = mean_d | Â_n(d)         − Â_m(d) |      ← (H-cell)   结构按格子固定
    E_canvas = mean_d | Â_n(d·n/m)     − Â_m(d) |      ← (H-canvas) 结构随画布缩放
    Δ = E_canvas − E_cell

（Â_n 在非整数处用相邻整数线性插值；m=24,n=32 时 d·4/3 ≤ 16 = n//2，不越界。）
构造上：完全按格子 → E_cell=0 → Δ>0；完全随画布 → E_canvas=0 → Δ<0。**边界 0 是构造出来的，
不是挑的**，没有任何自由参数。

## 预注册判据（跑之前写死，不许改）

对每个对比做**逐瓦片自助**（两档各自独立重采样，B=2000，seed=0），取 Δ 的 95% 百分位区间。

- **(G1)** CI 上界 < 0 → **产物是"随画布缩放"的**：架构先验确实落到了输出上。
  与 (M9)（真人=按格子）合起来 = **32px 输出的结构尺度是错的**，这是 32px 缺口的一个具体机制。
  授权**写一份**提案①（替换/削弱归一化谐波支）的预注册；**不授权训练、不授权判官**。
- **(G2)** CI 下界 > 0 → **产物是"按格子"的**：模型已经绕过了那张归一化表 →
  **`ToroidalBias` 这条线作为 32px 缺口的解释就此关闭**，不许再开相关臂。
- **(G3)** CI 跨 0 → **未判定**，不改任何东西，照实记账。

**判决只看对比 A。** B、C 是稳健性/方向自检，**不推翻 A，也不许拿来替 A 下判**。

## 三个对比（跑前定死，不许增删）

| 代号 | m | n | 是什么 |
|---|---|---|---|
| **A（主）** | TRD16c_rr4 @16 | TRD32_rr4 @32 | **交付物本身**（`final_test.sh`，E_mat 272 材质），比值 2 |
| B（次） | TRD24_rr4 @24 | TRD32_rr4 @32 | **同一检查点 v10**，比值 4/3 |
| C（自检） | 真人 train @16 | 真人 train @32 | (M9) 已判 = 按格子 → Δ 应 > 0；**验的是这把尺子的方向** |

## 操作检验（任一不过 → 该对比判决作废，照实写"没测到"）

- **(OP1)** 每组 k_used ≥ 3 的瓦片 ≥ 100 张。
- **(OP2)** 每组 pooled Â(1) > 0（尺子活着）。
- **(OP3)** 每个对比的有效自助样本 ≥ 1900。

## 跑之前就写明的混杂（结果是哪个方向都不许事后改口径）

1. **(L3) A 跨检查点**：TRD16c=v8、TRD32=v10（`final_test.sh:18,22`），且 `--ret_nname`
   30/100 不同。所以 A 严格说是"**我们交付物**的档间结构尺度"。B 是同一个 v10，
   但比值只有 1.333 → 分辨力弱。**两者都不是完美对照，这是本轮的主要局限。**
2. **PNG 还原秩网格会并掉亮度相同的颜色** → k_used 可能低于模型实际输出的色阶数。
   Â 已按机遇水平归一化，能吸收"色阶数不同"的水平差，但**不能**保证吸收形状差。
3. **C 用 train split、生成图用 E_mat 提示词** → 材质构成不同。C 只是**方向自检**，
   不是 A 的匹配对照。
4. `extra=False`（不含 64→32 派生瓦片），与 (M9) 口径一致；(M11)（`ca77839`）已证实
   派生包与原生包同尺，所以这个选择不影响 C。

## 本轮不做什么

**不调判官、不出胜率、不新增臂、不改任何默认值/配置/`final_test.sh`/`UNITS_PER_TILE`。**
无论判决如何，**都不直接授权训练**（(G1) 只授权再写一份预注册）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import tiles_data  # noqa: E402
from analysis.arch.scale_prior import tile_curves  # noqa: E402  尺子逐字复用

B_BOOT = 2000
MIN_TILES = 100
MIN_BOOT = 1900


def png_rows(root: Path, tag: str, size: int):
    """<tag>/<size>/*.png -> [{"idx": 秩网格}]，按亮度升序编号（同 canonicalise 规则）。"""
    files = sorted((root / tag / str(size)).glob("*.png"))
    out = []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
        assert a.shape[:2] == (size, size), (f, a.shape)
        flat = a.reshape(-1, 3)
        cols, inv = np.unique(flat, axis=0, return_inverse=True)
        order = np.argsort(cols.astype(np.float64) @ tiles_data.W_LUM, kind="stable")
        rank = np.empty(len(cols), dtype=np.int64)
        rank[order] = np.arange(len(cols))
        out.append({"idx": rank[inv].reshape(size, size), "k_used": int(len(cols))})
    return out


def interp(curve, x):
    """curve[j] 对应 d=j+1；在实数 d=x 处线性插值。"""
    lo = int(np.floor(x))
    hi = min(int(np.ceil(x)), len(curve))
    if lo < 1:
        lo = 1
    if lo == hi:
        return curve[lo - 1]
    w = x - lo
    return (1.0 - w) * curve[lo - 1] + w * curve[hi - 1]


def delta(cm, cn, m, n):
    """Δ = E_canvas − E_cell（cm/cn 是两档的 pooled Â 曲线）。"""
    ds = np.arange(1, m // 2 + 1)
    e_cell = np.mean([abs(interp(cn, d) - cm[d - 1]) for d in ds])
    e_canvas = np.mean([abs(interp(cn, d * n / m) - cm[d - 1]) for d in ds])
    return float(e_canvas - e_cell), float(e_cell), float(e_canvas)


def compare(name, cur_m, cur_n, m, n, out):
    ok1 = len(cur_m) >= MIN_TILES and len(cur_n) >= MIN_TILES
    pm, pn = cur_m.mean(0), cur_n.mean(0)
    ok2 = bool(pm[0] > 0 and pn[0] > 0)
    d, e_cell, e_canvas = delta(pm, pn, m, n)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(B_BOOT):
        bm = cur_m[rng.integers(0, len(cur_m), len(cur_m))].mean(0)
        bn = cur_n[rng.integers(0, len(cur_n), len(cur_n))].mean(0)
        boots.append(delta(bm, bn, m, n)[0])
    ok3 = len(boots) >= MIN_BOOT
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    if not (ok1 and ok2 and ok3):
        verdict = "OP_FAIL"
    else:
        verdict = "G1_canvas" if hi < 0 else ("G2_cell" if lo > 0 else "G3_undecided")
    rec = {"m": m, "n": n, "n_tiles": [len(cur_m), len(cur_n)],
           "op1": bool(ok1), "op2": ok2, "op3": bool(ok3), "n_boot": len(boots),
           "E_cell": e_cell, "E_canvas": e_canvas, "delta": d, "ci": [lo, hi],
           "verdict": verdict,
           "curve": {str(m): pm.tolist(), str(n): pn.tolist()}}
    out[name] = rec
    print(f"[{name}] m={m} n={n} tiles={len(cur_m)}/{len(cur_n)} "
          f"OP {int(ok1)}{int(ok2)}{int(ok3)}")
    print(f"[{name}] E_cell={e_cell:.4f} E_canvas={e_canvas:.4f} "
          f"Δ={d:+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="experiments/baselines")
    ap.add_argument("--out", default="/tmp/gen_scale.json")
    a = ap.parse_args()
    root = Path(a.gen_root)

    def gen_curves(tag, size):
        rows = [r for r in png_rows(root, tag, size) if r["k_used"] >= 3]
        print(f"  {tag}@{size}: {len(rows)} 张 (k_used>=3)")
        return tile_curves(rows, size)

    def real_curves(size):
        rows = [x for x in tiles_data.load(size=size, split="train") if x["k_used"] >= 3]
        print(f"  real@{size}: {len(rows)} 张 (k_used>=3)")
        return tile_curves(rows, size)

    out = {}
    print("读图…")
    g16 = gen_curves("TRD16c_rr4", 16)
    g24 = gen_curves("TRD24_rr4", 24)
    g32 = gen_curves("TRD32_rr4", 32)
    r16, r32 = real_curves(16), real_curves(32)

    compare("A_deliver_16_32", g16, g32, 16, 32, out)
    compare("B_v10_24_32", g24, g32, 24, 32, out)
    compare("C_real_16_32", r16, r32, 16, 32, out)

    out["primary"] = "A_deliver_16_32"
    out["primary_verdict"] = out["A_deliver_16_32"]["verdict"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("主判决（只看 A）：", out["primary_verdict"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
