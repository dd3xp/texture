#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""32px 训练分布自己内部一尺吗？——派生包（64→32）与原生 32px 的相关长度之比
（预注册：本文件先提交再跑，跑前不看任何数字）

## 为什么问这个：上一轮的因果故事有一个我没核过的前提

`8e364ae`/`4e965b8` 量到 (M9)：真人瓦片的相关长度**按格子对齐**（R = L32/L16 = 1.011，
CI [0.960, 1.177]）→ `ToroidalBias` 共用的归一化表是错设的。**这条数据判决本身不动。**

但那一轮**顺手写下的解释**是："训练包里 16px 瓦片比 32px 多 6.8 倍 → 冲突由 16px 先验赢
→ 32px 拿到错误的空间频率"（`scale_prior.py:17-18`、`docs/arch_progress.md`）。
本轮开工核对实际配方，这句话的两个输入**都不对**：

1. **瓦片数不是 6.8 倍。** 2732/403 是**分析口径**（`extra=False`，故意排除 64→32）。
   训练实际加载的是 `--extra_file train_extra_packs_only.json+train_64to32.json`：
   **16px 3177 / 32px 1877 = 1.69 倍**（本机实测，见本轮账本）。
2. **梯度份额根本不由瓦片数决定。** `train_trd.py:491` 每步以 `--p32 0.7` 的概率整批取 32px，
   `--batch 256` / `--batch32 64`。按格子算：16px 每步 0.3×256×256 = 1.97 万，
   32px 0.7×64×1024 = 4.59 万 → **32px 是 16px 的 2.3 倍**；偏置表打分的是**格子对**，
   按对算 0.3×256×256² vs 0.7×64×1024² → **32px 是 9.3 倍**。
   → **"16px 先验赢"这句话没有依据，本轮予以撤回**（(M9) 与那次架构改动不受影响：
   一张归一化表**不可能**同时满足两档，这是构造性的，与谁赢无关）。

撤回之后，"32px 为什么偏偏是坏的那一档"重新变成没有答案的问题。而核对配方时冒出一个
**上一轮的尺子照不到的**新嫌疑人：

> 训练用的 1877 张 32px 瓦片里，**1236 张（66%）是 `train_64to32.json`**
> ——由 64px 瓦片众数降采样 2 倍得来的派生瓦片。而 `scale_prior.py:62` 的口径
> **`extra=False`，明确把这 1236 张排除在外**。也就是说：(M9) 的 "32px 那条曲线"
> 只覆盖了模型实际 32px 训练数据的 **34%**，且是被多数票压过的那 34%。

如果 64px 瓦片的结构本身是按格子的（(M9) 的同一条结论），那么把它**降采样 2 倍**之后，
结构在格子单位上就**被压到一半**。于是 32px 的训练分布内部会是**两把尺子**：
原生那 34% 是一尺，派生那 66% 是半尺。**模型在 32px 上"满屏单像素噪点"**
（`docs/arch_progress.md` 2026-09-12）与"多数 32px 训练样本的相关长度只有一半"是同一个方向。

**零 GPU、零 API、零判官**：只读 `split="train"`，测试集一张不看。

## 量什么

尺子与 `scale_prior.py` **逐字相同**（直接 import 它的 `tile_curves` / `corr_length`）：
超出机遇的同色阶率 Â_n(d)，相关长度 L = Â 首次跌破半高的 d。

三组（都是 `size=32, split="train", decontam=True`）：
- **A = 原生**：`extra="train_extra_packs_only.json"`
- **B = 派生**：A 之外、`extra="...+train_64to32.json"` 之内的那些（按 (pack, material, idx) 配对求差）
- **MIX = A ∪ B**：模型 32px 实际训练分布

## 预注册判据（跑前写死，不许改）

主统计量 **R_d = L(B) / L(A)**，对瓦片自助重采样（B=2000，seed=0）取 95% 百分位区间。
- **(H-half)** 派生包把格子尺度压到一半 → 预测 **R_d ≈ 0.5**
- **(H-same)** 派生包与原生 32px 同尺 → 预测 **R_d ≈ 1**
- 分界取二者**几何中点 √0.5 ≈ 0.7071**（与上一轮取 √2 同一套办法）

- **(E1)** CI 上界 < 0.7071 → **32px 训练分布内部不一尺，且多数票（66%）投给更短的相关长度**
  → 登记为 32px 缺口的**第二个候选原因（数据侧）**。
- **(E2)** CI 下界 > 0.7071 → 派生包与原生同尺 → **这条候选当场死掉**，照实记。
- **(E3)** CI 跨界 → **未判定**。

⛔ **三种结局都不授权任何改动**：不动 `final_test.sh`、不动任何默认值、不动在跑的
`arch_cell` 训练与它的 ①②③ 判据、不开任何判官臂。真要据此改数据配方，**必须另行预注册**。

操作检验（任一不过 → 判决作废，照实写"没测到"）：
- **(OP1)** A、B 各 ≥300 张 k_used≥3 的瓦片。
- **(OP2)** A、B 的 Â(1) 都 > 0。
- **(OP3)** A、B 的 pooled 曲线都没被截断（d ≤ 16 内确实跌破半高）。
- **(OP4)** 分组不重不漏：|A| + |B| == |MIX|（decontam 在两次 load 之间没有改变去留）。

稳健性（预注册为**次要**，不推翻主判据）：B 有 91% 来自单独一个包（Sharpik），
所以再算一次**只取 A、B 共有材质名**子集的 R_d，以区分"降采样"与"包风格"。

次要（描述性，**不作判据**）：L(MIX)、L16（`extra` 同训练口径）、R_mix = L(MIX)/L16
——这才是共用偏置表实际看到的两档之比。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import tiles_data  # noqa: E402
from scale_prior import corr_length, tile_curves  # noqa: E402

R_STAR = 0.5 ** 0.5
B_BOOT = 2000
MIN_TILES = 300
PACKS = "train_extra_packs_only.json"
DOWN = "train_extra_packs_only.json+train_64to32.json"


def key(x):
    return (x["pack"], x["material"], x["idx"].tobytes())


def boot_ratio(cur_num, cur_den, seed=0):
    """L(num)/L(den) 的自助分布（重采样瓦片）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B_BOOT):
        ln = corr_length(cur_num[rng.integers(0, len(cur_num), len(cur_num))].mean(0))
        ld = corr_length(cur_den[rng.integers(0, len(cur_den), len(cur_den))].mean(0))
        if ln and ld:
            out.append(ln / ld)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/scale_prior_down.json")
    a = ap.parse_args()

    keep = lambda s: [x for x in s if x["k_used"] >= 3]                       # noqa: E731
    A = keep(tiles_data.load(size=32, split="train", extra=PACKS))
    MIX = keep(tiles_data.load(size=32, split="train", extra=DOWN))
    ka = {key(x) for x in A}
    B = [x for x in MIX if key(x) not in ka]
    S16 = keep(tiles_data.load(size=16, split="train", extra=DOWN))
    print(f"A 原生 {len(A)}   B 派生 {len(B)}   MIX {len(MIX)}   16px {len(S16)}")

    cur = {"A": tile_curves(A, 32), "B": tile_curves(B, 32),
           "MIX": tile_curves(MIX, 32), "S16": tile_curves(S16, 16)}
    pooled = {k: v.mean(0) for k, v in cur.items()}
    L = {k: corr_length(v) for k, v in pooled.items()}

    ok1 = len(A) >= MIN_TILES and len(B) >= MIN_TILES
    ok2 = pooled["A"][0] > 0 and pooled["B"][0] > 0
    ok3 = L["A"] is not None and L["B"] is not None
    ok4 = len(A) + len(B) == len(MIX)
    print(f"(OP1) n>=300: {ok1}  (OP2) A_hat(1)>0: {ok2}  (OP3) 未截断: {ok3}  (OP4) 不重不漏: {ok4}")
    for k in ("A", "B", "MIX", "S16"):
        print(f"[{k}] A_hat(d) =", " ".join(f"{v:.3f}" for v in pooled[k][:16]))
        print(f"[{k}] L = {L[k]}")

    out = {"op1": bool(ok1), "op2": bool(ok2), "op3": bool(ok3), "op4": bool(ok4),
           "n": {k: len(v) for k, v in (("A", A), ("B", B), ("MIX", MIX), ("S16", S16))},
           "curve": {k: v.tolist() for k, v in pooled.items()},
           "L": {k: L[k] for k in L}, "R_star": R_STAR}

    if not (ok1 and ok2 and ok3 and ok4):
        out["verdict"] = "OP_FAIL"
        print("操作检验未过 -> 判决作废")
    else:
        R_d = L["B"] / L["A"]
        boots = boot_ratio(cur["B"], cur["A"])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "E1_half" if hi < R_STAR else ("E2_same" if lo > R_STAR else "E3_undecided")
        out.update({"R_d": float(R_d), "ci": [float(lo), float(hi)],
                    "n_boot": len(boots), "verdict": verdict})
        print(f"R_d = L(B)/L(A) = {R_d:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   R* = {R_STAR:.4f}")
        print(f"判决：{verdict}")

        both = {x["material"] for x in A} & {x["material"] for x in B}
        ia = [i for i, x in enumerate(A) if x["material"] in both]
        ib = [i for i, x in enumerate(B) if x["material"] in both]
        m = {"n_materials": len(both), "n": {"A": len(ia), "B": len(ib)}}
        if ia and ib:
            la, lb = corr_length(cur["A"][ia].mean(0)), corr_length(cur["B"][ib].mean(0))
            m["L"] = {"A": la, "B": lb}
            if la and lb:
                m["R_d"] = lb / la
                print(f"稳健性（同材质名 {len(both)} 个，A {len(ia)} / B {len(ib)}）：R_d = {lb / la:.3f}")
        out["matched"] = m

        if L["MIX"] and L["S16"]:
            out["R_mix"] = L["MIX"] / L["S16"]
            print(f"次要（描述性）R_mix = L(MIX)/L16 = {out['R_mix']:.3f}"
                  f"   [上一轮同口径 extra=False 是 1.011]")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
