#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M27) 预注册：那张共用的归一化表，**真的在控制产物的周期**吗？零 GPU / 零 API / 零判官 / 零训练。

## 为什么问这个

(M26)（`a8ced52` -> `5f1f4e3`，`analysis/arch/table_conflict.py`）证明了一件**关于数据**的事：
同一个归一化输入 u=0.375 上，真人 16px 要谷（C=-0.0303 CI [-0.0494,-0.0155]）、
真人 32px 要峰（+0.0111 CI [+0.0049,+0.0156]）→ 一张只吃 u=d/n 的表在**符号层面**装不下两档。
它授权的唯一下一步是"**另行预注册『替换/削弱归一化谐波支』，且那份预注册必须自带识别检验**"。

但在花掉一次重训之前，还有一句同样**从没量过的隐含前提**：

> 那张表**在控制产物的周期**。

它不是自动成立的。`ToroidalBias` 只是注意力偏置的一项，网络还有内容项、CLIP 文本条件、
调色板记忆库（16px 消融里唯一承重的部件，`3258294`）。(M13)（`4fa50da`）给偏置表**补**格子特征
后三档 14 个对比全在噪声下限内 —— 当时留下的两种解释里，"(a) 偏置表根本不是那个把手"
**至今没有被直接检验过**。(M14)（`be122c4`）问过"先验有没有落到产物上"，主判据 A 判 `G3` 未判定。

**如果表不是把手，替换它就不该指望改变产物的周期 —— 那次重训不必花。**
本轮就是那次重训之前的**决策门**，而且它零成本。

## 24px 是唯一能把两个假设分开的画布（这才是本轮的支点）

v10 的训练尺寸是 `sizes=[16, 32]`（已核 `ckpt["args"]`），**24px 一张都没训过**；
`bias_freqs=8`、无 `bias_cells`（`bias.mlp.0.weight` 输入维 32 = 4x8）→ 它的偏置表是
**纯归一化**的 8 次谐波。于是模型在 24px 上画出来的周期结构，是"表把 u 上学到的形状搬到一个
没见过的画布上"与"内容先验（真人纹理的砖缝就是 4 像素一道）"两股力量的竞争 ——
而 **24 恰好把两者的预测错开**：

| 假设 | 峰应落在（n=24 的格子 d） | 理由 |
|---|---|---|
| **H_canvas**（表在控制）| **6**, 12 | 表只认 u；两档真人的主峰都在 u=0.25（(M26) 表格）→ 24x0.25=6 |
| **H_cell**（固定 4 像素梳齿）| **4**, **8**, 12 | (M25)/(M26)：16px 峰在 d=4/8、32px 峰全在 4 的倍数 |

16px 与 32px 上这两个假设**完全重合**（4 = 16x0.25、8 = 32x0.25）——这正是为什么必须用 24px，
也正是为什么这个读数至今没人做过。d=12 两个假设都预测是峰，且是 n=24 的环面折回**单边点**
（(M25) 第六节）-> ⛔ 不进任何判据。

## 量什么（仪器逐字复用，不新造统计量）

`scale_prior.tile_curves`（秩网格的超机遇同阶率 A_hat）、`period_scale.contrast`
（局部对比 C(d) = A_hat(d) - [A_hat(d-1)+A_hat(d+1)]/2）、`gen_scale.png_rows`
（生成 PNG -> 秩网格，按亮度升序编号）、`judge_cluster.fam_drop_last`（族定义）——四个都一个字不改。

- **主统计量：C_24(6)** —— H_canvas -> 峰 (>0)；H_cell -> d=6 正落在 4 格梳齿的两齿之间 = 谷 (<0)。
  与 (M25) 的 D_odd 同一个套路：**单个 d 上两个假设符号相反**，不需要跨 d 比较。
- **像素梳齿分数：P = [C_24(4) + C_24(8)] / 2** —— 只在 (D2) 的合取里出现（见下）。

**为什么不用"C(6) 减去 C(4)、C(8)"那种跨 d 的复合量**（跑前写死）：C 会被曲线**凸性**污染，
A_hat 越陡的地方被压得越负，而 d=4 比 d=6 陡得多。本文件先在**结构为零的合成衰减总体**上
量过这件事（400 张/档，箱式模糊半径 r=1..6）：

    r      1        2        3        4        5        6
    C(4) +0.0012  -0.0069  -0.0059  -0.0063  -0.0053  -0.0048
    C(6) +0.0005  +0.0004  -0.0027  -0.0024  -0.0029  -0.0025
    C(8) +0.0005  +0.0017  +0.0003  -0.0015  -0.0006  -0.0012

复合量 C(6)-[C(4)+C(8)]/2 在纯衰减上有 **+0.005 的系统正偏**（正好偏向本轮**预期**的那一边）
-> ⛔ 弃用。**单个 C(6) 的空值在 [-0.0029, +0.0005]**：几乎全是负的
-> 对 (D1) 是**保守**的（凸性只会把峰压没，造不出峰），对 (D2) 才需要保护 -> 见 (D2) 的合取。

## 预注册判据（跑之前写死，不许改）

重采样单位 = **提示词族**（(M17) 纪律推广到生成图：同族材质的提示词高度相关；
族 = `fam_drop_last` 去末词，更细 = 更保守）。B=2000，seed=0，95% 百分位区间。
**逐瓦片自助只作下界报告，⛔ 不进判据。**

- **(D1) `TABLE_BINDING`**：C_24(6) 的族级 CI 下界 > 0
  -> 在一个**没训过**的画布上，产物的梳齿仍落在**表说的位置**（u=0.25），而不是像素位置
  -> 那张表**确实是周期的把手**，(M26) 量到的符号矛盾因此是**有后果的**。
  授权：另行预注册"替换/削弱归一化谐波支"的**重训**，并可在那份预注册里把"24px 梳齿位置"
  当作**便宜的前置读数**（改完表后梳齿应当移到 4/8）。⛔ 本身仍不授权任何配置改动/重训/新臂。
- **(D2) `PIXEL_LOCKED`**：C_24(6) 的族级 CI 上界 < 0 **且** P 的族级 CI 下界 > 0
  （合取的理由跑前写死：凸性只能把 C 压负，**造不出**正的像素梳齿分数——上表里 C_null(4) 恒为负
  -> 正的 P 不可能是凸性伪造的）
  -> 产物的梳齿是**固定像素**的，尽管表只看 u -> 周期这一面**不是**那张表定的
  -> ⛔ "替换归一化谐波支能修好产物周期"这个**机制**判死，不许再拿它当重训的理由。
  ⚠ 这**不**推翻 (M26)（那是关于数据的读数，照旧成立），也**不**洗清 (M9)（衰减/局部性，另一回事）。
- **(D3)** 其余一律 `UNDECIDED`，不改任何东西，照实记账。⛔ 不许挑合上的那一半讲。

## 识别检验（(M26) 授权里点名要求的那一条，跑前写死）

同一条代码路径、同样的 `tile_curves`/`contrast`/判决函数，跑三个合成 24px 总体（seed 固定）：

- **(ID1) `stripe6`**：3 格块棋盘（周期 6，5% 随机翻面）-> 必须判出 **D1**
- **(ID2) `stripe4`**：2 格块棋盘（周期 4，5% 随机翻面）-> 必须判出 **D2**
- **(ID3) `decay`**：白噪声箱式模糊后按分位量化（**只有衰减、没有周期**），
  模糊半径按**观测数据自己的 A_hat(1)、A_hat(2)** 在 r=1..6 上匹配 -> 必须判出 **D3**

(ID1)(ID2) 任一不过 -> 尺子分不开两个假设；(ID3) 不过 -> 尺子在无结构数据上就会下判
-> 两种情况都是**全轮作废**，照实写"没测到"。

## 操作检验

- **(OP1)** k_used >= 3 的 24px 生成图 >= 100 张（共 272 张）。
- **(OP2)** pooled A_hat_24(1) > 0（尺子活着）。
- **(OP3)** 有效自助样本 >= 1900。
- **(OP4)** 描述性记号（**不是否决门**）：C_24(4)/C_24(6)/C_24(8) 至少一个族级 CI 下界 > 0，
  否则记 `no_comb_24 = true`（三处都没峰 -> 判决多半是 D3，照实写）。

## 稳健性（跑前定死，不推翻主判决，只给判决加后缀）

- **(R1) 去掉重排**：同一统计量跑未重排的 `TRD24`（1088 张 = 每材质 4 张）。rr4 是 CLIP 四选一，
  可能挑走某种结构 -> 若 R1 的 C_24(6) 与主判决**符号相反** -> 加 `_RERANK_FRAGILE`，只能引方向。
- **(R2) LOFO**：逐族去掉重算主统计量的符号，任一族能翻 -> 加 `_FRAGILE`。
- **(R3) 扣空偏置**：把匹配半径的 `decay` 总体的 C_null(d) 当固定偏置扣掉后重判，
  判决若改变 -> 加 `_NULL_SENSITIVE`，只能引方向。

## 口径与跑前就写明的混杂

1. **24px 从未被训练**（`sizes=[16,32]`）。这是本轮的**支点**也是**最大局限**：未训练画布上的
   外推行为未必等同于模型在 16/32 上的内部机制。⛔ 结论一律带"在一个未训练的画布上"这个限定语，
   不许外推成"32px 上表也不是把手"。
2. **(M14) 的次要对比 B**（同为 v10 的 24 vs 32，`experiments/gen_scale.json`）判过 `G1_canvas`
   （Δ=-0.0143 CI [-0.0167,-0.0066]）-> 方向上**已经有一条指向 (D1) 的旁证**，本轮不是盲猜；
   ⚠ 跑前记账：**(D1) 不算意外，(D2) 才是信息量大的那一边**。但那是**整条曲线形状**
   （被衰减主导）的读数，本轮量的是**梳齿位置**，两个统计量。
   ⛔ 无论哪边，都不许拿 (M14) 的 B 来替本轮下判（(M14) 第五节禁令照旧）。
3. PNG 还原秩网格会并掉亮度相同的颜色（`gen_scale.py` 混杂 2 照旧）。
4. **一张测试集真人瓦片都不看**；只读已生成的 PNG 与合成对照，不调判官、不出胜率、不碰配置。

用法（PNG 在远程 `/mnt/data/kw/RoundSquisheen/texture/experiments/baselines/`）：
    python analysis/arch/comb_lock.py --out /tmp/comb_lock.json
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_scale import png_rows                     # noqa: E402  PNG -> 秩网格，逐字复用
from period_scale import contrast                  # noqa: E402  局部对比 C(d)，逐字复用
from scale_prior import tile_curves                # noqa: E402  A_hat 曲线，逐字复用
from judge_cluster import fam_drop_last            # noqa: E402  族定义，逐字复用

B_BOOT = 2000
SEED = 0
MIN_TILES = 100
MIN_BOOT = 1900
N = 24
D_MAIN = 6              # H_canvas 预测峰、H_cell 预测谷
D_PIX = (4, 8)          # H_cell 预测峰（只在 (D2) 的合取里用）
R_GRID = (1, 2, 3, 4, 5, 6)


def c_main(curve):
    return contrast(curve, D_MAIN, N)


def c_pix(curve):
    return float(np.mean([contrast(curve, d, N) for d in D_PIX]))


def decide(ci6, cip):
    """跑前写死的判决函数；ID1/ID2/ID3 走的是同一个函数。"""
    if ci6[0] > 0:
        return "D1_TABLE_BINDING"
    if ci6[1] < 0 and cip[0] > 0:
        return "D2_PIXEL_LOCKED"
    return "D3_UNDECIDED"


def family_of(path: Path) -> str:
    """baked_clay_black_0.png -> 提示词 'baked clay black' -> 去末词族 'baked clay'。"""
    return fam_drop_last(re.sub(r"_\d+$", "", path.stem).replace("_", " "))


def load_gen(root: Path, tag: str, size: int):
    """返回 (curves[T, size//2], families[T], 总张数)。两处都是 sorted -> 顺序一致。"""
    files = sorted((root / tag / str(size)).glob("*.png"))
    rows = png_rows(root, tag, size)
    assert len(files) == len(rows), (len(files), len(rows))
    keep = [i for i, r in enumerate(rows) if r["k_used"] >= 3]
    return tile_curves([rows[i] for i in keep], size), [family_of(files[i]) for i in keep], len(rows)


def boot(curves, fams=None, b=B_BOOT, seed=SEED):
    """族级（fams 给定）或逐瓦片自助 -> (C(6) 的 b 个值, P 的 b 个值)。"""
    rng = np.random.default_rng(seed)
    if fams is None:
        groups = [np.asarray([i]) for i in range(len(curves))]
    else:
        by = defaultdict(list)
        for i, f in enumerate(fams):
            by[f].append(i)
        groups = [np.asarray(by[k]) for k in sorted(by)]
    s6, sp = [], []
    for _ in range(b):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        c = curves[sel].mean(0)
        s6.append(c_main(c))
        sp.append(c_pix(c))
    return np.asarray(s6), np.asarray(sp)


def ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------- 合成总体（识别检验 / 空偏置） ----------
def synth(kind, r=2, n_tiles=400, size=N, flip=0.05, k=4, seed=11):
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    rows = []
    for _ in range(n_tiles):
        if kind in ("stripe6", "stripe4"):
            blk = 3 if kind == "stripe6" else 2
            g = (((xs // blk) % 2) ^ ((ys // blk) % 2)).astype(np.int64)
            g = np.where(rng.random(g.shape) < flip, 1 - g, g)
            g = np.where(rng.random(g.shape) < 0.02, 2, g)      # 少量第三阶 -> k_used>=3
        elif kind == "decay":
            z = rng.standard_normal((size, size))
            zz = np.zeros_like(z)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    zz += np.roll(np.roll(z, dy, 0), dx, 1)     # 箱式模糊（环面卷积）
            g = np.digitize(zz, np.quantile(zz, np.arange(1, k) / k)).astype(np.int64)
        else:
            raise ValueError(kind)
        rows.append({"idx": g, "k_used": int(g.max()) + 1})
    return tile_curves([r_ for r_ in rows if r_["k_used"] >= 3], size)


def match_radius(target):
    """按 A_hat(1)、A_hat(2) 在 R_GRID 上给观测曲线匹配一个纯衰减总体。"""
    best, curves = None, {}
    for r in R_GRID:
        cur = synth("decay", r=r)
        curves[r] = cur
        err = float(sum((cur.mean(0)[j] - target[j]) ** 2 for j in (0, 1)))
        if best is None or err < best[1]:
            best = (r, err)
    return best[0], best[1], curves[best[0]]


def report(name, curves, fams, out, key):
    pooled = curves.mean(0)
    b6, bp = boot(curves, fams)
    ci6, cip = ci(b6), ci(bp)
    t6, _ = boot(curves, None)
    rec = {"n_tiles": int(len(curves)), "n_families": (len(set(fams)) if fams else None),
           "a_hat_1": float(pooled[0]), "C6": float(c_main(pooled)), "ci6_family": ci6,
           "ci6_tile": ci(t6), "P": c_pix(pooled), "cip_family": cip,
           "contrasts": {str(d): float(contrast(pooled, d, N)) for d in (2, 4, 6, 8, 10)},
           "curve": pooled.tolist(), "verdict": decide(ci6, cip)}
    out[key] = rec
    print("[%s] tiles=%d families=%s A_hat(1)=%.4f" % (name, len(curves), rec["n_families"], pooled[0]))
    print("  C_24(6)=%+.4f family CI [%+.4f, %+.4f] (tile CI [%+.4f, %+.4f], 仅下界)"
          % (rec["C6"], *ci6, *rec["ci6_tile"]))
    print("  P=[C(4)+C(8)]/2=%+.4f family CI [%+.4f, %+.4f]  -> %s" % (rec["P"], *cip, rec["verdict"]))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="experiments/baselines")
    ap.add_argument("--tag", default="TRD24_rr4")
    ap.add_argument("--r1_tag", default="TRD24")
    ap.add_argument("--out", default="/tmp/comb_lock.json")
    a = ap.parse_args()
    root = Path(a.gen_root)
    out = {"n": N, "d_main": D_MAIN, "d_pix": list(D_PIX), "B": B_BOOT, "seed": SEED}

    # ---------- 主判据 ----------
    curves, fams, n_all = load_gen(root, a.tag, N)
    main_rec = report("main %s@24" % a.tag, curves, fams, out, "main")
    main_rec["n_all"] = int(n_all)
    pooled = curves.mean(0)

    # ---------- 识别检验（含按数据匹配的空偏置总体） ----------
    r_hat, err, cur_decay = match_radius(pooled)
    print("[ID] 匹配衰减半径 r=%d (A_hat(1,2) 平方误差 %.5f)" % (r_hat, err))
    ident = {"matched_r": r_hat, "match_err": err}
    for kind, cur, need in (("stripe6", synth("stripe6"), "D1_TABLE_BINDING"),
                            ("stripe4", synth("stripe4"), "D2_PIXEL_LOCKED"),
                            ("decay", cur_decay, "D3_UNDECIDED")):
        rec = report("ID %s" % kind, cur, None, ident, kind)
        rec["required"] = need
        rec["pass"] = rec["verdict"] == need
        print("  -> 要求 %s : %s" % (need, "PASS" if rec["pass"] else "FAIL"))
    out["identification"] = ident
    id_ok = all(ident[k]["pass"] for k in ("stripe6", "stripe4", "decay"))

    # ---------- 操作检验 ----------
    op1 = len(curves) >= MIN_TILES
    op2 = bool(pooled[0] > 0)
    op3 = B_BOOT >= MIN_BOOT
    # OP4（描述性，不是否决门）：C(6) 或像素梳齿分数 P 至少一个族级 CI 下界 > 0
    op4 = (main_rec["ci6_family"][0] > 0) or (main_rec["cip_family"][0] > 0)
    main_rec["op"] = {"OP1": bool(op1), "OP2": op2, "OP3": bool(op3), "no_comb_24": not op4}
    print("[OP] OP1=%s OP2=%s OP3=%s no_comb_24=%s" % (op1, op2, op3, not op4))

    verdict = main_rec["verdict"]
    if not id_ok:
        verdict = "VOID_NO_IDENTIFICATION"
    elif not (op1 and op2 and op3):
        verdict = "OP_FAIL"

    # ---------- (R1) 去掉重排 ----------
    if verdict.startswith("D"):
        try:
            c1, f1, _ = load_gen(root, a.r1_tag, N)
            r1 = report("R1 %s@24" % a.r1_tag, c1, f1, out, "R1_norerank")
            if verdict.startswith("D1") and r1["C6"] < 0 or verdict.startswith("D2") and r1["C6"] > 0:
                verdict += "_RERANK_FRAGILE"
        except (AssertionError, OSError) as e:
            out["R1_norerank"] = {"error": str(e)}
            print("[R1] 不可用：%s" % e)

    # ---------- (R2) LOFO ----------
    if verdict.startswith(("D1", "D2")):
        sign = 1.0 if verdict.startswith("D1") else -1.0
        by = defaultdict(list)
        for i, f in enumerate(fams):
            by[f].append(i)
        flips = []
        for k in sorted(by):
            drop = set(by[k])
            sel = np.asarray([i for i in range(len(curves)) if i not in drop])
            if sign * c_main(curves[sel].mean(0)) <= 0:
                flips.append(k)
        out["lofo"] = {"n_families": len(by), "flips": flips}
        print("[R2] LOFO 翻侧族数 = %d / %d" % (len(flips), len(by)))
        if flips:
            verdict += "_FRAGILE"

    # ---------- (R3) 扣空偏置 ----------
    if verdict.startswith(("D1", "D2")):
        nul = cur_decay.mean(0)
        off6, offp = c_main(nul), c_pix(nul)
        b6, bp = boot(curves, fams)
        v3 = decide(ci(b6 - off6), ci(bp - offp))
        out["R3_null_offset"] = {"r": r_hat, "C6_null": float(off6), "P_null": float(offp),
                                 "verdict_after": v3}
        print("[R3] 空偏置 C6=%+.4f P=%+.4f -> 扣除后判决 %s" % (off6, offp, v3))
        if not v3.startswith(verdict.split("_")[0]):
            verdict += "_NULL_SENSITIVE"

    out["verdict"] = verdict
    print("==> %s" % verdict)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written %s" % a.out)


if __name__ == "__main__":
    main()
