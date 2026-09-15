#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M28) 预注册：(M27) 的"表是把手"只在**没训过**的 24px 上成立过 —— 在 32px 上呢？
零判官 / 零 API / 零训练 / 零配置改动；只多跑两次**推理**（同一个检查点，权重一个字不改）。

## 为什么必须先问这个

(M27)（`5ec5556` -> `4b6f7e4`，`analysis/arch/comb_lock.py`）判出 `D1_TABLE_BINDING`：模型在 24px
画布上的梳齿恰落在归一化表说的位置（C_24(6)=+0.0416 族级 [+0.0168,+0.0757]），像素梳齿分数贴 0。
但它**自己写死了限定语**：

> "这是**在一个未训练的画布上**的外推行为。⛔ 不许外推成 '32px 上表也是唯一把手' ——
>  16/32 上内容先验有训练信号，24 上没有。"

而 (M26)/(M27) 授权的唯一下一步——"替换/削弱归一化谐波支"的重训——整条理由链是
**"那张表在 32px 上把结构摆错了"**。这句话的前半截（表在 32px 上说了算）到今天为止
**只有在 24px 上的读数**。如果在 32px 上真正决定周期的是内容先验 / 调色板记忆库 / 范例，
那么换表不会改变 32px 的产物，**那次重训就不必花**。本轮就是重训之前的第二道决策门。

⚠ 跑前写明：本轮**不是** (M27) 的重复。(M27) 问"在没训过的画布上表说了算吗"（观察性、
靠 24 把两个假设错开）；本轮问"**在训过的画布上，改表会不会改产物**"（干预性）。

## 支点：16/32 上位置预测重合，所以只能用"干预"而不是"观察"

(M27) 第二节已经写死：H_canvas 与 H_cell 在 16px 与 32px 上预测**完全重合**（4=16x0.25、
8=32x0.25）——这正是当初必须绕到 24px 的原因。于是在 32px 上唯一能分开两者的办法是
**动那张表本身**，看产物跟不跟着动。

`ToroidalBias` 的**唯一**输入是 u = wrap(d)/n。本轮的干预 = 在 32px 画布上把它换成
u' = wrap(d)/16（`--bias_n 16`）：权重、画布尺寸、提示词、种子、调色板、范例**全部不变**，
只是"骗表说这块画布有 16 格"。

- 谐波按 u 是 **1 周期**的 -> u'=w/16 在 |w|>8 时自动折回，取值集合与表在 16px 上见过的
  **一模一样**（⛔ 没有任何外推）；特征仍只是**环绕偏移**的函数 -> 平移等变/可平铺不受影响。
  本机自检：override 后的特征逐位等于 16px 的表 2x2 平铺（见 (OP0)）。
- 16 | 32，所以折回点 w=+-16 处 u'=-+1，sin/cos 与 u'=0 处相接 -> 表在 32 环面上仍连续。
- ⛔ **反方向（n=16 骗成 m=32）不做**：那样 u'=w/32 只用到表的中间四分之一，峰会落在
  w=8 = n=16 的环面折回**单边点**（(M25) 第六节），本来就不进判据 -> 退化，无信息。
  同理 m=64 也退化。**m=16 是唯一干净的选择**，跑前写死。

两个假设在**同一块 32px 画布**上第一次错开：

| 假设 | 干预后梳齿应落在（n=32 的格子 d） | 理由 |
|---|---|---|
| **H_table**（表在把手）| **4**、12 | 表只认 u'；u'=0.25 -> w=4（w=12 同值，折回） |
| **H_content**（内容先验/调色板/范例在把手）| **8**、16（不动）| 表被骗了也没用，像素/内容尺度不变 |

d=8 在干预臂上是 u'=0.5 = 折回单边点，d=16 是 n=32 的折回单边点 -> **都不进主判据**
（只在 (D2) 的合取里用 d=8 的**绝对**值，见下）。

## 量什么（仪器逐字复用，不新造统计量）

`scale_prior.tile_curves`（超机遇同阶率 A_hat）、`period_scale.contrast`
（C(d) = A_hat(d) - [A_hat(d-1)+A_hat(d+1)]/2）、`gen_scale.png_rows`、
`judge_cluster.fam_drop_last` —— 四个一个字不改，与 (M25)/(M26)/(M27) 同一把尺子。

- **主统计量：ΔC(4) = C_over(4) - C_ctl(4)**，两臂**逐图配对**（同提示词、同种子、同调色板，
  文件名逐一对齐，代码里 assert）。
  ⚑ 为什么用配对差而不是 (M27) 的绝对 C：两臂在**同一个 d** 上比，(M27) 第五节点名的
  "跨 d 复合对比会把曲率差异折算成信号"这条**结构上不存在**——凸性对 d=4 的压低在两臂里
  是同一项，一阶抵消。⛔ 但不是零阶抵消（两臂图不同），所以 (D1) 仍加合取保护。
- **合取保护**：(D1) 另要求 **C_over(4) 的族级 CI 下界 > 0**。理由跑前写死、直接引 (M27)
  第五节实测的空总体表：纯衰减总体上 C_null(4) 在 r>=2 时**恒为负**（-0.0069..-0.0048）
  -> **正的绝对 C(4) 凸性造不出来**。
- **(D2) 的等价界**：ΔC(4) 的族级 CI **上界 < +0.0168**。这个数不是我挑的——它是 (M27)
  实测的 `D1_TABLE_BINDING` 效应量的族级 CI **下界**，即"表当把手"这件事**唯一一次被量到**
  的最弱形态。若连最乐观的读数都比它小，就说明表在 32px 上**没有**它在 24px 上那种把手作用。

## 预注册判据（跑之前写死，不许改）

重采样单位 = **提示词族**（`fam_drop_last` 去末词；(M17) 纪律，(M27) 照用）。B=2000，seed=0，
95% 百分位区间。逐图自助只作**下界**报告、⛔ 不进判据。

- **(D1) `TABLE_BINDS_32`**：ΔC(4) 族级 CI 下界 > 0 **且** C_over(4) 族级 CI 下界 > 0
  -> 在**训练过**的 32px 画布上，骗表就能把梳齿搬到表说的新位置 -> 表在 32px 上确实是把手
  -> "替换/削弱归一化谐波支"的重训**前提在 32px 上成立**，可按 (M26)/(M27) 的授权去预注册它。
  ⛔ 本身仍不授权任何配置改动/重训/新臂/判官调用。
- **(D2) `TABLE_WEAK_AT_32`**：ΔC(4) 族级 CI 上界 < +0.0168 **且** C_ctl(8) 的族级 CI 下界 > +0.0168
  **且** C_over(8) 的族级 CI 下界 > 0
  （= 老位置上本来就有一把够得着的梳齿、干预后它还在、而新位置上什么也没长出来）
  ⚑ **中间那一条是跑识别检验时补上的**（写在这里存档，改动发生在看任何真数据之前）：
  第一版 (D2) 只要求 C_over(8) > 0，结果 **(ID3) 纯衰减总体就能蒙混过关**——无结构数据上
  C(8) 也能有 +0.0004 的自助正下界。加上"控制臂本来就得有梳齿（>= (M27) 的 EQ_BOUND）"之后
  (ID3) 正确判回 D3。⚑ 这正是 (M27) 第五节那条教训的第二次兑现：**先在空总体上标定统计量，
  再决定判据长什么样**；⚠ 代价写在明处：若控制臂在 d=8 上本来就没有够得着的梳齿，本轮只能判 D3
  （"没有梳齿可搬"不等于"搬不动"）。
  -> 骗表没把梳齿搬走 -> 表在 32px 上**不是**周期的把手
  -> ⛔ "换表能修好 32px 的周期"这个**机制**判死，⛔ 不许再拿它当重训的理由。
  ⚠ 这**不**推翻 (M26)（数据侧的符号矛盾照旧成立）、**不**推翻 (M27)（那是 24px 的读数、
  两个总体），也**不**洗清 (M9)。
- **(D3)** 其余一律 `UNDECIDED`，不改任何东西，照实记账。⛔ 不许挑合上的那一半讲。

⚠ 跑前记账：**(D1) 不算意外**（(M27) 已在 24px 判过 D1，(M14) 的次要对比 B 也同向），
**(D2) 才是信息量大的那一边** —— 而 (D2) 恰好是**省钱**的那一边，所以 ⛔ 判据一个字不许放宽。

## 识别检验（(M26)/(M27) 都点名要求的那一条，跑前写死）

尺子必须**两个方向都报得出来**，否则任一结论都没有信息。同一条判决函数 `decide()`：

- **(ID1) 搬家**：合成 n=32 总体，"控制臂" = 周期 8 的块棋盘、"干预臂" = 周期 4
  -> 必须判出 **D1**（尺子看得见梳齿从 8 搬到 4）。
- **(ID2) 没搬**：控制臂与干预臂**都**是周期 8（只换翻面噪声的种子）
  -> 必须判出 **D2**（尺子报得出"什么也没动"）。
- **(ID3) 无结构**：两臂都是纯衰减（只有相关、没有周期）
  -> 必须判出 **D3**（尺子在没有梳齿的数据上不下判）。

任一不过 -> **全轮作废** `VOID_NO_IDENTIFICATION`，照实写"没测到"。

## 操作检验（任一不过 -> `OP_FAIL`，不下判）

- **(OP0)** 本机已自检（补丁当场跑的，记在账本）：`--bias_n` 不给时 `grid_offsets` 与打补丁前
  **逐位相同**（16/24/32 三档 max|Δ| = 0.0）；给 16 时 32px 的特征逐位等于 16px 表 2x2 平铺。
- **(OP1) 空操作臂**：再跑一次 `--bias_n 32`（= 把表的输入换成它本来的值）。产物 PNG 必须与
  控制臂**逐字节相同**。不同 -> 说明这个旋钮改的不只是除数 -> 作废。
  ⚑ 这一条同时是"尺子能报告没变化"的**精确**版本：ΔC 恒等于 0。
- **(OP2)** 两臂可用图（k_used>=3）各 >= 300 张，且**族集合相同**。
- **(OP3) 退化门（(M21) 教训：别用逐像素全等，要对着连续量画）**：干预臂"近乎纯色"的比例
  （最常见的秩占 >90% 格子）<= 0.10，且不超过控制臂的 3 倍。干预臂垮了 -> 位置读数无意义 -> 作废。
- **(OP4)** pooled A_hat(1) 两臂都 > 0。

## 稳健性（跑前定死，只给判决加后缀，不推翻主判决）

- **(R1) LOFO**：逐族去掉重算主统计量的符号，任一族能翻 -> `_FRAGILE`。
- **(R2) 非配对复核**：族级自助改成两臂**各自独立**重采样（不配对）后重判，判决若变
  -> `_PAIRING_SENSITIVE`，只能引方向。
- **(R3) 描述性（非判据）**：报两臂完整 C(d) 曲线与 d=12 的读数（H_table 预测 d=12 也该是峰）。

## 口径与跑前写明的混杂

1. 用 **v10**（`runs/trd_v10`，与 (M27) 同一个检查点，`sizes=[16,32]`、`bias_freqs=8`、无 `bias_cells`），
   E_mat 272 材质、每材质 2 张、**不重排**（(M27) 的 R1 已证去掉重排同判，且省一次 CLIP）。
2. 干预是**推理期**的，模型从没在"被骗的表"下训练过 -> 干预臂本身是**分布外**的。
   ⛔ 因此 (D1) 只能读成"表在 32px 上**有能力**支配周期"，⛔ 不许读成"32px 的毛病就是表造成的"；
   (D2) 只能读成"骗表搬不动梳齿"，⛔ 不许读成"表没参与"。**两边都必须带这个限定语。**
3. 一张**真人瓦片都不看**（连训练集都不看），只比模型自己的两臂 -> 与任何已发表胜率/分布指标无关。
4. PNG 还原秩网格会并掉亮度相同的颜色（`gen_scale.py` 混杂 2 照旧）。
5. ⛔ 本轮不碰 32px 准入条件（`6c64796`）①②③、不改 `final_test.sh`、不改任何默认值；
   `--bias_n` 默认 0 = 旧行为一个字不变。

用法：
    python analysis/arch/comb_shift.py --ctl_tag m28_ctl --over_tag m28_over \
        --noop_tag m28_noop --out /tmp/comb_shift.json
"""
import argparse
import hashlib
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

N = 32
D_MAIN = 4              # H_table 预测峰（u'=0.25）；H_content 预测这里什么都没有
D_STAY = 8              # 老位置（u=0.25），只在 (D2) 的合取里用绝对值
B_BOOT = 2000
SEED = 0
EQ_BOUND = 0.0168       # (M27) 实测 D1 效应量的族级 CI 下界 —— (D2) 的等价界，⛔ 不许改
MIN_TILES = 300
MAX_FLAT = 0.10
FLAT_SHARE = 0.90


def decide(d4_ci, c4_over_ci, c8_over_ci, c8_ctl_ci):
    """跑前写死的判决函数；ID1/ID2/ID3 走的是同一个函数。"""
    if d4_ci[0] > 0 and c4_over_ci[0] > 0:
        return "D1_TABLE_BINDS_32"
    if d4_ci[1] < EQ_BOUND and c8_ctl_ci[0] > EQ_BOUND and c8_over_ci[0] > 0:
        return "D2_TABLE_WEAK_AT_32"
    return "D3_UNDECIDED"


def family_of(path: Path) -> str:
    return fam_drop_last(re.sub(r"_\d+$", "", path.stem).replace("_", " "))


def flat_frac(rows):
    """近乎纯色的比例：最常见的秩占 >FLAT_SHARE 的格子（(OP3) 退化门）。"""
    if not rows:
        return 1.0
    n = 0
    for r in rows:
        g = np.asarray(r["idx"]).ravel()
        if np.bincount(g).max() > FLAT_SHARE * g.size:
            n += 1
    return n / len(rows)


def load_arm(root: Path, tag: str):
    files = sorted((root / tag / str(N)).glob("*.png"))
    rows = png_rows(root, tag, N)
    assert len(files) == len(rows), (tag, len(files), len(rows))
    return files, rows


def ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_boot(cc, co, fams, b=B_BOOT, seed=SEED, paired=True):
    """族级自助 -> (ΔC(4), C_over(4), C_over(8)) 各 b 个值。paired=False 时两臂独立重采样（R2）。"""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, f in enumerate(fams):
        by[f].append(i)
    keys = sorted(by)
    groups = [np.asarray(by[k]) for k in keys]
    d4, o4, o8, c8 = [], [], [], []
    for _ in range(b):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        sel2 = sel if paired else np.concatenate(
            [groups[j] for j in rng.integers(0, len(groups), len(groups))])
        mc, mo = cc[sel].mean(0), co[sel2].mean(0)
        d4.append(contrast(mo, D_MAIN, N) - contrast(mc, D_MAIN, N))
        o4.append(contrast(mo, D_MAIN, N))
        o8.append(contrast(mo, D_STAY, N))
        c8.append(contrast(mc, D_STAY, N))
    return np.asarray(d4), np.asarray(o4), np.asarray(o8), np.asarray(c8)


def report(name, cc, co, fams, out, key, paired=True):
    pc, po = cc.mean(0), co.mean(0)
    d4, o4, o8, k8 = paired_boot(cc, co, fams, paired=paired)
    rec = {"n_pairs": int(len(cc)), "n_families": len(set(fams)),
           "dC4": float(contrast(po, D_MAIN, N) - contrast(pc, D_MAIN, N)), "dC4_ci": ci(d4),
           "C4_ctl": float(contrast(pc, D_MAIN, N)), "C4_over": float(contrast(po, D_MAIN, N)),
           "C4_over_ci": ci(o4),
           "C8_ctl": float(contrast(pc, D_STAY, N)), "C8_over": float(contrast(po, D_STAY, N)),
           "C8_over_ci": ci(o8), "C8_ctl_ci": ci(k8),
           "C12_ctl": float(contrast(pc, 12, N)), "C12_over": float(contrast(po, 12, N)),
           "a_hat_1": [float(pc[0]), float(po[0])],
           "curve_ctl": pc.tolist(), "curve_over": po.tolist(),
           "verdict": decide(ci(d4), ci(o4), ci(o8), ci(k8))}
    out[key] = rec
    print("[%s] pairs=%d families=%d A_hat(1) ctl=%.4f over=%.4f"
          % (name, len(cc), rec["n_families"], pc[0], po[0]))
    print("  C(4)  ctl=%+.4f over=%+.4f  dC4=%+.4f family CI [%+.4f, %+.4f]"
          % (rec["C4_ctl"], rec["C4_over"], rec["dC4"], *rec["dC4_ci"]))
    print("  C_over(4) CI [%+.4f, %+.4f]   C(8) ctl=%+.4f CI [%+.4f, %+.4f] over=%+.4f CI [%+.4f, %+.4f]"
          % (*rec["C4_over_ci"], rec["C8_ctl"], *rec["C8_ctl_ci"], rec["C8_over"], *rec["C8_over_ci"]))
    print("  -> %s" % rec["verdict"])
    return rec


# ---------- 合成总体（识别检验） ----------
def checker(blk, n_tiles=400, size=N, flip=0.05, seed=11):
    """周期 2*blk 的块棋盘 + 5% 随机翻面（构造逐字照 comb_lock.synth 的 stripe 支）。"""
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    rows = []
    for _ in range(n_tiles):
        g = (((xs // blk) % 2) ^ ((ys // blk) % 2)).astype(np.int64)
        g = np.where(rng.random(g.shape) < flip, 1 - g, g)
        g = np.where(rng.random(g.shape) < 0.02, 2, g)
        rows.append({"idx": g, "k_used": int(g.max()) + 1})
    return tile_curves(rows, size)


def decay(n_tiles=400, size=N, r=2, k=4, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_tiles):
        z = rng.standard_normal((size, size))
        zz = np.zeros_like(z)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                zz += np.roll(np.roll(z, dy, 0), dx, 1)
        g = np.digitize(zz, np.quantile(zz, np.arange(1, k) / k)).astype(np.int64)
        rows.append({"idx": g, "k_used": int(g.max()) + 1})
    return tile_curves(rows, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="experiments/baselines")
    ap.add_argument("--ctl_tag", default="m28_ctl")
    ap.add_argument("--over_tag", default="m28_over")
    ap.add_argument("--noop_tag", default="m28_noop")
    ap.add_argument("--out", default="/tmp/comb_shift.json")
    a = ap.parse_args()
    root = Path(a.gen_root)
    out = {"n": N, "d_main": D_MAIN, "d_stay": D_STAY, "eq_bound": EQ_BOUND,
           "B": B_BOOT, "seed": SEED, "tags": [a.ctl_tag, a.over_tag, a.noop_tag]}

    # ---------- 识别检验（先跑：不过就不必看真数据） ----------
    ident = {}
    for name, cc, co, need in (
            ("ID1_moved", checker(4), checker(2), "D1_TABLE_BINDS_32"),
            ("ID2_stayed", checker(4), checker(4, seed=12), "D2_TABLE_WEAK_AT_32"),
            ("ID3_nostruct", decay(), decay(seed=12), "D3_UNDECIDED")):
        fams = ["f%d" % (i % 50) for i in range(len(cc))]
        rec = report("ID %s" % name, cc, co, fams, ident, name)
        rec["required"] = need
        rec["pass"] = rec["verdict"] == need
        print("  -> 要求 %s : %s" % (need, "PASS" if rec["pass"] else "FAIL"))
    out["identification"] = ident
    id_ok = all(ident[k]["pass"] for k in ident)

    # ---------- 两臂配对 ----------
    fc, rc = load_arm(root, a.ctl_tag)
    fo, ro = load_arm(root, a.over_tag)
    assert [f.name for f in fc] == [f.name for f in fo], "两臂文件名必须逐一对齐"
    keep = [i for i in range(len(rc)) if rc[i]["k_used"] >= 3 and ro[i]["k_used"] >= 3]
    cc = tile_curves([rc[i] for i in keep], N)
    co = tile_curves([ro[i] for i in keep], N)
    fams = [family_of(fc[i]) for i in keep]
    main_rec = report("main %s vs %s @32" % (a.ctl_tag, a.over_tag), cc, co, fams, out, "main")

    # ---------- 操作检验 ----------
    op1 = None
    noop_dir = root / a.noop_tag / str(N)
    if noop_dir.exists():
        same, diff = 0, []
        for f in sorted(noop_dir.glob("*.png")):
            g = root / a.ctl_tag / str(N) / f.name
            if not g.exists():
                continue
            if hashlib.sha256(f.read_bytes()).hexdigest() == hashlib.sha256(g.read_bytes()).hexdigest():
                same += 1
            else:
                diff.append(f.name)
        op1 = {"compared": same + len(diff), "identical": same, "differing": diff[:5]}
        print("[OP1] 空操作臂 --bias_n 32：%d/%d 逐字节相同" % (same, same + len(diff)))
    op1_ok = bool(op1 and op1["compared"] > 0 and not op1["differing"])
    ff_c, ff_o = flat_frac([rc[i] for i in keep]), flat_frac([ro[i] for i in keep])
    op2 = len(keep) >= MIN_TILES
    op3 = ff_o <= MAX_FLAT and ff_o <= 3 * max(ff_c, 1e-6)
    op4 = bool(cc.mean(0)[0] > 0 and co.mean(0)[0] > 0)
    out["op"] = {"OP1_noop_identical": op1_ok, "OP1_detail": op1, "OP2_pairs": int(len(keep)),
                 "OP2": bool(op2), "OP3": bool(op3), "flat_ctl": ff_c, "flat_over": ff_o,
                 "OP4": op4, "n_all": int(len(rc))}
    print("[OP] OP1=%s OP2=%s(%d 对) OP3=%s(近纯色 ctl %.3f over %.3f) OP4=%s"
          % (op1_ok, op2, len(keep), op3, ff_c, ff_o, op4))

    verdict = main_rec["verdict"]
    if not id_ok:
        verdict = "VOID_NO_IDENTIFICATION"
    elif not (op1_ok and op2 and op3 and op4):
        verdict = "OP_FAIL"

    # ---------- (R1) LOFO ----------
    if verdict.startswith(("D1", "D2")):
        by = defaultdict(list)
        for i, f in enumerate(fams):
            by[f].append(i)
        base = main_rec["dC4"]
        flips = []
        for k in sorted(by):
            drop = set(by[k])
            sel = np.asarray([i for i in range(len(cc)) if i not in drop])
            v = contrast(co[sel].mean(0), D_MAIN, N) - contrast(cc[sel].mean(0), D_MAIN, N)
            if np.sign(v) != np.sign(base):
                flips.append(k)
        out["lofo"] = {"n_families": len(by), "flips": flips}
        print("[R1] LOFO 翻侧族数 = %d / %d" % (len(flips), len(by)))
        if flips:
            verdict += "_FRAGILE"

    # ---------- (R2) 非配对复核 ----------
    if verdict.startswith(("D1", "D2")):
        r2 = report("R2 unpaired", cc, co, fams, out, "R2_unpaired", paired=False)
        if not r2["verdict"].startswith(verdict.split("_")[0]):
            verdict += "_PAIRING_SENSITIVE"

    out["verdict"] = verdict
    print("==> %s" % verdict)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written %s" % a.out)


if __name__ == "__main__":
    main()
