#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M24) 判出率 R 的跨臂全表：那 23pp 到底算不算一个需要机制去解释的现象
（预注册：本文件连同判据先提交，再跑）

## 为什么问这个

`6c64796` 立下的 32px 准入条件第 4 条写的是：

> ⚠ 理由已从"等 e4"换成"**尺子本身没校准**"：四边的环不闭合（S2 z=+2.68 p=0.0074），
> 同一台判官在四条边上判出率能差 23pp。**先弄清判出率为何差 23pp，不是开杠杆。**

这条件已经挡了好几轮。这条线上两个机制（刺激差异 `830b2d2`、位置分解 `367f841`）
**都已判死、闭合存档**，账本同时写死"⛔ 硬凑第三个机制＝p-hacking"。

但有一件从没做过的事：**那 23pp 是拿环上的两条边（e3 50.7% vs b32 74.3%）算出来的，
而这台仪器已经跑过 28 条 full 臂，R 的全表一次都没被整体看过。**
在问"为什么差 23pp"之前，先得知道**臂间差 23pp 在这台仪器上是不是家常便饭**。
如果是，那么要解释的就不是"某条边坏了"，而是"R 是逐比较的属性"——
这是对③**前提**的检验，不是给它硬凑第三个机制。

**零 GPU、零 API、零判官调用、零新臂、零训练、零配置改动、零重跑。**
只读已落盘的 `experiments/judge_full_*.json`。
`eval/b3_32.sh` 正在 GPU 6/7 上跑 B3@32（本轮开工已核活性：9/12 张齐、两支都活），
**本文件碰都不碰它**。

    python analysis/arch/resolve_spread.py

## 臂的名单不是我挑的

`ARMS = recheck_judge.EXPECT` —— 直接 import 账本里那 28 条**已登记**的 full 臂，
本文件**不增不减一条**（与 `judge_cluster_sweep.py` 同款纪律）。
`experiments/` 里还躺着若干未登记的 full JSON，**一概不取**。

## 口径（跑前写死）

- 每条臂 `R = decided / (decided + inconsistent)` —— 就是 `eval/judge_pairs.py:183`
  落盘的 `resolve_rate`（api_fail 已排除在分母外）。
- 每条臂 `W = a_wins / decided`（A 胜率，方向不翻）。
- **主分析集**＝ `n_ask = decided + inconsistent >= 30` 的臂。
  理由跑前写死：n=12 时 R 的二项 SE 约 14pp，**比要解释的 23pp 还大**，
  拿它进离散度检验是拿噪声当信号。被排除的是 4 条 B3 子集臂（n=10~12）。
  次要分析（含全部 28 条）**只报不判**。

## 操作检验（先跑，不过就整轮作废）

- **(OP1) 复现**：e3 = `judge_full_B2_vs_B2up16_32` 的 R 必须 = 0.507，
  b32 = `judge_full_TRD32_rr4_vs_B2_32` 的 R 必须 = 0.743（各 +-0.001，账本原值）。
  且两者之差必须 >= 0.23（这就是那 23pp 的出处）。不满足 -> `REPRO_FAIL`，本轮作废。
- **(OP2) 样本量**：主分析臂数 >= 20。
- **(OP3) 尺子非平凡**：主分析集里 R 的极差 > 0（全一样就没什么可测）。

## 主判据一 (Q1)：R 是不是"逐臂的真实属性"

H0：所有臂共享同一个 R，臂间差异全是二项抽样噪声。
统计量 `chi2 = sum_i (k_i - n_i*Rbar)^2 / (n_i*Rbar*(1-Rbar))`，`Rbar = sum k / sum n`。
**p 由参数自助算**（B=20000、seed=0，按各臂 n_i 从 Binom(n_i, Rbar) 重采样），
不用 chi2 分布（本机/远程都没 scipy，且不想依赖渐近）。

- **`R_HETERO`**：p < 0.01 -> R 是逐比较的真实属性，臂间差不是仪器噪声。
- **`R_HOMO`**：p >= 0.01 -> 全表与"单一 R + 二项噪声"相容，则 e3 掉到 50.7% 确实反常。

## 主判据二 (Q1b)：23pp 在这张表里有多常见

只在 `R_HETERO` 成立时读。把主分析集的臂**两两枚举**（精确，不抽样），
算 `g = #{|R_i - R_j| >= 0.23} / C(m,2)`。

- **`GAP_TYPICAL`**：g >= 0.10 -> 23pp 是这台仪器的**常见**臂间差
  -> 准入条件③的**前提**（"23pp 是个需要机制解释的异常"）**不成立**。
- **`GAP_RARE`**：g < 0.10 -> 23pp 确实罕见，③ 的前提保留，该线继续挂着。

阈值 0.23 不是本轮挑的：它是账本 `6c64796` 自己写下的那个数（0.743 - 0.507 = 0.236）。
0.10 这道线也跑前写死，不许事后改。

## 主判据三 (Q2)：把"判官噪声"那条提案一次做掉

账本里挂着的下一候选（`docs/arch_progress.md:3784`，**提案、非结果**）原话是：

> 判官本身有噪声——已知 57 道重复题里 14 道翻面（24.6%）。看内容的对被噪声打翻也会掉进
> inconsistent：偏爱 X 的对翻面->both-first，偏爱 Y 的->both-second，比例随该臂胜率走。

把它写成可证伪的形式：噪声把"看内容的对"打进 inconsistent 的概率，随该臂真效应变大而变小
-> **R 应随 |W - 0.5| 单调上升**。28 条臂的 W 从 0.26 铺到 1.00，正好是检验它的样本。

Spearman rho（自己实现，并列取平均秩），p 由置换检验（B=20000、seed=1）。

- **`NOISE_SUPPORTED`**：rho > 0 且 p < 0.05。
- **`NOISE_FALSIFIED`**：rho < 0 且 p < 0.05。
- **`NOISE_UNDECIDED`**：p >= 0.05。
  ⛔ 此时**不许**读成"提案被否"（(M13) 那条纪律：什么也没测到 != 更差）。
- 稳健性：LOAO（逐个去掉一条臂重算）rho 若有翻号，判决加 `_FRAGILE`。

## 判决之后能说什么、不能说什么（跑前写死）

**⛔ 一律不能说**：
- ⛔ 本轮**不授权**任何配置、默认值、新臂、判官调用、训练。
- ⛔ 无论判成什么，准入条件 ①（三档不单独构成证据）与 ②（不开 32px 新臂）**一个字不动**。
  本轮**只可能**动③的前提。
- ⛔ `GAP_TYPICAL` **不等于**"环闭合了"、**不等于**"32px 没问题"。
  环的判决（`6c64796`：S2 z=+2.68 证伪）本轮**不重判**，也**不许**引本轮结果去重判它。
- ⛔ g 不许读成"环那四条边之间的差是随机的"——本轮量的是**臂间差的分布**，
  不是任何一对特定边的因果。
- ⛔ (Q2) 的 rho 是**臂级相关**，28 个点里有大量共用图集的臂（所有 TRD32 臂共用同一批图），
  **不是独立样本** -> rho 只能当描述+方向，⛔ 不许把 p 当作强证据引。

**能说（若判成）**：
- `R_HETERO` + `GAP_TYPICAL`：③ 的前提被它自己的预注册判据否掉 ->
  "判出率为何差 23pp"这个问法本身要重写成"R 为什么是逐比较的"。**这不解锁任何东西**，
  只是把一道挡路的**伪问题**摘掉，并把该写进账本的事实写进去。
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "arch"))
from recheck_judge import EXPECT                      # noqa: E402  臂的名单：账本里那 28 条

EXP = ROOT / "experiments"
MIN_N = 30          # 主分析集门槛（跑前写死）
GAP = 0.23          # 要解释的那个差（6c64796 自己写下的数）
G_LINE = 0.10       # GAP_TYPICAL 的线
B_BOOT = 20000
E3 = "judge_full_B2_vs_B2up16_32.json"
B32 = "judge_full_TRD32_rr4_vs_B2_32.json"
# 环的四条边（`6c64796` / analysis/arch/tier_cycle.py），只作描述性标注，不进判据
RING = {"judge_full_TRD16c_rr4_vs_B2_16.json": "e1 TRD16 vs B2@16 (same canvas)",
        B32: "e2/b32 TRD32 vs B2@32 (same canvas)",
        E3: "e3 B2@32 vs B2@16up (same method)",
        "judge_full_TRD32_rr4_vs_TRD16cup_32.json": "e4 TRD32 vs TRD16up (same method)"}


def load_arms():
    arms = []
    for fn in sorted(EXPECT):
        p = EXP / fn
        if not p.exists():
            raise SystemExit(f"缺文件 {fn}：账本登记了但 experiments/ 里没有")
        d = json.loads(p.read_text(encoding="utf-8"))
        k, inc = d["decided"], d["inconsistent"]
        n = k + inc
        arms.append({"file": fn, "tag": d["tag"], "k": k, "n": n,
                     "R": k / n, "W": d["a_wins"] / k if k else float("nan"),
                     "api_fail": d.get("api_fail", 0)})
    return arms


def rankdata(x):
    """平均秩（处理并列），不依赖 scipy。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else float("nan")


def chi2_stat(k, n, rbar):
    k, n = np.asarray(k, float), np.asarray(n, float)
    return float((((k - n * rbar) ** 2) / (n * rbar * (1 - rbar))).sum())


def main():
    arms = load_arms()
    by_file = {a["file"]: a for a in arms}
    out = {"n_arms_all": len(arms)}

    # ---------------- 操作检验 ----------------
    r_e3, r_b32 = by_file[E3]["R"], by_file[B32]["R"]
    op1 = abs(r_e3 - 0.507) <= 0.001 and abs(r_b32 - 0.743) <= 0.001 and (r_b32 - r_e3) >= GAP
    main_arms = [a for a in arms if a["n"] >= MIN_N]
    op2 = len(main_arms) >= 20
    Rs = np.array([a["R"] for a in main_arms])
    op3 = bool(Rs.max() - Rs.min() > 0)
    out["op"] = {"OP1_repro": bool(op1), "R_e3": r_e3, "R_b32": r_b32, "gap_obs": r_b32 - r_e3,
                 "OP2_n_main": len(main_arms), "OP2": bool(op2), "OP3": op3}
    print("=" * 96)
    print("(M24) 判出率 R 的跨臂全表 —— 23pp 算不算需要机制解释的现象")
    print("=" * 96)
    print(f"[OP1] e3 R={r_e3:.4f}（账本 0.507）  b32 R={r_b32:.4f}（账本 0.743）  "
          f"差 {r_b32 - r_e3:.4f} >= {GAP} -> {'过' if op1 else '**不过**'}")
    print(f"[OP2] 主分析臂数 {len(main_arms)}/{len(arms)}（n_ask >= {MIN_N}） -> {'过' if op2 else '**不过**'}")
    print(f"[OP3] 主分析集 R 极差 {Rs.max() - Rs.min():.4f} -> {'过' if op3 else '**不过**'}")
    if not (op1 and op2 and op3):
        out["verdict"] = "REPRO_FAIL"
        (EXP / "resolve_spread.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n操作检验不过 -> REPRO_FAIL，本轮作废，不读任何主判据。")
        return 1

    # ---------------- 全表 ----------------
    print(f"\n全部 {len(arms)} 条已登记臂，按 R 排序（* = 主分析集，# = 环上的边）：")
    print(f"  {'臂':<52s} {'n_ask':>6s} {'R':>7s} {'W':>7s}  标注")
    for a in sorted(arms, key=lambda z: z["R"]):
        mark = "*" if a["n"] >= MIN_N else " "
        ring = ("#  " + RING[a["file"]]) if a["file"] in RING else ""
        print(f"{mark} {a['tag']:<52s} {a['n']:6d} {a['R']:7.3f} {a['W']:7.3f}  {ring}")

    # ---------------- (Q1) 超离散 ----------------
    ks = np.array([a["k"] for a in main_arms], float)
    ns = np.array([a["n"] for a in main_arms], float)
    rbar = float(ks.sum() / ns.sum())
    c_obs = chi2_stat(ks, ns, rbar)
    rng = np.random.default_rng(0)
    sim = rng.binomial(ns.astype(int)[None, :].repeat(B_BOOT, 0), rbar).astype(float)
    c_sim = (((sim - ns * rbar) ** 2) / (ns * rbar * (1 - rbar))).sum(1)
    p_q1 = float((1 + int((c_sim >= c_obs).sum())) / (B_BOOT + 1))
    q1 = "R_HETERO" if p_q1 < 0.01 else "R_HOMO"
    out["Q1"] = {"Rbar": rbar, "chi2": c_obs, "df_like": len(main_arms) - 1,
                 "p_boot": p_q1, "verdict": q1,
                 "R_min": float(Rs.min()), "R_max": float(Rs.max()),
                 "R_sd": float(Rs.std(ddof=1))}
    print(f"\n[Q1] 同质性：Rbar={rbar:.4f}，chi2={c_obs:.1f}（自由度约 {len(main_arms) - 1}），"
          f"参数自助 p={p_q1:.3g}（B={B_BOOT}）")
    print(f"     R 跨臂 sd={Rs.std(ddof=1):.4f}，范围 [{Rs.min():.3f}, {Rs.max():.3f}]  -> **{q1}**")

    # ---------------- (Q1b) 23pp 有多常见 ----------------
    pairs = list(combinations(range(len(main_arms)), 2))
    diffs = np.array([abs(main_arms[i]["R"] - main_arms[j]["R"]) for i, j in pairs])
    g = float((diffs >= GAP).mean())
    q1b = "GAP_TYPICAL" if g >= G_LINE else "GAP_RARE"
    out["Q1b"] = {"n_pairs": len(pairs), "g": g, "line": G_LINE, "verdict": q1b,
                  "median_gap": float(np.median(diffs)),
                  "q90_gap": float(np.quantile(diffs, 0.9))}
    print(f"\n[Q1b] 主分析集两两枚举 {len(pairs)} 对：|dR| >= {GAP} 的比例 g={g:.3f}"
          f"（线 {G_LINE}） -> **{q1b}**")
    print(f"      臂间差中位 {np.median(diffs):.3f}，90 分位 {np.quantile(diffs, 0.9):.3f}")

    # ---------------- (Q2) 判官噪声提案 ----------------
    eff = np.array([abs(a["W"] - 0.5) for a in main_arms])
    rho = spearman(eff, Rs)
    rng2 = np.random.default_rng(1)
    null = np.array([spearman(rng2.permutation(eff), Rs) for _ in range(B_BOOT // 10)])
    p_q2 = float((1 + int((np.abs(null) >= abs(rho)).sum())) / (len(null) + 1))
    loao = [spearman(np.delete(eff, i), np.delete(Rs, i)) for i in range(len(main_arms))]
    flip = any((r > 0) != (rho > 0) for r in loao)
    if p_q2 >= 0.05:
        q2 = "NOISE_UNDECIDED"
    else:
        q2 = "NOISE_SUPPORTED" if rho > 0 else "NOISE_FALSIFIED"
        if flip:
            q2 += "_FRAGILE"
    out["Q2"] = {"rho": rho, "p_perm": p_q2, "B": len(null), "loao_min": float(min(loao)),
                 "loao_max": float(max(loao)), "loao_sign_flip": bool(flip), "verdict": q2}
    print(f"\n[Q2] |W-0.5| vs R 的 Spearman rho={rho:+.3f}，置换 p={p_q2:.3g}（B={len(null)}）")
    print(f"     LOAO rho 范围 [{min(loao):+.3f}, {max(loao):+.3f}]，翻号={flip} -> **{q2}**")

    # ---------------- 描述性（不作判决） ----------------
    print("\n[描述性，不进判据] 环上四条边的 R：")
    for f, name in RING.items():
        a = by_file[f]
        print(f"     {name:<44s} R={a['R']:.3f}  W={a['W']:.3f}  n={a['n']}")
    small = [a for a in arms if a["n"] < MIN_N]
    if small:
        print(f"\n[次要，只报不判] 被 n<{MIN_N} 排除的 {len(small)} 条：",
              ", ".join(f"{a['tag']}(n={a['n']}, R={a['R']:.2f})" for a in small))

    out["verdict"] = f"{q1}+{q1b}+{q2}"
    out["arms"] = arms
    (EXP / "resolve_spread.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n判决：{out['verdict']}")
    print("授权：本轮不授权任何配置/默认值/新臂/判官调用；准入条件①②一个字不动。")
    print("产物 experiments/resolve_spread.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
