#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M18) 同一把刀转向**唯一承重的那个部件**：调色板记忆库的 66% 是族级的吗？
（预注册：本文件先提交再跑）

## 为什么问这个

上一轮 (M17)（`9145815`）在判官那半边立了一条**默认规矩**：

> **判官 full 的胜率结论今后须同时报族级 (U1) CI；二项 / Jeffreys 区间只能当下界报。**

那一轮只切了三条臂（e3 / e4 / b32），因为预注册把对象写死在 1 号缺口那条线上。
但**全项目唯一一条「正面」结论**当时一个字都没测：

> 16px 消融表（`3258294`，预注册 `83a6257`）：**四个部件里只有「调色板记忆库」承重**
> —— AB_pal **126/190 = 66.3%**，p=8.1e-06，Jeffreys [0.594, 0.727]；
> AB_ex 结构范例 87/175 = 49.7%（p=1.0）、AB_ct 解码顺序 86/154 = 55.8%（p=0.171）
> 是**真平局**。

这条结论是**路线图的地基**：它是我们迄今唯一知道"哪个部件在干活"的证据，
也是"32px 差不能归到唯一已知承重的部件头上"那句缩小范围的前提。
若它的 CI 与 b32 一样是假宽度，**地基就得重标**。

**零 GPU、零 API、零判官调用、零新臂、零配置改动、零重跑。**
只读已落盘的 `experiments/judge_full_TRD16c_rr4_vs_AB_*_rr4_16.json`。
`eval/b3_32.sh` 正在 GPU 6/7 上跑 B3@32（本轮开工已核活性），**本文件碰都不碰它**。

## 跑之前就已知的事：族的结构与 e3 **逐字相同**

三条消融臂用的都是同一套 `eval/prompts.py` 的 **E_mat 提示词**（272 条，270 个不同材质），
与 (M17) 的主对象 e3 **同一批**。按上一轮冻结的 (U1) 规则切出来是 **150 个族**，
最大族 15 对（`hardened clay stained`），与 e3 的族结构一致。

→ **本轮不发明任何新规则**：族定义直接从 `judge_cluster.py` import，一个字不改。
这一点很要紧——上一轮的 (U1) 是为 e3 冻结的，**不是**为本轮的臂挑的。

## 族的定义（沿用 `judge_cluster.py`，不许改）

- **(U1) 去末词族**（**主判据**）：`baked clay black` → `baked clay`。更细 = 更保守。
- **(U2) 首词族**（次判据，只报告，**不单独下判**）：`baked clay black` → `baked`。

## 主判据 (A)：给 AB_pal 换重采样单位

对象写死 = **AB_pal**（`judge_full_TRD16c_rr4_vs_AB_pal_rr4_16`，126/190）。
理由跑前写死：四条消融臂里**只有它**下过"承重"的判决，另外三条要么是真平局、
要么没有 full。平局臂的 CI 变宽**不可能改变一个平局**，所以它们进不了主判据。

- **(A1) 族级自助**：按 (U1) 的族有放回抽 n_fam 个族，取其全部判出对，
  重算胜率 = Σwins / Σn。B=2000、seed=0、95% 百分位区间。
- **(A2) 留一族法 (LOFO)**：每次删一个族，只看重算胜率落在 0.5 的哪一侧。

**判据（跑前写死，不许改）**

- **(W1) `robust`**：(A1) 的 95% CI **不含 0.5** 且 (A2) 每个合格 fold 与点估计同侧
  → 族级聚类不足以动摇 AB_pal，已发表 CI/p 照用，**消融表结论原样保留**。
- **(W2) `overstated`**：(A1) CI **含 0.5** 或有 fold 翻侧。
  - CI 跨 0.5、所有 fold 同侧 → **(W2a) 方向稳健、证据不足**。
  - 有 fold 翻侧 → **(W2b) 连方向都是族级的**。

⚠ 与 (M16)/(M17) 同款纪律：**(W2) ≠「调色板记忆库不承重」**，只等于**证据强度被高估**。
**不许**读成反向证据，**不许**据此说"那四个部件都不承重"，
**不许**动 `final_test.sh` 的主配置（那需要回验证集重选 + 另行预注册）。

⚠ **0.0002 纪律照旧**：CI 端点若落在 0.4999 / 0.5001 这类地方，
**按判据字面读**，不许写"基本显著"、不许改单侧、不许扩 B 重跑。

## 次判据 (B)：另外两条平局臂，**只报告，不下判**

`AB_ct`（86/154 = 55.8%）与 `AB_ex`（87/175 = 49.7%）照跑同一套 (A1)/(A2) 并登记，
但**跑前就写死不参与任何判决**：它们的已发表结论是"平局"，CI 变宽只会让平局更平，
**不可能翻案**。登记它们是为了让族级宽度有个同仪器的参照系。

## 次判据 (C)：族级聚集到底在不在（置换检验，无自由参数）

(M17) 第四节给过一个**描述性**说法（当时明写"不是判决"）：

> 同方法跨画布的两条边没有族级冗余（1.19×、0.99×）；**跨方法同画布那条有**（1.39×）。
> 解释：胜负对所有材质大体一样 → 族内不相关；而"TRD 相对基线在哪类材质上好或坏"
> 天然随材质族走 → 族内正相关。

**AB_pal 正是"跨方法同画布"那一类**（同为 16px，A/B 差一个部件）。所以本轮顺带
第一次**检验**那个说法，而不是继续把它当解释：

- **(C1) 族 × 胜负独立性**：列联表 族 × {A, B}（只含判出对），Pearson 卡方统计量，
  **置换 verdict 标签**（B=20000、seed=0，边际自动固定）。p<0.05 → 胜负随族聚集。
- **(C2) 族 × 可解性独立性**：列联表 族 × {resolved, inconsistent}，同样置换。

(C) **只描述，不参与 (W1)/(W2)**（与 (M16) 的 (B)、(M17) 的 (C) 同款定位）。
⚠ (C) 本轮**只对 AB_pal 跑**；跨方法那类臂的 b32 自己的 (C1) 仍然**没跑**，
它仍是提案，**不许**拿本轮的结果替它下判。

## 次判据 (D)：消融表的**对比**本身有没有被量过（描述性，跑前写死不是判决）

"只有调色板承重"这句话，靠的是**每条臂各自对 0.5**，外加 gen 半边独立同指一个部件
（`3258294`：三个口径全过噪声下限）。**「AB_pal 显著高于 AB_ct」从来没被直接检验过。**
本轮补一个描述性的数：按 (U1) 的族**配对**重采样（同一批族同时进两条臂，
两臂共用同一套 272 条提示词与同一个 A 侧 TRD16c），报
`rate(AB_pal) − rate(AB_ct)` 的 95% 区间（点差 = 0.663 − 0.558 = 0.105）。

⚠ **跑前写死的读法**：
- 该区间**含 0** → 只等于"**pal 比 ct 更承重**这件事本身**没有被直接量到**"，
  是一条**引用注意事项**；⛔ **不等于**消融表错了（表从来没声称过这个对比），
  也**不授权**任何改动。
- 该区间**不含 0** → 是对消融表的**加固**，同样不授权任何改动。

## 跑前就写死的三条方向预测（无论对错都登记）

⚠ 上一轮的三条预测**只兑现一条**，所以这里照录、不许事后改：

- **(P1)** (A) 判为 **W1_robust**。理由：已发表半宽 0.067，(M17) 实测的拓宽倍数区间是
  0.99×–1.56×，取上端得 [0.559, 0.767]，仍不含 0.5；**要翻案需要约 2.4× 的拓宽**。
- **(P2)** **(C1) p < 0.05**（AB_pal 是跨方法臂 → 按 (M17) 第四节的说法应当有族级聚集）。
  ⚠ 注意 e3 那条同样的检验给的是 p=0.160，(M16) 的包级也给 p=0.17
  ——**我假设的聚类已经连着两次没量到**。
- **(P3)** (D) 的区间**含 0**。

## 操作检验（任一不过 → 对应判决作废，照实写"没测到"）

- **(OP1) 复算锚点**：从 `records` 逐条重数，精确复现文件里的
  `a_wins`/`decided`/`inconsistent`/`api_fail`/`rate`。
- **(OP2) 退化锚点**：每对自成一族跑同一段 (A1) 代码，95% CI 与已发表 Jeffreys
  两端各差 < 0.03 → 证明只有**重采样单位**变了。
- **(OP3)** 每个判出对恰属一个族，各族对数之和 = `decided`。
- **(OP4)** 有效自助样本 ≥ 1900。
- **(OP5)** LOFO 的 fold 若剩余判出对 < 30 记 `skipped`；合格 fold < 2 → (A2) 作废。
- **(OP6) 复用检验**：本文件的族函数必须与 `judge_cluster.py` 的**同一个对象**
  （`is` 判等），杜绝"悄悄换了族定义"。

## 跑之前就写明的混杂（结果是哪个方向都不许事后改口径）

1. **不是盲设计**：126/190、86/154、87/175 三个数字跑前就在记忆与账本里；
   本轮可行性检查时我还 `print` 过每条臂 `records[0]` 的 verdict（各 1 条）。
   族定义是纯文本规则、且**上一轮就冻结**，与这三条臂的 verdict 无关——但既然看了就登记。
2. 族只有 150 个 → 族级 CI **必然变宽**。**"变宽"本身不是发现**；判据只看是否含 0.5。
3. 三条臂的 `inconsistent` 各不相同（82 / 118 / 97），所以 (D) 的配对是**族级配对**，
   不是对级配对。跑前接受，不许事后改成"只取两臂都判出的对"（那会另换一个总体）。
4. 本轮**不重判任何已下判决**，不碰 gen 半边（另一台仪器），不碰 `judge_pairs.py`、
   `final_test.sh`、`eval/prompts.py`、`palette_memory.py`。
5. 授权（无论结果）**仅限纸面**：给受影响的胜率加引用注意事项。
   **不授权**任何新臂、训练、判官调用、默认值改动、配置回选。

## 用法

    python analysis/arch/ablation_cluster.py
    python analysis/arch/ablation_cluster.py --json experiments/ablation_cluster.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_cluster as JC                                    # noqa: E402

B_BOOT = JC.B_BOOT
B_PERM = JC.B_PERM
SEED = JC.SEED

# 跑前写死的三条臂（不许跑完挑）；only_report=True 的不参与判决
ARMS = [
    ("AB_pal", "judge_full_TRD16c_rr4_vs_AB_pal_rr4_16",
     "主判据：调色板记忆库，消融表里唯一下过'承重'判决的部件", False),
    ("AB_ct", "judge_full_TRD16c_rr4_vs_AB_ct_rr4_16",
     "只报告：解码顺序，已发表结论=真平局", True),
    ("AB_ex", "judge_full_TRD16c_rr4_vs_AB_ex_rr4_16",
     "只报告：结构范例，已发表结论=真平局", True),
]


def paired_family_diff(recs_a, recs_b, famfn, b=B_BOOT, seed=SEED):
    """(D) 族级配对自助：同一批族同时进两条臂，报 rate_a - rate_b 的 95% 区间。"""
    fams = sorted({famfn(r["material"]) for r in recs_a} | {famfn(r["material"]) for r in recs_b})
    idx = {f: i for i, f in enumerate(fams)}
    wa = np.zeros(len(fams)); na = np.zeros(len(fams))
    wb = np.zeros(len(fams)); nb = np.zeros(len(fams))
    for r in recs_a:
        if r["verdict"] in ("A", "B"):
            i = idx[famfn(r["material"])]; wa[i] += r["verdict"] == "A"; na[i] += 1
    for r in recs_b:
        if r["verdict"] in ("A", "B"):
            i = idx[famfn(r["material"])]; wb[i] += r["verdict"] == "A"; nb[i] += 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(fams), size=(b, len(fams)))
    ta, tb = na[pick].sum(1), nb[pick].sum(1)
    ok = (ta > 0) & (tb > 0)
    d = wa[pick].sum(1)[ok] / ta[ok] - wb[pick].sum(1)[ok] / tb[ok]
    point = wa.sum() / na.sum() - wb.sum() / nb.sum()
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"point": float(point), "ci": [lo, hi], "contains_zero": bool(lo <= 0 <= hi),
            "n_families": len(fams), "n_eff_boot": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=ROOT / "experiments/ablation_cluster.json")
    a = ap.parse_args()

    # (OP6) 族函数必须就是上一轮那个对象
    op6 = (JC.FAMS["U1_drop_last"] is JC.fam_drop_last
           and JC.FAMS["U2_first"] is JC.fam_first)

    res = {"pre_registered": "analysis/arch/ablation_cluster.py",
           "reuses": "analysis/arch/judge_cluster.py (M17)",
           "B_boot": B_BOOT, "B_perm": B_PERM, "seed": SEED,
           "OP6_family_fn_identical": op6,
           "predictions": {"P1": "A -> W1_robust", "P2": "C1 p < 0.05", "P3": "D contains 0"},
           "arms": []}

    keep = {}
    for tag, name, note, only_report in ARMS:
        o, decided, recs = JC.analyse(tag, name, note)
        o["only_report_no_verdict"] = only_report
        if only_report:
            o["verdict"] = "REPORT_ONLY_" + o["verdict"]
        res["arms"].append(o)
        keep[tag] = recs

    # (C) 只对 AB_pal
    pal = keep["AB_pal"]
    dec = [r for r in pal if r["verdict"] in ("A", "B")]
    ans = [r for r in pal if r["verdict"] is not None]
    c1 = JC.perm_chi2([JC.fam_drop_last(r["material"]) for r in dec],
                      [r["verdict"] for r in dec])
    c2 = JC.perm_chi2([JC.fam_drop_last(r["material"]) for r in ans],
                      ["resolved" if r["verdict"] in ("A", "B") else "inconsistent" for r in ans])
    res["C1_fam_x_winner"] = {"chi2": c1[0], "p": c1[1], "n_fam": c1[2], "n": len(dec)}
    res["C2_fam_x_resolved"] = {"chi2": c2[0], "p": c2[1], "n_fam": c2[2], "n": len(ans)}

    # (D) pal vs ct 的族级配对差
    res["D_pal_minus_ct"] = paired_family_diff(keep["AB_pal"], keep["AB_ct"], JC.fam_drop_last)

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 打印 ----------
    print(f"(OP6) 族函数与 judge_cluster.py 同一对象：{op6}")
    for o in res["arms"]:
        p = o["published"]
        g = o["grains"]["U1_drop_last"]
        g2 = o["grains"]["U2_first"]
        print(f"\n=== {o['tag']}  {o['file']}")
        print(f"    {o['note']}")
        print(f"  已发表：A 胜 {p['a_wins']}/{p['decided']} = {p['rate']:.1%}  p={p['p']:.3g}  "
              f"Jeffreys [{p['jeffreys'][0]:.3f}, {p['jeffreys'][1]:.3f}]  api_fail={p['api_fail']}")
        d2 = o["OP2_degenerate"]
        print(f"  (OP1) 逐条重数精确复现：{o['OP1_recount_exact']}   "
              f"(OP2) 退化锚点 max_dev={d2['max_dev']:.4f} pass={d2['pass']}")
        print(f"  (U1) {g['n_families']} 族（最大 {g['max_family']} 对，有效 "
              f"{g['eff_families']:.1f}）  族级 CI [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]  "
              f"拓宽 {g['widening']:.2f}x  含 0.5={g['contains_half']}")
        lf = g["lofo"]
        print(f"       LOFO {lf['n_folds']} fold（skipped {lf['skipped']}）"
              f"全部同侧={lf['all_same_side']}  翻侧 {len(lf['flipped'])}")
        print(f"  (U2) {g2['n_families']} 族  CI [{g2['ci'][0]:.3f}, {g2['ci'][1]:.3f}]  "
              f"拓宽 {g2['widening']:.2f}x  含 0.5={g2['contains_half']}")
        print(f"  >>> {o['verdict']}")

    c = res["C1_fam_x_winner"]; cc = res["C2_fam_x_resolved"]
    print(f"\n=== (C) AB_pal 的族级聚集（置换 B={B_PERM}，只描述不下判）")
    print(f"  (C1) 族 x 胜负     chi2={c['chi2']:.2f}  p={c['p']:.4f}  "
          f"({c['n_fam']} 族 / {c['n']} 对)")
    print(f"  (C2) 族 x 可解性   chi2={cc['chi2']:.2f}  p={cc['p']:.4f}  "
          f"({cc['n_fam']} 族 / {cc['n']} 对)")

    d = res["D_pal_minus_ct"]
    print(f"\n=== (D) 消融表的对比本身（描述性，非判决）")
    print(f"  rate(AB_pal) - rate(AB_ct) = {d['point']:+.4f}  "
          f"族级 95% [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  含 0={d['contains_zero']}")

    print(f"\n=== 预测兑现（跑前写死）")
    pal_o = res["arms"][0]
    print(f"  (P1) A -> W1_robust      实际 {pal_o['verdict']}  "
          f"{'HIT' if pal_o['verdict'] == 'W1_robust' else 'MISS'}")
    print(f"  (P2) C1 p < 0.05         实际 p={c['p']:.4f}  "
          f"{'HIT' if c['p'] < 0.05 else 'MISS'}")
    print(f"  (P3) D 含 0              实际 {d['contains_zero']}  "
          f"{'HIT' if d['contains_zero'] else 'MISS'}")
    print(f"\n写入 {a.json}")


if __name__ == "__main__":
    main()
