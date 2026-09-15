#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M19) 把那把刀一次性划过**全部 28 条判官臂**，并检验 (M18) 留下的那条事后描述
（预注册：本文件先提交再跑）

## 为什么问这个

(M17)（`9145815`）在判官那半边立了一条**默认规矩**：

> **判官 full 的胜率结论今后须同时报族级 (U1) CI；二项 / Jeffreys 区间只能当下界报。**

这条规矩立了两轮，实际只切过 **6 条臂**（e3 / e4 / b32 于 `9145815`，
AB_pal / AB_ct / AB_ex 于 `5f38d8c`）。账本 `recheck_judge.py` 里登记在案的是 **28 条**。
剩下的 22 条**一个字都没测**，其中包括判官封盘那三档本身
（TRD16 vs B2/B7、TRD24、TRD32）与 tier_anchor 的锚臂。
→ 规矩立了却没执行完，这本身就是一笔欠账；**本轮一次性还清**。

同时，(M18) 第五节留下一条**明确标注为"提案、非结果、不许引用"**的事后描述：

> 三条消融臂全是跨方法同画布，拓宽却是 1.30x / 1.00x / 1.05x → 按臂的类别不能预测拓宽。
> 三条里唯一拓宽的，恰好是唯一**有真效应**的那条。
> 一个说得通但**没被检验**的解释：真胜率若处处 = 0.5（无效应），族就没有东西可调制
> → 无过离散；**有效应时效应随材质族变** → 才有过离散。

28 条臂的胜率从 41% 铺到 78%，正好是检验它的样本。**本文件就是那条提案的预注册。**

**零 GPU、零 API、零判官调用、零新臂、零训练、零配置改动、零重跑。**
只读已落盘的 `experiments/judge_full_*.json`。
`eval/b3_32.sh` 正在 GPU 6/7 上跑 B3@32（本轮开工已核活性：6/12 张齐、两支都活），
**本文件碰都不碰它**。

## 臂的名单不是我挑的

`ARMS = recheck_judge.EXPECT` —— 直接 import 账本里那 28 条**已登记**的 full 臂，
**本文件不增不减一条**。（`recheck_judge.py` 的规矩是"新臂跑完当轮就要登进去"，
所以那个 dict 就是全集。）B3@32 跑完之后的新臂**不在本轮范围内**，
它按 (M17) 的规矩自己报族级 CI。

## 族的定义：直接 import，一个字不改

`from judge_cluster import fam_drop_last, fam_first, cluster_boot, chi2`

- **(U1) 去末词族**（**主判据**）：`baked clay black` → `baked clay`。更细 = 更保守。
- **(U2) 首词族**：只报告。

(U1) 是 (M17) 为 e3 冻结的规则，**不是**为本轮任何一条臂挑的；本轮**不发明新规则**。

## 交付物 (A)：28 条臂的族级 CI 表（机械执行，无判断余地）

对每条臂逐条重数 → 族级自助 (B=2000, seed=0) → LOFO → 按 (M17) 冻结的判据机械打标签：

- **(W1) `robust`**：族级 95% CI 不含 0.5 **且** LOFO 每个合格 fold 同侧。
- **(W2a) `direction_only`**：CI 含 0.5、fold 全同侧。
- **(W2b) `direction_too`**：有 fold 翻侧。
- **`VOID_*`**：操作检验或 LOFO 不合格（合格 fold < 2，见 (OP5)）。

⚠ 与 (M16)/(M17)/(M18) 同款纪律（**不许改**）：**(W2) ≠「该结论是假的」**，
只等于**证据强度被高估**；⛔ 不许当反向证据。
⚠ 本轮**不重判任何已下的判决**，只给受影响的胜率加引用注意事项。

**(A-tally) 跑前写死的汇总量**：`n_downgraded` = 已发表 Jeffreys **不含** 0.5、
而族级 (U1) CI **含** 0.5 的臂数。这是本轮对项目最有用的一个数：
它说的是"照 (M17) 的新规矩，还有几条已发表的判官结论要挂注意事项"。

## 主判据 (B)：(M18) 第五节那条描述——「有效应才有过离散」

- **效应量** `effect = |rate − 0.5|`（逐条臂，来自重数）。
- **过离散**不能直接用"拓宽倍数"：`widening = 族级宽 / Jeffreys 宽` 有一条**机械通道**
  会自己产生本轮预测的方向 —— 设计效应 ≈ 1 + (m̄−1)ρ，而 ρ = var(p_f) / (p̄(1−p̄))，
  分母在 p 远离 0.5 时变小 → 同样的族间差异会被放大成更大的 ρ。
  **跑前就写死的中和办法**：对每条臂做**族内结构破坏的零标定**——
  把该臂 decided 对的胜负标签在**全臂范围内随机置换**（族的归属与族的大小一字不动），
  重算族级 CI 宽度，取 `B_NULL=100` 次的**中位数** `w_null`。
  置换后 p̄ 与族大小分布完全不变，**唯一被破坏的是族与胜负的对应** →
  `excess = w_obs / w_null` 就是**扣掉机械通道之后的过离散**。
  ⚠ 为免百分位区间在不同 B 下的宽度偏差污染比值，`w_obs` 与 `w_null` **都用 B=500**
  的同一条代码路径算（报告出去的 CI 永远是 B=2000 那条，两者不混用）。

- **(B1) 主检验**：Spearman(`effect`, `excess`)，置换 `effect`（B=20000, seed=0）求 p。
  **预测方向：正相关**（有效应 → 过离散）。
- **(B2) 机械混杂的控制**：过离散**天然随族变大而变大**（m̄ 越大，设计效应越大）。
  取 `mbar = decided / n_families`，算**偏 Spearman**（在秩上用标准偏相关公式）
  `r(effect, excess | mbar)`，同样置换 `effect` 求 p。

**判据（跑前写死，不许改）**

- **`SUPPORTED`**：(B1) rho > 0 且 p < 0.05 **且** (B2) 偏相关同号且 p < 0.05。
- **`UNDECIDED`**：(B1) 过、(B2) 不过（说明那条相关可能只是"族更大"）。
- **`NOT_DETECTED`**：(B1) p ≥ 0.05。**读作"什么也没测到"**，
  ⛔ **不是**"已证否「有效应才有过离散」"（(M13) 的教训，此处照搬）。

- **入选口径（跑前写死）**：`decided ≥ 50` **且** `n_families ≥ 10` 的臂进 (B)。
  按账本里的数字这会剔掉 4 条 B3 子集臂（n = 9/10/10/12），**剩 24 条**。
  ⛔ 跑完不许改这个阈值、不许剔任何一条"看着碍事"的臂。
  被剔的臂**照样进 (A) 的表**。

## 次判据 (C)：族 × 胜负的置换检验，铺到全部入选臂

(M18) 在 AB_pal 上第一次真的量到族级聚集（p=0.0009），而 (M16)/(M17) 连吃两次空手。
本轮对每条入选臂跑同一个 `perm_chi2`（B=2000, seed=0；**注意 B 比 (M18) 的 20000 小**，
因为要跑 24 遍，这会让最小可报 p ≈ 5e-4，**跑前写死，不许事后加 B 去追小 p**）。
**(C) 只描述，不参与任何判决**，但 `n_sig` = p<0.05 的臂数会被记下来。

## 操作检验（任一不过 → 对应判决作废，照实写"没测到"）

- **(OP1) 逐条重数**：每条臂的 (a_wins, decided, inconsistent, api_fail)
  必须**精确等于** `recheck_judge.EXPECT` 里账本抄录的四元组。
- **(OP2) 逐位复现**：e3 / e4 / b32 的族级 CI 必须与 `experiments/judge_cluster.json`
  **逐位相等**（1e-12），AB_pal / AB_ct / AB_ex 必须与 `experiments/ablation_cluster.json`
  逐位相等 —— 证明本轮走的是**同一条代码路径、同一个种子**，不是重新实现了一遍。
  ⚠ 这 6 条**不许**因为本轮跑出别的数就改口；对不上 = 整个 (A) 作废。
- **(OP3)** 每条臂各族对数之和 = decided。
- **(OP4)** 有效自助样本 ≥ 1900（B=2000 那条）。
- **(OP5)** LOFO fold 剩余判出对 < 30 记 skipped；合格 fold < 2 → 该臂 `VOID_lofo_invalid`。
- **(OP6)** 零标定的健全性：`w_null` 必须 > 0 且 `excess` 有限；
  另外**退化自检**——把族全部打散成"每对自成一族"时 `excess` 应当 ≈ 1（容差 0.15），
  对 e3 做一次。不过 → (B) 作废。

## 跑之前就写明的混杂（结果是哪个方向都不许事后改口径）

1. **不是盲设计**。28 条臂的胜率我跑前都知道（账本 + `recheck_judge.EXPECT` 就在眼前），
   6 条臂的族级 CI 也已经算过。**但 (B) 的两个统计量谁都没算过**，
   `excess` 这个量本轮才第一次定义。
2. **⚠⚠ 28 条臂不是 28 个独立单位** —— 这正是本项目连着三轮在教的那件事，
   本轮自己也逃不掉：大量臂共用同一批 E_mat 提示词，多条臂共用同一个参照腿
   （`TRD16c_rr4` 出现在 4 条臂里），三档梯度臂之间更是同一批图。
   → **(B) 的置换 p 是下界**（证据被高估），**跑前就这么写**。
   ⛔ 因此 `SUPPORTED` 只允许被引用成"**有迹象**"，⛔ **不许**当成已确立的机制，
   ⛔ 也**不许**据它去解释或改写任何一条已下的判决。要认真做须另行预注册
   （正确的做法是把臂本身也做成重采样单位，本轮明令不做）。
3. 族只有几十个 → 族级 CI **必然变宽**。"变宽"本身不是发现；判据只看是否含 0.5。
4. 本轮授权**仅限纸面**：给 (A) 表里被降级的臂加引用注意事项 + 更新账本。
   ⛔ **不授权**任何新臂、训练、判官调用、默认值改动、配置回选、对 `final_test.sh`
   或 `judge_pairs.py` 的任何改动。

## 跑前写死的预测（兑现率要记进账本）

- **(P1)** (A) 里 `n_downgraded ≥ 1`：28 条臂里**至少有一条**已发表结论要挂注意事项。
- **(P2)** `excess` 的**中位数 < 1.15**：多数臂**没有**可测的过离散
  （即 (M18) 在 AB_pal 上量到的 p=0.0009 是少数派，不是普遍现象）。
- **(P3)** (B) 主判据判 `SUPPORTED`（即 (B1)(B2) 都过）。

## 用法

    python analysis/arch/judge_cluster_sweep.py
    python analysis/arch/judge_cluster_sweep.py --json /tmp/judge_cluster_sweep.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/arch"))
from exact import jeffreys, binom_test                       # noqa: E402
from judge_cluster import fam_drop_last, fam_first, cluster_boot, chi2, hhi   # noqa: E402
from recheck_judge import EXPECT                             # noqa: E402

B_BOOT = 2000          # 报告出去的 CI
B_CAL = 500            # 零标定专用（w_obs 与 w_null 共用）
B_NULL = 100           # 置换次数
B_PERM_ARM = 2000      # (C) 每条臂的置换
B_PERM_RHO = 20000     # (B) 臂级置换
SEED = 0
MIN_FOLD = 30
MIN_DECIDED = 50       # 入选 (B) 的门槛（跑前写死）
MIN_FAM = 10

FAMS = {"U1_drop_last": fam_drop_last, "U2_first": fam_first}


# ---------- 秩统计 ----------
def rank(x):
    """平均秩（处理并列）。"""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    # 并列取平均
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def partial(rx, ry, rz):
    """偏相关（在秩上）。"""
    rxy, rxz, ryz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 0 else float("nan")


def perm_p(stat_fn, x, b=B_PERM_RHO, seed=SEED):
    """置换 x，双侧 p（|stat| >= |obs|）。"""
    obs = stat_fn(x)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(b):
        ge += abs(stat_fn(rng.permutation(x))) >= abs(obs) - 1e-12
    return float(obs), float((ge + 1) / (b + 1))


def perm_chi2_small(labels, cats, b=B_PERM_ARM, seed=SEED):
    rows = sorted(set(labels))
    cols = sorted(set(cats))
    ri = {k: i for i, k in enumerate(rows)}
    ci = {k: i for i, k in enumerate(cols)}
    r = np.array([ri[x] for x in labels])
    c = np.array([ci[x] for x in cats])
    obs = np.zeros((len(rows), len(cols)))
    np.add.at(obs, (r, c), 1)
    stat = chi2(obs)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(b):
        cp = rng.permutation(c)
        t = np.zeros_like(obs)
        np.add.at(t, (r, cp), 1)
        ge += chi2(t) >= stat - 1e-12
    return float(stat), float((ge + 1) / (b + 1))


# ---------- 零标定 ----------
def boot_width(units, b, seed):
    lo, hi, ne = cluster_boot(units, b=b, seed=seed)
    return hi - lo, ne


def calibrate(fam_of, wins, seed=SEED):
    """族归属固定，置换胜负标签 B_NULL 次，返回 (w_obs@B_CAL, w_null_median, excess)。"""
    fams = sorted(set(fam_of))
    fi = {f: i for i, f in enumerate(fams)}
    idx = np.array([fi[f] for f in fam_of])
    w = np.asarray(wins, float)

    def units_from(vec):
        wsum = np.zeros(len(fams))
        nsum = np.zeros(len(fams))
        np.add.at(wsum, idx, vec)
        np.add.at(nsum, idx, 1.0)
        return list(zip(wsum.tolist(), nsum.tolist()))

    w_obs, _ = boot_width(units_from(w), B_CAL, SEED)
    rng = np.random.default_rng(seed + 991)
    widths = []
    for i in range(B_NULL):
        widths.append(boot_width(units_from(rng.permutation(w)), B_CAL, SEED + 1 + i)[0])
    w_null = float(np.median(widths))
    return w_obs, w_null, (w_obs / w_null if w_null > 0 else float("nan"))


# ---------- 单臂 ----------
def analyse(name, exp):
    d = json.loads((ROOT / "experiments" / name).read_text(encoding="utf-8"))
    recs = d["records"]
    wins = sum(r["verdict"] == "A" for r in recs)
    dec = sum(r["verdict"] in ("A", "B") for r in recs)
    inc = sum(r["verdict"] == "inconsistent" for r in recs)
    fail = sum(r["verdict"] is None for r in recs)
    op1 = (wins, dec, inc, fail) == tuple(exp)

    decided = [r for r in recs if r["verdict"] in ("A", "B")]
    rate = wins / dec
    side = rate > 0.5
    jlo, jhi = jeffreys(wins, dec)

    out = {"file": name, "tag": d["tag"], "OP1_recount_exact": op1,
           "a_wins": wins, "decided": dec, "inconsistent": inc, "api_fail": fail,
           "rate": rate, "p": binom_test(wins, dec), "jeffreys": [jlo, jhi],
           "published_excludes_half": not (jlo <= 0.5 <= jhi),
           "grains": {}}

    for gname, fn in FAMS.items():
        groups = defaultdict(lambda: [0, 0])
        for r in decided:
            g = groups[fn(r["material"])]
            g[0] += r["verdict"] == "A"
            g[1] += 1
        units = [tuple(v) for v in groups.values()]
        sizes = [v[1] for v in groups.values()]
        lo, hi, ne = cluster_boot(units, b=B_BOOT, seed=SEED)
        folds, skipped = [], 0
        for k in sorted(groups):
            w2 = wins - groups[k][0]
            n2 = dec - groups[k][1]
            if n2 < MIN_FOLD:
                skipped += 1
                continue
            folds.append({"drop": k, "n": n2, "rate": w2 / n2,
                          "same_side": (w2 / n2 > 0.5) == side})
        out["grains"][gname] = {
            "n_families": len(groups), "hhi": hhi(sizes), "eff_families": 1 / hhi(sizes),
            "max_family": max(sizes), "mbar": dec / len(groups),
            "ci": [lo, hi], "width": hi - lo, "published_width": jhi - jlo,
            "widening": (hi - lo) / (jhi - jlo),
            "contains_half": bool(lo <= 0.5 <= hi),
            "n_eff_boot": ne, "OP4_pass": ne >= 1900,
            "OP3_pass": sum(sizes) == dec,
            "lofo": {"n_folds": len(folds), "skipped": skipped,
                     "all_same_side": bool(all(f["same_side"] for f in folds)),
                     "valid": len(folds) >= 2,
                     "worst": min(folds, key=lambda f: abs(f["rate"] - 0.5)) if folds else None,
                     "n_flipped": sum(not f["same_side"] for f in folds)},
        }

    g = out["grains"]["U1_drop_last"]
    if not (op1 and g["OP3_pass"] and g["OP4_pass"]):
        out["verdict"] = "VOID_operational_check_failed"
    elif not g["lofo"]["valid"]:
        out["verdict"] = "VOID_lofo_invalid"
    elif not g["contains_half"] and g["lofo"]["all_same_side"]:
        out["verdict"] = "W1_robust"
    elif g["lofo"]["all_same_side"]:
        out["verdict"] = "W2a_direction_only"
    else:
        out["verdict"] = "W2b_direction_too"
    # (A-tally)：已发表不含 0.5，族级含 0.5 -> 要挂注意事项
    out["downgraded"] = bool(out["published_excludes_half"] and g["contains_half"])
    out["eligible_B"] = bool(dec >= MIN_DECIDED and g["n_families"] >= MIN_FAM)
    out["_decided_recs"] = decided
    return out


# ---------- (OP2) 逐位复现 ----------
def op2_reproduce(arms):
    ref = {}
    for f, keymap in ((ROOT / "experiments/judge_cluster.json", None),
                      (ROOT / "experiments/ablation_cluster.json", None)):
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        for a in d["arms"]:
            ref[a["file"] + ".json"] = (a["tag"], a["grains"]["U1_drop_last"]["ci"])
    checked, bad = [], []
    by_name = {a["file"]: a for a in arms}
    for name, (tag, ci) in ref.items():
        if name not in by_name:
            bad.append(f"{name}: 参照里有、本轮名单里没有")
            continue
        got = by_name[name]["grains"]["U1_drop_last"]["ci"]
        ok = abs(got[0] - ci[0]) < 1e-12 and abs(got[1] - ci[1]) < 1e-12
        checked.append({"file": name, "tag": tag, "ref": ci, "got": got, "exact": ok})
        if not ok:
            bad.append(f"{name}: {got} != {ci}")
    return {"n_checked": len(checked), "pass": not bad and len(checked) >= 6,
            "detail": checked, "bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=ROOT / "experiments/judge_cluster_sweep.json")
    a = ap.parse_args()

    arms = [analyse(name, exp) for name, exp in EXPECT.items()]
    op2 = op2_reproduce(arms)

    # ---- (B) 零标定 + 相关 ----
    elig = [x for x in arms if x["eligible_B"]]
    for x in arms:
        if not x["eligible_B"]:
            x["excess"] = None
            continue
        fam_of = [fam_drop_last(r["material"]) for r in x["_decided_recs"]]
        wins = [int(r["verdict"] == "A") for r in x["_decided_recs"]]
        w_obs, w_null, exc = calibrate(fam_of, wins)
        x["calib"] = {"w_obs_cal": w_obs, "w_null_median": w_null}
        x["excess"] = exc

    # (OP6) 退化自检：e3 每对自成一族 -> excess 应 ~1
    e3 = next(x for x in arms if x["file"] == "judge_full_B2_vs_B2up16_32.json")
    fam_id = [f"pair{i}" for i in range(len(e3["_decided_recs"]))]
    w_e3 = [int(r["verdict"] == "A") for r in e3["_decided_recs"]]
    _, _, exc_deg = calibrate(fam_id, w_e3)
    op6 = {"degenerate_excess": exc_deg, "pass": abs(exc_deg - 1.0) < 0.15,
           "all_null_positive": all(x.get("calib", {}).get("w_null_median", 1) > 0 for x in elig),
           "all_finite": all(np.isfinite(x["excess"]) for x in elig)}

    eff = np.array([abs(x["rate"] - 0.5) for x in elig])
    exc = np.array([x["excess"] for x in elig])
    mbar = np.array([x["grains"]["U1_drop_last"]["mbar"] for x in elig])
    r_eff, r_exc, r_mb = rank(eff), rank(exc), rank(mbar)

    rho1, p1 = perm_p(lambda v: pearson(v, r_exc), r_eff)
    rho2, p2 = perm_p(lambda v: partial(v, r_exc, r_mb), r_eff)
    rho_mb = pearson(r_mb, r_exc)

    if not (op6["pass"] and op6["all_finite"] and op6["all_null_positive"]):
        verdict_B = "VOID_op6_failed"
    elif rho1 > 0 and p1 < 0.05 and rho2 > 0 and p2 < 0.05:
        verdict_B = "SUPPORTED"
    elif rho1 > 0 and p1 < 0.05:
        verdict_B = "UNDECIDED"
    else:
        verdict_B = "NOT_DETECTED"

    # ---- (C) 每条入选臂的族 x 胜负 ----
    for x in elig:
        s, p = perm_chi2_small([fam_drop_last(r["material"]) for r in x["_decided_recs"]],
                               [r["verdict"] for r in x["_decided_recs"]])
        x["C_fam_x_winner"] = {"chi2": s, "p": p}

    n_down = sum(x["downgraded"] for x in arms)
    med_exc = float(np.median(exc))
    n_sig = sum(x["C_fam_x_winner"]["p"] < 0.05 for x in elig)

    for x in arms:
        x.pop("_decided_recs", None)
    res = {"pre_registered": "analysis/arch/judge_cluster_sweep.py",
           "reuses": ["analysis/arch/judge_cluster.py", "analysis/arch/recheck_judge.py"],
           "B_boot": B_BOOT, "B_cal": B_CAL, "B_null": B_NULL, "seed": SEED,
           "MIN_DECIDED": MIN_DECIDED, "MIN_FAM": MIN_FAM,
           "n_arms": len(arms), "n_eligible": len(elig),
           "OP1_all_exact": all(x["OP1_recount_exact"] for x in arms),
           "OP2_reproduce": op2, "OP6": op6,
           "A_tally": {"n_downgraded": n_down,
                       "n_W1": sum(x["verdict"] == "W1_robust" for x in arms),
                       "n_W2a": sum(x["verdict"] == "W2a_direction_only" for x in arms),
                       "n_W2b": sum(x["verdict"] == "W2b_direction_too" for x in arms),
                       "n_void": sum(x["verdict"].startswith("VOID") for x in arms)},
           "B_correlation": {"n": len(elig), "rho_effect_excess": rho1, "p1": p1,
                             "partial_rho_given_mbar": rho2, "p2": p2,
                             "rho_mbar_excess": rho_mb,
                             "median_excess": med_exc, "verdict": verdict_B},
           "C_tally": {"n_sig": n_sig, "n": len(elig)},
           "predictions": {"P1_n_downgraded_ge_1": n_down >= 1,
                           "P2_median_excess_lt_1.15": med_exc < 1.15,
                           "P3_B_supported": verdict_B == "SUPPORTED"},
           "arms": arms}
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 打印 ----------
    print(f"(OP1) 28 条臂逐条重数全部等于账本：{res['OP1_all_exact']}")
    print(f"(OP2) 已切过的 {op2['n_checked']} 条臂逐位复现：{op2['pass']}"
          + ("" if op2["pass"] else f"  {op2['bad']}"))
    print(f"(OP6) 退化自检 excess={exc_deg:.3f}（应 ~1）-> {'通过' if op6['pass'] else '**不过**'}")

    print("\n=== (A) 28 条臂的族级 (U1) CI 表 "
          "（* = 已发表不含 0.5、族级含 0.5，须挂注意事项）")
    hdr = f"{'臂':46s} {'A胜/判出':>10s} {'率':>6s} {'已发表CI':>16s} {'族级CI':>16s} {'拓宽':>6s} {'判':<18s}"
    print(hdr)
    for x in sorted(arms, key=lambda y: -y["decided"]):
        g = x["grains"]["U1_drop_last"]
        star = "*" if x["downgraded"] else " "
        print(f"{star}{x['tag'][:45]:45s} {x['a_wins']:4d}/{x['decided']:<5d} "
              f"{x['rate']:5.1%} [{x['jeffreys'][0]:.3f},{x['jeffreys'][1]:.3f}] "
              f"[{g['ci'][0]:.3f},{g['ci'][1]:.3f}] {g['widening']:5.2f}x {x['verdict']:<18s}")
    t = res["A_tally"]
    print(f"\n  W1 {t['n_W1']}  W2a {t['n_W2a']}  W2b {t['n_W2b']}  VOID {t['n_void']}"
          f"   -> **须挂注意事项的臂 n_downgraded = {t['n_downgraded']}**")

    print(f"\n=== (B) 有效应才有过离散？（n={len(elig)} 条入选臂）")
    print(f"  excess 中位数 {med_exc:.3f}  最小 {exc.min():.3f}  最大 {exc.max():.3f}")
    print(f"  (B1) Spearman(effect, excess) rho={rho1:+.3f}  perm p={p1:.4f}")
    print(f"  (B2) 偏 Spearman | mbar       rho={rho2:+.3f}  perm p={p2:.4f}"
          f"   (参考 rho(mbar,excess)={rho_mb:+.3f})")
    print(f"  ==> **{verdict_B}**")
    print(f"\n=== (C) 族 x 胜负 p<0.05 的臂：{n_sig}/{len(elig)}")
    for x in sorted(elig, key=lambda y: y["C_fam_x_winner"]["p"])[:6]:
        print(f"  {x['tag'][:45]:45s} chi2={x['C_fam_x_winner']['chi2']:7.2f}  "
              f"p={x['C_fam_x_winner']['p']:.4f}  excess={x['excess']:.3f}")
    pr = res["predictions"]
    print(f"\n预测兑现：P1 {pr['P1_n_downgraded_ge_1']}  P2 {pr['P2_median_excess_lt_1.15']}  "
          f"P3 {pr['P3_B_supported']}")
    print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
