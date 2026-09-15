#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M17) 判官那半边的 272 对也是独立单位吗？——把 (M16) 的刀转向胜率 CI
（预注册：本文件先提交再跑）

## 为什么问这个

上一轮 (M16)（`001ff93`）证明**真人语料**那半边的 CI 是假宽度：把瓦片换成包重采样，
C 的 CI 宽了 5.4 倍、跨了 0。当时账本写死了一句话：

> **判官那半边一个字都没测，不许外推**；「272 对跨 12 个测试包、胜率二项 CI 同样把对当
> 独立单位」是**提案，须预注册**。

本文件就是那条提案。它要紧，因为**全项目 1 号缺口的全部承重梁都是判官胜率的二项 CI**：
e3（B2@32 胜 B2@16 = 107/138 = 78%，p=5.5e-11，`2c68bc4`）、e4（TRD@32 vs TRD@16
= 136/189 = 72%，`6c64796`）、以及"32px 是坏的那档"（TRD32 vs B2@32 = 41%，p=0.014，
`afa624c`）。这三个数字全部把**每一对当成一个独立单位**。

**零 GPU、零 API、零判官调用、零新臂、零配置改动。** 只读已落盘的
`experiments/judge_full_*.json`（每条记录含 `material` = 提示词与 `verdict`）。

## 跑之前已经发现的事：判官侧的聚类单位**不是包**

(M16) 的提案原话是"272 对跨 12 个测试包"。**这个前提是错的，本文件跑前已经证否**：
`eval/prompts.py` 的 E_mat 是 272 个**材质文件名**，而同一个文件名在 test split 里
横跨多个包——272 个里 **211 个（78%）** 属于 ≥2 个包（如 `acacia tree` 出现在 8 个包）。
更根本的是：判官比的两张图**都是模型生成的**，包只提供提示词文本，**不是对的产生单位**。

→ 按包聚类在这一侧无定义。但真正的聚类隐患就摆在记录里：E_mat 的提示词大量成族，
`judge_full_B2_vs_B2up16_32.json` 开头连着 16 条 `baked clay <颜色>`。
同一族里模型面对的几乎是同一张图换个色调，**它们的胜负不可能是独立抽样**。
本文件因此把重采样单位换成**提示词族**。

## 族的定义（跑前写死，两个粒度都报，不许跑完挑）

纯文本规则，与任何 verdict 无关：

- **(U1) 去末词族**（**主判据**）：提示词去掉最后一个词；只有一个词则取自身。
  `baked clay black` → `baked clay`；`acacia tree top` → `acacia tree`。
- **(U2) 首词族**（次判据）：提示词的第一个词。`baked clay black` → `baked`。

主判据故意选**更细**的 (U1)：族越细、有效自由度越高、CI 越窄 → **越难支持本轮的假设**。
若连 (U1) 都让 CI 跨 0.5，结论才硬。**(U2) 只报告，不单独下判。**

## 主判据 (A)：给 e3 换重采样单位

选 **e3 = `judge_full_B2_vs_B2up16_32`** 当主对象，理由跑前写死：它是 (M16) 之后仍然
**没有被任何降级触及**的那条，且是**全项目 1 号缺口的分母**（`2c68bc4`：免门重问 272 对，
可解率 50.7%、A 胜 107/138 = 77.5%、p=5.5e-11、Jeffreys [70.0%, 83.9%]）。

- **(A1) 族级自助**：按 (U1) 的族有放回抽 n_fam 个族，取其**全部判出对**，
  重算胜率 = Σwins / Σn。B=2000、seed=0，95% 百分位区间。
- **(A2) 留一族法 (LOFO)**：每次删掉一个族，重算胜率，**只看它落在 0.5 的哪一侧**。

**判据（跑前写死，不许改）**

- **(W1) `robust`**：(A1) 的 95% CI **不含 0.5** 且 (A2) 每个合格 fold 与点估计**同侧**
  → 272 对的聚类不足以动摇 e3，已发表 CI 照用，本轮什么都不改。
- **(W2) `overstated`**：(A1) CI **含 0.5** 或任一合格 fold 翻到另一侧。
  - CI 跨 0.5、但所有 fold 同侧 → **(W2a) 方向稳健、证据不足**。
  - 有 fold 翻侧 → **(W2b) 连方向都是族级的**。
- ⚠ 与 (M16) 同款纪律：(W2) **不等于"e3 是假的"**，只等于**证据强度被高估**。
  **不许**读成反向证据，**不许**据此说"32px 的画布增益不存在"。

## 次判据 (B)：同一把刀施加到另外两条承重梁（写死，全部报告）

- **(B-e4)** `judge_full_TRD32_rr4_vs_TRD16cup_32`（136/189 = 72%，环不闭合那条边）。
  ⚠ 该臂 `api_fail=1`（记忆已录）：失败对本来就不进 `decided`，本轮**照录不改**。
- **(B-32)** `judge_full_TRD32_rr4_vs_B2_32`（41%，p=0.014，"32px 输"的那条）。
  它的点估计在 0.5 **另一侧**，判据按"CI 是否含 0.5"对称适用。

## 次判据 (C)：胜负与可解率到底随不随族走（置换检验，无自由参数）

对 e3：

- **(C1) 族 × 胜负独立性**：列联表 族 × {A, B}（只含判出对），统计量 Pearson χ²，
  **置换 verdict 标签**（B=20000、seed=0，边际自动固定）。p<0.05 → 胜负随族聚集。
- **(C2) 族 × 可解性独立性**：列联表 族 × {resolved, inconsistent}，同样置换。
  p<0.05 → **可解率 50.7% 这个数字也是族级产物**。

(C) 只描述，**不参与 (W1)/(W2) 判决**（与 (M16) 的 (B) 同款定位）。

## 操作检验（任一不过 → 对应判决作废，照实写"没测到"）

- **(OP1) 复算锚点**：从 JSON 的 `records` 逐条重数，必须**精确复现**该文件已存的
  `a_wins` / `decided` / `inconsistent` / `api_fail` / `rate`（证明读的就是那批数）。
- **(OP2) 退化锚点**：把"每一对自成一族"当单位跑同一个 (A1) 代码路径，
  95% CI 必须与已发表 Jeffreys 区间两端各差 < 0.03
  → 证明 (A1) 与已发表数字之间**只有重采样单位变了**。
- **(OP3)** 每个判出对恰属一个族，各族对数之和 = `decided`。
- **(OP4)** 有效自助样本 ≥ 1900。
- **(OP5)** LOFO 的 fold 若剩余判出对 < 30 记 `skipped`，不参与判据；
  合格 fold < 2 → (A2) 作废。

## 跑之前就写明的混杂（结果是哪个方向都不许事后改口径）

1. **不是盲设计**：78% / 72% / 41% 三个已发表胜率我跑前就知道（记忆与账本里）。
   **披露**：可行性检查时 `json.dumps` 截断输出让我看到了 e3 的前 13 条记录
   （`baked clay` 那一串的 verdict）。族定义是**纯文本规则**，与 verdict 无关，
   且在看到那 13 条**之后**才写——但既然看了就照实登记。
2. 族只有几十个 → 族级 CI **必然变宽**。**"变宽"本身不是发现**；判据只看是否含 0.5。
3. (U1) 会把 `baked clay black` 与 `baked clay dark green` 分进两族（`dark` 留在族名里）。
   这是规则的机械后果，**偏细 = 偏保守**，跑前接受，不许事后改成"先剥颜色词"。
4. 本轮**不重判任何已下判决**。环闭合 (`6c64796`) 的 z 检验用的是二项方差，
   若 (W2) 成立它显然也受影响——但重判环需要四条边全部重做 + 单独预注册，
   **本轮明令不做，也不许在账本里顺手改环的判决**。
5. 授权（无论结果）**仅限纸面**：给受影响的胜率加引用禁令 + 一条默认规矩。
   **不授权**任何新臂、训练、判官调用、默认值改动。

## 用法

    python analysis/arch/judge_cluster.py            # 全部判据
    python analysis/arch/judge_cluster.py --json /tmp/judge_cluster.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/annotate"))
from exact import jeffreys                       # noqa: E402

B_BOOT = 2000
B_PERM = 20000
SEED = 0
MIN_FOLD = 30          # (OP5)

# 跑前写死的三条臂（不许跑完挑）
ARMS = [
    ("e3", "judge_full_B2_vs_B2up16_32", "主判据：B2@32 vs B2@16，1 号缺口的分母"),
    ("e4", "judge_full_TRD32_rr4_vs_TRD16cup_32", "次：TRD@32 vs TRD@16 放大，环的 e4"),
    ("b32", "judge_full_TRD32_rr4_vs_B2_32", "次：TRD@32 vs B2@32，'32px 输'那条"),
]


# ---------- 族定义（纯文本，与 verdict 无关） ----------
def fam_drop_last(prompt: str) -> str:
    """(U1) 去末词族：主判据。更细 = 更保守。"""
    w = prompt.split()
    return " ".join(w[:-1]) if len(w) > 1 else prompt


def fam_first(prompt: str) -> str:
    """(U2) 首词族：次判据，更粗。"""
    w = prompt.split()
    return w[0] if w else prompt


FAMS = {"U1_drop_last": fam_drop_last, "U2_first": fam_first}


# ---------- 统计 ----------
def cluster_boot(units, b=B_BOOT, seed=SEED):
    """units: [(wins, n), ...] 每个单位一项。按单位有放回重采样，返回 (lo, hi, n_eff)。"""
    if not units:
        return float("nan"), float("nan"), 0
    w = np.array([u[0] for u in units], float)
    n = np.array([u[1] for u in units], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(b, len(units)))
    tot = n[idx].sum(1)
    win = w[idx].sum(1)
    ok = tot > 0
    rates = win[ok] / tot[ok]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5)), int(ok.sum())


def chi2(table: np.ndarray) -> float:
    r = table.sum(1, keepdims=True)
    c = table.sum(0, keepdims=True)
    tot = table.sum()
    if tot == 0:
        return float("nan")
    exp = r * c / tot
    m = exp > 0
    return float((((table - exp) ** 2)[m] / exp[m]).sum())


def perm_chi2(labels, cats, b=B_PERM, seed=SEED):
    """置换 cats（第二维标签），行列边际固定。返回 (chi2, p, n_row, n_col)。"""
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
    return stat, (ge + 1) / (b + 1), len(rows), len(cols)


def hhi(counts) -> float:
    tot = sum(counts)
    return float(sum((c / tot) ** 2 for c in counts)) if tot else float("nan")


# ---------- 主流程 ----------
def load_arm(name):
    p = ROOT / "experiments" / f"{name}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d


def op1_recount(d):
    """(OP1) 从 records 重数，必须精确复现已存的汇总字段。"""
    recs = d["records"]
    wins = sum(r["verdict"] == "A" for r in recs)
    dec = sum(r["verdict"] in ("A", "B") for r in recs)
    inc = sum(r["verdict"] == "inconsistent" for r in recs)
    fail = sum(r["verdict"] is None for r in recs)
    rate = wins / dec if dec else float("nan")
    ok = (wins == d["a_wins"] and dec == d["decided"] and inc == d["inconsistent"]
          and fail == d["api_fail"] and abs(rate - d["rate"]) < 1e-12)
    return ok, dict(wins=wins, decided=dec, inconsistent=inc, api_fail=fail, rate=rate)


def analyse(tag, name, note):
    d = load_arm(name)
    op1, cnt = op1_recount(d)
    recs = d["records"]
    decided = [r for r in recs if r["verdict"] in ("A", "B")]
    point = cnt["rate"]
    side = point > 0.5                       # 点估计在 0.5 的哪一侧

    out = {"tag": tag, "file": name, "note": note, "published": {
        "a_wins": d["a_wins"], "decided": d["decided"], "rate": d["rate"],
        "p": d["p"], "jeffreys": d["jeffreys"], "api_fail": d["api_fail"],
        "resolve_rate": d.get("resolve_rate")}, "OP1_recount_exact": op1, "recount": cnt}

    # (OP2) 退化锚点：每对自成一族
    units1 = [(int(r["verdict"] == "A"), 1) for r in decided]
    lo1, hi1, ne1 = cluster_boot(units1)
    jlo, jhi = jeffreys(cnt["wins"], cnt["decided"])
    out["OP2_degenerate"] = {"ci": [lo1, hi1], "jeffreys": [jlo, jhi],
                             "max_dev": max(abs(lo1 - jlo), abs(hi1 - jhi)),
                             "pass": max(abs(lo1 - jlo), abs(hi1 - jhi)) < 0.03,
                             "n_eff": ne1, "OP4_pass": ne1 >= 1900}

    out["grains"] = {}
    for gname, fn in FAMS.items():
        groups = defaultdict(lambda: [0, 0])
        for r in decided:
            g = groups[fn(r["material"])]
            g[0] += r["verdict"] == "A"
            g[1] += 1
        units = [tuple(v) for v in groups.values()]
        sizes = [v[1] for v in groups.values()]
        lo, hi, ne = cluster_boot(units)
        contains_half = lo <= 0.5 <= hi

        # (A2) LOFO
        keys = sorted(groups)
        folds, skipped = [], 0
        for k in keys:
            w = cnt["wins"] - groups[k][0]
            n = cnt["decided"] - groups[k][1]
            if n < MIN_FOLD:
                skipped += 1
                continue
            folds.append({"drop": k, "n": n, "rate": w / n, "same_side": (w / n > 0.5) == side})
        all_same = all(f["same_side"] for f in folds)

        out["grains"][gname] = {
            "n_families": len(groups), "hhi": hhi(sizes), "eff_families": 1 / hhi(sizes),
            "max_family": max(sizes), "ci": [lo, hi], "width": hi - lo,
            "published_width": d["jeffreys"][1] - d["jeffreys"][0],
            "widening": (hi - lo) / (d["jeffreys"][1] - d["jeffreys"][0]),
            "contains_half": bool(contains_half),
            "n_eff_boot": ne, "OP4_pass": ne >= 1900,
            "OP3_pass": sum(sizes) == cnt["decided"],
            "lofo": {"n_folds": len(folds), "skipped": skipped, "all_same_side": bool(all_same),
                     "valid": len(folds) >= 2,
                     "worst": min(folds, key=lambda f: abs(f["rate"] - 0.5))if folds else None,
                     "flipped": [f for f in folds if not f["same_side"]]},
        }

    # 判决只认 (U1)
    g = out["grains"]["U1_drop_last"]
    if not (op1 and out["OP2_degenerate"]["pass"] and g["OP3_pass"] and g["OP4_pass"]):
        out["verdict"] = "VOID_operational_check_failed"
    elif not g["lofo"]["valid"]:
        out["verdict"] = "VOID_lofo_invalid"
    elif not g["contains_half"] and g["lofo"]["all_same_side"]:
        out["verdict"] = "W1_robust"
    elif g["lofo"]["all_same_side"]:
        out["verdict"] = "W2a_direction_only"
    else:
        out["verdict"] = "W2b_direction_too"
    return out, decided, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=ROOT / "experiments/judge_cluster.json")
    a = ap.parse_args()

    res = {"pre_registered": "analysis/arch/judge_cluster.py", "arms": [],
           "B_boot": B_BOOT, "B_perm": B_PERM, "seed": SEED}
    e3_decided = e3_recs = None
    for tag, name, note in ARMS:
        o, decided, recs = analyse(tag, name, note)
        res["arms"].append(o)
        if tag == "e3":
            e3_decided, e3_recs = decided, recs

    # (C) 只对 e3
    c1 = perm_chi2([fam_drop_last(r["material"]) for r in e3_decided],
                   [r["verdict"] for r in e3_decided])
    ans = [r for r in e3_recs if r["verdict"] is not None]
    c2 = perm_chi2([fam_drop_last(r["material"]) for r in ans],
                   ["resolved" if r["verdict"] in ("A", "B") else "inconsistent" for r in ans])
    res["C1_fam_x_winner"] = {"chi2": c1[0], "p": c1[1], "n_fam": c1[2], "n": len(e3_decided)}
    res["C2_fam_x_resolved"] = {"chi2": c2[0], "p": c2[1], "n_fam": c2[2], "n": len(ans)}

    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 打印 ----
    for o in res["arms"]:
        p = o["published"]
        print(f"\n=== {o['tag']}  {o['file']}")
        print(f"    {o['note']}")
        print(f"  已发表：A 胜 {p['a_wins']}/{p['decided']} = {p['rate']:.1%}  "
              f"p={p['p']:.3g}  Jeffreys [{p['jeffreys'][0]:.3f}, {p['jeffreys'][1]:.3f}]"
              f"  api_fail={p['api_fail']}")
        print(f"  (OP1) 逐条重数精确复现：{o['OP1_recount_exact']}")
        d2 = o["OP2_degenerate"]
        print(f"  (OP2) 退化锚点 CI [{d2['ci'][0]:.3f}, {d2['ci'][1]:.3f}]  "
              f"最大偏差 {d2['max_dev']:.4f} -> {'通过' if d2['pass'] else '**不过**'}")
        for gname, g in o["grains"].items():
            lf = g["lofo"]
            print(f"  [{gname}] 族数 {g['n_families']}  HHI {g['hhi']:.3f}"
                  f"（有效 {g['eff_families']:.1f}）  最大族 {g['max_family']}")
            print(f"      族级 CI [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]  宽 {g['width']:.4f}"
                  f"（已发表 {g['published_width']:.4f}，{g['widening']:.2f}×）  "
                  f"含 0.5：{'**是**' if g['contains_half'] else '否'}")
            print(f"      LOFO {lf['n_folds']} fold（skip {lf['skipped']}）  "
                  f"全同侧：{lf['all_same_side']}"
                  + (f"  最险 drop='{lf['worst']['drop']}' -> {lf['worst']['rate']:.1%}"
                     if lf["worst"] else ""))
        print(f"  ==> 判决（只认 U1）：**{o['verdict']}**")

    print(f"\n=== (C) e3 的聚集性（描述性，不参与判决）")
    print(f"  (C1) 族 × 胜负   χ²={c1[0]:.2f}  置换 p={c1[1]:.4f}  （{c1[2]} 族，n={len(e3_decided)}）")
    print(f"  (C2) 族 × 可解性 χ²={c2[0]:.2f}  置换 p={c2[1]:.4f}  （{c2[2]} 族，n={len(ans)}）")
    print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
