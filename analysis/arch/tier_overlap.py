"""三档判官结果（16 胜 / 24 平 / 32 输）到底是不是在**同一批材质**上比出来的？

零 GPU、零 API、零远程磁盘写入：只读三份**已封盘**的判定 JSON（`recheck_judge.py` 每轮复核它们），
输出写本机 `experiments/tier_overlap.json`。

---------------------------------------------------------------- 为什么现在问这一条
账本给下一条臂写死了准入条件：**任何关于"32px 的毛病"的候选解释，必须先说清它怎么与
16 胜 / 24 平 / 32 输三档相容**。而这三档的分母根本不一样：

    16px  114 胜 / 199 判出（272 对）  57.3%  p=0.047
    24px   94 胜 / 188 判出（272 对）  50.0%  p=1
    32px   83 胜 / 202 判出（272 对）  41.1%  p=0.014

三档跑的是**同一批 272 对、同一个材质顺序**（本脚本的操作检验 O1 逐条核对），但判官在每一档
**答不上来的那 70–84 对不是同一批**。于是"胜率随画布单调下降"至少有两种读法：

  (H1) **同一批材质上真的在变**：同一个材质，画布越大 TRD 越吃亏。
  (H2) **换了一批材质在比**：每一档能判出的子集不同，三个数字各自在自己的子集上算，
       排序可以纯粹由"谁被判出"造出来，逐材质什么都没变。

这正是本项目已经栽过一次的地方：`scale_diag.py` 的跨方法比较被作废，就是因为它的筛子在两组上的
通过率差三倍（TRD 27.6% vs B2 74.4%）——**先看覆盖率，再看效应**。三档的判出率 73% / 69% / 74%
彼此接近，所以这次不是"通过率差三倍"那种硬作废，但**接近不等于同一批**，得直接去看交集。

---------------------------------------------------------------- 尺子与数据（跑之前定死）
只用三份 JSON 里的 `records[i] = {pair, material, verdict}`，`verdict ∈ {A, B, inconsistent}`
（`inconsistent` = 正反两种顺序问出的答案不一致，已弃用 = "判官在这一对上没有分辨力"）。
**不引入任何新尺子**（`edge`、结构门、各向异性一概不碰，它们都已被禁止当判据）。
配对检验用 McNemar 的精确二项形式（只看不一致对），`math.comb` 精确、无 scipy。

---------------------------------------------------------------- 判据（跑之前写死）
 (R1) **先看覆盖率**：报三档判出集合 D16 / D24 / D32 的大小、两两交集、三档共同交集 C，
      并与"判出与否互相独立"时的期望交集对照。|C| < 100 则本脚本**只作描述**，不下判定。
 (R2) **主问题（H1 还是 H2）**：在 C 上重算三档胜率。
      - 若 C 上三档排序仍是 16 > 24 > 32，**且** 24 vs 32 的配对 McNemar p < 0.05
        -> H2 被排除，"三档"是逐材质的真实变化，账本的准入条件原样保留。
      - 若 C 上排序不保或配对检验够不着 p<0.05 -> **不能排除 H2**：三档差异里有多少来自
        "换了一批材质"未定，账本必须写上这一条限制（候选解释还要解释"谁被判出"）。
 (R3) **方向预测（写在跑之前）**：预测 H1 成立（C 上排序保住）。理由：三档判出率接近，
      且判出与否主要该由材质本身好不好比决定，而不是画布大小。
 (R4) **口径限制（沿用 `edge_vs_judge.py` 的 S1）**：16px 那一档来自**另一个 run**（`TRD16c_rr4`），
      24 / 32 两档才是同一个 `runs/trd_v10` 检查点。**所以主判据只用 24 vs 32**，
      16 那一档一并报出但不进判定。
 (R5) 本脚本**不产生任何关于 32px 内容层面的描述**，也**不足以开新臂**：它只回答
      "三档能不能当成同一批材质上的趋势来读"。无论结论如何都不改任何配置。
 (R6) 另报一栏**事后**观察（明确标注 post-hoc、不进判据）：只在某一档被判出的"独占"材质上的胜率——
      若独占子集的胜率与共同子集差很多，说明判出与否本身带信息。
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(ROOT, "experiments")

# (档位, 文件, 账本里的 (a_wins, decided))
TIERS = [
    ("16", "judge_full_TRD16c_rr4_vs_B2_16.json", (114, 199)),
    ("24", "judge_full_TRD24_rr4_vs_B2_24.json", (94, 188)),
    ("32", "judge_full_TRD32_rr4_vs_B2_32.json", (83, 202)),
]


def binom_p(k, n):
    """Exact two-sided binomial test against p=0.5 (no scipy on the servers)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) for i in range(n + 1)]
    total = float(2 ** n)
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-9) / total)


def main():
    recs, meta = {}, {}
    for tier, fn, ledger in TIERS:
        d = json.load(open(os.path.join(EXP, fn), encoding="utf-8"))
        recs[tier] = d["records"]
        meta[tier] = d
        # O1a: 三份 JSON 的胜负数必须与账本一致（防止读错文件）
        assert (d["a_wins"], d["decided"]) == ledger, (tier, d["a_wins"], d["decided"], ledger)

    # O1b: 三档必须是同一批对、同一个材质顺序
    n_pairs = len(recs["16"])
    assert all(len(recs[t]) == n_pairs for t, _, _ in TIERS), "三档对数不同"
    for i in range(n_pairs):
        a, b, c = recs["16"][i], recs["24"][i], recs["32"][i]
        assert a["pair"] == b["pair"] == c["pair"] == i, ("pair 错位", i)
        assert a["material"] == b["material"] == c["material"], ("材质错位", i)

    verd = {t: [r["verdict"] for r in recs[t]] for t in recs}
    dec = {t: set(i for i, v in enumerate(verd[t]) if v in ("A", "B")) for t in verd}
    # O1c: 判出数与 JSON 自报的一致
    for t, _, _ in TIERS:
        assert len(dec[t]) == meta[t]["decided"], (t, len(dec[t]), meta[t]["decided"])

    out = {"n_pairs": n_pairs, "coverage": {}, "overlap": {}, "common": {}, "posthoc": {}}

    # ---- (R1) 覆盖率
    for t, _, _ in TIERS:
        k = sum(1 for i in dec[t] if verd[t][i] == "A")
        out["coverage"][t] = {
            "decided": len(dec[t]),
            "resolvable_rate": round(len(dec[t]) / n_pairs, 3),
            "a_wins": k,
            "rate": round(k / len(dec[t]), 3),
            "p": binom_p(k, len(dec[t])),
        }

    C = dec["16"] & dec["24"] & dec["32"]
    for x, y in (("16", "24"), ("16", "32"), ("24", "32")):
        obs = len(dec[x] & dec[y])
        exp = len(dec[x]) * len(dec[y]) / n_pairs  # 判出与否互相独立时的期望
        out["overlap"][f"{x}&{y}"] = {"obs": obs, "exp_if_independent": round(exp, 1)}
    exp3 = len(dec["16"]) * len(dec["24"]) * len(dec["32"]) / n_pairs ** 2
    out["overlap"]["all3"] = {"obs": len(C), "exp_if_independent": round(exp3, 1)}

    # ---- (R2) 共同子集上的三档胜率
    Cs = sorted(C)
    for t, _, _ in TIERS:
        k = sum(1 for i in Cs if verd[t][i] == "A")
        out["common"][t] = {"n": len(Cs), "a_wins": k,
                            "rate": round(k / len(Cs), 3) if Cs else None,
                            "p": binom_p(k, len(Cs))}

    # 配对 McNemar（精确二项，只看不一致对）
    for x, y in (("16", "32"), ("24", "32"), ("16", "24")):
        b = sum(1 for i in Cs if verd[x][i] == "A" and verd[y][i] == "B")  # x 赢 y 输
        c = sum(1 for i in Cs if verd[x][i] == "B" and verd[y][i] == "A")
        out["common"][f"mcnemar_{x}_vs_{y}"] = {
            "x_win_y_lose": b, "x_lose_y_win": c, "discordant": b + c,
            "p": binom_p(b, b + c),
            # 翻转的材质名一并留下：这是下一个候选解释要正面对上的具体清单（线索，不是证据）
            "mats_x_win_y_lose": [recs[x][i]["material"] for i in Cs
                                  if verd[x][i] == "A" and verd[y][i] == "B"],
            "mats_x_lose_y_win": [recs[x][i]["material"] for i in Cs
                                  if verd[x][i] == "B" and verd[y][i] == "A"]}

    # ---- (R6) 事后：只在某一档判出的"独占"材质
    for t, _, _ in TIERS:
        others = set()
        for u, _, _ in TIERS:
            if u != t:
                others |= dec[u]
        only = sorted(dec[t] - others)
        k = sum(1 for i in only if verd[t][i] == "A")
        out["posthoc"][f"only_{t}"] = {"n": len(only), "a_wins": k,
                                       "rate": round(k / len(only), 3) if only else None,
                                       "p": binom_p(k, len(only))}

    dst = os.path.join(EXP, "tier_overlap.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("== 覆盖率（分母都是 %d 对）" % n_pairs)
    for t, _, _ in TIERS:
        c = out["coverage"][t]
        print("  %spx  判出 %3d (%.0f%%)   TRD %3d/%3d = %.1f%%  p=%.4g"
              % (t, c["decided"], 100 * c["resolvable_rate"], c["a_wins"],
                 c["decided"], 100 * c["rate"], c["p"]))
    print("== 判出集合的交集（obs vs 独立时的期望）")
    for k, v in out["overlap"].items():
        print("  %-8s %3d   期望 %.1f" % (k, v["obs"], v["exp_if_independent"]))
    print("== 三档共同判出的 %d 个材质上" % len(Cs))
    for t, _, _ in TIERS:
        v = out["common"][t]
        print("  %spx  TRD %3d/%3d = %.1f%%  p=%.4g"
              % (t, v["a_wins"], v["n"], 100 * v["rate"], v["p"]))
    for key in ("mcnemar_16_vs_32", "mcnemar_24_vs_32", "mcnemar_16_vs_24"):
        v = out["common"][key]
        print("  %-16s 翻转 %d:%d（共 %d）  p=%.4g"
              % (key, v["x_win_y_lose"], v["x_lose_y_win"], v["discordant"], v["p"]))
    print("== 事后（不进判据）：只在该档判出的材质")
    for t, _, _ in TIERS:
        v = out["posthoc"][f"only_{t}"]
        print("  只在 %spx 判出 n=%2d  TRD %2d 胜 = %s  p=%.4g"
              % (t, v["n"], v["a_wins"],
                 "%.1f%%" % (100 * v["rate"]) if v["rate"] is not None else "n/a", v["p"]))
    print("写入 %s" % dst)


if __name__ == "__main__":
    main()
