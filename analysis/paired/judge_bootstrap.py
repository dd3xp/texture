"""§3.3 的判读对那 72 条人工标注有多敏感？——自助重采样。

全篇最薄的一环是人工标注：72 对（A4）+ 24 对（B2），且**出自同一个人**。
§3.3 的整个判官准入协议就validate在这 72 条上，结论是三个判官分成三档：
`claude-opus-5` 保住方向与显著性、`gemini-3.1-pro` 压到不显著并抹平分层、
`gpt-5.6-sol` 把结论倒转。

标注者数量我改变不了，但**能量清楚这个判读有多稳**：把 72 条按条目有放回重采样，
每次重算三个判官的口径，看「谁属于哪一档」的判读复现率。
若某判官的档位在重采样下频繁翻转，§3.3 就不该照现在这样写。

口径与 `fig_judges.py` 一致：
  - 胜率分母只留 human/vlm 都 ∈ {baseline, model}（排除 artist 票）
  - 逐条一致率用全部 real
无 scipy：二项检验用 `analysis/exact.py` 的精确实现。
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test                                  # noqa: E402

A = ROOT / "experiments/annotate"
JUDGES = [("claude-opus-5", "vlm_a4_claude-opus-5.json"),
          ("gemini-3.1-pro", "vlm_a4_gemini.json"),
          ("gpt-5.6-sol", "vlm_a4_gpt-5_6-sol.json")]


def strat_gap(sub, strat):
    """分层证据：模型在 seeded 层与 plain 层的胜率之差。

    人的口径是 seeded 6% vs plain 33%（差 -27pp）——先验起作用那层反而更差。
    判官若把这个差抹平（接近 0）就是「抹平分层」，
    与胜率的显著性是**两条独立的腿**：一条脆，另一条未必。
    """
    se = [r for r in sub if strat.get(r["idx"]) == "seeded"]
    pl = [r for r in sub if strat.get(r["idx"]) == "plain"]
    if not se or not pl:
        return None
    f = lambda g: sum(1 for r in g if r["vlm"] == "model") / len(g)
    return f(se) - f(pl)


def verdict(sub, real, strat):
    """把一个判官在一批记录上的表现归成一档。

    档位定义照 §3.3 正文：
      preserve —— 方向对（判官也认为基线胜）**且**显著（p<0.05）
      compress —— 方向对但不显著
      invert   —— 方向反了（判官认为模型胜）
    """
    if not sub:
        return None, None, None
    k = sum(1 for r in sub if r["vlm"] == "baseline")
    rate = k / len(sub)
    p = binom_test(k, len(sub))
    agree = np.mean([r["human"] == r["vlm"] for r in real]) if real else float("nan")
    if rate <= 0.5:
        v = "invert"
    elif p < 0.05:
        v = "preserve"
    else:
        v = "compress"
    return v, rate, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", type=int, default=4000, help="重采样次数")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    strat = {int(r["idx"]): r["stratum"] for r in csv.DictReader(
        (A / "a4_labels.csv").open(encoding="utf-8"))}

    data = {}
    for name, f in JUDGES:
        recs = [r for r in json.loads((A / f).read_text(encoding="utf-8"))
                if r["kind"] == "real"]
        data[name] = {r["idx"]: r for r in recs}
    # **每个判官在自己的样本上自助**——这才对得上论文的口径：
    # §3.3 的 73%/62%/42% 各自算在该判官答上来的那些条目上，不是算在交集上。
    # 三者答上来的条目并不相同（API 失败与正反不一致弃样所致），
    # 交集只有 18 条，限制到交集会把样本压垮，测的就不是论文的主张了。
    own = {n: sorted(d) for n, d in data.items()}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"各判官答上来的 real 条目：" +
          "，".join(f"{n} {len(v)}" for n, v in own.items()) +
          f"；三者交集仅 {len(common)} 条")

    def slice_(name, idxs):
        recs = [data[name][i] for i in idxs]
        sub = [r for r in recs if r["human"] in ("baseline", "model")
               and r["vlm"] in ("baseline", "model")]
        return sub, recs

    print(f"\n{'判官':<16}{'点估计档位':>12}{'基线胜率':>10}{'逐条一致':>10}")
    print("-" * 50)
    point = {}
    for name, _ in JUDGES:
        sub, real = slice_(name, own[name])
        v, rate, ag = verdict(sub, real, strat)
        point[name] = v
        print(f"{name:<16}{v:>12}{rate:>10.1%}{ag:>10.1%}")

    rng = np.random.default_rng(a.seed)
    tally = {n: Counter() for n, _ in JUDGES}
    gaps = {n: [] for n, _ in JUDGES}
    rates = {n: [] for n, _ in JUDGES}
    tallies_v = {n: [] for n, _ in JUDGES}
    sep = 0
    for _ in range(a.B):
        vs = {}
        for name, _ in JUDGES:
            pool = own[name]
            idxs = list(rng.choice(pool, size=len(pool), replace=True))
            sub, real = slice_(name, idxs)
            v, _r, _ = verdict(sub, real, strat)
            tally[name][v] += 1
            vs[name] = v
            rates[name].append(_r)
            tallies_v[name].append(v)
            g = strat_gap(sub, strat)
            if g is not None:
                gaps[name].append(g)
        # 「三档分得开」= 三个判官落在三个不同的档位
        if len(set(vs.values())) == 3:
            sep += 1

    print(f"\n自助重采样 B={a.B}（按条目有放回）")
    print(f"{'判官':<16}{'preserve':>10}{'compress':>10}{'invert':>10}"
          f"{'点估计档复现率':>16}")
    print("-" * 62)
    for name, _ in JUDGES:
        t = tally[name]
        rep = t[point[name]] / a.B
        print(f"{name:<16}{t['preserve']/a.B:>10.1%}{t['compress']/a.B:>10.1%}"
              f"{t['invert']/a.B:>10.1%}{rep:>16.1%}")
    print(f"\n三个判官落在三个不同档位的比例：{sep/a.B:.1%}")

    # 第二条腿：分层。人是 -27pp（seeded 更差）；判官若抹平则趋近 0。
    print("")
    print("分层差（模型胜率 seeded - plain），人的口径 -27pp")
    print(f"{'判官':<16}{'点估计':>10}{'自助 2.5%':>12}{'自助 97.5%':>12}"
          f"{'保住负号的比例':>16}")
    print("-" * 66)
    for name, _ in JUDGES:
        pt = strat_gap(slice_(name, own[name])[0], strat)
        arr = np.array(gaps[name])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        neg = float((arr < 0).mean())
        print(f"{name:<16}{pt*100:>9.0f}pp{lo*100:>11.0f}pp{hi*100:>11.0f}pp"
              f"{neg:>16.1%}")

    # 第三条腿：**效应保留量的排序**，不依赖 p 值跨不跨 0.05 这条线。
    # 若 opus > gemini > gpt 的排序在重采样下很稳，论文就该靠这条腿而不是显著性。
    order = sum(1 for i in range(len(rates["claude-opus-5"]))
                if rates["claude-opus-5"][i] > rates["gemini-3.1-pro"][i]
                > rates["gpt-5.6-sol"][i])
    pair = sum(1 for i in range(len(rates["claude-opus-5"]))
               if rates["claude-opus-5"][i] > rates["gemini-3.1-pro"][i])
    both = sum(1 for i in range(len(rates["claude-opus-5"]))
               if tallies_v["claude-opus-5"][i] == "preserve"
               and tallies_v["gemini-3.1-pro"][i] != "preserve")
    B = len(rates["claude-opus-5"])
    print("")
    print("效应保留量（基线胜率，人 86%）：")
    for name, _ in JUDGES:
        arr = np.array(rates[name])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        print(f"  {name:<16}点估计 {np.mean(arr):.1%}  自助 [{lo:.0%}, {hi:.0%}]")
    print(f"  opus > gemini             ：{pair/B:.1%} 的重采样")
    print(f"  opus > gemini > gpt 全序   ：{order/B:.1%}")
    print(f"  「opus 显著而 gemini 不显著」：{both/B:.1%}  <- 论文现在靠的就是这条")
    print("判读：复现率高 -> §3.3 的三档划分对这 72 条标注稳健；")
    print("      复现率低 -> 该划分是这批标注的偶然产物，正文措辞需放宽。")


if __name__ == "__main__":
    main()
