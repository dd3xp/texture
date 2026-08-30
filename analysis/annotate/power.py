"""A4 的功效：47 对有种子样本能测出多大的效应？

该在负责人花 12 分钟标注**之前**回答。
若这个规模只够测出巨大差异，就应当先说清楚，而不是标完再解释。

用精确二项检验（本来就是这么读的），不是正态近似——n=47 时两者有差别。
"""

import sys

import numpy as np
from scipy import stats


def power_at(n: int, p_true: float, alpha: float = 0.05, sims: int = 0) -> float:
    """真实胜率为 p_true 时，双尾精确二项检验在 n 次里达到显著的概率。

    精确算：对每个可能的胜数求二项检验 p 值，把显著的那些概率加起来。
    """
    ks = np.arange(n + 1)
    sig = np.array([stats.binomtest(int(k), n, 0.5).pvalue < alpha for k in ks])
    return float(stats.binom.pmf(ks, n, p_true)[sig].sum())


def main():
    ns = [int(x) for x in (sys.argv[1:] or [47, 20, 67, 120])]
    ps = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    print("双尾精确二项检验，alpha=0.05")
    print(f"\n{'n':>6}" + "".join(f"{p:>9.0%}" for p in ps))
    print("-" * (6 + 9 * len(ps)))
    for n in ns:
        print(f"{n:>6}" + "".join(f"{power_at(n, p):>9.0%}" for p in ps))
    print("\n列 = 真实胜率，格 = 测出显著的概率")

    n = ns[0]
    print(f"\n=== n={n}（A4 有种子层）===")
    for p in ps:
        pw = power_at(n, p)
        tag = "够" if pw >= 0.8 else ("勉强" if pw >= 0.5 else "不够")
        print(f"  真实胜率 {p:.0%} -> 功效 {pw:>4.0%}  [{tag}]")
    # 反过来：什么胜率才有 80% 功效
    lo = next((p for p in np.arange(0.5, 1.0, 0.005) if power_at(n, p) >= 0.8), None)
    print(f"\n  要有 80% 功效，真实胜率需达到 {lo:.0%} 以上")
    k = next(k for k in range(n // 2, n + 1)
             if stats.binomtest(k, n, 0.5).pvalue < 0.05)
    print(f"  n={n} 时，观察到 {k}/{n}（{k/n:.0%}）才达到 p<0.05")
    print("\n判读：A4 是**探路**不是终审。它能可靠回答的只有"
          "「有没有大到 ~70% 的效应」；\n"
          "      55–65% 的真实效应在这个规模下大概率测不出，"
          "那种情况必须扩到 A5 才有结论。")


if __name__ == "__main__":
    main()
