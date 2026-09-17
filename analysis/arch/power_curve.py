#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判官这台仪器能测到多大的效应 —— 精确二项功效曲线（零 API、零 GPU）。

为什么要有它：(M40) 的下一步写死了「⛔ 不许靠加臂碰运气，按 (M40) 的话去解决功效本身」，
而 (M37)(M38)(M40) 三轮实测的可解率是 63.2% -> 56.0% -> 50.4%（在往下掉），
对应实得 n = 79 / 70 / 63。本文件把"这台仪器在给定 n 上最小能测到多少"算成一张表，
供以后**跑前**选杠杆用。

⛔ 三条使用纪律（写在代码里，免得以后被当成别的东西用）：
 1. 这里算的是**裸配对二项**的最小可测效应（MDE）。主判据是**族级 (U1) CI**，
    它比裸二项**更宽**（(M37)-(M40) 实测拓宽 0.95x-1.08x）=> 这张表是**下界/最乐观值**。
 2. ⛔ 不许用它去给任何**已经跑完**的臂"重新解释"判决（那是事后功效分析，没有意义）。
    它只回答跑前的那个问题：这条杠杆若有用、其效应有没有可能落在可测范围内。
 3. ⛔ 不许用它放宽任何已定的门槛。
"""
import argparse
from math import comb, exp, lgamma, log


def two_sided_reject_set(n, alpha=0.05):
    """精确二项双侧检验（p0=0.5）在 n 次试验下的拒绝域 = 使 p 值 <= alpha 的 k 集合。
    ⚠ 全程整数比较（2*min(S_lo,S_hi)*den <= num*2^n）：n 上千时 2.0**n 会 OverflowError。"""
    num, den = alpha.as_integer_ratio()
    c = [comb(n, k) for k in range(n + 1)]
    cum, s = [], 0
    for k in range(n + 1):
        s += c[k]
        cum.append(s)
    tot = 1 << n
    out = set()
    for k in range(n + 1):
        lo = cum[k]
        hi = tot - (cum[k - 1] if k else 0)
        if 2 * min(lo, hi) * den <= num * tot:
            out.add(k)
    return out


def crit_rate(n, alpha=0.05):
    """账本一直在说的那个"门槛"：**观测到**多大的胜率才够显著（= 拒绝域上边界 / n）。
    ⚠ 这是对**观测值**的要求，⛔ 不是对**真效应**的要求 —— 两者差着一个功效，见 mde()。"""
    r = two_sided_reject_set(n, alpha)
    hi = [k for k in r if k > n / 2.0]
    return (min(hi) / n) if hi else None


def power(n, p, reject):
    """真胜率为 p 时落进拒绝域的概率。⚠ 对数空间求和：n 大时 comb*p**k 会上/下溢。"""
    lp, lq = log(p), log(1.0 - p)
    base = lgamma(n + 1)
    return sum(exp(base - lgamma(k + 1) - lgamma(n - k + 1) + k * lp + (n - k) * lq)
               for k in reject)


def mde(n, alpha=0.05, target=0.80, step=0.0005):
    """最小可测效应：使功效 >= target 的最小真胜率（单调，直接扫）。"""
    if n < 6:
        return None
    reject = two_sided_reject_set(n, alpha)
    if not reject:
        return None
    p = 0.5
    while p < 1.0:
        p += step
        if power(n, p, reject) >= target:
            return p
    return None


def n_for(p_true, alpha=0.05, target=0.80, nmax=4000):
    """要在真胜率 p_true 上拿到 target 功效，至少需要多少条可解材质。"""
    lo, hi = 6, nmax
    if power(hi, p_true, two_sided_reject_set(hi, alpha)) < target:
        return None
    while lo < hi:                       # 功效对 n 非严格单调（离散），二分后再线性收一下
        mid = (lo + hi) // 2
        if power(mid, p_true, two_sided_reject_set(mid, alpha)) >= target:
            hi = mid
        else:
            lo = mid + 1
    n = lo
    while n <= nmax and power(n, p_true, two_sided_reject_set(n, alpha)) < target:
        n += 1
    return n


HIST = [                                 # 轮次, 可解率, 实得 n, 判决
    ("M37 检查点", 0.632, 79, "显著 (CKPT_LATE_WINS)"),
    ("M38 CFG   ", 0.560, 70, "CFG_NULL"),
    ("M40 代次  ", 0.504, 63, "GEN_NULL"),
]
N_MAT = 125                              # V_mat 测试集的材质数（judge_pairs 的分母上限）


def selftest():
    ok = True
    # 已知锚点：(M40) 写死"n=63 只测得到 >=63.5% 的效应" = 观测门槛，必须逐位复现
    c = crit_rate(63)
    print("[selftest] n=63 -> 观测门槛 %.4f（账本记 0.635）" % c)
    ok &= abs(c - 0.635) <= 0.002
    ok &= crit_rate(63) < mde(63)        # 真效应的要求必然严于观测门槛
    # n=79 实测 26/79=32.9% 显著 => 该 n 的拒绝域必须含 26
    ok &= 26 in two_sided_reject_set(79)
    # n=70 实测 42/70=60% 不显著 => 该 n 的拒绝域必须不含 42
    ok &= 42 not in two_sided_reject_set(70)
    # n=63 实测 33/63 不显著
    ok &= 33 not in two_sided_reject_set(63)
    # 功效随 n 增大不减（抽查）
    ok &= power(200, 0.6, two_sided_reject_set(200)) > power(100, 0.6, two_sided_reject_set(100))
    print("[selftest] %s" % ("通过" if ok else "【禁】失败"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--target", type=float, default=0.80)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())

    print("精确二项双侧, alpha=%.3f, 功效目标=%.2f" % (a.alpha, a.target))
    print("\n[一] 三轮实测。两列是**两件不同的事**，账本此前只记了第一列：")
    print("     观测门槛 = 观测到多大胜率才算显著；MDE = 真效应要多大，才有 %.0f%% 把握观测到它"
          % (a.target * 100))
    for name, rr, n, verdict in HIST:
        print("  %s 可解率 %.1f%%  n=%-3d  观测门槛=%.1f%%  MDE=%.1f%%   判决: %s"
              % (name, rr * 100, n, crit_rate(n, a.alpha) * 100,
                 mde(n, a.alpha, a.target) * 100, verdict))

    print("\n[二] 这台仪器的上限：V_mat 只有 %d 条材质" % N_MAT)
    for rr in (0.504, 0.56, 0.632, 0.75, 0.90, 1.00):
        n = int(round(N_MAT * rr))
        print("  可解率 %5.1f%% -> n=%-3d  观测门槛=%.1f%%  MDE=%.1f%%"
              % (rr * 100, n, crit_rate(n, a.alpha) * 100, mde(n, a.alpha, a.target) * 100))

    print("\n[三] 反过来：想测到某个效应，需要多少条**可解**材质")
    for p in (0.70, 0.65, 0.625, 0.60, 0.575, 0.55):
        n = n_for(p, a.alpha, a.target)
        need = n / N_MAT
        tag = "" if need <= 1.0 else "  <- 【禁】超过 V_mat 全部 %d 条，单靠这台仪器做不到" % N_MAT
        print("  真胜率 %.1f%% -> 需 n=%-4d（= V_mat 的 %.2f 倍可解率）%s" % (p * 100, n, need, tag))

    print("\n[四] 为什么不能靠「换个更大的集合」买功效")
    print("  判官的分母 = `--set` 的条目数（judge_pairs.py:107 `load_set`）：V_mat=125（验证集）、E_mat=272（测试集）。")
    for nm, N in (("V_mat", 125), ("E_mat", 272)):
        for rr in (0.5, 1.0):
            n = int(round(N * rr))
            print("    %s 可解率 %3.0f%% -> n=%-3d MDE=%.1f%%" % (nm, rr * 100, n, mde(n) * 100))
    print("  ⛔ E_mat 是**测试集**（CLAUDE.md：只在最终报数时用一次）=> 选杠杆/挑配置一律只能用 V_mat，")
    print("     所以上面 [二] 那个 %.1f%% 就是这类活儿的**真上限**，⛔ 不许为了买功效去动测试集。" % (mde(125) * 100))


if __name__ == "__main__":
    main()
