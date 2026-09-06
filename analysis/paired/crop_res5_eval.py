"""对 crop_scale_study 系列 JSON 做统计，并对照预注册判据（paper.md）。

预注册（写于运行前，git 6442f0d 为时间戳）：
  1. 48 档胜率 < 16 档胜率；
  2. 五档 {16,24,32,48,64} 上，胜率与分辨率 Spearman ρ < 0 且 p < 0.05；
  3. 若 ρ ≥ 0，或 48/64 胜率不低于 16 档 → 预测证伪，如实报告。

jzs_train 环境无 scipy：二项检验用 math.comb 精确算，
Spearman 的 p 用置换检验（逐对层面：分辨率 vs 胜负 0/1，与 run1 口径一致）。
"""

import argparse
import json
import math
import random
from pathlib import Path


def binom_p(w, n):
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, n - w) + 1)) / 2 ** n)


def rank(xs):  # 平均秩（处理并列）
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", type=Path)
    ap.add_argument("--perms", type=int, default=100000)
    args = ap.parse_args()
    recs = json.loads(args.json.read_text())

    fired = [r for r in recs if r["fired"]]
    judged = [r for r in fired if r.get("vlm") in ("before", "after")]
    inc = sum(1 for r in fired if r.get("vlm") == "inconsistent")
    miss = len(fired) - len(judged) - inc
    sizes = sorted({r["size"] for r in recs})
    print(f"{args.json}: 共 {len(recs)} 例，触发 {len(fired)}，"
          f"有效判断 {len(judged)}，不一致弃 {inc}，API 失败 {miss}")

    w = sum(1 for r in judged if r["vlm"] == "after")
    print(f"总胜率 {w}/{len(judged)} = {w/len(judged):.0%}  p={binom_p(w, len(judged)):.3g}")
    tier = {}
    for n in sizes:
        s = [r for r in judged if r["size"] == n]
        ww = sum(1 for r in s if r["vlm"] == "after")
        tier[n] = (ww, len(s))
        print(f"  {n:>2}px: {ww}/{len(s)} = {ww/len(s):.0%}  p={binom_p(ww, len(s)):.3g}"
              if s else f"  {n:>2}px: 无有效判断")

    # 逐对层面 Spearman + 置换 p（单侧，方向为负）
    x = [r["size"] for r in judged]
    y = [1 if r["vlm"] == "after" else 0 for r in judged]
    rho = spearman(x, y)
    rng = random.Random(0)
    hits = 0
    ys = y[:]
    for _ in range(args.perms):
        rng.shuffle(ys)
        if spearman(x, ys) <= rho:
            hits += 1
    p_perm = (hits + 1) / (args.perms + 1)
    print(f"Spearman(分辨率, 胜负) ρ={rho:+.3f}  置换 p(单侧,ρ≤观测)={p_perm:.4g}")

    if 48 in tier and 64 in tier and 16 in tier:
        r16 = tier[16][0] / tier[16][1] if tier[16][1] else float("nan")
        r48 = tier[48][0] / tier[48][1] if tier[48][1] else float("nan")
        r64 = tier[64][0] / tier[64][1] if tier[64][1] else float("nan")
        c1 = r48 < r16
        c2 = rho < 0 and p_perm < 0.05
        falsified = rho >= 0 or r48 >= r16 or r64 >= r16
        print("\n== 预注册判据 ==")
        print(f"1. 48档({r48:.0%}) < 16档({r16:.0%}): {'通过' if c1 else '不通过'}")
        print(f"2. ρ<0 且 p<0.05: {'通过' if c2 else '不通过'} (ρ={rho:+.3f}, p={p_perm:.4g})")
        print(f"3. 证伪条件(ρ≥0 或 48/64≥16档): {'触发——预测证伪' if falsified else '未触发'}")


if __name__ == "__main__":
    main()
