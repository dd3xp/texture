"""预注册二（单元惯例说，写于运行前，git 8207f3f 为时间戳）的评估。

对照 crop_ctrl.json（原提示词）与 crop_fewunits.json（加 few large blocks 修饰词），
判据（跑前固定，paper.md）：
  1. 修饰后渲染图主周期显著变大（单元数变少）——先验证操作有效；
  2. 若单元数确实变少，裁剪胜率应低于对照组；
  3. 若单元数变少而胜率不降（差值 <5 个百分点或反向）→ 单元惯例说证伪。

jzs_train 无 scipy：操作检验用逐提示词配对符号检验（math.comb 精确二项），
胜率差附 Fisher 精确检验（超几何，仅供参考，不进判据——判据只看方向与幅度）。
"""

import argparse
import json
import statistics
import math
from pathlib import Path


def binom_p(w, n):
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, n - w) + 1)) / 2 ** n)


def fisher_two_sided(a, b, c, d):
    """2x2 表 [[a,b],[c,d]] 的 Fisher 精确检验（双侧，按概率求和）。"""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def hyp(x):
        return (math.comb(c1, x) * math.comb(n - c1, r1 - x)) / math.comb(n, r1)

    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    p_obs = hyp(a)
    return sum(p for x in range(lo, hi + 1) if (p := hyp(x)) <= p_obs * (1 + 1e-9))


def load(path):
    recs = json.loads(Path(path).read_text())
    fired = [r for r in recs if r["fired"]]
    judged = [r for r in fired if r.get("vlm") in ("before", "after")]
    wins = sum(1 for r in judged if r["vlm"] == "after")
    # period 每提示词只有一个（各 size 重复记录），去重取每提示词一份
    per = {r["prompt"]: r["period"] for r in recs}
    return recs, fired, judged, wins, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", type=Path, default=Path("experiments/crop_ctrl.json"))
    ap.add_argument("--treat", type=Path, default=Path("experiments/crop_fewunits.json"))
    args = ap.parse_args()

    (_, f_c, j_c, w_c, per_c) = load(args.ctrl)
    (_, f_t, j_t, w_t, per_t) = load(args.treat)

    for tag, f, j, w in [("对照", f_c, j_c, w_c), ("少单元", f_t, j_t, w_t)]:
        r = w / len(j) if j else float("nan")
        print(f"{tag}: 触发 {len(f)}，有效 {len(j)}，裁后胜 {w}/{len(j)} = {r:.0%}"
              f"  p={binom_p(w, len(j)):.3g}" if j else f"{tag}: 无有效判断")

    # —— 判据 1：操作检验（配对符号检验，同提示词 period 变大计一胜）——
    common = sorted(set(per_c) & set(per_t))
    up = sum(1 for p in common if per_t[p] > per_c[p])
    # 真中位数：sorted(v)[n//2] 在 n 为偶数时取的是上中位数（B12 因此把
    # 30.1/16.8/8.8 记成了 32.5/17.4/9.3）。这里只用于打印，判据看的是符号检验。
    med_c = statistics.median(per_c[p] for p in common)
    med_t = statistics.median(per_t[p] for p in common)
    p_sign = binom_p(up, len(common))
    ok1 = up > len(common) / 2 and p_sign < 0.05
    print(f"\n== 判据 1（操作有效性）==")
    print(f"period 中位数：对照 {med_c:.0f}px → 少单元 {med_t:.0f}px；"
          f"变大 {up}/{len(common)} 提示词，符号检验 p={p_sign:.3g} → "
          f"{'通过' if ok1 else '不通过（操作未生效，判据 2/3 无从谈起）'}")

    if not ok1 or not j_c or not j_t:
        return

    # —— 判据 2/3：胜率方向与幅度 ——
    r_c, r_t = w_c / len(j_c), w_t / len(j_t)
    diff = r_c - r_t
    p_f = fisher_two_sided(w_c, len(j_c) - w_c, w_t, len(j_t) - w_t)
    print(f"\n== 判据 2/3（胜率差）==")
    print(f"对照 {r_c:.0%} vs 少单元 {r_t:.0%}，差 {diff:+.1%}（Fisher p={p_f:.3g}，仅参考）")
    if diff >= 0.05:
        print("→ 判据 2 通过：少单元组胜率低 ≥5 个百分点，与单元惯例说预测一致")
    else:
        print("→ 判据 3 触发：胜率未按预测下降（差 <5pp 或反向）——单元惯例说证伪，机制需再修正")


if __name__ == "__main__":
    main()
