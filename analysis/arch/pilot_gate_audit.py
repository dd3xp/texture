"""零 API、零 GPU：把 20 条判官试点的**空对照**汇总，量出这台仪器真正的地板。

为什么写：每条臂开跑前的试点都附 5 对"两边同一张图"的空对照，用来给可解率定地板。
20 条臂跑下来，这 100 对空对照**从没被合并看过一次**——单条臂只有 5 对，看不出名堂。
合起来能回答两件之前只能靠推理的事：

1. **地板不是 50%。** 论文期 `crop_scale_study.py` 的推导是：判官若不看内容、以概率 q 选第一张，
   跨序一致率 = 2q(1-q) <= 0.5，q=0.5 时取到 0.5。**那是上界，不是这台仪器的实测值。**
   合并 100 对空对照实测 **12%** → 反推 q ~ 0.94：`judge_pairs.py` 的判官在两图相同时
   **几乎总是挑同一个位置**，换序就翻面，于是判成 inconsistent。
2. 于是"可解率低"要分两种读法：低到 12% 附近 = 判官只在按位置作答（真的没信息）；
   60% 出头 = **远高于地板**，只是没到我们为了保功效而画的 65% 那条线。
   **把 62% 写成"判官没分辨力"是错的**，脚本里那句提示词要按这个改。

⚠ 本脚本一个胜负字段都不读（只读 kind/answered/resolved），不会把试点变成偷看结果。

    python analysis/arch/pilot_gate_audit.py
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from exact import binom_test, jeffreys     # noqa: E402

FIELDS = ("kind", "answered", "resolved")   # 白名单：verdict 之类一律不碰


def main():
    files = sorted(glob.glob(str(ROOT / "experiments" / "judge_pilot_*.json")))
    if not files:
        print("没有试点 JSON（净克隆里应当有，见 .gitignore 的豁免）")
        return 1
    R = RA = N = NA = 0
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        recs = [{k: x[k] for k in FIELDS} for x in d["records"]]
        r = [x for x in recs if x["kind"] == "real" and x["answered"]]
        n = [x for x in recs if x["kind"] == "null" and x["answered"]]
        rr, nn = sum(x["resolved"] for x in r), sum(x["resolved"] for x in n)
        R += rr; RA += len(r); N += nn; NA += len(n)
        rows.append((d["tag"], rr, len(r), nn, len(n), d["min_rate"], d["pass"]))

    print(f"{'臂':52s} {'真题':>9s} {'空对照':>8s}  门槛  过?")
    for tag, rr, rn, nn, nnn, mr, ok in rows:
        print(f"{tag:52s} {rr:3d}/{rn:<3d}{rr / rn:5.0%} {nn:2d}/{nnn:<2d}{nn / nnn:4.0%}  {mr:.2f}  "
              f"{'OK' if ok else 'X'}")

    print(f"\n合并真题   {R}/{RA} = {R / RA:.1%}  Jeffreys [{jeffreys(R, RA)[0]:.1%},{jeffreys(R, RA)[1]:.1%}]")
    lo, hi = jeffreys(N, NA)
    print(f"合并空对照 {N}/{NA} = {N / NA:.1%}  Jeffreys [{lo:.1%},{hi:.1%}]  "
          f"vs 50% 的二项 p = {binom_test(N, NA):.3g}")
    print("\n读法：空对照 = 两边同一张图，判官若按内容作答无从判起。实测远低于 2q(1-q) 的上界 0.5，")
    print("     说明它在平局上几乎总按位置作答（q ~ 0.94）→ **这台仪器的可解率地板约 12%，不是 50%**。")

    fails = [x for x in rows if not x[6]]
    if fails:
        print(f"\n未过门槛的 {len(fails)} 条（都**没有** full，结论只能是'没到我们要的功效'，不是'判官没分辨力'）：")
        for tag, rr, rn, nn, nnn, mr, _ in fails:
            print(f" - {tag}: 真题 {rr}/{rn} = {rr / rn:.0%}，空对照 {nn}/{nnn}；"
                  f"差 {int(mr * rn - rr) + (1 if mr * rn > rr else 0)} 条就过线")
    print("\n注意：n=15 时门槛的分辨力只有 6.7pp——pb05 以 10/15=67% 过线、pb10 以 8/13=62% 落线，")
    print("  两者相差不到一条记录。**别把'过/不过'读成两种性质不同的结果。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
