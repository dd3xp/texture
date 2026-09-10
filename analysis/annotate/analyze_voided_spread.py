"""**例外分析**：那份被自己闸门作废的可辨认性标注，用户要求给出结论。

⚠ **这个脚本是一个被明确标注的例外，不是正常路径。**
`reduce_paired_orders.py` 的作废闸门**保持原样、对以后所有轮次照旧有效**——
我没有给它加 `--force`，因为覆盖开关只会把症状藏起来。

--- 为什么允许这个例外（理由在看结果之前就已提交：`1238ac8`）---

`study_spread_v2.csv` 未通过同图检查（4/4 错，闸门阈值 >2）。但闸门声明的目的是
"两张卡没有被比较"，而**三个与结果无关的量证明标注者在比较**：

  两序选到同一臂  **34/39 = 87.2%**（纯选边策略此项应≈0%，因两序选同侧必选到不同臂；
                  且 87.2% 落在本项目引用的判官区间 83.5–89.6% 内）
  位置偏好        左 40/76 = 52.6%，p=0.73（对比接缝那次 71%，p=0.0027）
  用过"分不出"    真题里 2 次 —— 标注者知道该选项且用过

**闸门失效的根因是我的**：页面从未告知"有几对是故意放的同一张图"
（`grep` 当时 0 处命中），所以那道检查测的是**猜规则的能力**，不是比较。
已修（模板现在明写），但那份标注是在修之前做的。

--- 报告纪律（写在看结果之前）---

1. 结论**无论方向都照报**，不挑；
2. 报告里必须**同时出现"该次标注未通过注意力检查"**这句话，不许只报数字；
3. 证据强度**低于一份干净标注**：这是"闸门可能误伤"而非"闸门确认通过"，
   任何据此的主张都要写成**待一份干净标注复现**；
4. 判据沿用 `spread_recognise.py`（`e9e3045`）与 `analyze_pair_study.py` 的
   `spread` 条目，**不新设**：主判据 = 重标版被选 >50% 且二项 p<0.05。
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test, jeffreys                        # noqa: E402

CSV = ROOT / "experiments/annotate/study_spread_v2.csv"
TARGET, OTHER = "after", "before"


def main():
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8-sig")))
    real = [r for r in rows if r["kind"] == "real"]
    print("=" * 68)
    print("⚠ 例外分析：该次标注**未通过注意力检查**（同图 4/4 错，闸门判作废）。")
    print("  允许分析的理由见本文件文档串，且已于 1238ac8 在看结果之前提交。")
    print("  证据强度低于一份干净标注，任何主张须写成待干净标注复现。")
    print("=" * 68)

    # 与结果无关的质量指标（复算，供读者自行判断）
    g = defaultdict(list)
    for r in real:
        g[r["pair"]].append(r["chosen"])
    paired = {k: v for k, v in g.items() if len(v) == 2}
    agree = sum(1 for v in paired.values() if v[0] == v[1])
    side = Counter(r["choice"] for r in real)
    L, R = side.get("left", 0), side.get("right", 0)
    print(f"\n质量（与胜负无关）：两序同臂 {agree}/{len(paired)} = {agree/len(paired):.1%}"
          f"；位置 左{L}/右{R} p={binom_test(L, L + R):.3g}"
          f"；用过分不出 {side.get('tie', 0)} 次")

    # 归约：两序一致才计入，取那个一致的臂
    kept = [v[0] for v in paired.values() if v[0] == v[1] and v[0] in (TARGET, OTHER)]
    ties = sum(1 for v in paired.values() if v[0] == v[1] and v[0] == "tie")
    w, t = sum(1 for x in kept if x == TARGET), len(kept)
    disc = len(paired) - len(kept) - ties
    print(f"\n归约：{len(paired)} 对 -> 两序一致且非平局 {t} 对"
          f"（不一致弃 {disc}，一致判平局 {ties}）")
    if not t:
        print("无有效判断"); return 2
    p = binom_test(w, t)
    lo, hi = jeffreys(w, t)
    print(f"\n主判据（更容易认出）：重标版被选 {w}/{t} = {w/t:.0%}"
          f"  p={p:.3g}  [{lo:.0%},{hi:.0%}]")
    if w / t > 0.5 and p < 0.05:
        print("  -> 方向为正且显著。**但这是一份未通过注意力检查的标注**，")
        print("     只能作为「值得用干净标注复现」的依据，不能直接当结论写进正文。")
    else:
        print("  -> 不成立。⚠ 判官那条腿已证明无分辨力（合并 12/18 p=0.238），")
        print("     人这条腿在这份（未通过检查的）标注上也没给出显著方向。")
    # 分层（描述性）
    strat = defaultdict(lambda: [0, 0])
    pr = {r["pair"]: r for r in real}
    for k, v in paired.items():
        if v[0] == v[1] and v[0] in (TARGET, OTHER):
            s = strat[pr[k]["stratum"]]
            s[1] += 1
            s[0] += v[0] == TARGET
    print("\n分层（描述性，不参与判定）：")
    for k in sorted(strat):
        a, b = strat[k]
        print(f"  {k:<6} {a}/{b} = {a/b:.0%}  p={binom_test(a, b):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
