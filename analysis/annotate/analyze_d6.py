"""复算 `study_labels.csv`（D6）——论文引用了它的 60%，却没有脚本能算出来。

这份标注是仓库里**最大的一组人工盲比**（163 条 real + 8 条注意力检查），
但直到 2026-09-09 为止**没有任何脚本或文档引用它**，只有论文 §4.2 里
「不带先验的同一模型输给基线 60%」这一句间接用到。数字没错，可是
净克隆里没法验证——这个脚本补上这个洞。

它包含三组比较，`left`/`right` 两列决定是哪一组：
  baseline vs model   133 对（去平局 116）—— 论文 §4.2 的 60%
  artist   vs baseline 17 对（去平局 16）—— §4.1 主张的第三个测量
  artist   vs model    13 对（去平局 11）

⚠ 与 B2（`b2_labels.csv`）有 **4/17 材质重叠**，且**同一个标注者**，
所以真人 vs 基线那组只能算「另一组比较」，不能叫独立复现。
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test, jeffreys                        # noqa: E402

CSV = ROOT / "experiments/annotate/study_labels.csv"
B2 = ROOT / "experiments/annotate/b2_labels.csv"

PAIRS = [("baseline", "model", "baseline", "基线 vs 模型（§4.2 的 60%）"),
         ("artist", "baseline", "artist", "真人 vs 基线（§4.1 第三个测量）"),
         ("artist", "model", "artist", "真人 vs 模型")]


def main():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    chk = [x for x in rows if x["kind"] == "check"]
    ok = sum(x["chosen"] == "good" for x in chk)
    print(f"{CSV.name}：{len(rows)} 行，注意力检查 {ok}/{len(chk)}")
    if chk and ok < len(chk):
        print("  ⚠ 注意力检查未全对")

    for a, b, win, label in PAIRS:
        g = [x for x in rows if x["kind"] == "real"
             and sorted((x["left"], x["right"])) == sorted((a, b))]
        nt = [x for x in g if x["choice"] != "tie"]
        if not nt:
            continue
        k = sum(x["chosen"] == win for x in nt)
        lo, hi = jeffreys(k, len(nt))
        print(f"\n{label}")
        print(f"  {len(g)} 对，平局弃 {len(g) - len(nt)}，有效 {len(nt)}")
        print(f"  {win} 胜 {k}/{len(nt)} = {k/len(nt):.1%}"
              f"  p={binom_test(k, len(nt)):.3g}  Jeffreys [{lo:.0%},{hi:.0%}]")

    # 与 B2 的重叠——真人vs基线那组不能宣称独立
    ab = {x["material"] for x in rows if x["kind"] == "real"
          and sorted((x["left"], x["right"])) == ["artist", "baseline"]}
    b2 = {x["material"] for x in csv.DictReader(B2.open(encoding="utf-8"))
          if x["kind"] == "real"}
    ov = ab & b2
    print(f"\n与 B2 的材质重叠：{len(ov)}/{len(ab)}"
          f" —— {'部分重叠，勿称独立复现' if ov else '无重叠'}")
    print("标注者与 B2、A4 同为一人（论文限制节已述）。")


if __name__ == "__main__":
    main()
