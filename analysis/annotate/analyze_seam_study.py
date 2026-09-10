"""分析接缝对齐的人工盲比（`study_seam.html` 导出的 CSV）。

判据**已在跑之前写死**（预注册 `d4aad16`，题面见
`analysis/annotate/build_study_seam.py` 的文档串）。本脚本只是把那些判据
照搬成代码，**不引入任何新的口径**。写在 CSV 到达之前，正是为了让
「拿到数据再决定怎么算」这件事没有发生的余地。

  **注意力检查**：好图 vs 同图重糊。**错 >1 个则该次标注作废**，不出主判据。
  **主判据**：两轮材质合并（57 对），标注者偏好接缝对齐 >50% 且二项 p<0.05
             → 人能看出来，方法三的说法从「客观测量」升级为「人工盲比验证」。
  **证伪**：≤50% 或不显著 → 人也看不出来。接缝对齐是**客观更对、观感无差别**
             的改动；交付仍保留，但**所有「更好看」的说法都要撤掉**。
  **次判据（描述性，不参与判定）**：按材质集分层（orig42 / holdout60）。
             两层方向一致 → VLM 泛化轮的不复现是**判官分辨力**问题；
             方向相反 → 是**材质集**问题，比判官问题严重，要单独查。

平局按本项目一贯口径**弃用**（B2、D6 皆如此），并在输出里报出弃了几个。

CSV 列（`task_template.html` 导出）：
  idx,material,kind,struct,stratum,left,right,choice,chosen,ms
`chosen` 对 real 条目是 `seam` / `center` / `tie`；对 check 条目是 `good` / `blur`。

跑法：`python analysis/annotate/analyze_seam_study.py [路径.csv]`
自检：`python analysis/annotate/analyze_seam_study.py --selftest`
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test, jeffreys                        # noqa: E402

DEFAULT = ROOT / "experiments/annotate/study_seam.csv"
MAX_CHECK_WRONG = 1          # 预注册：错 >1 个即作废
STRATA = [("orig42", "原 42 条（判官曾给 10/12）"),
          ("holdout60", "留出 60 条（判官 12/18 未复现）")]


def load(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tally(rows):
    """返回 (选接缝数, 有效对数, 平局数)。"""
    win = tot = tie = 0
    for r in rows:
        c = (r.get("chosen") or "").strip()
        if c == "tie":
            tie += 1
        elif c in ("seam", "center"):
            tot += 1
            win += c == "seam"
    return win, tot, tie


def report(name, win, tot, tie, indent="") -> bool:
    if not tot:
        print(f"{indent}{name}：无有效判断（平局 {tie}）")
        return False
    p = binom_test(win, tot)
    lo, hi = jeffreys(win, tot)
    sig = win / tot > 0.5 and p < 0.05
    print(f"{indent}{name}：接缝对齐被选 {win}/{tot} = {win/tot:.0%}  "
          f"p={p:.3g}  [{lo:.0%},{hi:.0%}]   （平局弃 {tie}）")
    return sig


def analyse(path: Path) -> int:
    rows = load(path)
    real = [r for r in rows if r.get("kind") == "real"]
    check = [r for r in rows if r.get("kind") == "check"]
    print(f"{path.name}：{len(rows)} 行，real {len(real)}，注意力检查 {len(check)}")

    # —— 注意力检查（必须排在主判据之前）——
    wrong = sum(1 for r in check if (r.get("chosen") or "").strip() != "good")
    print(f"\n注意力检查：错 {wrong}/{len(check)}（预注册阈值：错 >{MAX_CHECK_WRONG} 即作废）")
    if not check:
        print("  -> 警告：这份 CSV 里没有检查条目，无法验标注质量；**不出主判据**。")
        return 2
    if wrong > MAX_CHECK_WRONG:
        print("  -> **该次标注作废**，主判据不予评估（判据已在跑前固定）。")
        return 1
    print("  -> 通过")

    # —— 主判据 ——
    win, tot, tie = tally(real)
    print()
    ok = report("主判据（两轮合并）", win, tot, tie)
    if ok:
        print("  -> **成立**：人能看出接缝对齐更好，方法三从「客观测量」"
              "升级为「人工盲比验证」。")
    else:
        print("  -> 不成立：**人也没看出来**。接缝对齐是客观更对、观感无差别的改动；")
        print("     交付仍保留（不会更差，平坦守卫排除了作弊），"
              "但**所有「更好看」的说法都要撤掉**，")
        print("     正文只能写「改善了可测的平铺连续性」。")

    # —— 次判据（描述性，不参与判定）——
    print("\n次判据（描述性，预先指定，不参与判定）——按材质集分层：")
    dirs = []
    for tag, label in STRATA:
        sub = [r for r in real if r.get("stratum") == tag]
        w, t, ti = tally(sub)
        report(f"{label}", w, t, ti, indent="  ")
        if t:
            dirs.append(w / t > 0.5)
    if len(dirs) == 2:
        if dirs[0] == dirs[1]:
            print("  -> 两层方向一致：VLM 泛化轮的不复现更像**判官分辨力**问题。")
        else:
            print("  -> 两层方向**相反**：这是**材质集**问题，比判官问题严重，"
                  "要单独查，别当判官噪声带过。")

    # 左右位置有没有系统偏好——不是判据，是标注质量的旁证
    side = Counter((r.get("choice") or "").strip() for r in real)
    l, r_ = side.get("left", 0), side.get("right", 0)
    if l + r_:
        print(f"\n左右选择分布：左 {l} / 右 {r_}"
              f"（二项 p={binom_test(l, l + r_):.3g}）—— 位置偏好的旁证，非判据")
    return 0


def selftest() -> int:
    """用合成 CSV 把每条分支都走一遍。判据是死的，分支不能是。"""
    import tempfile

    def make(rows):
        d = Path(tempfile.mkdtemp()) / "t.csv"
        with open(d, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "material", "kind", "struct",
                                              "stratum", "left", "right", "choice",
                                              "chosen", "ms"])
            w.writeheader()
            for i, (kind, stratum, chosen) in enumerate(rows):
                w.writerow({"idx": i, "material": "m", "kind": kind, "struct": 0,
                            "stratum": stratum, "left": "seam", "right": "center",
                            "choice": "left", "chosen": chosen, "ms": 100})
        return d

    def rows(kind, stratum, chosen, n):
        return [(kind, stratum, chosen)] * n

    cases = [
        ("检查错 2 个 -> 作废",
         rows("check", "orig42", "blur", 2) + rows("check", "orig42", "good", 3)
         + rows("real", "orig42", "seam", 40), 1),
        ("没有检查条目 -> 不出主判据",
         rows("real", "orig42", "seam", 40), 2),
        ("主判据成立",
         rows("check", "orig42", "good", 5)
         + rows("real", "orig42", "seam", 30) + rows("real", "holdout60", "seam", 10)
         + rows("real", "holdout60", "center", 5), 0),
        ("主判据不成立（对半开）",
         rows("check", "orig42", "good", 5)
         + rows("real", "orig42", "seam", 20) + rows("real", "orig42", "center", 20), 0),
        ("全平局 -> 无有效判断",
         rows("check", "orig42", "good", 5) + rows("real", "orig42", "tie", 20), 0),
        ("两层方向相反",
         rows("check", "orig42", "good", 5)
         + rows("real", "orig42", "seam", 18) + rows("real", "orig42", "center", 2)
         + rows("real", "holdout60", "center", 18) + rows("real", "holdout60", "seam", 2),
         0),
    ]
    bad = 0
    for name, rs, want in cases:
        print(f"\n{'=' * 62}\n[自检] {name}（期望退出码 {want}）\n{'=' * 62}")
        got = analyse(make(rs))
        if got != want:
            print(f"**退出码不符：{got} != {want}**")
            bad += 1
    print(f"\n自检 {len(cases)} 例，不符 {bad}")
    return 1 if bad else 0


def main():
    args = [a for a in sys.argv[1:]]
    if "--selftest" in args:
        raise SystemExit(selftest())
    path = Path(args[0]) if args else DEFAULT
    if not path.exists():
        raise SystemExit(f"没有 {path}\n"
                         f"用户标完 `experiments/annotate/study_seam.html` 后导出 CSV "
                         f"放到这里，或把路径当参数传进来。")
    raise SystemExit(analyse(path))


if __name__ == "__main__":
    main()
