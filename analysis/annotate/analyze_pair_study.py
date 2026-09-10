"""分析另外两份排队中的人工盲比：`study_crop.html` 与 `study_ab60.html`。

接缝那份（`analyze_seam_study.py`）在 CSV 到达之前就把判据写死了，这两份没有。
三份 CSV 是同一批交回来的，若等数据到了再决定怎么算，**判据就是看着数据定的**。
所以本脚本写在 CSV 之前，判据照抄两份研究**建页时自己写下的问题**
（`build_study_crop.py` / `build_study_ab.py` 的文档串），不新设口径。

不改 `analyze_seam_study.py`：那份已预注册（`d4aad16`）并验过钥，
不该为了少写几行去动它。

--- crop：按结构尺度裁剪，修前 vs 修后（39 对 + 3 检查）---
建页时的问题："B5 的『15/16 可用』只是我的目视判断，单观察者。"
VLM 判官已在三个口径上给出 72–89% 偏好裁后，但判官是压缩的、且是同一家模型。
  主判据：标注者偏好 after >50% 且二项 p<0.05
          -> 主效应得到**人工确认**，不再只有 VLM 一条腿。
  证伪：<=50% 或不显著 -> 人看不出来。裁剪的依据退回到客观测量与 VLM，
          正文里凡是"看起来更好"的说法都要注明未经人工确认。

--- ab60：真人 vs 降采样基线（60 对 + 6 检查）---
建页时的问题："真人若明显赢，那个差距就是唯一值得做的东西。"
先验很要紧：同一比较在 24 对上是 **10/23 = 43%，p 不显著**（`b2_labels.csv`），
即**没测到差距**。60 对是加功效的重做，因此两个方向都有信息量。
  主判据：artist 被选 >50% 且 p<0.05 -> 差距真实存在且人眼可见。
  证伪：不显著 -> 24 对那个零结果在更大样本上复现，
          "基线已接近真人"的说法获得**更强**的支持（而不是"没结论"）。
  次判据（描述性，建页时就分好的层）：structured vs plain。
          两层差距的方向是本项目的核心不对称，方向不一致要单独查。

平局按本项目一贯口径**弃用**并报出个数；注意力检查错 >1 个则该次标注作废，
不出主判据——与接缝那份同规矩。

跑法：python analysis/annotate/analyze_pair_study.py crop|ab60 [路径.csv]
自检：python analysis/annotate/analyze_pair_study.py --selftest
退出码：0 正常 / 1 标注作废 / 2 没有检查条目
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test, jeffreys                        # noqa: E402

MAX_CHECK_WRONG = 1          # 预注册：错 >1 个即作废

STUDIES = {
    "crop": {
        "csv": ROOT / "experiments/annotate/study_crop.csv",
        "target": "after", "other": "before",
        "title": "裁剪修前/修后（B6）",
        "hit": "主效应得到人工确认：裁后更好不再只有 VLM 一条腿。",
        "miss": ("人没看出来。裁剪的依据退回客观测量与 VLM 判官，"
                 "正文凡是「看起来更好」的说法都要注明未经人工确认。"),
        "strata": [],
    },
    "ab60": {
        "csv": ROOT / "experiments/annotate/study_ab60.csv",
        "target": "artist", "other": "baseline",
        "title": "真人 vs 降采样基线（B2 加功效重做）",
        "hit": ("真人与基线的差距真实且人眼可见——那个差距就是值得做的东西，"
                "先看它是不是集中在有结构的那一层。"),
        "miss": ("24 对上的零结果（10/23）在更大样本上复现："
                 "基线已接近真人。这是**更强的证据**，不是没结论。"),
        "strata": [("structured", "有几何结构"), ("plain", "无几何结构")],
    },
}


def load(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tally(rows, target, other):
    """返回 (选 target 的数, 有效对数, 平局数)。"""
    win = tot = tie = 0
    for r in rows:
        c = (r.get("chosen") or "").strip()
        if c == "tie":
            tie += 1
        elif c in (target, other):
            tot += 1
            win += c == target
    return win, tot, tie


def report(name, target, win, tot, tie, indent="") -> bool:
    if not tot:
        print(f"{indent}{name}：无有效判断（平局 {tie}）")
        return False
    p = binom_test(win, tot)
    lo, hi = jeffreys(win, tot)
    print(f"{indent}{name}：{target} 被选 {win}/{tot} = {win/tot:.0%}  "
          f"p={p:.3g}  [{lo:.0%},{hi:.0%}]   （平局弃 {tie}）")
    return win / tot > 0.5 and p < 0.05


def analyse(key: str, path: Path) -> int:
    cfg = STUDIES[key]
    target, other = cfg["target"], cfg["other"]
    rows = load(path)
    real = [r for r in rows if r.get("kind") == "real"]
    check = [r for r in rows if r.get("kind") == "check"]
    print(f"{path.name}（{cfg['title']}）：{len(rows)} 行，"
          f"real {len(real)}，注意力检查 {len(check)}")

    # —— 注意力检查（必须排在主判据之前）——
    wrong = sum(1 for r in check if (r.get("chosen") or "").strip() != "good")
    print(f"\n注意力检查：错 {wrong}/{len(check)}"
          f"（预注册阈值：错 >{MAX_CHECK_WRONG} 即作废）")
    if not check:
        print("  -> 警告：这份 CSV 里没有检查条目，无法验标注质量；**不出主判据**。")
        return 2
    if wrong > MAX_CHECK_WRONG:
        print("  -> **该次标注作废**，主判据不予评估（判据已在跑前固定）。")
        return 1
    print("  -> 通过")

    # —— 主判据 ——
    win, tot, tie = tally(real, target, other)
    print()
    ok = report("主判据", target, win, tot, tie)
    print(f"  -> {'**成立**：' if ok else '不成立：'}{cfg['hit'] if ok else cfg['miss']}")

    # —— 次判据（描述性，不参与判定）——
    if cfg["strata"]:
        print("\n次判据（描述性，预先指定，不参与判定）——按建页时的分层：")
        rates = []
        for tag, label in cfg["strata"]:
            sub = [r for r in real if r.get("stratum") == tag]
            w, t, ti = tally(sub, target, other)
            report(label, target, w, t, ti, indent="  ")
            if t:
                rates.append((label, w / t))
        if len(rates) == 2:
            (la, ra), (lb, rb) = rates
            print(f"  -> 两层差 {ra - rb:+.0%}（{la} - {lb}）。"
                  "本项目的核心不对称预期是有结构那层差距更大；"
                  "若方向相反，要单独查，别当噪声带过。")

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
                            "stratum": stratum, "left": chosen, "right": "x",
                            "choice": "left", "chosen": chosen, "ms": 100})
        return d

    def rows(kind, stratum, chosen, n):
        return [(kind, stratum, chosen)] * n

    cases = [
        ("crop 检查错 2 个 -> 作废", "crop",
         rows("check", "cropped", "blur", 2) + rows("check", "cropped", "good", 1)
         + rows("real", "cropped", "after", 39), 1),
        ("crop 没有检查条目 -> 不出主判据", "crop",
         rows("real", "cropped", "after", 39), 2),
        ("crop 主判据成立", "crop",
         rows("check", "cropped", "good", 3)
         + rows("real", "cropped", "after", 30) + rows("real", "cropped", "before", 9), 0),
        ("crop 对半开 -> 不成立", "crop",
         rows("check", "cropped", "good", 3)
         + rows("real", "cropped", "after", 20) + rows("real", "cropped", "before", 19), 0),
        ("crop 全平局 -> 无有效判断", "crop",
         rows("check", "cropped", "good", 3) + rows("real", "cropped", "tie", 39), 0),
        ("ab60 主判据成立 + 两层同向", "ab60",
         rows("check", "structured", "good", 6)
         + rows("real", "structured", "artist", 30) + rows("real", "structured", "baseline", 6)
         + rows("real", "plain", "artist", 18) + rows("real", "plain", "baseline", 6), 0),
        ("ab60 零结果复现（对半开）", "ab60",
         rows("check", "structured", "good", 6)
         + rows("real", "structured", "artist", 18) + rows("real", "structured", "baseline", 18)
         + rows("real", "plain", "artist", 15) + rows("real", "plain", "baseline", 15), 0),
        ("ab60 两层方向相反", "ab60",
         rows("check", "structured", "good", 6)
         + rows("real", "structured", "artist", 6) + rows("real", "structured", "baseline", 30)
         + rows("real", "plain", "artist", 24) + rows("real", "plain", "baseline", 6), 0),
    ]
    bad = 0
    for name, key, rs, want in cases:
        print(f"\n{'=' * 62}\n[自检] {name}（期望退出码 {want}）\n{'=' * 62}")
        got = analyse(key, make(rs))
        if got != want:
            print(f"**退出码不符：{got} != {want}**")
            bad += 1
    print(f"\n自检 {len(cases)} 例，不符 {bad}")
    return 1 if bad else 0


def main():
    args = sys.argv[1:]
    if "--selftest" in args:
        raise SystemExit(selftest())
    if not args or args[0] not in STUDIES:
        raise SystemExit(f"用法：analyze_pair_study.py {'|'.join(STUDIES)} [路径.csv]")
    key = args[0]
    path = Path(args[1]) if len(args) > 1 else STUDIES[key]["csv"]
    if not path.exists():
        raise SystemExit(f"没有 {path}\n"
                         f"用户标完 `experiments/annotate/study_{key}.html` 后导出 CSV "
                         f"（页面已按自己的文件名导出）放到这里，或把路径当参数传进来。")
    raise SystemExit(analyse(key, path))


if __name__ == "__main__":
    main()
