"""零 API、零 GPU：量**每一对给判官看的两张图本身差多少**，回答"同一台仪器判出率为何差 23pp"。

为什么写（`6c64796` 定下的下一步）：环不闭合，不可加几乎全压在 e3 这条边判出率只有 51%、
而另三条 70–74% 上。上一轮把这归因为"弃样口径"，并打算去改记分方式。**那之前必须先排除一个
更朴素的解释：判出率低是因为给判官看的两张图本身更像。** 极端情形是两张图**逐像素相同**——
那样的对就是一道**空对照**，判官在它上面按定义没有任何信息，只能按位置作答（实测地板 17.8%）。

这个解释对 e3 特别可疑：e3 = B2@32 vs B2@16↑，两边都出自**同一条确定性的下采样基线**。
对于**纯色/近纯色**材质（E_mat 里有大量 `baked clay <色>` 这类），把源图降到 32 与降到 16 再
最近邻放大 2x，**结果可以完全一样**。而 e4 = TRD@32 vs TRD@16↑ 两边都是模型生成的，
即使材质是纯色也几乎不会逐像素相同。**若如此，e3 的低判出率是刺激的属性，不是判官的属性，
"弃样口径"整条路就走错了。**

⚠ 本脚本**一个胜负字段都不读**（只读 verdict 是否 in ("A","B")，即"判没判出"，不读判给了谁），
照抄 `pilot_gate_audit.py` 的纪律，免得把完整性检查变成偷看结果。胜负怎么处理**另行预注册**。

在 emnlp 上跑（图只在远端）：

    python analysis/arch/pair_identity.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "eval"))
from exact import binom_test, jeffreys              # noqa: E402
from judge_pairs import FLOOR, first_tile           # noqa: E402  （同一套取图逻辑，不另写一遍）
from prompts import load_set                        # noqa: E402

EXP = ROOT / "experiments"
BASE = EXP / "baselines"

# 环的四条边（`analysis/arch/tier_cycle.py` 同一套），外加两条同画布参照。
EDGES = [
    ("e1", "TRD16c_rr4", "B2",        16, "TRD@16 vs B2@16      （同画布跨方法）"),
    ("e2", "TRD32_rr4",  "B2",        32, "TRD@32 vs B2@32      （同画布跨方法）"),
    ("e3", "B2",         "B2up16",    32, "B2@32  vs B2@16↑     （同方法跨画布，两边都是基线）"),
    ("e4", "TRD32_rr4",  "TRD16cup",  32, "TRD@32 vs TRD@16↑    （同方法跨画布，两边都是模型）"),
]


def build_pairs(a, b, size):
    """复刻 `judge_pairs.py:108-116` 的取对逻辑，保证 pair 下标与判定 JSON 对得上。"""
    items = load_set("E_mat")[0]
    da, db = BASE / a / str(size), BASE / b / str(size)
    pairs = []
    for e in items:
        slug = e["material"].rsplit(".", 1)[0]
        ta, tb = first_tile(da, slug), first_tile(db, slug)
        if ta is not None and tb is not None:
            pairs.append((e["prompt"], ta, tb))
    return pairs


def summarise(key, a, b, size, desc):
    pairs = build_pairs(a, b, size)
    f = EXP / f"judge_full_{a}_vs_{b}_{size}.json"
    recs = json.loads(f.read_text(encoding="utf-8"))["records"]
    assert len(recs) == len(pairs), f"{key}: 对数 {len(recs)} != 重建的 {len(pairs)}"
    for r, (label, _, _) in zip(recs, pairs):
        assert r["material"] == label, f"{key}: 第 {r['pair']} 对材质对不上（{r['material']} vs {label}）"

    diff, resolved = [], []
    for r, (_, ta, tb) in zip(recs, pairs):
        if r["verdict"] is None:                     # API 失败的对：没有读数，整对排除
            continue
        # 每通道绝对差的均值（0–255）。⚠ 别用"有多少像素不相等"：±1 的差也记成"不同"，
        # 那个数在四条边上都是 100%，什么都分不出来（第一版就是这么废掉的）。
        diff.append(float(np.abs(ta.astype(np.int16) - tb.astype(np.int16)).mean()))
        resolved.append(r["verdict"] in ("A", "B"))
    diff, resolved = np.array(diff), np.array(resolved)

    print(f"\n{'=' * 78}\n{key}  {desc}   （{a} vs {b} @{size}，n={len(diff)}）")
    near = diff < 2.0                                # 平均每通道差 <2/255：肉眼基本同一张图
    print(f"  **近乎同一张图的对（平均每通道差 <2/255）：{near.sum()}/{len(diff)} = {near.mean():.1%}** "
          f"（其中逐像素完全相同 {int((diff == 0).sum())} 对）← 这些对接近空对照")
    print(f"  平均每通道差：中位 {np.median(diff):.1f}/255   "
          f"四分位 [{np.quantile(diff, .25):.1f}, {np.quantile(diff, .75):.1f}]")

    # 判出率按"两张图差多少"分四层（等频）。若判出率随差异单调上升 = 判官在按内容作答。
    print(f"  {'分层（平均每通道差）':<26}{'对数':>5}{'判出率':>9}   {'Jeffreys':>14}")
    q = np.quantile(diff, [.25, .5, .75])
    edges_ = [(-1, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)]
    for lo_, hi_ in edges_:
        m = (diff > lo_) & (diff <= hi_)
        if not m.any():
            continue
        k, n = int(resolved[m].sum()), int(m.sum())
        jl, jh = jeffreys(k, n)
        name = f"({max(lo_, 0):.1f}, {min(hi_, diff.max()):.1f}]"
        print(f"  {name:<26}{n:>5}{k / n:>9.1%}   [{jl:>5.0%},{jh:>5.0%}]")
    k, n = int(resolved[near].sum()), int(near.sum())
    if n:
        print(f"  -> 近同那层 {k}/{n} = {k / n:.1%} vs 地板 {FLOOR:.1%}："
              f"双侧 p={binom_test(k, n, FLOOR):.3g}")
    return key, len(diff), int(near.sum()), float(np.median(diff)), float(resolved.mean())


def main():
    print("四条边：两张图本身差多少，以及判出率随这个差异怎么变")
    out = []
    for key, a, b, size, desc in EDGES:
        out.append(summarise(key, a, b, size, desc))
    print(f"\n{'=' * 78}\n汇总：\n  {'边':<4}{'对数':>5}{'近同对':>7}{'近同占比':>9}"
          f"{'中位差/255':>11}{'总判出率':>9}")
    for key, n, near, med, rr in out:
        print(f"  {key:<4}{n:>5}{near:>7}{near / n:>9.1%}{med:>11.1f}{rr:>9.1%}")
    print("\n判读（**跑之前写死**）：若 e3 的近同对明显多于另三条边，且近同那层的判出率贴着地板，")
    print("  则 e3 的低判出率是**刺激的属性**（有一批对根本没有可判的内容），")
    print("  而不是「判官在这条边上更不稳」——那样的话，去改弃样记分方式就是在治错的病。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
