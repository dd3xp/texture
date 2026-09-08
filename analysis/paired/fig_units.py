"""Fig 7：结构单元数惯例失配 + 剂量-反应（5.4 / B10・B13・B14）。

左轴：三种渲染尺寸下 SDXL 每瓦片结构单元数（canvas/period）的逐提示词分布
（本地 JSON：crop_ctrl=1024、crop_render512=512、crop_render384=384；
口径与 B14 一致：period>0 的全部提示词、按 prompt 去重、不限门触发——
中位 27.7/13.8/9.4 与 plan.md B14 表精确吻合）。横线 = 真人惯例 ~3.2
（B5 5979 张瓦片 3.20 与 B9 125 个高分源 3.33 两个独立来源一致；
旧值 4.5 出自加宽周期检测器之前的口径，已弃用——见 main.tex 附录）。
右轴：同三条件的裁剪胜率（B13/B14：88%→54%→17%，VLM 口径）。
单元数越接近真人惯例，裁剪收益越小以至有害——机制的剂量-反应图。
"""

import json
import statistics
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE = "#4477aa", "#cc6644"

CONDS = [  # (canvas, json, win, n_pairs, p_binom)
    (1024, "experiments/crop_ctrl.json", 0.88, "38/43", "2.5e-7"),
    (512, "experiments/crop_render512.json", 0.54, None, "0.749"),
    (384, "experiments/crop_render384.json", 0.17, None, "0.0015"),
]


def units(path, canvas):
    recs = json.loads((ROOT / path).read_text())
    seen = {}
    for r in recs:
        if r["period"] and r["period"] > 0 and r["prompt"] not in seen:
            seen[r["prompt"]] = canvas / r["period"]
    return sorted(seen.values())


def main():
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    xs = list(range(len(CONDS)))
    data = [units(p, c) for c, p, *_ in CONDS]

    bp = ax.boxplot(data, positions=xs, widths=0.45, showfliers=False,
                    medianprops=dict(color=BLUE, lw=1.5),
                    boxprops=dict(color=BLUE), whiskerprops=dict(color=BLUE),
                    capprops=dict(color=BLUE))
    for i, d in enumerate(data):
        ax.plot([i + 0.28] * len(d), d, ".", color=BLUE, ms=3, alpha=0.45)
        # n 为偶数时 sorted(d)[len(d)//2] 取的是上中位数，会与箱线图自己画的
        # 中位线对不上（384px 因此标成 9.5，真值 9.37）。用真中位数。
        med = statistics.median(d)
        ax.annotate(f"med {med:.1f}", (i - 0.28, med),
                    ha="right", va="center", fontsize=7.5, color=BLUE)
    ax.axhline(3.2, color="black", lw=1.2, ls="--")
    ax.annotate("human convention ≈ 3.2 units/tile (tiles 3.20, sources 3.33)",
                (-0.42, 2.8), fontsize=8, va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}px render" for c, *_ in CONDS])
    ax.set_ylabel("structural units per tile (canvas / period)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_yscale("log")
    ax.set_yticks([2, 3.2, 10, 30, 100])
    ax.set_yticklabels(["2", "3.2", "10", "30", "100"])
    ax.minorticks_off()

    ax2 = ax.twinx()
    wins = [w for *_, w, _, _ in [(c, p, w, n, pb) for c, p, w, n, pb in CONDS]]
    wins = [c[2] for c in CONDS]
    ax2.plot(xs, wins, "s-", color=ORANGE, lw=1.5, ms=5)
    for i, (c, _, w, _, pb) in enumerate(CONDS):
        ax2.annotate(f"{w:.0%}\np={pb}", (i, w), textcoords="offset points",
                     xytext=(14, -4), fontsize=7.5, color=ORANGE)
    ax2.axhline(0.5, color=ORANGE, lw=0.7, ls=":", alpha=0.6)
    ax2.set_ylabel("crop win rate (VLM screen)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(0, 1.05)
    ax2.spines[["top"]].set_visible(False)

    ax.spines[["top"]].set_visible(False)
    ax.set_title("Unit-convention mismatch drives the crop benefit (dose–response)",
                 fontsize=9)
    fig.tight_layout()
    out = ROOT / "figures" / "fig7_units.png"
    fig.savefig(out, dpi=300)
    for (c, p, *_), d in zip(CONDS, data):
        print(f"{c}: n={len(d)} median={statistics.median(d):.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
