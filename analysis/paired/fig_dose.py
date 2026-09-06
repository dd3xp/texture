"""图：剂量–反应——单元数越少，裁剪越无用甚至有害。

论文里最强的因果证据（B13/B14）：
操控生成器的结构单元数（经渲染分辨率），裁剪的收益单调变化，
512 恰好落在无效果的分水岭上。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RUNS = [("crop_render384.json", 384), ("crop_render512.json", 512),
        ("crop_ctrl.json", 1024)]


def main():
    E = Path("experiments")
    pts = []
    for f, S in RUNS:
        d = json.loads((E / f).read_text(encoding="utf-8"))
        per = np.array([r["period"] for r in d if r.get("period", 0) > 0])
        units = S / np.median(per)
        j = [r for r in d if r.get("vlm") in ("before", "after")]
        w = sum(1 for r in j if r["vlm"] == "after")
        lo, hi = stats.beta.ppf([.025, .975], w + .5, len(j) - w + .5)
        pts.append((units, w / len(j), lo, hi, len(j), S))

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    u = [p[0] for p in pts]
    r = [p[1] for p in pts]
    err = np.array([[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]])
    ax.errorbar(u, r, yerr=err, fmt="o-", color="#c0392b", lw=2, ms=7,
                capsize=4, zorder=3)
    ax.axhline(.5, color="#888", ls="--", lw=1)
    ax.text(u[0], .52, "no effect", fontsize=8, color="#666")
    ax.axhspan(0, .5, color="#f2f2f2", zorder=0)
    for x, y, _, _, n, S in pts:
        ax.annotate(f"{S}px render\n{y:.0%}  (n={n})", (x, y),
                    textcoords="offset points", xytext=(6, -22 if y < .6 else 8),
                    fontsize=8)
    ax.set_xlabel("Structural units per tile in the source render")
    ax.set_ylabel("Crop preferred (win rate)")
    ax.set_title("Cropping helps only when the source over-draws structure",
                 fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(.98, .04, "shaded: cropping hurts", transform=ax.transAxes,
            ha="right", fontsize=7.5, color="#777")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_dose_response.png", dpi=200)
    print("单元数 / 胜率 / n：")
    for x, y, lo, hi, n, S in pts:
        print(f"  {S:>5}px 渲染  单元 {x:5.1f}  胜率 {y:.0%} "
              f"[{lo:.0%},{hi:.0%}]  n={n}")
    print("写入 figures/fig_dose_response.png")


if __name__ == "__main__":
    main()
