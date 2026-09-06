"""Fig 8：五档分辨率 × 两独立口径的裁剪胜率——梯度证伪图（5.4）。

数据：experiments/crop_res5.json（gemini/种子21）与 crop_res5b.json（sonnet/种子99）。
胜率口径与 crop_res5_eval.py 一致：fired 且 vlm∈{before,after}，after 计胜。
误差棒为 Wilson 95% 区间。虚线 = 纯奈奎斯特解释预测的方向（递减）。
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def wilson(w, n, z=1.96):
    if not n:
        return float("nan"), 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def tiers(path):
    recs = json.loads((ROOT / path).read_text())
    judged = [r for r in recs if r["fired"] and r.get("vlm") in ("before", "after")]
    out = {}
    for s in sorted({r["size"] for r in judged}):
        sub = [r for r in judged if r["size"] == s]
        out[s] = (sum(1 for r in sub if r["vlm"] == "after"), len(sub))
    return out


def main():
    runs = [("gemini / seed 21 (ρ=−0.14, p=0.28)", "experiments/crop_res5.json", "#4477aa", "o"),
            ("sonnet / seed 99 (ρ=+0.04, p=0.64)", "experiments/crop_res5b.json", "#cc6644", "s")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for i, (label, path, color, marker) in enumerate(runs):
        t = tiers(path)
        xs = sorted(t)
        ps, los, his = zip(*(wilson(*t[s]) for s in xs))
        xoff = [x * (1.0 + 0.012 * (i * 2 - 1)) for x in xs]
        ax.errorbar(xoff, ps, yerr=[[p - lo for p, lo in zip(ps, los)],
                                    [hi - p for p, hi in zip(ps, his)]],
                    fmt=marker + "-", color=color, capsize=3, label=label, lw=1.5, ms=5)
        for x, s, p in zip(xoff, xs, ps):
            w, n = t[s]
            ax.annotate(f"{w}/{n}", (x, p), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color=color)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.annotate("chance", (66, 0.505), fontsize=7, color="gray")
    # 纯奈奎斯特解释预测的方向示意
    ax.annotate("", xy=(60, 0.32), xytext=(18, 0.44),
                arrowprops=dict(arrowstyle="->", color="gray", ls="--", lw=1.2))
    ax.annotate("pure-Nyquist prediction\n(falsified)", (22, 0.27), fontsize=8,
                color="gray", ha="center")
    ax.set_xscale("log")
    ax.set_xticks([16, 24, 32, 48, 64])
    ax.set_xticklabels(["16", "24", "32", "48", "64"])
    ax.minorticks_off()
    ax.set_xlabel("output resolution (px)")
    ax.set_ylabel("crop win rate (VLM screen)")
    ax.set_ylim(0.2, 1.05)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures" / "fig8_gradient.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
