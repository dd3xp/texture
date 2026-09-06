"""Fig 4：VLM 判官失效画像（3.3 / B6）。

数字来源 docs/plan.md B6 节（gemini-3.1-pro-preview，A4 的 72 对人工标注复判）：
- 主效应：人 41/47=87%（p<1e-4）vs VLM 29/47=62%（p=0.14）——压缩到失去显著性；
- 分层（模型胜率）：有种子层 人 6% vs VLM 37%；无种子层 人 33% vs VLM 42%
  ——A4 的关键不对称被抹平（VLM 分层 n 未单独记录，故不画区间）；
- 位置偏好：首轮两个判官选左均 ~67%（应 50%），须正反两问去偏。
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE = "#4477aa", "#cc6644"


def wilson(w, n, z=1.96):
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, p - (c - h), (c + h) - p


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                   gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel A: effect compression on the same 47 pairs
    ph, lo_h, hi_h = wilson(41, 47)
    pv, lo_v, hi_v = wilson(29, 47)
    ax1.bar([0, 1], [ph, pv], width=0.55, color=[BLUE, ORANGE])
    ax1.errorbar([0, 1], [ph, pv], yerr=[[lo_h, lo_v], [hi_h, hi_v]],
                 fmt="none", ecolor="black", capsize=4, lw=1)
    ax1.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["human\n41/47 = 87%\np < 1e-4", "VLM judge\n29/47 = 62%\np = 0.14 (n.s.)"],
                        fontsize=8)
    ax1.set_ylabel("baseline win rate (same 47 pairs)")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("(a) effect compressed below significance", fontsize=9)

    # Panel B: stratification flattened (model win rate per stratum)
    x = [0, 1]
    human = [0.06, 0.33]
    vlm = [0.37, 0.42]
    w = 0.32
    ax2.bar([i - w / 2 for i in x], human, w, color=BLUE, label="human")
    ax2.bar([i + w / 2 for i in x], vlm, w, color=ORANGE, label="VLM judge")
    for i, (h, v) in enumerate(zip(human, vlm)):
        ax2.annotate(f"{h:.0%}", (i - w / 2, h), ha="center", va="bottom", fontsize=8)
        ax2.annotate(f"{v:.0%}", (i + w / 2, v), ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["seeded prior\n(prior actively hurts)", "unseeded"], fontsize=8)
    ax2.set_ylabel("specialized-model win rate")
    ax2.set_ylim(0, 0.65)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax2.set_title("(b) stratification flattened", fontsize=9)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("VLM judge: direction survives, magnitude and structure do not"
                 "  (position bias 67% before order-debiasing)", fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "figures" / "fig4_judge.png"
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
