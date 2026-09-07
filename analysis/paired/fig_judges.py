"""图：三个判官、三种失败方式，以及"逐条一致率不是判据"。

同一验证集（A4 的人工标注）、同一协议（正反两问去偏）：
  gemini-3.1-pro  压平到不显著、分层抹平
  gpt-5.6-sol     方向倒转
  claude-opus-5   方向、显著性、分层顺序都保住（但仍压缩）

第三个面板是方法学要点：gemini 与 opus-5 的**逐条一致率相同**，
结论复现能力却完全不同——故逐条一致率不能作为判官可用性的判据。
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

JUDGES = [("gemini-3.1-pro", "vlm_a4_gemini.json", "#e67e22"),
          ("gpt-5.6-sol", "vlm_a4_gpt-5_6-sol.json", "#8e44ad"),
          ("claude-opus-5", "vlm_a4_claude-opus-5.json", "#16a085")]
HUMAN = "#2c6fbb"


def main():
    A = Path("experiments/annotate")
    strat = {int(r["idx"]): r["stratum"] for r in csv.DictReader(
        (A / "a4_labels.csv").open(encoding="utf-8"))}
    rows = []
    for name, f, col in JUDGES:
        p = A / f
        if not p.exists():
            print(f"缺 {f}，跳过"); continue
        recs = json.loads(p.read_text(encoding="utf-8"))
        sub = [r for r in recs if r["kind"] == "real"
               and r["human"] in ("baseline", "model")
               and r["vlm"] in ("baseline", "model")]
        real = [r for r in recs if r["kind"] == "real"]
        agree = np.mean([r["human"] == r["vlm"] for r in real])
        hb = np.mean([r["human"] == "baseline" for r in sub])
        vb = np.mean([r["vlm"] == "baseline" for r in sub])
        pv = stats.binomtest(sum(1 for r in sub if r["vlm"] == "baseline"),
                             len(sub), 0.5).pvalue
        seed = [r for r in sub if strat.get(r["idx"]) == "seeded"]
        hs = np.mean([r["human"] == "model" for r in seed]) if seed else np.nan
        vs = np.mean([r["vlm"] == "model" for r in seed]) if seed else np.nan
        rows.append((name, col, agree, hb, vb, pv, hs, vs, len(sub)))
        print(f"{name:<16} 逐条 {agree:.1%}  基线胜 人{hb:.0%}/判官{vb:.0%} "
              f"(p={pv:.3f})  有种子层 人{hs:.0%}/判官{vs:.0%}  n={len(sub)}")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    names = [r[0] for r in rows]
    x = np.arange(len(rows))

    ax = axes[0]
    ax.bar(x - .2, [r[3] for r in rows], .38, color=HUMAN, label="human")
    ax.bar(x + .2, [r[4] for r in rows], .38, color=[r[1] for r in rows],
           label="judge")
    ax.axhline(.5, color="#888", ls=":", lw=1)
    for i, r in enumerate(rows):
        ax.text(i + .2, r[4] + .02, f"p={r[5]:.2f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-", "-\n", 1) for n in names], fontsize=7.5)
    ax.set_ylabel("baseline win rate")
    ax.set_title("(a) does it reproduce the conclusion?", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left")

    ax = axes[1]
    ax.bar(x - .2, [r[6] for r in rows], .38, color=HUMAN)
    ax.bar(x + .2, [r[7] for r in rows], .38, color=[r[1] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i - .2, r[6] + .02, f"{r[6]:.0%}", ha="center", fontsize=7.5)
        ax.text(i + .2, r[7] + .02, f"{r[7]:.0%}", ha="center", fontsize=7.5)
    ax.set_ylim(0, .8); ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-", "-\n", 1) for n in names], fontsize=7.5)
    ax.set_ylabel("model win rate, seeded stratum")
    ax.set_title("(b) is the stratification preserved?", fontsize=9.5)

    ax = axes[2]
    ax.bar(x, [r[2] for r in rows], .5, color=[r[1] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, r[2] + .012, f"{r[2]:.1%}", ha="center", fontsize=8.5)
    ax.axhline(.5, color="#888", ls=":", lw=1)
    ax.set_ylim(0, .8); ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-", "-\n", 1) for n in names], fontsize=7.5)
    ax.set_ylabel("per-item agreement with human")
    ax.set_title("(c) per-item agreement says nothing", fontsize=9.5)

    fig.suptitle("Three judges, three failure modes — and (c) shows why "
                 "per-item agreement cannot admit a judge", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig("figures/fig4_judges.png", dpi=200)
    print("\n写入 figures/fig4_judges.png")


if __name__ == "__main__":
    main()
