"""Forest plot of the pairwise-judge arms reported in paper_draft.tex (numbers from experiments/judge_full_*.json
and experiments/judge_cluster_sweep.json; family CI = cluster interval, None where leave-one-family-out is void)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [  # label, wins, decided, family CI
    ("TRD vs B7 (same-data diffusion)", 131, 208, (0.540, 0.719)),
    ("TRD+rr vs B5 (artist retrieval)", 119, 202, (0.502, 0.668)),
    ("TRD+rr vs B2 (render+crop+down)", 114, 199, (0.49994, 0.641)),
    ("TRD vs B3 (SD-$\\pi$XL, 12 mat.)", 11, 12, None),
    ("Colour: TRD vs B2", 114, 189, (0.525, 0.680)),
    ("Colour: TRD vs B7", 117, 193, (0.524, 0.674)),
    ("Colour: TRD vs B5 (+recolour)", 111, 208, (0.434, 0.627)),
    ("Ablation: full vs no palette memory", 126, 190, (0.574, 0.747)),
]


def main(out=Path(__file__).resolve().parents[1] / "figures/fig_judge_forest.png"):
    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)
    for i, (lab, w, n, ci) in enumerate(ARMS):
        y = len(ARMS) - 1 - i
        r = w / n
        robust = ci is not None and ci[0] > 0.5
        col = "#1f5fa8" if robust else "#999999"
        if ci is not None:
            ax.plot(ci, [y, y], color=col, lw=2.2, solid_capstyle="butt")
        ax.plot(r, y, "o", color=col, ms=5)
        ax.text(1.005, y, f"{w}/{n}", va="center", fontsize=7, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(ARMS)))
    ax.set_yticklabels([a[0] for a in ARMS][::-1], fontsize=7.5)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("win rate of the first method (de-biased judge); bar = family-level 95% CI", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
