"""图 1：逐格一致率的分布——作者之间比随机还低。

这是全文的开篇主张（B1）：低分辨率下不存在唯一正确答案，
因此任何基于参考的逐像素指标测的都不是质量。
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        pal = np.array(s["palette"], np.uint8)
        a = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        # 归一到 12 档，消除各作者色数不同的影响
        q = np.clip((a.astype(float) / max(len(pal) - 1, 1) * 11).round(), 0, 11)
        bymat.setdefault(s["material"], []).append(q)

    # **配对零假设**：对同一对 (A, B)，比 agree(A,B) 与 agree(A, shuffle(B))。
    # 打乱 B 保留了 B 自己的边缘分布、只破坏空间排布，
    # 且两边用的是同一对图——比"把 A 自己打乱"更严格，
    # 因为真实比较里 A 与 B 的边缘分布本就不同。
    same, null = [], []
    rng = np.random.default_rng(0)
    for m, arrs in bymat.items():
        for x, y in combinations(arrs, 2):
            same.append(float((x == y).mean()))
            null.append(float((x == rng.permutation(y.ravel()).reshape(16, 16)).mean()))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(0, 0.6, 49)
    # 论文是英文的，图注一律英文（中文字形在 DejaVu 里也缺失）
    ax.hist(null, bins=bins, color="#bbbbbb", alpha=.85, density=True,
            label=f"Paired null: B shuffled (median {np.median(null):.3f})")
    ax.hist(same, bins=bins, histtype="step", lw=2.0, color="#c0392b",
            density=True,
            label=f"Two artists, same material (median {np.median(same):.3f})")
    ax.axvline(np.median(null), color="#777", ls="--", lw=1)
    ax.axvline(np.median(same), color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("Per-cell agreement")
    ax.set_ylabel("Density")
    ax.set_title("Independent artists agree below chance on individual pixels",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig1_agreement.png", dpi=200)
    from scipy import stats
    d = np.array(same) - np.array(null)
    print(f"作者对 {len(same)}（配对零假设同数）")
    print(f"作者间中位 {np.median(same):.4f}   配对零假设中位 {np.median(null):.4f}")
    print(f"配对差中位 {np.median(d):+.4f}   低于零假设的比例 {(d < 0).mean():.1%}")
    print(f"Wilcoxon p={stats.wilcoxon(same, null).pvalue:.3g}")
    print("写入 figures/fig1_agreement.png")


if __name__ == "__main__":
    main()
