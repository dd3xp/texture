"""把 batch_pack 出的一批纹理拼成一张总览图，看得见才知道能不能用。

三档并排，门未触发的材质标出来——它们是照原图降采样的结果，
不是失败，但读图时要知道哪些走了裁剪、哪些没走。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=ROOT / "experiments/pack")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", type=Path, default=ROOT / "figures/pack_sheet.png")
    a = ap.parse_args()

    man = json.loads((a.pack / "manifest.json").read_text(encoding="utf-8"))
    info = {(r["material"], r["size"]): r for r in man if r["variant"] == a.variant}
    sizes = sorted({s for _, s in info})
    mats = sorted({m for m, _ in info})

    n = len(mats)
    fig, axes = plt.subplots(n, len(sizes), figsize=(1.5 * len(sizes) + 1.6, 1.5 * n))
    axes = np.atleast_2d(axes)
    for r, m in enumerate(mats):
        for c, s in enumerate(sizes):
            ax = axes[r, c]
            f = a.pack / a.variant / str(s) / f"{m.replace(' ', '_')}.png"
            ax.imshow(np.asarray(Image.open(f).convert("RGB")),
                      interpolation="nearest")
            if r == 0:
                ax.set_title(f"{s}x{s}", fontsize=9)
            if c == 0:
                rec = info[(m, s)]
                tag = (f"1/{1/rec['frac']:.1f}" if rec["frac"] < 1
                       else ("no period" if rec["period"] <= 0
                             else f"aniso {rec['aniso']:.2f}" if rec["aniso"] < 0.20
                             else "source too small"))
                ax.set_ylabel(f"{m}\n{tag}", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fired = sum(1 for m in mats if info[(m, sizes[0])]["frac"] < 1)
    fig.suptitle(f"{len(mats)} materials, {a.variant} pipeline --- "
                 f"crop applied to {fired}/{len(mats)}; "
                 f"the rest are labelled with why not", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"写入 {a.out}（{fired}/{len(mats)} 走了裁剪）")


if __name__ == "__main__":
    main()
