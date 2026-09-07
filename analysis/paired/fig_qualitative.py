"""图：方法的定性结果——同一张 1024 渲染图，裁与不裁的区别（§5.1/§5.3）。

论文此前没有任何一张实际输出。对一篇纹理生成的文章这是硬伤：
读者要能自己看出"直接降采样糊成一片、按主周期裁剪后结构回来了"。

每行一个材质，四列：
  1024 渲染源 | 直接降到 16 | 按主周期裁剪后降到 16 | 裁剪窗口在源图上的位置
高分源由 `scripts/render_fig.sh` 在 emnlp 上渲染（种子 21，与主 run 一致），
本脚本只做裁剪与降采样，纯 CPU，可随时重跑调版式。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import auto_crop, dominant_period, anisotropy      # noqa: E402
from make_texture import extract_palette, quantize                 # noqa: E402

SIZE, COLORS = 16, 12
SRC = ROOT / "experiments/figqual"


def to_tile(a: np.ndarray, seed: int = 0) -> np.ndarray:
    small = np.asarray(Image.fromarray(a.astype(np.uint8))
                       .resize((SIZE,) * 2, Image.BOX))
    return quantize(small, extract_palette(small, COLORS, seed=seed))


def main():
    rows = []
    for d in sorted(SRC.glob("*/")):
        f = d / "hires_0.png"
        if not f.exists():
            continue
        a = np.asarray(Image.open(f).convert("RGB")).astype(float)
        per, ani = dominant_period(a), anisotropy(a)
        cropped, frac = auto_crop(a, SIZE)
        rows.append((d.name.replace("_", " "), a, cropped, frac, per, ani))
    if not rows:
        print(f"没有源图，先在 emnlp 上跑 scripts/render_fig.sh（存到 {SRC}）")
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(8.2, 2.05 * n))
    axes = np.atleast_2d(axes)
    for r, (name, a, cropped, frac, per, ani) in enumerate(rows):
        side = int(round(frac * min(a.shape[:2])))
        y0 = x0 = (min(a.shape[:2]) - side) // 2
        panels = [
            (a.astype(np.uint8), f"1024px render\nperiod {per:.0f}px, aniso {ani:.2f}"),
            (to_tile(a), "downsample to 16"),
            (to_tile(cropped), f"crop 1/{1/frac:.1f}, then 16" if frac < 1
             else "gate did not fire"),
            (a.astype(np.uint8), f"crop window ({side}px)"),
        ]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            if c == 3 and frac < 1:
                ax.add_patch(Rectangle((x0, y0), side, side, fill=False,
                                       ec="#e02020", lw=1.6))
            if r == 0:
                ax.set_title(title.split(chr(10))[0], fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
            if chr(10) in title:
                ax.set_xlabel(title.split(chr(10))[1], fontsize=6.5)
            elif c == 2 and frac < 1:
                ax.set_xlabel(f"1/{1/frac:.1f}", fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
        print(f"{name:<24} period={per:6.1f}  aniso={ani:.2f}  frac={frac:.3f}")

    fig.suptitle("Same render, with and without the structural-period crop",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "figures/fig_qualitative.png"
    fig.savefig(out, dpi=200)
    print("写入", out)


if __name__ == "__main__":
    main()
