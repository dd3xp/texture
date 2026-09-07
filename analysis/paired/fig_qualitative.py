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
# 论文版只放 4 行（页数所限）。挑的是覆盖面：干净的周期结构、长周期、
# 极端裁剪比，以及**一个门主动不触发的例子**——不能只展示成功的那半边。
# 三行而非四行：四行 4x4 的图接近正方形，放进论文会把正文挤到第 10 页
# （ICLR 正文限 9 页）。留下的三个覆盖：清晰周期结构、温和裁剪比、门不触发。
PICK = ["brick_wall", "woven_basket_surface", "clay_roof_tiles"]
HEAD = ["1024px render", "downsample to 16",
        "crop to the period, then 16", "crop window on the source"]


def to_tile(a: np.ndarray, seed: int = 0) -> np.ndarray:
    small = np.asarray(Image.fromarray(a.astype(np.uint8))
                       .resize((SIZE,) * 2, Image.BOX))
    return quantize(small, extract_palette(small, COLORS, seed=seed))


def main():
    rows = []
    for name in PICK:
        d = SRC / name
        f = d / "hires_0.png"
        if not f.exists():
            print(f"缺 {f}"); continue
        a = np.asarray(Image.open(f).convert("RGB")).astype(float)
        per, ani = dominant_period(a), anisotropy(a)
        cropped, frac = auto_crop(a, SIZE)
        rows.append((d.name.replace("_", " "), a, cropped, frac, per, ani))
    if not rows:
        print(f"没有源图，先在 emnlp 上跑 scripts/render_fig.sh（存到 {SRC}）")
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(8.6, 2.02 * n))
    axes = np.atleast_2d(axes)
    for r, (name, a, cropped, frac, per, ani) in enumerate(rows):
        side = int(round(frac * min(a.shape[:2])))
        y0 = x0 = (min(a.shape[:2]) - side) // 2
        fired = frac < 1.0
        note = (f"cropped 1/{1/frac:.1f}" if fired
                else f"gate declined: aniso {ani:.2f} < 0.20")
        panels = [
            (a.astype(np.uint8), f"period {per:.0f}px, aniso {ani:.2f}"),
            (to_tile(a), ""),
            (to_tile(cropped), note),
            (a.astype(np.uint8), f"{side}px window" if fired else "no crop"),
        ]
        for c, (img, sub) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest")
            if c == 3 and fired:
                ax.add_patch(Rectangle((x0, y0), side, side, fill=False,
                                       ec="#e02020", lw=1.6))
            if r == 0:
                ax.set_title(HEAD[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
            if sub:
                ax.set_xlabel(sub, fontsize=6.5)
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
