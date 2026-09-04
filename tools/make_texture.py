"""高分辨率图 -> 像素画纹理。交付给模组用的工具。

**为什么是基线管线而不是我们训练的模型**：B2 盲比显示降采样基线与真人手绘
不可区分（真人胜率 43%，p=0.68），而 A4 显示我们的模型输给基线 83:17。
把输的那个封装出去没有道理。

调色板两种取法，`--palette` 切换：
  extract  从源图自身提取（k-means，按亮度排序）——更接近真人做法
  derive   从一个纯色派生 K 档亮度阶——只有一个颜色时用
           A4a 量出现行 spread=0.55 使对比度达真人的 2–3 倍，
           真人反解中位是 0.292（`analysis/structure_grain/pal_target.py`）
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

W = np.array([0.299, 0.587, 0.114])
SPREAD_ARTIST_MEDIAN = 0.292


def derive_palette(hex_color: str, k: int, spread: float) -> np.ndarray:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    out = []
    for i in range(k):
        t = i / max(k - 1, 1)
        v = float(np.clip(vv * (1 - spread) + spread * vv * 2 * t, 0.04, 1.0))
        s = float(np.clip(ss * (1.15 - 0.3 * t), 0.0, 1.0))
        out.append([int(round(c * 255)) for c in colorsys.hsv_to_rgb(hh, s, v)])
    pal = np.array(out, np.uint8)
    return pal[np.argsort(pal.astype(float) @ W)]


def extract_palette(img: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """从源图提取 k 色，按亮度排序。用 k-means++ 初始化的简单 Lloyd 迭代。"""
    x = img.reshape(-1, 3).astype(float)
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
        c.append(x[rng.choice(len(x), p=d / max(d.sum(), 1e-9))])
    c = np.array(c)
    for _ in range(30):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    pal = np.clip(c, 0, 255).astype(np.uint8)
    return pal[np.argsort(pal.astype(float) @ W)]


def recolor_to(pal: np.ndarray, hex_color: str) -> np.ndarray:
    """把提取到的调色板整体挪到目标色相/饱和，保留其亮度结构。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, ss, _ = colorsys.rgb_to_hsv(r, g, b)
    out = []
    for c in pal:
        _, s0, v0 = colorsys.rgb_to_hsv(*(c / 255.0))
        s = float(np.clip(ss * (s0 / max(s0, 1e-6)) * 0.85 + s0 * 0.15, 0, 1))
        out.append([int(round(v * 255)) for v in colorsys.hsv_to_rgb(hh, s, v0)])
    pal = np.array(out, np.uint8)
    return pal[np.argsort(pal.astype(float) @ W)]


def quantize(img: np.ndarray, pal: np.ndarray) -> np.ndarray:
    d = ((img.reshape(-1, 1, 3).astype(float) - pal.astype(float)[None]) ** 2).sum(-1)
    return pal[d.argmin(1)].reshape(img.shape).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="高分辨率图 -> 像素画纹理（降采样基线，B2 验证与真人不可区分）")
    ap.add_argument("source", type=Path, help="高分辨率源图")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--palette", choices=["extract", "derive"], default="extract")
    ap.add_argument("--color", help="derive 模式的纯色；extract 模式下给了就整体改色")
    ap.add_argument("--spread", type=float, default=SPREAD_ARTIST_MEDIAN,
                    help=f"derive 模式的对比度，真人中位 {SPREAD_ARTIST_MEDIAN}")
    ap.add_argument("--upscale", type=int, default=16)
    ap.add_argument("-o", "--out", type=Path, default=Path("texture.png"))
    args = ap.parse_args()

    src = np.asarray(Image.open(args.source).convert("RGB"))
    small = np.asarray(Image.fromarray(src).resize((args.size,) * 2, Image.BOX))

    if args.palette == "derive":
        if not args.color:
            raise SystemExit("--palette derive 需要 --color")
        pal = derive_palette(args.color, args.colors, args.spread)
    else:
        pal = extract_palette(small, args.colors)
        if args.color:
            pal = recolor_to(pal, args.color)

    out = quantize(small, pal)
    lum = pal.astype(float) @ W
    print(f"{args.source.name} -> {args.size}x{args.size}，{args.colors} 色，"
          f"调色板={args.palette}")
    print(f"  亮度跨度 {lum[-1]-lum[0]:.1f}（真人中位 63.0）")
    u = args.upscale
    Image.fromarray(out).resize((args.size * u,) * 2, Image.NEAREST).save(args.out)
    print(f"  写入 {args.out}")


if __name__ == "__main__":
    main()
