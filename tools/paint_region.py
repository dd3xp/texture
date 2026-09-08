"""纯色图 + 材质名 -> 那块区域被画上低分辨率像素画纹理。

这是用户最初定义的交付形态。此前仓库里只有两半：
  `make_texture.py`  高分源 -> 瓦片
  `from_prompt.py`   材质名 -> 瓦片
都止步于"出一张瓦片"，没有"把瓦片贴回原图那块区域"这一步。

用法（两条路径）：

  # 1. 已有瓦片（纯 CPU，不需要 GPU）
  python tools/paint_region.py in.png --tile tile.png -o out.png

  # 2. 从材质名现生成（需要 GPU 与 diffusers，只在 emnlp 上跑）
  python tools/paint_region.py in.png "mossy stone" --size 16 -o out.png

区域怎么定：先把**占据画面边框的颜色**判为背景，再取其余颜色里最多的那个，
该颜色的全部像素即区域。整张纯色时退回边框色本身。
（不能直接取「出现最多的颜色」：图形嵌在背景里时背景常常更大——
实测灰底上的棕块，灰占 63.5%，会选错。）`--color` 可显式指定，`--tol` 给容差。

配色：瓦片的调色板整体挪到区域的颜色上（`recolor_to`，保亮度结构），
所以"纯棕色 + 木头"给出的是棕色调的木纹，而不是模型自己想画的颜色。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_texture import extract_palette, quantize, recolor_to    # noqa: E402
from downsample import auto_crop                                   # noqa: E402


def dominant_color(rgb: np.ndarray) -> tuple[int, int, int]:
    """要上纹理的那块区域的颜色。

    不能简单取"出现最多的颜色"：模组里的常见形态是**一个图形嵌在背景里**，
    背景往往比图形还大（实测一张灰底上的棕色块，灰占 63.5%，会被选错）。
    故先把**占据画面边框的颜色**判为背景，再在其余颜色里取最多的。
    整张纯色时没有"其余颜色"，退回边框色本身。
    """
    flat = rgb.reshape(-1, rgb.shape[-1])[:, :3]
    border = np.concatenate([rgb[0, :, :3], rgb[-1, :, :3],
                             rgb[:, 0, :3], rgb[:, -1, :3]])
    bg = Counter(map(tuple, border)).most_common(1)[0][0]
    counts = Counter(map(tuple, flat))
    del counts[bg]
    return counts.most_common(1)[0][0] if counts else bg


def region_mask(rgb: np.ndarray, color, tol: int = 0) -> np.ndarray:
    d = np.abs(rgb[..., :3].astype(int) - np.array(color, dtype=int)).max(axis=-1)
    return d <= tol


def tile_over(mask: np.ndarray, tile: np.ndarray, scale: int) -> np.ndarray:
    """把 tile 按 scale 倍放大后平铺满 mask 的外接框。

    相位对齐到外接框左上角而不是整图原点：区域被挪动时纹理不会跟着漂。
    """
    ys, xs = np.nonzero(mask)
    y0, x0 = ys.min(), xs.min()
    H, W = mask.shape
    big = np.kron(tile, np.ones((scale, scale, 1), dtype=tile.dtype))
    th, tw = big.shape[:2]
    yy = (np.arange(H) - y0) % th
    xx = (np.arange(W) - x0) % tw
    return big[np.ix_(yy, xx)]


def auto_scale(mask: np.ndarray, size: int, target_tiles: float = 4.0) -> int:
    """默认让区域短边容下约 `target_tiles` 张瓦片，至少 1 像素/纹素。"""
    ys, xs = np.nonzero(mask)
    short = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    return max(1, int(round(short / (target_tiles * size))))


def tile_from_prompt(prompt: str, size: int, colors: int, seed: int,
                     steps: int, render: int, lora: str) -> np.ndarray:
    """现生成一张瓦片。只在有 GPU 的机器上可用。"""
    import torch
    from diffusers import StableDiffusionXLPipeline
    from from_prompt import TMPL, NEG, load_lora

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    load_lora(pipe, lora)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator("cuda").manual_seed(seed)
    im = pipe(TMPL.format(p=prompt), negative_prompt=NEG,
              num_inference_steps=steps, generator=g,
              height=render, width=render).images[0]
    a = np.asarray(im).astype(float)
    a, frac = auto_crop(a, size)
    if frac < 1.0:
        print(f"检出周期结构，裁 1/{1/frac:.1f}", flush=True)
    small = np.asarray(Image.fromarray(a.astype(np.uint8))
                       .resize((size,) * 2, Image.BOX))
    return quantize(small, extract_palette(small, colors, seed=seed)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("image", type=Path, help="输入图（纯色区域待上纹理）")
    ap.add_argument("prompt", nargs="?", help="材质名，如 wood；给了 --tile 就不需要")
    ap.add_argument("--tile", type=Path, help="用现成瓦片，跳过生成（纯 CPU）")
    ap.add_argument("--size", type=int, default=16, choices=[16, 24, 32])
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--color", help="区域颜色 RRGGBB；默认取图中最多的那个颜色")
    ap.add_argument("--tol", type=int, default=0, help="颜色匹配容差（0=精确）")
    ap.add_argument("--scale", type=int, help="每个纹素占几个输出像素；默认自适应")
    ap.add_argument("--keep-hue", action="store_true",
                    help="保留瓦片原本的颜色，不挪到区域色相")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024, help="生成时的渲染分辨率")
    ap.add_argument("--lora", default="none",
                    help="默认不用 adapter：4 材质同种子对比下 base 更利落"
                         "（`experiments/lora_compare.png`，目视未盲比），"
                         "且论文所有已报结果都是 base。传仓库名可开启。")
    ap.add_argument("-o", "--out", type=Path, default=Path("painted.png"))
    a = ap.parse_args()

    if not a.tile and not a.prompt:
        raise SystemExit("要么给材质名，要么给 --tile")

    im = Image.open(a.image).convert("RGB")
    rgb = np.asarray(im)
    color = (tuple(int(a.color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
             if a.color else dominant_color(rgb))
    mask = region_mask(rgb, color, a.tol)
    if not mask.any():
        raise SystemExit(f"没有像素匹配颜色 {color}（容差 {a.tol}）")
    print(f"区域颜色 #{color[0]:02x}{color[1]:02x}{color[2]:02x}，"
          f"覆盖 {mask.mean():.1%} 的像素")

    if a.tile:
        t = np.asarray(Image.open(a.tile).convert("RGB"))
        if t.shape[0] != a.size:          # 允许传放大过的瓦片
            t = np.asarray(Image.open(a.tile).convert("RGB")
                           .resize((a.size,) * 2, Image.NEAREST))
        tile = t.astype(np.uint8)
    else:
        tile = tile_from_prompt(a.prompt, a.size, a.colors, a.seed,
                                a.steps, a.render, a.lora)

    if not a.keep_hue:
        pal = extract_palette(tile, a.colors, seed=a.seed)
        tile = quantize(tile, recolor_to(pal, "%02x%02x%02x" % color)).astype(np.uint8)

    scale = a.scale or auto_scale(mask, a.size)
    print(f"瓦片 {a.size}x{a.size}，每纹素 {scale} 像素 -> 单块 {a.size * scale} 像素")

    painted = tile_over(mask, tile, scale)
    out = rgb.copy()
    out[mask] = painted[mask]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(a.out)
    print(f"写入 {a.out}")


if __name__ == "__main__":
    main()
