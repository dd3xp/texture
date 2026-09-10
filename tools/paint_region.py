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
                     steps: int, render: int, lora: str,
                     best_of: int = 4) -> np.ndarray:
    """现生成一张瓦片。只在有 GPU 的机器上可用。

    **默认多采样**（`best_of=4`）。实测（`analysis/paired/bestof_units.py`）：
    采 4 个样、取「门触发且真的裁了」之中裁剪比最大的一版，
    有效裁剪率 **52% -> 90%**（新增 16、失去 0，p=3.05e-5）；
    新救回的 16 个材质里判官偏好新版 **10/12 = 83%**（p=0.039，下界）。
    代价是 4 倍渲染时间，`--best-of 1` 可退回单样本。

    ⚠ **已验证的只是「多采样提高覆盖」**，不是「取裁剪比最大」这条规则。
    对照 `bestof_rule.py` 已跑（判据 commit `6c12667`）：19 个材质上两条规则
    选中了不同样本，判官判了 16 个而**其中 12 个正反两问自相矛盾（75%）**，
    可用的 4 个里 frac 规则 3/4（p=0.625）——按跑前判据不成立。
    所选 frac 之差中位 0.164，不是两张图本来就一样；是判官分不出。
    故「取最大」保留为一个**任意但无害的定序**（可复跑、不需要额外随机数），
    **不要**据此宣称按 frac 挑更好。
    """
    import torch
    from diffusers import StableDiffusionXLPipeline
    from from_prompt import TMPL, NEG, load_lora

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    load_lora(pipe, lora)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    cands = []
    for k in range(max(1, best_of)):
        g = torch.Generator("cuda").manual_seed(seed + 1000 * k)
        im = pipe(TMPL.format(p=prompt), negative_prompt=NEG,
                  num_inference_steps=steps, generator=g,
                  height=render, width=render).images[0]
        src = np.asarray(im).astype(float)
        # 接缝对齐（方法三，`downsample.seam_offset`）：贴图是平铺用的，
        # 窗口没落在周期格点上会在墙面留一条断线。
        # 接缝比在两轮 57/57 个材质上全降（客观、已泛化）。
        # ⚠ **观感上没有证据**：判官 10/12 只在原 42 条上成立、60 个零重叠材质
        # 12/18 未复现，而人工盲比 34/55=62% p=0.105 **主判据不成立**
        # （且那次位置偏好显著）。别写成"更好看"，只能说"接得上"。
        cropped, frac = auto_crop(src, size, seam_align=True)
        # frac>=0.999 = 窗口被钳成整图，门显示触发但等于没裁，不算有效
        cands.append({"k": k, "img": cropped, "frac": frac,
                      "eff": frac < 0.999})
        print(f"  样本{k+1}/{best_of} "
              f"{'裁 1/%.1f' % (1 / frac) if frac < 0.999 else '无有效裁剪'}",
              flush=True)

    eff = [c for c in cands if c["eff"]]
    best = max(eff, key=lambda c: c["frac"]) if eff else cands[0]
    if eff:
        print(f"选中样本 {best['k']+1}（裁 1/{1/best['frac']:.1f}）", flush=True)
    else:
        print("四个样本都没有有效裁剪，退回第一个（与单样本管线一致）", flush=True)

    small = np.asarray(Image.fromarray(best["img"].astype(np.uint8))
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
    ap.add_argument("--best-of", type=int, default=4,
                    help="采几个样再挑（默认 4）。实测有效裁剪率 52%%->90%%；传 1 退回单样本")
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
                                a.steps, a.render, a.lora, a.best_of)

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
