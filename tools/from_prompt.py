"""文字 -> 像素画纹理：SDXL 出高分辨率图，再走降采样基线。

这是模组场景的完整管线：需要的材质在任何材质包里都不存在，
所以没有现成高分源，先让 SDXL 画一张。

后半段（降采样 + 量化）就是 B2 盲比中与真人手绘不可区分的那条基线。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_texture import extract_palette, quantize, recolor_to, W
from downsample import auto_crop, dominant_period

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--color", help="可选：把调色板整体挪到这个色相")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upscale", type=int, default=16)
    ap.add_argument("-o", "--out", type=Path, default=Path("texture.png"))
    ap.add_argument("--keep-hires", type=Path)
    ap.add_argument("--lora", default="nerijs/pixel-art-xl",
                    help="像素画 LoRA；--lora none 关闭")
    ap.add_argument("--no-autocrop", dest="autocrop", action="store_false",
                    help="关闭按结构尺度自动裁剪")
    ap.add_argument("--hires", type=int, default=1024,
                    help="SDXL 渲染分辨率。1024 下一块砖约 200px，"
                         "降到 16px 后砖缝不足 1px 被平均抹平；调低可让结构存活")
    args = ap.parse_args()

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    if args.lora and args.lora != "none":
        try:
            pipe.load_lora_weights(args.lora)
            print(f"已载入 LoRA {args.lora}", flush=True)
        except Exception as e:
            print(f"LoRA 载入失败（{e}），继续用基础模型", flush=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    outs, hires = [], []
    for i in range(args.n):
        g = torch.Generator("cuda").manual_seed(args.seed + i)
        im = pipe(TMPL.format(p=args.prompt), negative_prompt=NEG,
                  num_inference_steps=args.steps, generator=g,
                  height=args.hires, width=args.hires).images[0]
        hires.append(im)
        a = np.asarray(im).astype(float)
        if args.autocrop:
            a, frac = auto_crop(a, args.size)
            if frac < 1.0:
                print(f"  样本{i+1} 检出周期结构，裁 1/{1/frac:.1f}", flush=True)
        small = np.asarray(Image.fromarray(a.astype(np.uint8))
                           .resize((args.size,) * 2, Image.BOX))
        pal = extract_palette(small, args.colors, seed=i)
        if args.color:
            pal = recolor_to(pal, args.color)
        outs.append(quantize(small, pal))
        lum = pal.astype(float) @ W
        print(f"  样本{i+1} 亮度跨度 {lum[-1]-lum[0]:.1f}", flush=True)

    u = args.upscale
    w = args.size * u
    canvas = Image.new("RGB", (w * len(outs) + 4 * (len(outs) - 1), w), "white")
    for i, o in enumerate(outs):
        canvas.paste(Image.fromarray(o.astype(np.uint8)).resize((w, w), Image.NEAREST),
                     (i * (w + 4), 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"写入 {args.out}")
    if args.keep_hires:
        args.keep_hires.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(hires):
            im.save(args.keep_hires / f"hires_{i}.png")


if __name__ == "__main__":
    main()
