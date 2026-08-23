"""完整基线：SDXL 按 prompt 生成高分辨率纹理 → 降采样 → 量化 → 硬裁进区域。

S2 的基线需要一张现成的高分辨率纹理，不能按材质名生成，这一点上不如
SD-πXL 通用。本脚本补上这一环，构成真正应当被打败的对手：

    prompt ──SDXL──> 1024x1024 纹理 ──面积降采样──> NxN ──量化──> 调色板 ──硬裁──> 区域

与 SD-πXL 的关键差别：区域约束发生在**生成之后**，是一个硬操作，
所以轮廓天然完全保持。SD-πXL 试图让区域约束在生成**过程中**成立，
而它的架构里没有表达这件事的通道（见 related-work.md）。
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from run import read_palette, quantize_to, region_mask, iou

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def generate(prompts: dict[str, str], out: Path, seed: int, steps: int) -> dict[str, Path]:
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    made = {}
    for name, prompt in prompts.items():
        f = out / f"src_{name}.png"
        if not f.exists():
            g = torch.Generator("cuda").manual_seed(seed)
            img = pipe(prompt=prompt, negative_prompt="3d render, perspective, object, shadow",
                       num_inference_steps=steps, guidance_scale=7.0,
                       height=1024, width=1024, generator=g).images[0]
            img.save(f)
        made[name] = f
        print(f"  生成 {name}: {f.name}")
    del pipe
    torch.cuda.empty_cache()
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--palette", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--bg", type=int, nargs=3, default=[255, 255, 255])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("experiments/dq_full"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    prompts = {
        "wood": "seamless wood plank texture, flat top down view, tileable material",
    }

    print("== SDXL 生成高分辨率纹理 ==")
    srcs = generate(prompts, args.out, args.seed, args.steps)

    palette = read_palette(args.palette)
    bg = tuple(args.bg)
    fg = palette[~(palette == np.array(bg)).all(-1)]
    src_img = Image.open(args.input)

    print(f"\n{'材质':<8}{'尺寸':<10}{'轮廓 IoU':>10}{'用色数':>8}{'调色板合规':>12}")
    print("-" * 50)
    results = []
    for name, f in srcs.items():
        tex = Image.open(f).convert("RGB")
        for n in args.sizes:
            mask = region_mask(src_img, bg, n)
            t = quantize_to(np.asarray(tex.resize((n, n), Image.BOX), dtype=np.float64), fg)
            out = np.where(mask[..., None], t, np.array(bg, dtype=np.float64))
            om = (out != np.array(bg)).any(-1)
            used = np.unique(out.reshape(-1, 3), axis=0)
            ok = all(bool((palette == c).all(-1).any()) for c in used)
            results.append({"material": name, "size": n, "iou": round(iou(mask, om), 4),
                            "n_colors": len(used), "palette_compliant": ok})
            print(f"{name:<8}{str(n)+'x'+str(n):<10}{iou(mask, om):>10.4f}{len(used):>8}"
                  f"{'是' if ok else '否':>12}")
            Image.fromarray(out.astype(np.uint8)).save(args.out / f"{name}_{n}.png")
            Image.fromarray(out.astype(np.uint8)).resize((512, 512), Image.NEAREST).save(
                args.out / f"{name}_{n}_x512.png")

    (args.out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()
