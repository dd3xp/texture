"""批量出一套可直接用的低分辨率纹理（模组交付用，不是论文实验）。

与论文实验一致：**不带 adapter 的 SDXL base**。

`--lora` 可以开像素画 LoRA，但**默认关**。修好离线加载后跑了 4 个材质的
对比（`--compare`，见 `experiments/lora_compare.png`）：带 LoRA 的一批边缘更糊、
对比度更低、砖石结构更难读。⚠ 这是**目视判断，未做盲比**，按本项目的规矩
不能当成结论——但既然没有证据说它更好，交付就没有理由偏离论文的流水线。

流程与论文一致：渲染 → 双条件门（周期 + 各向异性 ≥0.20）→ 按主周期裁剪
→ 降到目标尺寸 → 调色板量化。门不触发的材质**照常输出但会标注**——
它们只是不裁，不是失败。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# 模组里最常见的方块材质。刻意包含几个各向同性的（沙、砾石、树叶），
# 用来看门在交付批次上的拒绝率是否与实验一致（约一半）。
MATERIALS = [
    "brick wall", "stone brick wall", "cobblestone", "mossy cobblestone",
    "wooden planks floor", "oak log bark", "sandstone block", "smooth stone",
    "clay roof tiles", "woven basket surface", "iron metal plate",
    "gold block surface", "coal ore in stone", "iron ore in stone",
    "packed dirt", "grass turf top", "coarse gravel", "dry sand",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--lora", default="none",
                    help="默认不用 adapter，与论文流水线一致；见模块文档")
    ap.add_argument("--compare", action="store_true",
                    help="每个材质额外出一张不带 LoRA 的，用于对比")
    ap.add_argument("--prompts", type=Path,
                    help="外挂材质表（JSON 数组）。缺省用内置的 18 个。")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/pack")
    a = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline
    from downsample import auto_crop, dominant_period, anisotropy
    from make_texture import extract_palette, quantize
    from from_prompt import TMPL, NEG, load_lora

    pool = (json.loads(a.prompts.read_text(encoding='utf-8'))
            if a.prompts else MATERIALS)
    mats = pool[:a.limit] if a.limit else pool
    a.out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    variants = ([("base", "none"), ("lora", "nerijs/pixel-art-xl")]
                if a.compare else [("base", a.lora)])
    recs = []
    for vname, lora in variants:
        # 每个变体重建一次 adapter 状态，避免上一轮的权重残留
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        load_lora(pipe, lora)
        for mi, m in enumerate(mats):
            g = torch.Generator("cuda").manual_seed(a.seed + mi)
            im = pipe(TMPL.format(p=m), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=a.render, width=a.render).images[0]
            src = np.asarray(im).astype(float)
            per, ani = dominant_period(src), anisotropy(src)
            slug = m.replace(" ", "_")
            for size in a.sizes:
                cropped, frac = auto_crop(src, size)
                small = np.asarray(Image.fromarray(cropped.astype(np.uint8))
                                   .resize((size,) * 2, Image.BOX))
                tile = quantize(small, extract_palette(small, a.colors, seed=mi))
                d = a.out / vname / str(size)
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile.astype(np.uint8)).save(d / f"{slug}.png")
                recs.append({"variant": vname, "material": m, "size": size,
                             "period": per, "aniso": ani, "frac": frac,
                             "gate_fired": frac < 1.0})
            print(f"[{vname}] {mi+1}/{len(mats)} {m:<24} "
                  f"period={per:6.1f} aniso={ani:.2f} "
                  f"{'裁 1/%.1f' % (1/recs[-1]['frac']) if recs[-1]['gate_fired'] else '门未触发'}",
                  flush=True)

    (a.out / "manifest.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    fired = [r for r in recs if r["size"] == a.sizes[0] and r["variant"] == "base"]
    if fired:
        n = sum(r["gate_fired"] for r in fired)
        rate = n / len(fired)
        print(f"\n门在 {n}/{len(fired)} 个材质上触发（{rate:.0%}）")
        # 判读跟着数走。此前这里无条件打印「与实验批次的约一半一致」，
        # 小样本全是方向性材质时会打出「100%…约一半一致」的自相矛盾。
        if len(fired) < 8:
            print("      样本太小，不与实验批次的约 50% 作比较")
        elif 0.35 <= rate <= 0.65:
            print("      与实验批次的约 50% 一致")
        else:
            print("      与实验批次的约 50% **不一致**，看一眼材质构成")
    print(f"写入 {a.out}")


if __name__ == "__main__":
    main()
