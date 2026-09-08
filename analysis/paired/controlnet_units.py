"""「每图 ~25 个结构单元」是模型烧死的惯例，还是提示词模式的产物？

论文限制节写着最致命的一条：单元惯例只在 **SDXL 一个生成器**上量过。
换底模需要联网（机器离线，只缓存了 SDXL），但本地有 **ControlNet（canny/depth）**——
可以不换模型而换**生成条件**：给它一张单元数正确的条件图，看它照条件画还是照自己的惯例画。

两种结果都有价值，且指向不同的论文改动：
  A. **跟随条件**（渲染单元数接近条件图的 ~4.5）
     -> 惯例不是烧死的，是提示词模式下的产物；
        而且意味着存在**比裁剪更好的解法**（直接约束生成），论文目前缺的正是「方法」贡献。
  B. **不跟随**（仍是 ~25）
     -> 惯例是模型层面的，机制主张变强，裁剪作为后处理更有理由。

--- 判据（跑之前写下并 commit）---
条件图按构造含 4.5 个横向单元（砖行），故「条件图单元数正确」无需检验。
操作检验：ControlNet 必须真的起作用——条件组渲染的**单元数中位**必须与对照组
        （纯提示词，历史值 ~27.7）**有显著差异**（配对符号检验 p<0.05）。
        无差异 = ControlNet 没接上或强度太低，两条结论都不能下。
主判据（二选一，跑前定义，不事后挑）：
  条件组单元数中位 **≤ 9**（即落进 4.5 的两倍内）-> 判为 A「跟随条件」；
  条件组单元数中位 **≥ 18** -> 判为 B「不跟随」；
  落在 9–18 之间 -> **判为不确定**，不写进论文任何一侧。
纯测量，不需要判官——单元数是客观量（`dominant_period`）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis"):
    sys.path.insert(0, str(ROOT / sub))
from downsample import dominant_period, anisotropy            # noqa: E402
from exact import binom_test                                   # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"

# 有方向性周期的材质——只有这类能量出「单元数」
MATERIALS = [
    "brick wall", "stone brick wall", "wooden planks floor", "clay roof tiles",
    "sandstone block wall", "herringbone brick paving", "subway tile wall",
    "log cabin wall", "roman tile roof", "cinder block wall",
    "slate roof shingles", "teak deck boards",
]
TARGET_UNITS = 4.5      # 真人惯例（论文 §5.2 的 3.2 是生产口径；4.5 是流水线常数）


def brick_canny(size: int, units: float) -> Image.Image:
    """构造一张含 `units` 个横向砖行的边缘图，供 canny ControlNet 使用。

    直接画白线黑底 = 已经是边缘图，不需要再跑 canny 检测器。
    行高 = size/units；每行错缝半块，与真人砖材质一致。
    """
    a = np.zeros((size, size), np.uint8)
    row_h = size / units
    col_w = row_h * 2.2                       # 砖长宽比约 2.2
    for r in range(int(units) + 2):
        y = int(r * row_h)
        if 0 <= y < size:
            a[y:y + 2, :] = 255               # 横缝
        off = (col_w / 2) if r % 2 else 0     # 错缝
        c = 0
        while off + c * col_w < size:
            x = int(off + c * col_w)
            if 0 <= x < size:
                a[y:min(y + int(row_h), size), x:x + 2] = 255   # 竖缝
            c += 1
    return Image.fromarray(np.stack([a] * 3, -1))


def units_of(img: np.ndarray) -> float:
    per = dominant_period(img)
    return float(img.shape[0] / per) if per > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--scale", type=float, default=0.8,
                    help="ControlNet conditioning scale")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/controlnet_units.json")
    ap.add_argument("--keep", type=Path, default=ROOT / "experiments/cnet")
    a = ap.parse_args()

    import torch
    from diffusers import (StableDiffusionXLPipeline,
                           StableDiffusionXLControlNetPipeline, ControlNetModel)

    mats = MATERIALS[:a.limit] if a.limit else MATERIALS
    cond = brick_canny(a.render, TARGET_UNITS)
    a.keep.mkdir(parents=True, exist_ok=True)
    cond.save(a.keep / "condition.png")
    print(f"条件图已存：{a.render}px，构造含 {TARGET_UNITS} 个横向单元")

    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    base.set_progress_bar_config(disable=True)

    cn = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-mid", torch_dtype=torch.float16)
    cpipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=cn,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    cpipe.set_progress_bar_config(disable=True)

    recs = []
    for i, m in enumerate(mats):
        g1 = torch.Generator("cuda").manual_seed(a.seed + i)
        plain = base(TMPL.format(p=m), negative_prompt=NEG,
                     num_inference_steps=a.steps, generator=g1,
                     height=a.render, width=a.render).images[0]
        g2 = torch.Generator("cuda").manual_seed(a.seed + i)
        condi = cpipe(TMPL.format(p=m), negative_prompt=NEG, image=cond,
                      controlnet_conditioning_scale=a.scale,
                      num_inference_steps=a.steps, generator=g2,
                      height=a.render, width=a.render).images[0]
        pa, ca = np.asarray(plain).astype(float), np.asarray(condi).astype(float)
        slug = m.replace(" ", "_")
        plain.save(a.keep / f"{slug}_plain.png")
        condi.save(a.keep / f"{slug}_cond.png")
        recs.append({"material": m, "units_plain": units_of(pa),
                     "units_cond": units_of(ca),
                     "aniso_plain": float(anisotropy(pa)),
                     "aniso_cond": float(anisotropy(ca))})
        print(f"[{i+1}/{len(mats)}] {m:<26} 单元数 纯提示 {recs[-1]['units_plain']:6.1f}"
              f"  条件 {recs[-1]['units_cond']:6.1f}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    pl = [r["units_plain"] for r in recs if r["units_plain"] > 0]
    co = [r["units_cond"] for r in recs if r["units_cond"] > 0]
    both = [(r["units_plain"], r["units_cond"]) for r in recs
            if r["units_plain"] > 0 and r["units_cond"] > 0]
    if not both:
        print("两组都检出周期的材质为 0，无法评估"); return
    down = sum(1 for p, c in both if c < p)
    ps = binom_test(down, len(both))
    mp, mc = float(np.median(pl)), float(np.median(co))
    print(f"\n检出周期：纯提示 {len(pl)}/{len(recs)}，条件 {len(co)}/{len(recs)}，"
          f"两者都有 {len(both)}")
    print(f"操作检验：单元数中位 纯提示 {mp:.1f} -> 条件 {mc:.1f}；"
          f"变小 {down}/{len(both)}，符号检验 p={ps:.3g}")
    if ps >= 0.05:
        print("  -> **ControlNet 没产生显著差异**，两条结论都不能下（可能强度太低或没接上）")
        return
    print("  -> 操作有效，评估主判据")
    if mc <= 9:
        print(f"判读：**A 跟随条件**（中位 {mc:.1f} ≤ 9）-> 惯例不是模型烧死的；"
              "存在比裁剪更直接的解法，值得作为方法写进论文")
    elif mc >= 18:
        print(f"判读：**B 不跟随**（中位 {mc:.1f} ≥ 18）-> 惯例在模型层面；"
              "机制主张变强，裁剪作为后处理更有理由")
    else:
        print(f"判读：**不确定**（中位 {mc:.1f} 落在 9–18）-> 按跑前约定，"
              "不写进论文任何一侧")


if __name__ == "__main__":
    main()
