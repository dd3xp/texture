"""M2 基线出图：大模型 + 降采样这一族（B1 / B2 / B4），同一批提示词、同一套种子。

  B1  SDXL 1024 → 盒式降采样到 size → 量化 12 色          （"大模型 + 降采样"，每材质 n 个样本）
  B2  本仓库现有管线：同一批 n 张渲染里按 `auto_crop` 选有效裁剪比最大的一张，
      接缝对齐裁剪 → 盒式降采样 → 量化                    （每材质 1 张，n=4 即 best-of-4）
  B4  SDXL + 像素画 LoRA（nerijs/pixel-art-xl）→ 盒式降采样 → 量化   （`--lora pixelart`）

提示词集来自 `eval/prompt_sets.json`（已在出结果前定死）；`--set V_mat` 在验证集上出基线
（配 `--out experiments/baselines_val`，给新架构调参时对照 B2，不碰测试集）。模板与本仓库一贯的一致。
种子 = seed + 材质序号 + 1000*样本号（与 batch_pack 同式），所有基线可复现。
只存瓦片（16/24/32 PNG）与清单，不存 1024 渲染（服务器磁盘紧张）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import auto_crop                               # noqa: E402
from make_texture import extract_palette, quantize             # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"


def to_tile(img: np.ndarray, size: int, colors: int, seed: int) -> np.ndarray:
    small = np.asarray(Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.BOX))
    return quantize(small, extract_palette(small, colors, seed=seed)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--lora", choices=["none", "pixelart"], default="none")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/baselines")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "eval"))
    from prompts import load_set                  # E_* 测试集 / V_* 验证集（验证集只用于调参）
    prompts, _ = load_set(a.set)
    if a.limit:
        prompts = prompts[:a.limit]

    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    tags = ["B4"] if a.lora == "pixelart" else ["B1", "B2"]
    if a.lora == "pixelart":
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
        pipe.fuse_lora(lora_scale=1.0)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for t in tags:
        for s in a.sizes:
            (a.out / t / str(s)).mkdir(parents=True, exist_ok=True)

    man_path = a.out / f"manifest_{'_'.join(tags)}.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    t0 = time.time()
    for mi, e in enumerate(prompts):
        slug = e["material"].rsplit(".", 1)[0]
        if slug in man:
            continue
        rec = {"prompt": e["prompt"], "samples": []}
        renders = []
        for k in range(a.n):
            g = torch.Generator("cuda").manual_seed(a.seed + mi + 1000 * k)
            im = pipe(TMPL.format(p=e["prompt"]), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=1024, width=1024).images[0]
            arr = np.asarray(im).astype(float)
            renders.append(arr)
            tag = tags[0]
            for s in a.sizes:
                Image.fromarray(to_tile(arr, s, a.colors, mi)).save(
                    a.out / tag / str(s) / f"{slug}_{k}.png")
            _, frac = auto_crop(arr, a.sizes[0])
            rec["samples"].append({"k": k, "frac": float(frac)})
        if a.lora == "none":                       # B2：同一批渲染上的 best-of-n
            eff = [c for c in rec["samples"] if c["frac"] < 0.999]
            best = max(eff, key=lambda c: c["frac"])["k"] if eff else 0
            rec["b2_pick"] = best
            for s in a.sizes:
                crop, _ = auto_crop(renders[best], s, seam_align=True)
                Image.fromarray(to_tile(crop, s, a.colors, mi)).save(
                    a.out / "B2" / str(s) / f"{slug}.png")
        man[slug] = rec
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        el = time.time() - t0
        print(f"[{mi+1}/{len(prompts)}] {e['prompt']:<32} {el/60:6.1f} min", flush=True)
    print("done", man_path)


if __name__ == "__main__":
    main()
