"""评测方法二：学到的惯例，能不能强过事后裁剪？

判据在 `model/train_lora.py` 的文档串里，已随训练脚本一起 commit（9e56e5a）：
  操作检验：微调模型的单元数中位须 < 基线的一半（27.7 -> 阈值 13.9），
          且逐提示词配对符号检验 p<0.05。不过 = 没学到惯例，主判据不评估。
  主判据：「微调直出、不裁」vs「基线 + 裁剪」，经验证判官偏好前者 >50% 且 p<0.05。
  次判据（描述性）：「微调 + 裁剪」vs「基线 + 裁剪」，看是否叠加。

⚠ **「留出」的确切含义**：训练按**材质包**切分（49 包训练、15 包留出），
所以模型没见过留出包对同一材质的画法，但**材质名本身可能在训练包里出现过**。
这防的是「背下某个包的具体瓦片」，不防「见过这个词」。
本实验要检验的是**惯例**（每图画几个单元）而不是具体像素，该切分对此是够的——
但不能据此宣称「未见过的材质」，措辞须准确。

评测提示词沿用 `crop_scale_study.py` 的 42 条，与此前全部结果口径一致。
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/annotate"):
    sys.path.insert(0, str(ROOT / sub))
from downsample import auto_crop, dominant_period, anisotropy   # noqa: E402
from make_texture import extract_palette, quantize              # noqa: E402
from exact import binom_test, jeffreys                          # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"
Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")
BASE_UNITS = 27.7          # B14：基线 1024 上的单元数中位


def tile_of(src, size, colors, seed, crop: bool):
    a = src
    frac = 1.0
    if crop:
        a, frac = auto_crop(src, size)
    small = np.asarray(Image.fromarray(a.astype(np.uint8))
                       .resize((size,) * 2, Image.BOX))
    return quantize(small, extract_palette(small, colors, seed=seed)).astype(np.uint8), frac


def judge_pair(ask, model, label, size, imgA, imgB, base_url, key):
    """A/B 各问一次，返回 'A'/'B'/'inconsistent'/None。"""
    import base64, io

    def b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    q = Q.format(n=size, label=label)
    x, y = b64(imgA), b64(imgB)
    o1, o2 = ask(model, q, [x, y], base_url, key), ask(model, q, [y, x], base_url, key)
    if not o1 or not o2:
        return None
    p1 = "A" if o1.upper().startswith("A") else "B"
    p2 = "B" if o2.upper().startswith("A") else "A"
    return p1 if p1 == p2 else "inconsistent"


def report(name, wins, tot, inc, good_msg, bad_msg):
    if not tot:
        print(f"{name}：无有效判断"); return
    p = binom_test(wins, tot); lo, hi = jeffreys(wins, tot)
    print(f"{name}：{wins}/{tot} = {wins/tot:.0%}  p={p:.3g}  [{lo:.0%},{hi:.0%}]"
          f"   （弃 {inc}）")
    print(f"   -> {good_msg if wins/tot > 0.5 and p < 0.05 else bad_msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", type=Path, default=ROOT / "runs/lora_convention")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--scale", type=float, default=1.0, help="LoRA 强度")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/lora_eval.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/loraeval")
    a = ap.parse_args()

    import re
    src = (ROOT / "analysis/paired/crop_scale_study.py").read_text(encoding="utf-8")
    prompts = re.findall(r'"([^"]+)"',
                         re.search(r"PROMPTS\s*=\s*\[(.*?)\]", src, re.S).group(1))
    if a.limit:
        prompts = prompts[:a.limit]

    import torch
    from diffusers import StableDiffusionXLPipeline
    dt = torch.bfloat16
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dt,
        variant="fp16", use_safetensors=True).to("cuda")
    base.set_progress_bar_config(disable=True)
    tuned = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dt,
        variant="fp16", use_safetensors=True).to("cuda")
    # 离线模式下必须显式给 weight_name，否则 diffusers 只报一句就失败
    # （`tools/from_prompt.py` 的 load_lora 修的是同一个坑，这里是本地目录）。
    tuned.load_lora_weights(str(a.lora), weight_name="pytorch_lora_weights.safetensors")
    tuned.fuse_lora(lora_scale=a.scale)
    tuned.set_progress_bar_config(disable=True)
    a.tiles.mkdir(parents=True, exist_ok=True)

    recs = []
    for pi, p in enumerate(prompts):
        imgs = {}
        for tag, pipe in (("base", base), ("lora", tuned)):
            g = torch.Generator("cuda").manual_seed(a.seed + pi)
            im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=a.render, width=a.render).images[0]
            imgs[tag] = np.asarray(im).astype(float)
        u = {t: (a.render / dominant_period(v) if dominant_period(v) > 0 else 0.0)
             for t, v in imgs.items()}
        slug = p.replace(" ", "_")
        out = {"prompt": p, "units_base": u["base"], "units_lora": u["lora"],
               "aniso_base": float(anisotropy(imgs["base"])),
               "aniso_lora": float(anisotropy(imgs["lora"]))}
        for tag, crop in (("base_crop", True), ("lora_raw", False), ("lora_crop", True)):
            srcimg = imgs["base" if tag == "base_crop" else "lora"]
            t_, f_ = tile_of(srcimg, a.size, a.colors, pi, crop)
            Image.fromarray(t_).save(a.tiles / f"{slug}_{tag}.png")
            out[f"frac_{tag}"] = f_
        recs.append(out)
        print(f"[{pi+1}/{len(prompts)}] {p:<32} 单元数 base {u['base']:6.1f}"
              f"  lora {u['lora']:6.1f}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    # —— 操作检验 ——
    both = [(r["units_base"], r["units_lora"]) for r in recs
            if r["units_base"] > 0 and r["units_lora"] > 0]
    lo_u = [r["units_lora"] for r in recs if r["units_lora"] > 0]
    if not both or not lo_u:
        print("\n操作检验：检出周期的样本不足，无法评估"); return
    med = statistics.median(lo_u)
    down = sum(1 for b, l in both if l < b)
    ps = binom_test(down, len(both))
    print(f"\n操作检验：LoRA 单元数中位 {med:.1f}（基线 {BASE_UNITS}，阈值 {BASE_UNITS/2:.1f}）；"
          f"变小 {down}/{len(both)}，符号检验 p={ps:.3g}")
    if not (med < BASE_UNITS / 2 and ps < 0.05):
        print("  -> **没学到惯例**，主判据不评估（判据已在跑前固定）")
        return
    print("  -> 学到了，评估主判据")
    if a.no_judge:
        return

    from vlm_judge import ask
    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    tally = {k: [0, 0, 0] for k in ("main", "sec")}   # wins, tot, inconsistent
    for r in recs:
        slug = r["prompt"].replace(" ", "_")
        ld = lambda t: np.asarray(Image.open(a.tiles / f"{slug}_{t}.png").convert("RGB"))
        for key_, chal in (("main", "lora_raw"), ("sec", "lora_crop")):
            v = judge_pair(ask, a.model, r["prompt"], a.size,
                           ld("base_crop"), ld(chal), base_url, key)
            if v is None:
                continue
            if v == "inconsistent":
                tally[key_][2] += 1
            else:
                tally[key_][1] += 1
                tally[key_][0] += v == "B"       # B = 挑战者
            r[f"vlm_{key_}"] = v
        print(f"  {r['prompt']:<32} 主 {r.get('vlm_main')}  次 {r.get('vlm_sec')}",
              flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    report("主判据（LoRA 直出不裁 vs 基线+裁剪）", *tally["main"],
           "**成立**：学到的惯例强过事后裁剪（判官压缩，故为下界）",
           "不成立：只能说没测到大效应，学惯例不比裁剪好")
    report("次判据（LoRA+裁剪 vs 基线+裁剪）", *tally["sec"],
           "两者叠加", "没测到叠加")


if __name__ == "__main__":
    main()
