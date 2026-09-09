"""方法一：多采样 + 按单元数选样（best-of-N）。

**动机来自我们自己的机制，不是拍脑袋。** B14 量到单元数驱动质量
（1024/512/384 → 88%/54%/17%）。同一个提示词换个种子，SDXL 画的单元数
差别很大（B24 里 SD1.5 的四分位是 7.3–26.3）。那么：**多采几个，
挑那个本来就画得最接近真人惯例的**，应当同时改善两件事：

1. **适用率**——门目前只在 45–58% 的提示词上触发（无周期或各向异性不足）。
   四个样本里只要有一个有可检出的方向性周期，这个材质就能被处理。
2. **裁剪烈度**——裁剪比 = 周期 × 4.5 / 源边长。周期越大，裁得越轻、保留内容越多。
   而周期大恰恰意味着这一版本来就画得更接近真人惯例。
   所以「在触发的样本里挑裁剪比最大的」= 「挑生成时最接近真人惯例的」。

选样规则（跑前固定，不事后调）：
  在 N 个样本中，优先取**门触发**的；其中取**裁剪比最大**的那个；
  若无一触发，退回第一个样本（与现管线一致，不制造虚假优势）。

--- 判据（跑之前写下并 commit）---
① 适用率（客观量，无需判官）：best-of-4 的门触发率必须**高于**单样本基线，
   且配对符号检验 p<0.05。不过则本方法在适用率上无效。
② 质量（经验证判官 claude-opus-5，正反两问去偏、不一致弃用）：
   在**两者都触发**的材质上，best-of-4 的成品被偏好 >50% 且二项 p<0.05。
   —— 这一条检验的是「挑裁剪比大的」是否真的更好，而不只是更多触发。
③ 证伪：①②都不成立 -> 多采样对本方法无益，如实记录并关闭这条。
方向性限定：判官压缩效应，为正作下界；为负只能说「没测到大效应」。

注意：本实验**不改变**裁剪规则本身，只改变喂给它的那张源图。
"""
import argparse
import json
import os
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


def to_tile(src, size, colors, seed):
    cropped, frac = auto_crop(src, size)
    small = np.asarray(Image.fromarray(cropped.astype(np.uint8))
                       .resize((size,) * 2, Image.BOX))
    return quantize(small, extract_palette(small, colors, seed=seed)).astype(np.uint8), frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=4, help="每个提示词采几个样")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/bestof_units.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/bestof")
    a = ap.parse_args()

    import re
    src = (ROOT / "analysis/paired/crop_scale_study.py").read_text(encoding="utf-8")
    prompts = re.findall(r'"([^"]+)"',
                         re.search(r"PROMPTS\s*=\s*\[(.*?)\]", src, re.S).group(1))
    if a.limit:
        prompts = prompts[:a.limit]

    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    a.tiles.mkdir(parents=True, exist_ok=True)

    recs = []
    for pi, p in enumerate(prompts):
        cands = []
        for k in range(a.N):
            g = torch.Generator("cuda").manual_seed(a.seed + pi + 1000 * k)
            im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=a.render, width=a.render).images[0]
            arr = np.asarray(im).astype(float)
            per, ani = dominant_period(arr), anisotropy(arr)
            fired = per > 0 and ani >= 0.20
            tile, frac = to_tile(arr, a.size, a.colors, pi)
            cands.append({"k": k, "period": float(per), "aniso": float(ani),
                          "fired": bool(fired), "frac": float(frac), "tile": tile})
        # 选样规则：优先门触发；其中裁剪比最大（= 生成时最接近真人惯例）
        fired_c = [c for c in cands if c["fired"]]
        best = max(fired_c, key=lambda c: c["frac"]) if fired_c else cands[0]
        base = cands[0]
        slug = p.replace(" ", "_")
        Image.fromarray(base["tile"]).save(a.tiles / f"{slug}_single.png")
        Image.fromarray(best["tile"]).save(a.tiles / f"{slug}_best.png")
        recs.append({"prompt": p,
                     "single_fired": base["fired"], "single_frac": base["frac"],
                     "best_k": best["k"], "best_fired": best["fired"],
                     "best_frac": best["frac"],
                     "n_fired": len(fired_c),
                     "periods": [c["period"] for c in cands],
                     "anisos": [c["aniso"] for c in cands]})
        print(f"[{pi+1}/{len(prompts)}] {p:<32} 单样本{'触发' if base['fired'] else '未触发'}"
              f"  {len(fired_c)}/{a.N} 触发  选中 k={best['k']}"
              f" frac={best['frac']:.3f}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    # —— 判据① 适用率 ——
    s_fire = sum(r["single_fired"] for r in recs)
    b_fire = sum(r["best_fired"] for r in recs)
    gained = sum(1 for r in recs if r["best_fired"] and not r["single_fired"])
    lost = sum(1 for r in recs if r["single_fired"] and not r["best_fired"])
    p1 = binom_test(gained, gained + lost) if (gained + lost) else 1.0
    print(f"\n判据① 适用率：单样本 {s_fire}/{len(recs)} = {s_fire/len(recs):.0%}"
          f" -> best-of-{a.N} {b_fire}/{len(recs)} = {b_fire/len(recs):.0%}")
    print(f"   新增触发 {gained}，失去 {lost}，符号检验 p={p1:.3g}"
          f"  -> {'**成立**' if gained > lost and p1 < 0.05 else '不成立'}")

    if a.no_judge:
        return
    both = [r for r in recs if r["single_fired"] and r["best_fired"]
            and r["best_k"] != 0]
    if not both:
        print("\n判据②：没有「两者都触发且选了不同样本」的材质，不予评估"); return

    from vlm_judge import ask
    import base64, io

    def b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    win = tot = inc = 0
    for r in both:
        slug = r["prompt"].replace(" ", "_")
        S = b64(np.asarray(Image.open(a.tiles / f"{slug}_single.png").convert("RGB")))
        B = b64(np.asarray(Image.open(a.tiles / f"{slug}_best.png").convert("RGB")))
        q = Q.format(n=a.size, label=r["prompt"])
        o1, o2 = ask(a.model, q, [S, B], base_url, key), ask(a.model, q, [B, S], base_url, key)
        if not o1 or not o2:
            continue
        p1_ = "single" if o1.upper().startswith("A") else "best"
        p2_ = "best" if o2.upper().startswith("A") else "single"
        if p1_ != p2_:
            inc += 1; r["vlm"] = "inconsistent"
        else:
            tot += 1; win += p1_ == "best"; r["vlm"] = p1_
        print(f"  {r['prompt']:<32} -> {r['vlm']}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    if not tot:
        print("判据②：无有效判断"); return
    pv = binom_test(win, tot); lo, hi = jeffreys(win, tot)
    print(f"\n判据② 质量（两者都触发、且选了不同样本的 {len(both)} 个材质）：")
    print(f"   best 被选 {win}/{tot} = {win/tot:.0%}  p={pv:.3g}  [{lo:.0%},{hi:.0%}]"
          f"   （弃 {inc}）")
    print(f"   -> {'**成立**（判官压缩，故为下界）' if win/tot > 0.5 and pv < 0.05 else '不成立（只能说没测到大效应）'}")


if __name__ == "__main__":
    main()
