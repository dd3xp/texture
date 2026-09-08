"""规模化验收 B5（按结构尺度裁剪），用 VLM 粗筛。

B6 已量出 VLM 判官的性质：**方向对、幅度压缩**
（A4 的 87% 被压成 62%，p=0.14），并会抹平分层。
所以它**不能下结论**，但可以用来粗筛——
而 B5 的效应很大（目视 1/4 vs 15/16），压缩后应当仍可见。

设计：同一提示词、同一种子渲染一次，只差裁剪那一步，
**正反各问一次去位置偏好**（B6 实测两模型选左都在 67%）。

产出的是**粗筛证据**，论文里的结论仍以人工盲比为准。
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import auto_crop, dominant_period                  # noqa: E402
from make_texture import extract_palette, quantize                 # noqa: E402

PROMPTS = [
    # 周期性建材
    "brick wall", "stone brick wall", "mossy cracked stone bricks",
    "sandstone block wall", "clay roof tiles", "slate roof shingles",
    "wooden planks floor", "parquet wood floor", "bamboo mat",
    "checkered tiled floor", "hexagonal floor tiles", "metal grate panel",
    "chain link fence", "woven basket surface", "corrugated metal sheet",
    "stacked stone wall", "concrete blocks wall", "adobe mud bricks",
    # 颗粒/无周期
    "coarse gravel path", "desert sand dunes", "grass turf top",
    "snow covered ground", "cracked dry mud", "volcanic ash ground",
    "iron ore in grey stone", "gold ore in dark rock", "coal ore vein",
    "moss covered forest floor", "fallen autumn leaves", "cobweb corner",
    # 有机/其他
    "tree bark oak", "birch tree bark", "cactus skin",
    "reptile scales", "fish scales", "honeycomb wax",
    "rusty iron plate", "polished marble slab", "cracked obsidian glass",
    "woven wool cloth", "burlap sack fabric", "quilted padding",
]
TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"
Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")


def b64(a):
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).resize((256, 256), Image.NEAREST).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(model, prompt, imgs, base, key, retries=3):
    content = [{"type": "text", "text": prompt}]
    for b in imgs:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b}})
    body = {"model": model, "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    for a in range(retries):
        try:
            r = requests.post(base.rstrip("/") + "/v1/chat/completions",
                              headers={"Authorization": "Bearer " + key},
                              json=body, timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        time.sleep(2 + 3 * a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--fewer-units", action="store_true",
                    help="加修饰词让 SDXL 少画结构单元——检验单元惯例说（预注册于 8207f3f）")
    ap.add_argument("--render-size", type=int, default=1024,
                    help="SDXL 渲染分辨率。384 时单元数显著变少（render_res_probe 真中位 30.1→8.8，−70.9%，MW p=0.0066；旧记的 32.5→9.3 是 med() 取上中位数所致，已修），是唯一通过操作检验的少单元杠杆")
    ap.add_argument("--prompts", type=Path,
                    help="外挂提示词表（JSON 数组）。缺省用内置的 42 个。"
                         "泛化复现用：见 experiments/prompts_holdout60.json")
    ap.add_argument("--out", type=Path, default=Path("experiments/crop_scale_study.json"))
    args = ap.parse_args()
    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要 VLM_BASE_URL / VLM_API_KEY")

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    recs = []
    prompts = (json.loads(args.prompts.read_text(encoding='utf-8'))
               if args.prompts else PROMPTS)
    for pi, p in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(args.seed + pi)
        pr = TMPL.format(p=p)
        if args.fewer_units:
            pr += ", very few large blocks, macro close-up, minimal detail"
        im = pipe(pr, negative_prompt=NEG,
                  num_inference_steps=args.steps, generator=g,
                  height=args.render_size, width=args.render_size).images[0]
        a = np.asarray(im).astype(float)
        for n in args.sizes:
            c, frac = auto_crop(a, n)
            fired = frac < 0.999

            def tile(arr):
                sm = np.asarray(Image.fromarray(arr.astype(np.uint8))
                                .resize((n, n), Image.BOX))
                return quantize(sm, extract_palette(sm, 12))
            before, after = tile(a), tile(c)
            rec = {"prompt": p, "size": n, "fired": bool(fired),
                   "frac": float(frac),
                   "period": float(dominant_period(a))}
            if fired:
                q = Q.format(n=n, label=p)
                B, A = b64(before), b64(after)
                o1 = ask(args.model, q, [B, A], base, key)
                o2 = ask(args.model, q, [A, B], base, key)
                if o1 and o2:
                    p1 = "before" if o1.upper().startswith("A") else "after"
                    p2 = "after" if o2.upper().startswith("A") else "before"
                    rec["vlm"] = p1 if p1 == p2 else "inconsistent"
            recs.append(rec)
        print(f"[{pi+1}/{len(prompts)}] {p:<32} "
              + " ".join(f"{r['size']}:{'裁' if r['fired'] else '不裁'}"
                         f"{r.get('vlm','-')[:4]}" for r in recs[-len(args.sizes):]),
              flush=True)

    args.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    fired = [r for r in recs if r["fired"]]
    judged = [r for r in fired if r.get("vlm") in ("before", "after")]
    inc = sum(1 for r in fired if r.get("vlm") == "inconsistent")
    print(f"\n提示词 {len(prompts)}，尺寸 {args.sizes}，共 {len(recs)} 例")
    print(f"  裁剪触发 {len(fired)}/{len(recs)} = {len(fired)/len(recs):.0%}")
    print(f"  VLM 有效判断 {len(judged)}，正反不一致弃用 {inc}")
    def binom_p(w, n):  # 双侧精确二项检验（jzs_train 环境无 scipy）
        import math
        return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, n - w) + 1)) / 2 ** n)

    if judged:
        w = sum(1 for r in judged if r["vlm"] == "after")
        print(f"  **裁剪后胜 {w}/{len(judged)} = {w/len(judged):.0%}**"
              f"   p={binom_p(w, len(judged)):.3g}")
        for n in args.sizes:
            s = [r for r in judged if r["size"] == n]
            if s:
                ww = sum(1 for r in s if r["vlm"] == "after")
                print(f"    {n}x{n}: {ww}/{len(s)} = {ww/len(s):.0%}"
                      f"   p={binom_p(ww, len(s)):.3g}")
    print("\n这是**粗筛证据**（B6：VLM 会压缩效应），结论以人工盲比为准。")


if __name__ == "__main__":
    main()
