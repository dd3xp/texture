"""方法一的对照：收益来自「候选变多」，还是来自「按 frac 挑」？

`bestof_units.py` 证明了多采样把有效裁剪率从 52% 提到 90%，
新救回的 16 个材质判官偏好 83%。但**那个收益有两个可能的来源**：

  (a) 候选变多 —— 四个样本里只要有一个能裁，这个材质就被覆盖了。
      这与「挑哪一个」无关，随便挑一个能裁的都行。
  (b) 选样规则 —— 「取 frac 最大」= 取生成时最接近真人惯例的那版，
      比随便挑一个更好。

`bestof_units.py` 的判据②（在本来就能处理的材质上换样本）是 3/6，n 太小，
分不开这两者。本脚本在**同一批候选**上直接比两条规则，一次渲染回答这个问题。

  规则 A（frac）：有效裁剪的样本里，取 frac 最大的
  规则 B（随机）：有效裁剪的样本里，用固定种子均匀随机取一个

--- 判据（跑之前写下并 commit）---
只取**两条规则选中了不同样本**的材质（选中同一个则无从比较）。
主判据：经验证判官（claude-opus-5，正反去偏、不一致弃用）偏好规则 A
        >50% 且二项 p<0.05 -> 选样规则本身有效，(b) 成立。
证伪：≤50% 或不显著 -> **收益来自多采样本身，与挑哪一个无关**；
        方法一的主张须收窄为「多采样提高覆盖」，`docs/loop.md` 里那句
        「不能说按 frac 挑有效」就此坐实，正文若写到也必须这样写。
方向性限定：判官压缩效应，为正作下界；为负只能说「没测到大效应」。

⚠ 随机规则用固定种子（`--pick-seed`），使本实验可复跑；
   它不是「最差情形」，只是「无信息选样」的一个实例。
"""
import argparse
import json
import os
import random
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
    ap.add_argument("-N", type=int, default=4)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21, help="与 bestof_units 同种子，候选池一致")
    ap.add_argument("--pick-seed", type=int, default=7, help="随机规则的选样种子")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/bestof_rule.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/bestof_rule")
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
    rng = random.Random(a.pick_seed)

    recs = []
    for pi, p in enumerate(prompts):
        cands = []
        for k in range(a.N):
            # 与 bestof_units.py 完全相同的种子公式 -> 候选池逐张一致
            g = torch.Generator("cuda").manual_seed(a.seed + pi + 1000 * k)
            im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=a.render, width=a.render).images[0]
            arr = np.asarray(im).astype(float)
            per, ani = dominant_period(arr), anisotropy(arr)
            tile, frac = to_tile(arr, a.size, a.colors, pi)
            cands.append({"k": k, "period": float(per), "aniso": float(ani),
                          "frac": float(frac), "tile": tile,
                          "eff": bool(per > 0 and ani >= 0.20 and frac < 0.999)})
        eff = [c for c in cands if c["eff"]]
        if not eff:
            recs.append({"prompt": p, "n_eff": 0, "same": True})
            print(f"[{pi+1}/{len(prompts)}] {p:<32} 无有效候选，跳过", flush=True)
            continue
        by_frac = max(eff, key=lambda c: c["frac"])
        by_rand = eff[rng.randrange(len(eff))]
        same = by_frac["k"] == by_rand["k"]
        slug = p.replace(" ", "_")
        if not same:
            Image.fromarray(by_frac["tile"]).save(a.tiles / f"{slug}_frac.png")
            Image.fromarray(by_rand["tile"]).save(a.tiles / f"{slug}_rand.png")
        recs.append({"prompt": p, "n_eff": len(eff), "same": same,
                     "k_frac": by_frac["k"], "frac_frac": by_frac["frac"],
                     "k_rand": by_rand["k"], "frac_rand": by_rand["frac"]})
        print(f"[{pi+1}/{len(prompts)}] {p:<32} 有效 {len(eff)}/{a.N}  "
              f"frac 规则 k={by_frac['k']}({by_frac['frac']:.3f})  "
              f"随机 k={by_rand['k']}({by_rand['frac']:.3f})"
              f"{'  同一张' if same else ''}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    diff = [r for r in recs if not r["same"] and r.get("n_eff", 0) > 1]
    print(f"\n两条规则选中不同样本的材质：{len(diff)}/{len(recs)}")
    if not diff:
        print("没有可比的材质（规则总是选同一张），无从评估"); return
    if a.no_judge:
        return

    from vlm_judge import ask
    import base64, io

    def b64(p_):
        arr = np.asarray(Image.open(p_).convert("RGB"))
        buf = io.BytesIO()
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    win = tot = inc = 0
    for r in diff:
        slug = r["prompt"].replace(" ", "_")
        F = b64(a.tiles / f"{slug}_frac.png")
        R = b64(a.tiles / f"{slug}_rand.png")
        q = Q.format(n=a.size, label=r["prompt"])
        o1, o2 = ask(a.model, q, [F, R], base_url, key), ask(a.model, q, [R, F], base_url, key)
        if not o1 or not o2:
            continue
        p1 = "frac" if o1.upper().startswith("A") else "rand"
        p2 = "rand" if o2.upper().startswith("A") else "frac"
        pick = p1 if p1 == p2 else "inconsistent"
        if pick == "inconsistent":
            inc += 1
        else:
            tot += 1; win += pick == "frac"
        r["vlm"] = pick
        print(f"  {r['prompt']:<32} -> {pick}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    if not tot:
        print("无有效判断"); return
    pv = binom_test(win, tot); lo, hi = jeffreys(win, tot)
    print(f"\n主判据：frac 规则被选 {win}/{tot} = {win/tot:.0%}  p={pv:.3g}"
          f"  [{lo:.0%},{hi:.0%}]   （弃 {inc}）")
    if win / tot > 0.5 and pv < 0.05:
        print("  -> **成立**：选样规则本身有效（判官压缩，故为下界）")
    else:
        print("  -> 不成立：**收益来自多采样本身，与挑哪一个无关**；")
        print("     方法一的主张须收窄为「多采样提高覆盖」。")


if __name__ == "__main__":
    main()
