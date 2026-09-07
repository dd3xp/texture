"""第 30 轮仪器验钥：按 build_study_crop.py 的确定性管线重生成全部 (prompt, seed)，
与 study_crop.html 内嵌图逐像素比对，验证 left/right 答案钥匙。

需在 emnlp GPU 上、repo 根目录运行（HF_HUB_OFFLINE=1，SDXL 种子确定）。
第 30 轮实跑结果：39/39 KEY-OK、0 SWAPPED、0 NO-MATCH
（experiments/spotcheck_crop_key.{txt,json}）。
"""
import base64, io, json, re, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "tools")
from downsample import auto_crop
from make_texture import extract_palette, quantize

PROMPTS = ["brick wall", "stone brick wall", "mossy cracked stone bricks",
           "wooden planks floor", "cobblestone path", "clay roof tiles",
           "woven basket surface", "checkered tiled floor",
           "tree log bark side", "sandstone block wall",
           "metal grate panel", "stacked slate shingles",
           "red brick pavement", "wooden fence planks", "corrugated metal roof",
           "stacked log wall", "subway tile wall", "parquet wood flooring",
           "adobe mud brick wall", "bamboo mat weave"]
TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"
SIZE, COLORS, STEPS = 16, 12, 28

html = Path("experiments/annotate/study_crop.html").read_text(encoding="utf-8")
items = json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", html, re.S).group(1))


def img(b):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))


from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
pipe.set_progress_bar_config(disable=True)

report = []
for p in PROMPTS:
    gen = {}
    for k in range(3):
        g = torch.Generator("cuda").manual_seed(1000 + k)
        im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                  num_inference_steps=STEPS, generator=g,
                  height=1024, width=1024).images[0]
        a = np.asarray(im).astype(float)
        cropped, frac = auto_crop(a, SIZE)
        if frac >= 0.999:
            continue

        def to_tile(arr, seed):
            small = np.asarray(Image.fromarray(arr.astype(np.uint8))
                               .resize((SIZE,) * 2, Image.BOX))
            return quantize(small, extract_palette(small, COLORS, seed))
        gen[k] = {"before": to_tile(a, k), "after": to_tile(cropped, k),
                  "frac": frac}
    for j, it in enumerate(items):
        if it["kind"] != "real" or it["material"] != p:
            continue
        L, R = img(it["limg"]), img(it["rimg"])
        side = {it["left"]: L, it["right"]: R}
        verdict = "NO-MATCH"
        for k, gg in gen.items():
            ok_b = np.array_equal(side["before"], gg["before"])
            ok_a = np.array_equal(side["after"], gg["after"])
            sw_b = np.array_equal(side["after"], gg["before"])
            sw_a = np.array_equal(side["before"], gg["after"])
            if ok_b and ok_a:
                verdict = f"KEY-OK (seed {1000+k}, frac={gg['frac']:.3f})"
                break
            if sw_b and sw_a:
                verdict = f"KEY-SWAPPED (seed {1000+k})"
                break
        report.append((j, p, it["left"], it["right"], round(it["struct"], 3), verdict))
        print(f"idx={j:2d} {p:28s} L={it['left']:6s} R={it['right']:6s} "
              f"struct={it['struct']:.3f}  {verdict}", flush=True)

okn = sum(1 for r in report if r[5].startswith("KEY-OK"))
print(f"\n钥匙验证: {okn}/{len(report)} KEY-OK；"
      f"SWAPPED={sum(1 for r in report if 'SWAPPED' in r[5])}；"
      f"NO-MATCH={sum(1 for r in report if r[5]=='NO-MATCH')}")
Path("experiments/spotcheck_crop_key.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
