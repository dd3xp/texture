"""B6：按结构尺度裁剪（B5）修前 vs 修后 的盲比。

B5 的"15/16 可用"目前只是我的目视判断，单观察者。
本项目已经栽过一次：凭画廊收下的缝档位改动，后来被尺子证明有害（A3y）。
所以把它钉死。

设计：同一句提示词、**同一个随机种子**，只差裁剪那一步——
渲染图完全相同，因此比较是干净的单变量。
只收自动裁剪**实际触发**的样本（未触发时两图相同，没有可比性）。
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import auto_crop                                   # noqa: E402
from make_texture import extract_palette, quantize                 # noqa: E402

PROMPTS = ["brick wall", "stone brick wall", "mossy cracked stone bricks",
           "wooden planks floor", "cobblestone path", "clay roof tiles",
           "woven basket surface", "checkered tiled floor",
           "tree log bark side", "sandstone block wall",
           "metal grate panel", "stacked slate shingles"]
TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"


def b64(a):
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--n-check", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/annotate/study_crop.html"))
    args = ap.parse_args()

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    rng = random.Random(0)
    items, sides, skipped = [], {}, 0
    for p in PROMPTS:
        for k in range(args.per_prompt):
            g = torch.Generator("cuda").manual_seed(1000 + k)
            im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                      num_inference_steps=args.steps, generator=g,
                      height=1024, width=1024).images[0]
            a = np.asarray(im).astype(float)
            cropped, frac = auto_crop(a, args.size)
            if frac >= 0.999:
                skipped += 1
                continue

            def to_tile(arr, seed):
                small = np.asarray(Image.fromarray(arr.astype(np.uint8))
                                   .resize((args.size,) * 2, Image.BOX))
                return quantize(small, extract_palette(small, args.colors, seed))
            imgs = {"before": to_tile(a, k), "after": to_tile(cropped, k)}
            kk = sides.setdefault(("before", "after"), [0, 0])
            first = kk[0] <= kk[1]
            kk[0 if first else 1] += 1
            l, r = ("before", "after") if first else ("after", "before")
            items.append({"material": p, "label": p, "kind": "real",
                          "struct": round(1.0 / frac, 3), "stratum": "cropped",
                          "left": l, "right": r,
                          "limg": b64(imgs[l]), "rimg": b64(imgs[r])})
            print(f"  {p} #{k+1}  裁 1/{1/frac:.1f}", flush=True)

    for it in rng.sample(items, min(args.n_check, len(items))):
        good = np.asarray(Image.open(io.BytesIO(base64.b64decode(it["limg"]))))
        blur = np.stack([ndimage.gaussian_filter(good[..., c].astype(float), 3.0)
                         for c in range(3)], -1)
        gg = {"good": good, "blur": blur}
        kk = sides.setdefault(("good", "blur"), [0, 0])
        first = kk[0] <= kk[1]
        kk[0 if first else 1] += 1
        l, r = ("good", "blur") if first else ("blur", "good")
        items.append({"material": it["material"], "label": it["label"],
                      "kind": "check", "struct": 0.0, "stratum": "cropped",
                      "left": l, "right": r, "limg": b64(gg[l]), "rimg": b64(gg[r])})

    rng.shuffle(items)
    n_real = sum(1 for i in items if i["kind"] == "real")
    print(f"\n生成 {len(items)} 对：修前vs修后 {n_real}，"
          f"注意力检查 {len(items)-n_real}，未触发裁剪而跳过 {skipped}")
    for (x, y), kk in sorted(sides.items()):
        print(f"  左右平衡 {x} vs {y}: {kk[0]} / {kk[1]}")

    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)),
                        encoding="utf-8")
    print(f"写入 {args.out}  ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
