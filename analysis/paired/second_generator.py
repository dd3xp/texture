"""结构单元惯例是 SDXL 独有的，还是文生图模型的共性？

论文限制节里最致命的一条：「~25 单元/瓦片」只在**一个生成器**上量过，
所以「生成器画得比真人密得多」究竟是普遍现象还是 SDXL 的癖好，无从判断。
本机可联网，故下载 SD 1.5 权重、传到 GPU 机器上补这个测量。

**渲染分辨率必须对齐**。B14 已证明单元数随渲染分辨率变（1024/512/384 →
27.7/13.8/9.4），所以拿 SD1.5@512 去比 SDXL@1024 是把模型和分辨率混在一起。
SD 1.5 原生 512，故与 **SDXL@512（中位 13.8，`crop_render512.json`）** 对比。
⚠ 该口径下 SDXL 低于其原生分辨率，属于让 SDXL 吃亏的比法，报告时须写明。

--- 判据（跑之前写下并 commit）---
可比性检验：SD1.5 须在 ≥50% 的提示词上检出方向性周期，否则两组量的不是同一件事，
        主判据不评估。
**主判据（论文真正需要的）**：单元惯例**失配**是否也出现在 SD1.5 上——
        其单元数中位是否 ≥ 真人惯例的 2 倍（真人 3.2，故阈值 6.4）。
        是 -> 失配跨模型成立，论文的机制主张不再局限于 SDXL；
        否 -> SD1.5 本身就接近真人惯例，**机制是 SDXL 特有的**，
             §5 的适用范围必须收窄到 SDXL，这是对论文不利的结果，照实写。
次判据（描述性，不作结论）：SD1.5 的中位是否落在 SDXL@512 的 13.8 的 2 倍内（6.9–27.6）。
纯测量，无需判官。
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis"):
    sys.path.insert(0, str(ROOT / sub))
from downsample import dominant_period, anisotropy            # noqa: E402
from exact import binom_test                                   # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"

ARTIST_UNITS = 3.2          # 生产口径（论文 §5.2）
SDXL_512_UNITS = 13.8       # B14，crop_render512.json


def units_of(img: np.ndarray) -> float:
    per = dominant_period(img)
    return float(img.shape[0] / per) if per > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sd15",
                    help="SD1.5 权重目录（本机下载后拷到 GPU 机器；传绝对路径亦可）")
    ap.add_argument("--render", type=int, default=512)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/second_generator.json")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "analysis/paired"))
    import re
    src = (ROOT / "analysis/paired/crop_scale_study.py").read_text(encoding="utf-8")
    prompts = re.findall(r'"([^"]+)"',
                         re.search(r"PROMPTS\s*=\s*\[(.*?)\]", src, re.S).group(1))
    if a.limit:
        prompts = prompts[:a.limit]

    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        a.model, torch_dtype=torch.float16, variant="fp16",
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    recs = []
    for i, p in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(a.seed + i)
        im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                  num_inference_steps=a.steps, generator=g,
                  height=a.render, width=a.render).images[0]
        arr = np.asarray(im).astype(float)
        u, an = units_of(arr), float(anisotropy(arr))
        recs.append({"prompt": p, "units": u, "aniso": an,
                     "render": a.render, "model": "sd15"})
        print(f"[{i+1}/{len(prompts)}] {p:<32} 单元数 {u:6.1f}  aniso {an:.2f}",
              flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    got = [r["units"] for r in recs if r["units"] > 0]
    rate = len(got) / len(recs)
    print(f"\n检出方向性周期 {len(got)}/{len(recs)} = {rate:.0%}")
    if rate < 0.5:
        print("  -> **可比性检验不过**（<50%），主判据不评估")
        return
    med = statistics.median(got)
    print(f"  SD1.5@{a.render} 单元数中位 **{med:.1f}**"
          f"   （真人 {ARTIST_UNITS}，SDXL@512 {SDXL_512_UNITS}）")
    print(f"  主判据：≥ 真人的 2 倍（{ARTIST_UNITS*2:.1f}）？", end=" ")
    if med >= ARTIST_UNITS * 2:
        print(f"**是**（{med:.1f}）-> 失配跨模型成立，机制不局限于 SDXL")
    else:
        print(f"**否**（{med:.1f}）-> SD1.5 接近真人惯例，"
              "**机制是 SDXL 特有的**，§5 适用范围须收窄")
    lo, hi = SDXL_512_UNITS / 2, SDXL_512_UNITS * 2
    inside = lo <= med <= hi
    print(f"  次判据（描述性）：落在 SDXL@512 的 2 倍区间 [{lo:.1f}, {hi:.1f}] 内？"
          f" {'是' if inside else '否'}")
    print("\n警告：该比法下 SDXL 低于其原生 1024，属让 SDXL 吃亏的口径，报告时须写明。")


if __name__ == "__main__":
    main()
