"""操作探针：哪种提示词操作真能让 SDXL 少画结构单元（period 变大）？

背景：预注册二的修饰词（very few large blocks, ...）未通过操作检验
（period 中位 35→37px，16/42 变大，p=0.164）——见 crop_fewunits_eval.py。
本脚本**只测 period、不做 VLM 判断**，属于操作搜索，不是假设检验；
找到显著改变 period 的操作后，再对该操作单独预注册胜率预测。

变体（与对照唯一差别）：
  neg   —— 负面提示词追加 "many small units, dense repeating pattern, fine details"
  count —— 正向追加 ", only four large units visible, each unit fills a quarter of the frame"
  corner—— 正向追加 ", extreme macro close-up of one small corner, structure larger than the frame"

省时：取全部 42 提示词的偶数位（21 个，覆盖三类材质），种子与 crop_ctrl 相同。
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import dominant_period                              # noqa: E402
from crop_scale_study import PROMPTS, TMPL, NEG                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--out", type=Path, default=Path("experiments/period_probe.json"))
    args = ap.parse_args()

    variants = {
        "base": ("", ""),
        "neg": ("", ", many small units, dense repeating pattern, fine details"),
        "count": (", only four large units visible, each unit fills a quarter of the frame", ""),
        "corner": (", extreme macro close-up of one small corner, structure larger than the frame", ""),
    }

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    prompts = PROMPTS[::2]  # 21 个，三类材质都覆盖
    recs = []
    for pi, p in enumerate(prompts):
        row = {"prompt": p}
        for tag, (pos, neg) in variants.items():
            g = torch.Generator("cuda").manual_seed(args.seed + PROMPTS.index(p))
            im = pipe(TMPL.format(p=p) + pos, negative_prompt=NEG + neg,
                      num_inference_steps=args.steps, generator=g,
                      height=1024, width=1024).images[0]
            row[tag] = float(dominant_period(np.asarray(im).astype(float)))
        recs.append(row)
        print(f"[{pi+1}/{len(prompts)}] {p:<32} "
              + " ".join(f"{t}:{row[t]:.0f}" for t in variants), flush=True)
        args.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1))

    import math

    def binom_p(w, n):
        return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, n - w) + 1)) / 2 ** n)

    print("\n== 对 base 的配对比较（period 变大计一胜）==")
    base = [r["base"] for r in recs]
    # 用真中位数：sorted(v)[n//2] 在 n 为偶数时取的是上中位数。
    # 本脚本 n=21（奇数）时两者相同，故此前的数没错；改掉是防止样本量一变就悄悄偏。
    med = statistics.median(base)
    print(f"base 中位 {med:.0f}px")
    for tag in variants:
        if tag == "base":
            continue
        up = sum(1 for r in recs if r[tag] > r["base"])
        medt = statistics.median(r[tag] for r in recs)
        print(f"{tag:>6}: 中位 {medt:.0f}px，变大 {up}/{len(recs)}，符号检验 p={binom_p(up, len(recs)):.3g}")


if __name__ == "__main__":
    main()
