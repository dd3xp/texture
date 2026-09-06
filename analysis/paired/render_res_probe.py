"""操作探针二：降低渲染分辨率（1024→512→384）能否让 SDXL 少画结构单元？

背景：预注册二的提示词杠杆全部失败——修饰词（B11，p=0.164/0.953）与
三种更强变体（period_probe.py：neg p=0.383、count p=0.189、corner p=1.0）
都掰不动 SDXL 的单元数。本脚本改用**渲染分辨率杠杆**。

操作检验判据（写于运行之前，B11 已写死，措辞与实现一起固定）：
  以**单元数 n_units = size / period** 为量，低分辨率组对 1024 组
  1. Mann-Whitney（置换法，20000 次，种子 0）p < 0.05；
  2. 中位单元数**下降 ≥20%**。
  两条都满足才算操作生效；生效后另行预注册胜率预测，本脚本不做 VLM。

注意：period 原始像素值在不同渲染尺寸间不可直接比（尺寸本身变了），
归一化到单元数后才是"SDXL 画了几个单元"这一惯例量。
period=0（未检出）的记录不进统计，但计数报告。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import dominant_period                              # noqa: E402
from crop_scale_study import PROMPTS, TMPL, NEG                     # noqa: E402

SIZES = (1024, 512, 384)


def mw_u(x, y):
    u = 0.0
    for a in x:
        for b in y:
            u += 1.0 if a > b else (0.5 if a == b else 0.0)
    return u


def mw_p(x, y, iters=20000, seed=0):
    """两侧 Mann-Whitney，置换法（无 scipy 环境）。"""
    obs = mw_u(x, y)
    mu = len(x) * len(y) / 2.0
    pool = list(x) + list(y)
    n = len(x)
    rng = random.Random(seed)
    cnt = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(mw_u(pool[:n], pool[n:]) - mu) >= abs(obs - mu) - 1e-9:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--out", type=Path, default=Path("experiments/render_res_probe.json"))
    args = ap.parse_args()

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    prompts = PROMPTS[::2]  # 与 period_probe 同一 21 个子集
    recs = []
    for pi, p in enumerate(prompts):
        row = {"prompt": p}
        for s in SIZES:
            g = torch.Generator("cuda").manual_seed(args.seed + PROMPTS.index(p))
            im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                      num_inference_steps=args.steps, generator=g,
                      height=s, width=s).images[0]
            row[str(s)] = float(dominant_period(np.asarray(im).astype(float)))
        recs.append(row)
        print(f"[{pi+1}/{len(prompts)}] {p:<32} "
              + " ".join(f"{s}:{row[str(s)]:.0f}" for s in SIZES), flush=True)
        args.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1))

    def med(v):
        v = sorted(v)
        return v[len(v) // 2]

    print("\n== 操作检验：单元数 n_units = size/period，对 1024 组比较 ==")
    hi = [1024 / r["1024"] for r in recs if r["1024"] > 0]
    print(f"1024: 检出 {len(hi)}/{len(recs)}，中位单元数 {med(hi):.1f}")
    for s in SIZES[1:]:
        lo = [s / r[str(s)] for r in recs if r[str(s)] > 0]
        if not lo or not hi:
            print(f"{s}: 检出不足，无法检验")
            continue
        p = mw_p(hi, lo)
        chg = (med(lo) - med(hi)) / med(hi)
        ok = p < 0.05 and chg <= -0.20
        print(f"{s:>4}: 检出 {len(lo)}/{len(recs)}，中位单元数 {med(lo):.1f}，"
              f"变化 {chg:+.1%}，MW p={p:.4g} → 操作{'生效' if ok else '未生效'}")


if __name__ == "__main__":
    main()
