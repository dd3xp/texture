"""门拒绝的材质：改在**生成端**降分辨率，能不能救回来？

背景。交付批次里 18 个材质有 7 个门不触发，输出接近平涂
（亮度跨度中位 0.071 vs 触发组 0.262，真人 0.292）。
两条后处理的路已经试过：
  - 按特征尺度裁剪 —— **关闭**（`isotropic_scale.py`）：这些材质 84% 的径向
    相关长度就是 1 格，没有可对齐的结构单元，裁剪没有对齐目标。
  - 拉亮度跨度 —— 只治标（`spread_rescale.py`）：明暗层次回来了，
    材质并没有更可辨认。

所以该动生成端。机制本身就给了杠杆：论文 §5.5 量到渲染分辨率决定每图单元数
（1024→27.7、512→13.8、384→9.4）。对**有周期**的材质，裁剪和降渲染分辨率
是同一件事的两种做法（B13/B14 已验）。对**无周期**的材质裁剪做不了，
但降渲染分辨率照样能让每个颗粒在输出格上占更多地方——**这一格从没测过**。
B13/B14 测的是「已触发材质在 384 下裁剪收益消失」，不是「未触发材质在 384 下更好」。

--- 判据（跑之前写下并 commit）---
操作检验：384 渲染得到的 16px 瓦片，其**图内实际亮度跨度**中位应比 1024 的高
        ≥0.05，且逐材质配对符号检验 p<0.05。不过则操作无效，主判据不评估。
主判据：经验证判官（claude-opus-5，正反两问去偏、不一致弃用）
        偏好 384 版 >50% 且二项 p<0.05 -> 生成端分辨率是这半边的杠杆。
证伪：≤50% 或不显著 -> 降渲染分辨率对各向同性材质也没用；
        生成端不是杠杆，如实报告并关闭这条路。
方向性限定：判官压缩效应 —— 为正可作下界；为负只能说「没测到大效应」。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/annotate", "analysis/paired"):
    sys.path.insert(0, str(ROOT / sub))
from make_texture import extract_palette, quantize, W          # noqa: E402
from exact import binom_test, jeffreys                          # noqa: E402
from downsample import dominant_period, anisotropy              # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"
Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")

# 各向同性/颗粒材质。前 7 个就是交付批次里门未触发的那批，其余为同类扩样，
# 好让 n 不至于太小。清单写死以便复核。
MATERIALS = [
    "grass turf top", "dry sand", "oak log bark", "iron ore in stone",
    "sandstone block", "smooth stone", "coarse gravel",
    "packed snow", "dry dirt ground", "moss patch", "coal ore vein",
    "red desert sand", "crushed limestone", "pine needle litter",
    "rough concrete", "wet mud",
]


def realized_spread(img: np.ndarray) -> float:
    u = np.unique(img.reshape(-1, 3), axis=0).astype(float)
    lum = np.sort(u @ W)
    return float(lum[-1] - lum[0]) / 255.0


def to_tile(src: np.ndarray, size: int, colors: int, seed: int) -> np.ndarray:
    small = np.asarray(Image.fromarray(src.astype(np.uint8))
                       .resize((size,) * 2, Image.BOX))
    return quantize(small, extract_palette(small, colors, seed=seed)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--big", type=int, default=1024)
    ap.add_argument("--small", type=int, default=384)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/render_small_iso.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/iso_small")
    a = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    mats = MATERIALS[:a.limit] if a.limit else MATERIALS
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    recs = []
    for mi, m in enumerate(mats):
        srcs = {}
        for tag, res in (("big", a.big), ("small", a.small)):
            g = torch.Generator("cuda").manual_seed(a.seed + mi)
            im = pipe(TMPL.format(p=m), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=res, width=res).images[0]
            srcs[tag] = np.asarray(im).astype(float)
        # 只留门确实不触发的材质：这条实验是给那一半的
        per, ani = dominant_period(srcs["big"]), anisotropy(srcs["big"])
        fired = per > 0 and ani >= 0.20
        for size in a.sizes:
            tb = to_tile(srcs["big"], size, a.colors, mi)
            ts = to_tile(srcs["small"], size, a.colors, mi)
            d = a.tiles / str(size)
            d.mkdir(parents=True, exist_ok=True)
            slug = m.replace(" ", "_")
            Image.fromarray(tb).save(d / f"{slug}_big.png")
            Image.fromarray(ts).save(d / f"{slug}_small.png")
            recs.append({"material": m, "size": size, "gate_fired": bool(fired),
                         "period": float(per), "aniso": float(ani),
                         "spread_big": realized_spread(tb),
                         "spread_small": realized_spread(ts)})
        print(f"[{mi+1}/{len(mats)}] {m:<22} period={per:6.1f} aniso={ani:.2f}"
              f" {'门触发(不计入)' if fired else ''}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    use = [r for r in recs if not r["gate_fired"]]
    print(f"\n门不触发的记录 {len(use)}/{len(recs)} 条 —— 只用这些评判")
    if not use:
        print("没有门不触发的材质，实验不适用"); return

    # —— 操作检验 ——
    diffs = [r["spread_small"] - r["spread_big"] for r in use]
    med = float(np.median(diffs))
    up = sum(d > 0 for d in diffs); nz = sum(d != 0 for d in diffs)
    p_sign = binom_test(up, nz) if nz else 1.0
    print(f"操作检验：图内实际跨度中位差 {med:+.3f}"
          f"（1024 {np.median([r['spread_big'] for r in use]):.3f}"
          f" -> 384 {np.median([r['spread_small'] for r in use]):.3f}），"
          f"变大 {up}/{nz}，符号检验 p={p_sign:.3g}")
    if not (med >= 0.05 and p_sign < 0.05):
        print("  -> **操作无效**（判据要求中位差 ≥0.05 且 p<0.05），主判据不予评估")
        return
    print("  -> 操作有效，继续主判据")

    if a.no_judge:
        return

    from vlm_judge import ask
    import base64, io

    def b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    win = tot = incons = 0
    for r in use:
        d = a.tiles / str(r["size"]); slug = r["material"].replace(" ", "_")
        B = b64(np.asarray(Image.open(d / f"{slug}_big.png").convert("RGB")))
        S = b64(np.asarray(Image.open(d / f"{slug}_small.png").convert("RGB")))
        q = Q.format(n=r["size"], label=r["material"])
        o1, o2 = ask(a.model, q, [B, S], base_url, key), ask(a.model, q, [S, B], base_url, key)
        if not o1 or not o2:
            continue
        p1 = "big" if o1.upper().startswith("A") else "small"
        p2 = "small" if o2.upper().startswith("A") else "big"
        if p1 != p2:
            incons += 1; r["vlm"] = "inconsistent"
        else:
            tot += 1; win += p1 == "small"; r["vlm"] = p1
        print(f"  {r['material']:<22}{r['size']:>3}px -> {r['vlm']}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    if not tot:
        print("没有有效判断"); return
    pv = binom_test(win, tot); lo, hi = jeffreys(win, tot)
    print(f"\n384 版被选 {win}/{tot} = {win/tot:.0%}  p={pv:.3g}  [{lo:.0%},{hi:.0%}]"
          f"   （正反不一致弃 {incons}）")
    if win / tot > 0.5 and pv < 0.05:
        print("判读：主判据成立 -> 生成端分辨率是这半边的杠杆（判官压缩，故为**下界**）")
    else:
        print("判读：主判据不成立 -> **只能说没测到大效应**；判官压缩，")
        print("      不能据此声称降渲染分辨率无效，但这条路暂不继续。")


if __name__ == "__main__":
    main()
