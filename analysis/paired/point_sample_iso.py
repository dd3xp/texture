"""门拒绝的材质：**换降采样器**——每格取一个真实源像素，而不是求平均。

这是各向同性那半边的第四条路，前三条已关：
  ① 按特征尺度裁剪（`isotropic_scale.py`）——没有可对齐的结构单元；
  ② 拉亮度跨度（`spread_rescale.py`）——只治症状，材质没更可辨认；
  ③ 降渲染分辨率（`render_small_iso.py`）——操作有效但判官显著反向。

**为什么还值得试第四条**。B20 测出真人在这些材质上画的是**逐格颜色**，
而 `tools/downsample.py` 里五种降采样器（box/median/extremum/contrast/bimodal）
**全都把格内像素归约成一个统计量**，一律摧毁格内方差——这正是跨度从真人的 0.292
掉到 0.128 的机制。点采样每格取一个**真实存在的源颜色**，保住颜色分布。
论文 §5.1 说「五种降采样器全败」，那测的是**有周期**的渲染图（砖墙）；
对各向同性内容换降采样器从没测过，是空白格不是已排除项。

**风险**：点采样会把平均化伪影换成混叠噪声，可能只是「另一种难看」。判据来裁。

--- 判据（跑之前写下并 commit）---
操作检验：point 版的**图内实际亮度跨度**中位需 ≥0.20（真人 0.292，box 现为 0.128），
        且对 box 的逐材质配对符号检验 p<0.05。不过则操作无效，主判据不评估。
主判据：经验证判官（claude-opus-5，正反两问去偏、不一致弃用）偏好 point 版
        >50% 且二项 p<0.05 -> 各向同性那半该换降采样器。
证伪：≤50% 或不显著 -> 点采样只是把平均伪影换成混叠噪声，第四条路关闭。
方向性限定：判官压缩效应 —— 为正作下界；为负只能说「没测到大效应」。
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
from make_texture import extract_palette, quantize, W          # noqa: E402
from downsample import dominant_period, anisotropy            # noqa: E402
from exact import binom_test, jeffreys                        # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"
Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")


def realised_spread(img: np.ndarray) -> float:
    u = np.unique(img.reshape(-1, 3), axis=0).astype(float)
    lum = np.sort(u @ W)
    return float(lum[-1] - lum[0]) / 255.0


def box_tile(src, size):
    return np.asarray(Image.fromarray(src.astype(np.uint8))
                      .resize((size,) * 2, Image.BOX))


def point_tile(src, size):
    """每格取**格中心那一个真实源像素**，不做任何平均。

    与五种现有降采样器的根本区别：它们输出的是格内统计量（可能是源图里
    从不存在的颜色），点采样输出的一定是源图真有的颜色，因此保住颜色分布。
    取中心而非随机，是为了确定性可复现。
    """
    H, Wd = src.shape[:2]
    ys = ((np.arange(size) + 0.5) * H / size).astype(int).clip(0, H - 1)
    xs = ((np.arange(size) + 0.5) * Wd / size).astype(int).clip(0, Wd - 1)
    return src[np.ix_(ys, xs)].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/point_sample_iso.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/iso_point")
    a = ap.parse_args()

    # 复用两批交付的材质与种子，这样「门是否触发」与已入库的 manifest 一致
    jobs = []
    sys.path.insert(0, str(ROOT / "scripts"))
    from batch_pack import MATERIALS as PACK18
    for i, m in enumerate(PACK18):
        jobs.append((m, 21 + i))
    for i, m in enumerate(json.loads(
            (ROOT / "experiments/prompts_holdout60.json").read_text(encoding="utf-8"))):
        jobs.append((m, 77 + i))

    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    recs = []
    for k, (m, seed) in enumerate(jobs):
        g = torch.Generator("cuda").manual_seed(seed)
        im = pipe(TMPL.format(p=m), negative_prompt=NEG,
                  num_inference_steps=a.steps, generator=g,
                  height=a.render, width=a.render).images[0]
        src = np.asarray(im).astype(float)
        per, ani = dominant_period(src), anisotropy(src)
        if per > 0 and ani >= 0.20:
            print(f"[{k+1}/{len(jobs)}] {m:<24} 门触发，跳过（本实验只管被拒的那半）",
                  flush=True)
            continue
        b = quantize(box_tile(src, a.size),
                     extract_palette(box_tile(src, a.size), a.colors, seed=0)).astype(np.uint8)
        pt = point_tile(src, a.size)
        pt = quantize(pt, extract_palette(pt, a.colors, seed=0)).astype(np.uint8)
        d = a.tiles / str(a.size)
        d.mkdir(parents=True, exist_ok=True)
        slug = m.replace(" ", "_")
        Image.fromarray(b).save(d / f"{slug}_box.png")
        Image.fromarray(pt).save(d / f"{slug}_point.png")
        recs.append({"material": m, "seed": seed, "size": a.size,
                     "period": float(per), "aniso": float(ani),
                     "spread_box": realised_spread(b),
                     "spread_point": realised_spread(pt)})
        print(f"[{k+1}/{len(jobs)}] {m:<24} 跨度 box {recs[-1]['spread_box']:.3f}"
              f" -> point {recs[-1]['spread_point']:.3f}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    if not recs:
        print("没有门拒绝的材质"); return

    pts = [r["spread_point"] for r in recs]
    bxs = [r["spread_box"] for r in recs]
    up = sum(p > b for p, b in zip(pts, bxs))
    nz = sum(p != b for p, b in zip(pts, bxs))
    ps = binom_test(up, nz) if nz else 1.0
    med = float(np.median(pts))
    print(f"\n门拒绝的材质 {len(recs)} 个")
    print(f"操作检验：point 跨度中位 {med:.3f}（box {np.median(bxs):.3f}，真人 0.292），"
          f"变大 {up}/{nz}，符号检验 p={ps:.3g}")
    if not (med >= 0.20 and ps < 0.05):
        print("  -> **操作无效**（判据要求中位 ≥0.20 且 p<0.05），主判据不予评估")
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

    win = tot = inc = 0
    for r in recs:
        d = a.tiles / str(a.size); slug = r["material"].replace(" ", "_")
        B = b64(np.asarray(Image.open(d / f"{slug}_box.png").convert("RGB")))
        P = b64(np.asarray(Image.open(d / f"{slug}_point.png").convert("RGB")))
        q = Q.format(n=a.size, label=r["material"])
        o1, o2 = ask(a.model, q, [B, P], base_url, key), ask(a.model, q, [P, B], base_url, key)
        if not o1 or not o2:
            continue
        p1 = "box" if o1.upper().startswith("A") else "point"
        p2 = "point" if o2.upper().startswith("A") else "box"
        if p1 != p2:
            inc += 1; r["vlm"] = "inconsistent"
        else:
            tot += 1; win += p1 == "point"; r["vlm"] = p1
        print(f"  {r['material']:<24} -> {r['vlm']}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    if not tot:
        print("没有有效判断"); return
    pv = binom_test(win, tot); lo, hi = jeffreys(win, tot)
    print(f"\npoint 版被选 {win}/{tot} = {win/tot:.0%}  p={pv:.3g}  [{lo:.0%},{hi:.0%}]"
          f"   （正反不一致弃 {inc}）")
    if win / tot > 0.5 and pv < 0.05:
        print("判读：主判据成立 -> 各向同性那半该换降采样器（判官压缩，故为**下界**）")
    else:
        print("判读：主判据不成立 -> **只能说没测到大效应**；点采样把平均伪影")
        print("      换成了混叠噪声而未获益，第四条路关闭。")


if __name__ == "__main__":
    main()
