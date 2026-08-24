"""M1：造判别测试集。

需要一批"人一眼能分好坏、而现有七个特征分不开"的成对样本，
用来当后续造尺子的靶子。

每一对 = 同一材质、同一尺寸、同一调色板、同一条降采样+量化尾巴，
只有高分辨率源不同：
    (a) 真人手绘的高分辨率材质包纹理
    (b) SDXL 按材质名生成的纹理

调色板从该材质**真人 16x16 版本**提取，两个源共用，
这样调色板不构成混淆变量。

但共用调色板有个陷阱：两个源的色彩分布不同时（亮度/对比度不匹配），
其中一个的像素会**全部落到同一个最近调色板条目**上，塌成纯色。
第一版就踩了这个坑——SDXL 的 wood/brick/dirt 全部塌成平板，
那是构造假象，不是 SDXL 的性质。
所以量化前先做**逐通道均值/标准差匹配**，把源对齐到真人参考的色彩统计上。

不做区域硬裁——M1 要测的是纹理本身读不读得出材质，不是区域保真。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from compare import load_rgb, features, auc          # noqa: E402
from structure import structure_features             # noqa: E402

# 材质名 -> SDXL prompt 里用的自然说法
MATERIALS = {
    "default_wood.png": "wood planks",
    "default_stone.png": "stone",
    "default_cobble.png": "cobblestone",
    "default_brick.png": "brick wall",
    "default_sand.png": "sand",
    "default_gravel.png": "gravel",
    "default_dirt.png": "dirt soil",
    "default_grass.png": "grass",
    "default_snow.png": "snow",
    "default_ice.png": "ice",
    "default_stone_brick.png": "stone brick wall",
    "default_mossycobble.png": "mossy cobblestone",
    "default_sandstone.png": "sandstone",
    "default_obsidian.png": "obsidian rock",
    "default_tree.png": "tree bark",
}
SIZES = [16, 32, 64]
PROMPT = "seamless {} texture, flat top down view, tileable material"
NEG = "3d render, perspective, object, shadow, vignette"

OLD_FEATS = ["n_colors", "flat_frac", "edge_sharp",
             "period_max", "comp_density", "comp_size_mean", "run_mean"]


def palette_from(img: np.ndarray, k: int) -> np.ndarray:
    """从真人 16x16 版本取自适应调色板，两个源共用。"""
    p = Image.fromarray(img.astype(np.uint8)).convert(
        "P", palette=Image.ADAPTIVE, colors=max(2, k))
    pal = np.array(p.getpalette()[: max(2, k) * 3], dtype=np.float64).reshape(-1, 3)
    return pal


def match_stats(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """逐通道把 a 的均值/标准差对齐到 ref。

    不做这一步的话，共用调色板会把色彩分布偏离参考的那个源整片压到
    单一颜色上，产生"纹理消失"的假象。
    """
    out = np.empty_like(a)
    for c in range(3):
        sa, sr = a[..., c].std(), ref[..., c].std()
        scale = (sr / sa) if sa > 1e-6 else 1.0
        out[..., c] = (a[..., c] - a[..., c].mean()) * scale + ref[..., c].mean()
    return np.clip(out, 0, 255)


def apply_tail(tex: Image.Image, n: int, pal: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """降采样 → 色彩统计对齐到参考 → 量化到给定调色板。"""
    a = np.asarray(tex.resize((n, n), Image.BOX), dtype=np.float64)
    a = match_stats(a, ref)
    d = ((a.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
    return pal[d.argmin(1)].reshape(a.shape)


def all_feats(a: np.ndarray) -> dict:
    return {**features(a), **structure_features(a)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--out", type=Path, default=Path("experiments/metric/testset"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (args.out / "img").mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text())
    mats = [m for m in MATERIALS if m in pairs]
    print(f"材质 {len(mats)} 种，尺寸 {SIZES} → 预期 {len(mats)*len(SIZES)} 对\n")

    # --- SDXL 生成高分辨率源 ---
    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    sdxl_src = {}
    for m in mats:
        f = args.out / f"sdxl_{m}"
        if not f.exists():
            g = torch.Generator("cuda").manual_seed(args.seed)
            pipe(prompt=PROMPT.format(MATERIALS[m]), negative_prompt=NEG,
                 num_inference_steps=args.steps, guidance_scale=7.0,
                 height=1024, width=1024, generator=g).images[0].save(f)
        sdxl_src[m] = f
        print(f"  SDXL 源 {m}")
    del pipe
    torch.cuda.empty_cache()

    # --- 走同一条尾巴，产出成对样本 ---
    rows = []
    for m in mats:
        # 该材质的真人 16x16 参考，用来定色数和调色板
        ref_path = next(iter(pairs[m]["low"].values()))
        ref = load_rgb(ref_path, 16)
        k = features(ref)["n_colors"]
        pal = palette_from(ref, k)

        # 真人高分辨率源：优先 CC0 的 drummyfish / hand_painted
        highs = pairs[m]["high"]
        pick = next((p for name, p in highs.items()
                     if "drummyfish" in name or "hand_painted" in name), None)
        art_src = pick or next(iter(highs.values()))

        for tag, src in (("artist", art_src), ("sdxl", sdxl_src[m])):
            tex = Image.open(src).convert("RGB")
            for n in SIZES:
                a = apply_tail(tex, n, pal, ref)
                Image.fromarray(a.astype(np.uint8)).save(
                    args.out / "img" / f"{m[:-4]}_{n}_{tag}.png")
                rows.append({"material": m, "size": n, "source": tag,
                             "palette_k": k, **all_feats(a)})

    # --- 验收：旧特征在这个测试集上应当分不开 ---
    art = [r for r in rows if r["source"] == "artist"]
    sdx = [r for r in rows if r["source"] == "sdxl"]
    print(f"\n样本 {len(rows)}（真人源 {len(art)} / SDXL 源 {len(sdx)}），"
          f"成对 {len(art)} 对\n")
    print(f"{'旧特征':<18}{'真人源中位':>12}{'SDXL源中位':>12}{'AUC':>9}{'|AUC-.5|':>10}")
    print("-" * 62)
    worst = 0.0
    for f in OLD_FEATS:
        pa = np.array([r[f] for r in art], float)
        ps = np.array([r[f] for r in sdx], float)
        u = auc(pa, ps)
        worst = max(worst, abs(u - 0.5))
        print(f"{f:<18}{np.median(pa):>12.3f}{np.median(ps):>12.3f}{u:>9.3f}{abs(u-0.5):>10.3f}")

    print(f"\n旧特征最大分离度 |AUC-0.5| = {worst:.3f}"
          f"  → AUC {0.5+worst:.3f}")
    ok = len(art) >= 30 and (0.5 + worst) < 0.65
    print(f"验收（≥30 对 且 旧特征 AUC<0.65）: {'通过' if ok else '未通过'}")

    with open(args.out / "testset.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"写入 {args.out/'testset.csv'}")


if __name__ == "__main__":
    main()
