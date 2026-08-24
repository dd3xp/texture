"""M6：方案 A 的最小可行实验——生成端引导，只改 prompt，不训练。

检验 method.md 里三个方案的**共同前提**：
    把高分辨率源的低频能量推高，能改善降采样后的材质可读性吗？

两个条件，同材质同种子，只差 prompt：
    baseline —— 现用的中性 prompt
    steered  —— 加入"粗大形状/高对比/少量平涂色/减少细纹理"的引导

测两件事，缺一不可：
    1. 引导**有没有真的推高低频占比**（若没有，实验无效，不是前提被证伪）
    2. 推高之后，**可读性有没有改善**（这才是前提本身）

只有 1 成立而 2 不成立，才算证伪了共同前提。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from learn_material import SmallCNN                   # noqa: E402
from compare import load_rgb, features                # noqa: E402
from build_testset import match_stats, palette_from   # noqa: E402
from materials import load as load_materials          # noqa: E402
from m4_survival import predictors                    # noqa: E402

BASE_P = "seamless {} texture, flat top down view, tileable material"
STEER_P = ("seamless {} texture, flat top down view, tileable material, "
           "bold simple shapes, large forms, high contrast, "
           "few flat colors, graphic poster style")
BASE_N = "3d render, perspective, object, shadow, vignette"
STEER_N = (BASE_N + ", fine grain, subtle detail, photographic detail, "
           "noise, soft gradient, busy texture")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("experiments/metric/material_cnn_gray.pt"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--n-materials", type=int, default=24)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("experiments/metric/steering"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cls = ck["classes"]
    assert ck.get("gray")
    dev = "cuda"
    net = SmallCNN(len(cls)).to(dev).eval()
    net.load_state_dict(ck["state"])

    def survival(a: np.ndarray, mat: str) -> float:
        x = a / 255.0
        g = x @ np.array([0.299, 0.587, 0.114], np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        t = torch.tensor(np.repeat(g[..., None], 3, -1).astype(np.float32))
        with torch.no_grad():
            p = net(t.permute(2, 0, 1)[None].to(dev)).softmax(-1)[0].cpu()
        return float(p[cls[mat]])

    pairs = json.loads(args.pairs.read_text())
    mats = [m for m in load_materials() if m in pairs and m in cls][: args.n_materials]
    print(f"材质 {len(mats)} 种 × 2 条件\n")

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to(dev)
    pipe.set_progress_bar_config(disable=True)

    rows = []
    for m in mats:
        name = load_materials()[m]
        ref = load_rgb(next(iter(pairs[m]["low"].values())), args.size)
        if ref is None:
            continue
        pal = palette_from(ref, features(ref)["n_colors"])
        rec = {"material": m}
        for cond, (pp, nn) in (("base", (BASE_P, BASE_N)),
                               ("steer", (STEER_P, STEER_N))):
            f = args.out / f"{m[:-4]}_{cond}.png"
            if not f.exists():
                g = torch.Generator(dev).manual_seed(args.seed)
                pipe(prompt=pp.format(name), negative_prompt=nn,
                     num_inference_steps=args.steps, guidance_scale=7.0,
                     height=1024, width=1024, generator=g).images[0].save(f)
            tex = Image.open(f).convert("RGB")
            pr = predictors(tex)
            x = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            x = match_stats(x, ref)
            d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
            rec[f"{cond}_lowfreq"] = pr["lowfreq_ratio"]
            rec[f"{cond}_contrast"] = pr["contrast"]
            rec[f"{cond}_surv"] = survival(pal[d.argmin(1)].reshape(x.shape), m)
        rows.append(rec)
        print(f"  {m[:-4][:26]:<26} lowfreq {rec['base_lowfreq']:.4f}→{rec['steer_lowfreq']:.4f}"
              f"   surv {rec['base_surv']:.5f}→{rec['steer_surv']:.5f}")

    del pipe
    torch.cuda.empty_cache()

    def cmp(key, label):
        a = np.array([r[f"base_{key}"] for r in rows])
        b = np.array([r[f"steer_{key}"] for r in rows])
        w = stats.wilcoxon(a, b)
        print(f"{label:<16}{np.median(a):>12.5f}{np.median(b):>12.5f}"
              f"{int((b > a).sum()):>8}/{len(rows)}{w.pvalue:>12.3g}")
        return w.pvalue, int((b > a).sum())

    print(f"\n{'量':<16}{'baseline':>12}{'steered':>12}{'steer胜':>10}{'p':>12}")
    print("-" * 64)
    p_lf, w_lf = cmp("lowfreq", "低频能量占比")
    cmp("contrast", "对比度")
    p_sv, w_sv = cmp("surv", "可读性存活分")

    print("\n判读：")
    if p_lf >= 0.05:
        print("  引导**没能**显著推高低频占比 → 实验无效，不能据此否定前提")
    elif p_sv < 0.05 and w_sv > len(rows) / 2:
        print("  低频推高了，可读性也显著改善 → **共同前提成立**，方案 A 有效")
    elif p_sv < 0.05:
        print("  低频推高了，可读性却显著变差 → 前提方向相反")
    else:
        print("  低频推高了，但可读性无显著改善 → **共同前提被证伪**，"
              "方案 B/C 的依据同样不成立")

    (args.out / "m6_steering.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
