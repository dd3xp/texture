"""M6（续）：直接操纵低频占比，检验共同前提。

m6_steering.py 想用 prompt 把低频占比推高，失败了（0.0388→0.0299，p=0.128）。
操纵失败就不能对前提下结论。

改为**直接操纵**：对真实高分辨率源做不同强度的低通滤波，
低通必然抬高低频能量占比，然后看降采样后的可读性怎么变。
这是对「低频占比 ↑ ⇒ 可读性 ↑」最直接的因果检验，
不依赖 prompt 是否听话，也不需要 GPU 生成。

若可读性随低通强度单调上升 → 前提成立，方案 A/B/C 的共同依据站得住。
若不变或下降 → 前提被证伪，三个方案一起放弃。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from learn_material import SmallCNN                   # noqa: E402
from compare import load_rgb, features                # noqa: E402
from build_testset import match_stats, palette_from   # noqa: E402
from m4_survival import predictors                    # noqa: E402

SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0]  # 相对 128 参考尺寸的高斯低通强度


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("experiments/metric/material_cnn_gray.pt"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--max-per-material", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("experiments/metric"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cls = ck["classes"]
    assert ck.get("gray")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = SmallCNN(len(cls)).to(dev).eval()
    net.load_state_dict(ck["state"])

    def survival(a: np.ndarray, mat: str) -> float:
        g = (a / 255.0) @ np.array([0.299, 0.587, 0.114], np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        t = torch.tensor(np.repeat(g[..., None], 3, -1).astype(np.float32))
        with torch.no_grad():
            p = net(t.permute(2, 0, 1)[None].to(dev)).softmax(-1)[0].cpu()
        return float(p[cls[mat]])

    pairs = json.loads(args.pairs.read_text())
    by_sigma = {s: [] for s in SIGMAS}
    lf_by_sigma = {s: [] for s in SIGMAS}
    n = 0
    for m, v in pairs.items():
        if m not in cls:
            continue
        ref = load_rgb(next(iter(v["low"].values())), args.size)
        if ref is None:
            continue
        pal = palette_from(ref, features(ref)["n_colors"])
        for path in list(v["high"].values())[: args.max_per_material]:
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            base = np.asarray(tex.resize((128, 128), Image.BOX), np.float64)
            n += 1
            for s in SIGMAS:
                b = base if s == 0 else np.stack(
                    [ndimage.gaussian_filter(base[..., c], s) for c in range(3)], -1)
                im = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8))
                lf_by_sigma[s].append(predictors(im)["lowfreq_ratio"])
                x = np.asarray(im.resize((args.size,) * 2, Image.BOX), float)
                x = match_stats(x, ref)
                d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
                by_sigma[s].append(survival(pal[d.argmin(1)].reshape(x.shape), m))

    print(f"样本 {n} 个高分辨率源 × {len(SIGMAS)} 档低通\n")
    print(f"{'低通 sigma':<12}{'低频占比中位':>14}{'可读性存活分中位':>18}{'均值':>12}")
    print("-" * 58)
    for s in SIGMAS:
        print(f"{s:<12.1f}{np.median(lf_by_sigma[s]):>14.4f}"
              f"{np.median(by_sigma[s]):>18.6f}{np.mean(by_sigma[s]):>12.6f}")

    a = np.array(by_sigma[0.0])
    print(f"\n{'对比 sigma=0':<14}{'胜出':>10}{'Wilcoxon p':>14}")
    print("-" * 40)
    for s in SIGMAS[1:]:
        b = np.array(by_sigma[s])
        w = stats.wilcoxon(a, b)
        print(f"sigma={s:<8.1f}{int((b > a).sum()):>6}/{len(a)}{w.pvalue:>14.3g}")

    # 低频占比 与 存活分 的整体相关（跨所有 sigma 档）
    allx = np.concatenate([lf_by_sigma[s] for s in SIGMAS])
    ally = np.concatenate([by_sigma[s] for s in SIGMAS])
    rho, pv = stats.spearmanr(allx, ally)
    print(f"\n跨全部低通档：低频占比 vs 存活分  Spearman ρ={rho:+.3f}  p={pv:.3g}")

    (args.out / "m6_lowpass.json").write_text(json.dumps({
        "n_sources": n, "sigmas": SIGMAS,
        "lowfreq_median": {str(s): float(np.median(lf_by_sigma[s])) for s in SIGMAS},
        "survival_median": {str(s): float(np.median(by_sigma[s])) for s in SIGMAS},
        "spearman_rho": float(rho), "spearman_p": float(pv),
    }, indent=1))


if __name__ == "__main__":
    main()
