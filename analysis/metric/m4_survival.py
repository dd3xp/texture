"""M4：什么样的高分辨率源经得起降采样？

M3 发现不同 high 源经受降采样的能力差异极大，单个源足以把结论带偏。
这里把那个方差拆开，并找出**降采样之前**就能测到的预测因子——
那正是生成端应当优化的量。

存活分 = 该源降采样+量化到 16x16 后，灰度材质分类器给出的真类概率。
预测因子全部在**高分辨率原图**上计算（统一缩到 128 以便可比）：

- lowfreq_ratio  低频能量占比。粗母题集中在低频，细纹理在高频
- contrast       亮度标准差
- coarse_edge    高斯模糊后的梯度幅值均值，即"粗尺度上还有多少边"
- dom_scale      自相关首峰滞后，母题的空间周期
- eff_colors     有效色数
"""

import argparse
import json
import sys
from collections import defaultdict
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

REF = 128  # 预测因子统一在这个尺寸上算


def predictors(tex: Image.Image) -> dict:
    a = np.asarray(tex.resize((REF, REF), Image.BOX), np.float64)
    g = a @ np.array([0.299, 0.587, 0.114])

    # 频谱：低频能量占比（半径 < 1/8 奈奎斯特）
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    yy, xx = np.mgrid[:REF, :REF] - REF // 2
    r = np.sqrt(yy ** 2 + xx ** 2)
    low = float(F[r < REF / 16].sum() / (F.sum() + 1e-9))

    # 粗尺度边：先模糊掉细节，再看还剩多少梯度
    b = ndimage.gaussian_filter(g, sigma=REF / 32)
    gy, gx = np.gradient(b)
    coarse = float(np.sqrt(gy ** 2 + gx ** 2).mean())

    # 主周期
    prof = g.mean(1) - g.mean()
    ac = np.correlate(prof, prof, "full")[REF - 1:]
    ac = ac / (ac[0] + 1e-12)
    dom = 0.0
    for lag in range(1, len(ac) - 1):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.1:
            dom = float(lag)
            break

    q = (a // 8).astype(int).reshape(-1, 3)
    return {"lowfreq_ratio": low, "contrast": float(g.std()),
            "coarse_edge": coarse, "dom_scale": dom,
            "eff_colors": float(len(np.unique(q, axis=0)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("experiments/metric/material_cnn_gray.pt"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("experiments/metric"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cls = ck["classes"]
    assert ck.get("gray"), "必须用灰度模型"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    rows = []
    for m, v in pairs.items():
        if m not in cls:
            continue
        ref = load_rgb(next(iter(v["low"].values())), args.size)
        if ref is None:
            continue
        pal = palette_from(ref, features(ref)["n_colors"])
        for pack, path in v["high"].items():
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            x = match_stats(x, ref)
            d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
            s = survival(pal[d.argmin(1)].reshape(x.shape), m)
            rows.append({"material": m, "pack": pack, "survival": s, **predictors(tex)})

    print(f"样本 {len(rows)}（材质 × high 源）\n")

    # 逐包存活率
    byp = defaultdict(list)
    for r in rows:
        byp[r["pack"]].append(r["survival"])
    print(f"{'high 材质包':<34}{'n':>6}{'存活分中位':>12}{'均值':>12}")
    print("-" * 66)
    for p_, v in sorted(byp.items(), key=lambda kv: -np.median(kv[1])):
        print(f"{p_[:33]:<34}{len(v):>6}{np.median(v):>12.5f}{np.mean(v):>12.5f}")

    # 预测因子与存活分的相关
    print(f"\n{'预测因子（高分辨率原图上测）':<28}{'Spearman ρ':>13}{'p':>12}")
    print("-" * 54)
    keys = ["lowfreq_ratio", "contrast", "coarse_edge", "dom_scale", "eff_colors"]
    surv = np.array([r["survival"] for r in rows])
    corr = {}
    for k in keys:
        x = np.array([r[k] for r in rows])
        rho, pv = stats.spearmanr(x, surv)
        corr[k] = {"rho": float(rho), "p": float(pv)}
        print(f"{k:<28}{rho:>+13.3f}{pv:>12.3g}")

    best = max(corr.items(), key=lambda kv: abs(kv[1]["rho"]))
    print(f"\n最强预测因子: {best[0]}  ρ={best[1]['rho']:+.3f}  p={best[1]['p']:.3g}")

    (args.out / "m4_survival.json").write_text(json.dumps(
        {"n": len(rows), "by_pack": {k: float(np.median(v)) for k, v in byp.items()},
         "correlations": corr}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
