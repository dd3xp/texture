"""M3：用 M2 的灰度材质分类器重测原立论。

立论：低分辨率纹理是艺术家的重新设计，不是高分辨率纹理的降采样。

操作化：如果立论成立，那么
    真人在 16px 直接画的版本，其结构性材质身份 应当显著高于
    把高分辨率版降采样到 16px 的版本。

A 组 = 留出材质包（分类器没见过的作者）的原生 16x16
B 组 = high 组材质包的高分辨率纹理，降采样+量化到 16x16

两组的作者都没被分类器见过，所以"见过/没见过"不构成组间差异。
按材质配对做 Wilcoxon 符号秩检验，并统计逐材质胜负，
不只看一组中位数——上一轮就是只看中位数，容易被少数材质带偏。

**已知局限**（写在结论里）：A 组来自 low 包、B 组来自 high 包，
两个包群本身是不同的作者群体，所以差异里混有"包群风格"的成分，
无法与"绘制方式"完全分离。
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
from learn_material import SmallCNN, HELD_OUT     # noqa: E402
from compare import load_rgb, features            # noqa: E402
from build_testset import match_stats, palette_from  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("experiments/metric/material_cnn_gray.pt"))
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("experiments/metric"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cls, gray = ck["classes"], bool(ck.get("gray", False))
    assert gray, "M3 必须用灰度模型——彩色模型的读数被调色板签名污染"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = SmallCNN(len(cls)).to(dev).eval()
    net.load_state_dict(ck["state"])

    def prep(a: np.ndarray) -> torch.Tensor:
        g = a @ np.array([0.299, 0.587, 0.114], np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        x = np.repeat(g[..., None], 3, -1).astype(np.float32)
        return torch.tensor(x).permute(2, 0, 1)[None].to(dev)

    def scores(a: np.ndarray, mat: str) -> tuple[float, int]:
        """返回 (真类概率, top-1 是否正确)。两个指标一起看——
        它们在 M3 里给出过不同方向的结论，必须在同一份配对数据上对照。"""
        with torch.no_grad():
            p = net(prep(a / 255.0 if a.max() > 1.5 else a)).softmax(-1)[0].cpu()
        return float(p[cls[mat]]), int(int(p.argmax()) == cls[mat])

    pairs = json.loads(args.pairs.read_text())
    rows = []
    for m, v in pairs.items():
        if m not in cls:
            continue
        # A 组：留出作者的原生 16x16
        A = []
        for pack, path in v["low"].items():
            if pack not in HELD_OUT:
                continue
            a = load_rgb(path, args.size)
            if a is not None and Image.open(path).size == (args.size, args.size):
                A.append(scores(a, m))
        if not A:
            continue
        # B 组：high 包降采样+量化。调色板取自该材质任一 low 版本
        ref = load_rgb(next(iter(v["low"].values())), args.size)
        if ref is None:
            continue
        k = features(ref)["n_colors"]
        pal = palette_from(ref, k)
        B = []
        for path in v["high"].values():
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            x = match_stats(x, ref)
            d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
            B.append(scores(pal[d.argmin(1)].reshape(x.shape), m))
        if not B:
            continue
        rows.append({"material": m,
                     "a_med": float(np.median([x[0] for x in A])),
                     "b_med": float(np.median([x[0] for x in B])),
                     "a_top1": float(np.mean([x[1] for x in A])),
                     "b_top1": float(np.mean([x[1] for x in B])),
                     "n_a": len(A), "n_b": len(B)})

    a = np.array([r["a_med"] for r in rows])
    b = np.array([r["b_med"] for r in rows])
    print(f"配对材质 {len(rows)} 种   A 组样本 {sum(r['n_a'] for r in rows)}"
          f"   B 组样本 {sum(r['n_b'] for r in rows)}\n")

    print(f"{'组':<28}{'中位数':>12}{'均值':>12}")
    print("-" * 52)
    print(f"{'A 真人原生 16px（留出作者）':<28}{np.median(a):>12.5f}{a.mean():>12.5f}")
    print(f"{'B 高分辨率降采样到 16px':<28}{np.median(b):>12.5f}{b.mean():>12.5f}")

    wins = int((a > b).sum())
    w = stats.wilcoxon(a, b)
    # 效应量：配对差值的 Cliff's delta 近似（胜率转换）
    delta = 2 * wins / len(rows) - 1
    print(f"\nA 胜出材质数: {wins}/{len(rows)} = {wins/len(rows):.1%}")
    print(f"Wilcoxon 符号秩检验: W={w.statistic:.0f}, p={w.pvalue:.3g}")
    print(f"效应量 Cliff's delta ≈ {delta:+.3f}")
    print(f"中位数比值 A/B = {np.median(a)/max(np.median(b),1e-12):.2f}x")

    # top-1 指标上的同一份配对检验
    at = np.array([r["a_top1"] for r in rows])
    bt = np.array([r["b_top1"] for r in rows])
    wt = int((at > bt).sum()); lt = int((at < bt).sum())
    print("")
    print("--- 换 top-1 指标，同一份配对数据 ---")
    print(f"A 整体 top-1 = {at.mean():.1%}   B 整体 top-1 = {bt.mean():.1%}")
    nz = at != bt
    if nz.sum():
        w2 = stats.wilcoxon(at[nz], bt[nz])
        print(f"逐材质 A 胜 {wt} / B 胜 {lt} / 平 {len(rows)-wt-lt}"
              f"   Wilcoxon p={w2.pvalue:.3g}")

    verdict = ("立论成立（A 显著高于 B）" if w.pvalue < 0.05 and wins / len(rows) > 0.5
               else "立论证伪（B 不低于 A）" if w.pvalue < 0.05
               else "仍无法判定（差异不显著）")
    print(f"\n判定: {verdict}")

    (args.out / "m3_premise.json").write_text(json.dumps({
        "n_materials": len(rows), "a_median": float(np.median(a)),
        "b_median": float(np.median(b)), "a_wins": wins,
        "wilcoxon_p": float(w.pvalue), "cliffs_delta": delta,
        "verdict": verdict, "per_material": rows,
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
