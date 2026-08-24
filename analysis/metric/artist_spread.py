"""M3（改造版）：用作者间差异当尺子，检验原立论。不需要质量真值。

M3 原规格是"用 M2 找到的指标重测立论"，但 M2 阻塞（无质量真值）。
改用一个**不需要真值**的设计：

每种材质有 14–17 位不同作者画的原生 16x16 版本。作者之间本就有差异，
这个差异的幅度就是天然的尺子——

    d_内     = 同材质、两位不同作者的原生版本之间的距离
    d_降采样 = 某作者的原生版本 与 高分辨率降采样结果 之间的距离

若 d_降采样 落在 d_内 的分布之内 → 降采样结果不过是"另一位作者的画法"，
                                   立论（低分辨率是重新设计）证伪。
若 d_降采样 明显大于 d_内         → 降采样落在真人作画空间之外，立论成立。

距离定义在七个特征构成的标准化空间里。这些特征分辨力有限（AUC 0.69–0.75），
但两个距离用的是**同一把尺子**，比较是相对的，所以结论仍然有效。
另加一个跨材质对照 d_跨：不同材质之间的距离，作为"明显不同"的参照刻度。
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import load_rgb, features                 # noqa: E402
from structure import structure_features               # noqa: E402
from build_testset import match_stats, palette_from    # noqa: E402
from materials import load as load_materials           # noqa: E402

FEATS = ["n_colors", "flat_frac", "edge_sharp",
         "period_max", "comp_density", "comp_size_mean", "run_mean"]


def descriptor(a: np.ndarray) -> np.ndarray:
    f = {**features(a), **structure_features(a)}
    return np.array([f[k] for k in FEATS], float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("experiments/metric"))
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text())
    mats = [m for m in load_materials() if m in pairs and len(pairs[m]["low"]) >= 4]
    print(f"材质 {len(mats)} 种（每种至少 4 位作者的原生 {args.size}x{args.size}）\n")

    natives, downs = {}, {}
    for m in mats:
        vs = []
        for path in pairs[m]["low"].values():
            a = load_rgb(path, args.size)
            if a is not None and Image.open(path).size == (args.size, args.size):
                vs.append(a)
        if len(vs) < 4:
            continue
        natives[m] = vs

        # 降采样组：高分辨率源 → 色彩对齐到该材质首个真人版 → 量化到同色数调色板
        ref = vs[0]
        k = features(ref)["n_colors"]
        pal = palette_from(ref, k)
        ds = []
        for path in pairs[m]["high"].values():
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            a = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            a = match_stats(a, ref)
            d = ((a.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
            ds.append(pal[d.argmin(1)].reshape(a.shape))
        if ds:
            downs[m] = ds

    mats = [m for m in natives if m in downs]
    print(f"两组齐全的材质: {len(mats)}")

    # 标准化用的全局尺度：所有原生样本的特征标准差
    allv = np.array([descriptor(a) for m in mats for a in natives[m]])
    sd = allv.std(0)
    sd[sd < 1e-9] = 1.0

    def dist(x, y):
        return float(np.linalg.norm((descriptor(x) - descriptor(y)) / sd))

    d_in, d_down, d_cross = [], [], []
    for m in mats:
        for a, b in combinations(natives[m][:6], 2):
            d_in.append(dist(a, b))
        for a in natives[m][:6]:
            for b in downs[m][:6]:
                d_down.append(dist(a, b))
    for m1, m2 in combinations(mats, 2):
        d_cross.append(dist(natives[m1][0], natives[m2][0]))

    d_in, d_down, d_cross = map(np.array, (d_in, d_down, d_cross))
    print(f"\n{'距离':<22}{'n':>7}{'中位数':>10}{'均值':>10}{'p25':>9}{'p75':>9}")
    print("-" * 68)
    for name, d in (("d_内 (作者 vs 作者)", d_in),
                    ("d_降采样 (作者 vs 降)", d_down),
                    ("d_跨 (不同材质)", d_cross)):
        print(f"{name:<22}{len(d):>7}{np.median(d):>10.3f}{d.mean():>10.3f}"
              f"{np.percentile(d,25):>9.3f}{np.percentile(d,75):>9.3f}")

    r = np.median(d_down) / np.median(d_in)
    r_cross = np.median(d_cross) / np.median(d_in)
    print(f"\n比值 d_降采样 / d_内 = {r:.3f}")
    print(f"参照 d_跨材质 / d_内 = {r_cross:.3f}   ← 「明显不同」的刻度")
    print(f"\n判读：比值接近 1 → 降采样等同另一位作者的画法（立论证伪）")
    print(f"      比值接近 {r_cross:.2f} → 降采样像换了种材质（立论成立）")

    (args.out / "artist_spread.json").write_text(json.dumps({
        "n_materials": len(mats),
        "d_within_median": float(np.median(d_in)),
        "d_downsample_median": float(np.median(d_down)),
        "d_cross_material_median": float(np.median(d_cross)),
        "ratio_down_over_within": r,
        "ratio_cross_over_within": r_cross,
    }, indent=1))


if __name__ == "__main__":
    main()
