"""检验「材质 = 结构 + 噪点」这个分解，以及噪点是否落在调色板相邻档。

猜想（项目负责人提出）：
    一块纹理由两部分组成——
      结构：木板的横条、砖的分缝，低频、成片
      噪点：颜色略有不同的条或点，是像素画风格的来源

由此得到一个可证伪的预测：
    像素画里的噪点不是任意颜色，而是**调色板里相邻的那一档**
    （艺术家做明暗过渡取隔壁色阶）。
    降采样+量化则是"平均后就近取色"，不受相邻约束，偏移应当散得多。

操作化：
    把调色板按亮度排序 → 每个像素得到一个档位 index
    结构 = 局部众数档位（3x3 众数滤波，代表这一片的基色）
    噪点 = 该像素档位 − 局部众数档位
    看噪点分布有多集中在 ±1

这个实验只用已有数据，不需要 GPU。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metric"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from compare import load_rgb, features                # noqa: E402
from build_testset import match_stats, palette_from   # noqa: E402
from learn_material import HELD_OUT                   # noqa: E402


def to_rank(a: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """把图映射成调色板档位。调色板按亮度排序，所以相邻档 = 相邻明度。"""
    lum = pal @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    spal = pal[order]
    d = ((a.reshape(-1, 1, 3) - spal.reshape(1, -1, 3)) ** 2).sum(-1)
    return d.argmin(1).reshape(a.shape[:2])


def mode_filter(r: np.ndarray, k: int = 3) -> np.ndarray:
    """k x k 众数滤波，得到"这一片的基色档位"，即结构。"""
    p = k // 2
    pad = np.pad(r, p, mode="edge")
    out = np.empty_like(r)
    for y in range(r.shape[0]):
        for x in range(r.shape[1]):
            win = pad[y:y + k, x:x + k].ravel()
            out[y, x] = Counter(win.tolist()).most_common(1)[0][0]
    return out


def grain_stats(a: np.ndarray, pal: np.ndarray) -> dict:
    r = to_rank(a, pal)
    s = mode_filter(r)
    g = (r - s).ravel()
    n = len(g)
    absg = np.abs(g)
    return {
        "frac_zero": float((absg == 0).mean()),          # 纯结构、无噪点的像素
        "frac_pm1": float((absg == 1).mean()),           # 相邻一档
        "frac_ge2": float((absg >= 2).mean()),           # 跳档
        # 在**有噪点的像素里**，相邻一档占多少——这是猜想的核心判据
        "pm1_among_nonzero": float((absg == 1).sum() / max((absg > 0).sum(), 1)),
        "grain_energy": float(absg.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--max-high", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("experiments/structure_grain"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = json.loads(args.pairs.read_text())
    A, B = [], []
    for m, v in pairs.items():
        ref = load_rgb(next(iter(v["low"].values())), args.size)
        if ref is None:
            continue
        pal = palette_from(ref, features(ref)["n_colors"])
        if len(pal) < 3:
            continue

        # A 组：真人原生（留出作者，避免与任何训练集重叠）
        for pk, path in v["low"].items():
            if pk not in HELD_OUT:
                continue
            a = load_rgb(path, args.size)
            if a is not None and Image.open(path).size == (args.size, args.size):
                A.append(grain_stats(a, pal))

        # B 组：高分辨率降采样 + 量化
        for path in list(v["high"].values())[: args.max_high]:
            try:
                tex = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = np.asarray(tex.resize((args.size,) * 2, Image.BOX), float)
            x = match_stats(x, ref)
            d = ((x.reshape(-1, 1, 3) - pal.reshape(1, -1, 3)) ** 2).sum(-1)
            B.append(grain_stats(pal[d.argmin(1)].reshape(x.shape), pal))

    keys = ["frac_zero", "frac_pm1", "frac_ge2", "pm1_among_nonzero", "grain_energy"]
    labels = {
        "frac_zero": "纯结构像素占比",
        "frac_pm1": "相邻一档噪点占比",
        "frac_ge2": "跳档(>=2)占比",
        "pm1_among_nonzero": "噪点中相邻一档的比例",
        "grain_energy": "噪点强度(平均绝对档差)",
    }
    print(f"A 真人原生 {len(A)} 张   B 降采样+量化 {len(B)} 张\n")
    print(f"{'量':<26}{'A 真人':>10}{'B 降采样':>11}{'Mann-Whitney p':>17}")
    print("-" * 66)
    res = {}
    for k in keys:
        pa = np.array([x[k] for x in A])
        pb = np.array([x[k] for x in B])
        u = stats.mannwhitneyu(pa, pb)
        res[k] = {"a": float(np.median(pa)), "b": float(np.median(pb)),
                  "p": float(u.pvalue)}
        print(f"{labels[k]:<26}{np.median(pa):>10.3f}{np.median(pb):>11.3f}{u.pvalue:>17.3g}")

    print("\n判读：")
    r = res["pm1_among_nonzero"]
    if r["p"] < 0.05 and r["a"] > r["b"]:
        print("  真人的噪点显著更集中在相邻一档 → 猜想成立，"
              "「噪点=隔壁色阶」是可以写进模型的约束")
    elif r["p"] < 0.05:
        print("  方向相反：降采样的噪点反而更集中在相邻一档 → 猜想不成立")
    else:
        print("  两者无显著差异 → 该判据不支持猜想（但不排除分解本身有用）")

    (args.out / "grain_stats.json").write_text(
        json.dumps({"n_a": len(A), "n_b": len(B), "stats": res}, indent=1,
                   ensure_ascii=False))


if __name__ == "__main__":
    main()
