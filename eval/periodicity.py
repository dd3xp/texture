"""结构代理指标（零 GPU、零 API）：各方法瓦片的"超出偶然的周期一致率" S，按材质的真人结构强弱分两半报。

判官的差距全在结构性材质上（`analysis/arch/judge_structure_split.py`，修正版：
真人对 B2 在 HI 半 84%，TRD+重排 47%；LO 半两者持平）。S 的定义与那里相同：
亮度序网格在所有非平凡循环位移下的最大逐格一致率，减去该瓦片自己的偶然一致率 Σp_c²
（纯色 = 0，4×4 拼贴 ≈ 0.48，打乱 ≈ 0.1）。HI/LO = 按验证集真人瓦片 S 的中位数切 V-mat 材质。

    python eval/periodicity.py --methods REALval B2val v4_ret v7_retm
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from prompts import load_set                  # noqa: E402
from tiles_data import load, canonicalise     # noqa: E402


def score(grid):
    g = np.asarray(grid)
    n = g.size
    _, cnt = np.unique(g, return_counts=True)
    chance = float(((cnt / n) ** 2).sum())
    best = 0.0
    for dy in range(g.shape[0]):
        for dx in range(g.shape[1]):
            if dy or dx:
                best = max(best, float((g == np.roll(g, (dy, dx), (0, 1))).mean()))
    return best - chance


def grid_of(tile):
    cols, inv = np.unique(tile.reshape(-1, 3), axis=0, return_inverse=True)
    return canonicalise(inv.reshape(tile.shape[:2]).astype(np.int64), cols.astype(np.uint8))[0]


def tile_of(root: Path, name, slug, size):
    for d in (root / name / str(size), ROOT / "experiments/baselines_val" / name.replace("val", "") / str(size)):
        for fn in (f"{slug}_0.png", f"{slug}.png"):
            if (d / fn).exists():
                return np.asarray(Image.open(d / fn).convert("RGB"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    a = ap.parse_args()
    prompts, split = load_set(a.set)
    real = {}
    for s in load(16, split):
        real.setdefault(s["material"], []).append(score(s["idx"]))
    mats = [e for e in prompts if e["material"] in real]
    art = np.array([np.mean(real[e["material"]]) for e in mats])
    cut = float(np.median(art))
    hi = art > cut
    print(f"{len(mats)} 个材质，真人 S 中位 {cut:.3f}；HI {hi.sum()} / LO {(~hi).sum()}")
    print(f"{'真人（该材质全部 val 瓦片均值）':<28} HI {art[hi].mean():.3f}  LO {art[~hi].mean():.3f}")
    out = {"cut": cut, "artist": {"HI": float(art[hi].mean()), "LO": float(art[~hi].mean())}}
    for m in a.methods:
        sc = []
        for e in mats:
            t = tile_of(a.root, m, e["material"].rsplit(".", 1)[0], 16)
            sc.append(np.nan if t is None else score(grid_of(t)))
        sc = np.array(sc)
        out[m] = {"HI": float(np.nanmean(sc[hi])), "LO": float(np.nanmean(sc[~hi])),
                  "missing": int(np.isnan(sc).sum())}
        print(f"{m:<28} HI {out[m]['HI']:.3f}  LO {out[m]['LO']:.3f}   缺 {out[m]['missing']}")
    p = ROOT / f"experiments/periodicity_{a.set}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
