"""一个不赢者通吃的竖缝检测器。

A3s 查明：`_col_period` 在 lag 3–8 上取自相关最大值，
周期 8 只有 2 根缝，噪声在别的 lag 上的伪峰就能夺冠。
但**周期是先验给的、我们本来就知道**，根本不需要 argmax。

改成在**已知 lag** 上检验：
把列廓线按该 lag 折叠，比"最暗那一相位"与其余相位的差，
用列廓线自身的标准差归一，再和随机置换的零分布比。

验收（先定死）：在同一批瓦片上，新检测器必须
  1. 真人 vs 种子+噪声 的分离度比旧检测器**更好**；
  2. 且不是靠把两边都推高——真人的绝对值也要合理。
不满足就说明新检测器只是换了个噪声，不用它。
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, _col_period)


def bands_of(rows, size=16):
    out, prev = [], 0
    for r in (rows or []) + ([size] if rows and rows[-1] != size - 1 else []):
        if r > prev:
            out.append((prev, r))
        prev = r + 1
    return out


def seam_z(seg: np.ndarray, period: int, n_perm: int = 200,
           rng=None) -> float:
    """在已知周期上检验竖缝强度，返回相对随机置换零分布的 z 值。

    不做 argmax：周期是先验给的。
    零分布用列的随机置换——保留列廓线的取值分布，破坏其空间排列。
    """
    col = seg.mean(0)
    if col.std() < 1e-9:
        return 0.0

    def stat(c):
        ph = int(np.argmin([c[i::period].mean() for i in range(period)]))
        m = np.zeros(len(c), bool)
        m[ph::period] = True
        if m.all() or not m.any():
            return 0.0
        return float((c[~m].mean() - c[m].mean()) / (c.std() + 1e-9))

    obs = stat(col)
    rng = rng or np.random.default_rng(0)
    null = np.array([stat(rng.permutation(col)) for _ in range(n_perm)])
    return float((obs - null.mean()) / (null.std() + 1e-9))


def rate_old(tile, bands, period):
    return sum(_col_period(tile[y0:y1].astype(float))[0] == period
               for y0, y1 in bands) / max(len(bands), 1)


def rate_new(tile, bands, period, thresh=2.0, rng=None):
    return sum(seam_z(tile[y0:y1].astype(float), period, rng=rng) > thresh
               for y0, y1 in bands) / max(len(bands), 1)


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16 and s["split"] == "train":
            bymat.setdefault(s["material"], []).append(s)
    sel = json.loads(Path("data/tiles/bond_select.json").read_text())

    print(f"{'材质':<28}{'周期':>4}"
          f"{'旧:真人':>9}{'旧:噪声':>9}{'旧:差':>8}"
          f"{'新:真人':>9}{'新:噪声':>9}{'新:差':>8}")
    print("-" * 84)
    do, dn = [], []
    for m in sorted(sel):
        if m not in bymat:
            continue
        per = sel[m]["period"]
        tr = bymat[m]
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        bands = bands_of(pr["rows"])
        nk = len(tr[0]["palette"])
        brd, egs = learn_border(raw), learn_edges(raw)
        marg = np.bincount(np.concatenate([t.ravel() for t in raw]),
                           minlength=nk)[:nk].astype(float)
        marg = marg / marg.sum()

        rng = np.random.default_rng(11)
        ao = float(np.mean([rate_old(t, bands, per) for t in raw]))
        an = float(np.mean([rate_new(t, bands, per, rng=rng) for t in raw]))
        no, nn = [], []
        for i in range(12):
            r2 = np.random.default_rng(9300 + i)
            sd = add_border_union(make_seed(pr, nk, rng=r2), brd, egs, nk)
            free = sd < 0
            t2 = sd.copy()
            t2[free] = r2.choice(nk, size=int(free.sum()), p=marg)
            no.append(rate_old(t2, bands, per))
            nn.append(rate_new(t2, bands, per, rng=rng))
        no, nn = float(np.mean(no)), float(np.mean(nn))
        do.append(ao - no)
        dn.append(an - nn)
        print(f"{m:<28}{per:>4}{ao:>9.2f}{no:>9.2f}{ao-no:>8.2f}"
              f"{an:>9.2f}{nn:>9.2f}{an-nn:>8.2f}")
    print(f"\n真人与噪声的分离度中位：旧 {np.median(do):+.2f}   新 {np.median(dn):+.2f}")
    print("  新的更大 -> 检测器改好了；相近或更小 -> 没改好，别换")


if __name__ == "__main__":
    main()
