"""真人是"面内均匀、面间不同"，还是到处都花？

由 `seed_survives.py` 逼出来的问题：
周期 8 的 1 像素竖缝，在同边缘分布的随机噪声下检出率只有 0.08，
真人却有 0.47。真人不是靠把缝放得更对——缝就一根像素。
猜想是**真人让砖面内部保持均匀，缝才显出来**；
我们的 T=1.3 在整幅图上均匀地制造变化，把面也填花了。

A3o 当初用全局量（每图色数、相邻同色比例）选了 T=1.3，
那个量**区分不了**"面内均匀面间不同"和"到处都花"——两者可以有同样的全局统计。

**第一版量错了**，记在这里：原先比的是"面内方差 vs 面间方差"，
实测真人是 11.87（远大于 1）。但所有砖面本就是同一种材质、均值天然接近，
面间方差必然小——**这个比值几乎必然大于 1，它根本不回答缝显不显**。
问题 formulate 错了，不是猜想被推翻。

该问的是**缝列相对于非缝列暗多少**：
`_col_period` 靠列廓线的自相关检出周期，
失败是因为非缝列和缝列一样暗。所以量

    (非缝列均值 − 缝列均值) / 列廓线标准差

真人若显著为正而"种子+噪声"接近 0，就说明缝的对比度不够，
该修的是缝本身而不是模型。
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from structural_prior import learn_prior, _col_period               # noqa: E402


def bands_of(rows, size=16):
    out, prev = [], 0
    for r in (rows or []) + ([size] if rows and rows[-1] != size - 1 else []):
        if r > prev:
            out.append((prev, r))
        prev = r + 1
    return out


def seam_contrast(tile: np.ndarray, bands, period: int,
                  phase: int | None = None) -> float | None:
    """缝列比非缝列暗多少，以列廓线标准差为单位。

    正值越大 = 缝越显。这才是自相关能不能检出周期的直接原因。
    """
    out = []
    for y0, y1 in bands:
        seg = tile[y0:y1].astype(float)
        if seg.shape[0] < 2:
            continue
        col = seg.mean(0)
        sd = col.std()
        if sd < 1e-6:
            continue
        ph = phase
        if ph is None:
            _, ph = _col_period(seg)
            if ph is None:
                ph = int(np.argmin([col[i::period].mean()
                                    for i in range(period)]))
        m = np.zeros(16, bool)
        m[ph::period] = True
        if m.all() or not m.any():
            continue
        out.append(float((col[~m].mean() - col[m].mean()) / sd))
    return float(np.mean(out)) if out else None


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16 and s["split"] == "train":
            bymat.setdefault(s["material"], []).append(s)
    sel = json.loads(Path("data/tiles/bond_select.json").read_text())

    print(f"{'材质':<30}{'周期':>5}{'真人缝对比':>12}{'种子+噪声':>12}{'样本':>6}")
    print("-" * 70)
    allr, alln = [], []
    for m in sorted(sel):
        if m not in bymat:
            continue
        per = sel[m]["period"]
        tr = bymat[m]
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        bands = bands_of(pr["rows"])
        rs = [r for r in (seam_contrast(t, bands, per) for t in raw)
              if r is not None]
        if not rs:
            continue
        # 对照：种子放好的缝 + 匹配边缘分布的随机噪声
        nk = len(tr[0]["palette"])
        marg = np.bincount(np.concatenate([t.ravel() for t in raw]),
                           minlength=nk)[:nk].astype(float)
        marg = marg / marg.sum()
        from structural_prior import (learn_border, learn_edges, make_seed,
                                      add_border_union)
        brd, egs = learn_border(raw), learn_edges(raw)
        ns = []
        for i in range(12):
            rng = np.random.default_rng(9200 + i)
            sd_ = add_border_union(make_seed(pr, nk, rng=rng), brd, egs, nk)
            free = sd_ < 0
            t2 = sd_.copy()
            t2[free] = rng.choice(nk, size=int(free.sum()), p=marg)
            v = seam_contrast(t2, bands, per)
            if v is not None:
                ns.append(v)
        allr += rs
        alln += ns
        print(f"{m:<30}{per:>5}{np.median(rs):>12.2f}"
              f"{(np.median(ns) if ns else float('nan')):>12.2f}{len(rs):>6}")
        print(f"
缝对比度中位：真人 {np.median(allr):+.2f}   种子+噪声 {np.median(alln):+.2f}")
        print("  两者相近 -> 缝不弱，检不出是别的原因："
              "_col_period 在 lag 3-8 上取自相关最大值，赢者通吃；")
        print("  周期 4 有 4 根缝真峰稳赢，周期 8 只有 2 根，噪声的伪峰就能夺冠。")


if __name__ == "__main__":
    main()
