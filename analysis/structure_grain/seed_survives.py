"""种子放好的结构，填充之后还在不在？——以及是被谁淹的。

竖缝格子是种子钉死的，模型不会改写它们。
可 A3q 量到生成图的竖缝检出率只有 0.15，真人是 0.47。
如果**种子本身**的检出率接近 1，那被淹掉的责任在填充不在先验，
瓶颈就得换个地方修。

**必须有对照**：种子图里未定格子若填常数，竖缝对比度人为满格，
检出率 1.00 说明不了什么。所以再做一组：未定格子填**该材质的边缘分布随机抽样**
（有同样的用色统计、没有任何结构）。

  - 随机噪声下仍高 -> 是模型的填充在特定地破坏结构，该修填充。
  - 随机噪声下同样掉 -> 任何噪声都能淹掉 1 像素的缝，
    问题在缝太弱（对比度/宽度），不在模型。

不需要 GPU：全是先验与重采样。
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


def rate(tile, bands, period):
    return sum(_col_period(tile[y0:y1].astype(float))[0] == period
               for y0, y1 in bands) / max(len(bands), 1)


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16 and s["split"] == "train":
            bymat.setdefault(s["material"], []).append(s)
    sel = json.loads(Path("data/tiles/bond_select.json").read_text()) \
        if Path("data/tiles/bond_select.json").exists() else {}

    print(f"{'材质':<30}{'周期':>5}{'真人':>8}{'种子+常数':>11}"
          f"{'种子+噪声':>11}{'种子格数':>10}")
    print("-" * 78)
    seeds, noisy, arts = [], [], []
    for m in sorted(sel):
        if not sel[m].get("use_bond") or m not in bymat:
            continue
        tr = bymat[m]
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        brd, egs = learn_border(raw), learn_edges(raw)
        per = sel[m]["period"]
        bands = bands_of(pr["rows"])
        nk = len(tr[0]["palette"])
        art = float(np.mean([rate(t, bands, per) for t in raw]))
        # 该材质真人瓦片的索引边缘分布——噪声对照要匹配它
        marg = np.bincount(np.concatenate([t.ravel() for t in raw]),
                           minlength=nk)[:nk].astype(float)
        marg = marg / marg.sum()
        vals, vnoise, cells = [], [], []
        for i in range(12):
            rng = np.random.default_rng(9100 + i)
            sd = add_border_union(make_seed(pr, nk, rng=rng), brd, egs, nk)
            free = sd < 0
            vals.append(rate(np.where(free, nk - 1, sd), bands, per))
            draw = rng.choice(nk, size=int(free.sum()), p=marg)
            noise = sd.copy()
            noise[free] = draw
            vnoise.append(rate(noise, bands, per))
            cells.append(int((~free).sum()))
        s_, n_ = float(np.mean(vals)), float(np.mean(vnoise))
        seeds.append(s_)
        noisy.append(n_)
        arts.append(art)
        print(f"{m:<30}{per:>5}{art:>8.2f}{s_:>11.2f}{n_:>11.2f}"
              f"{np.mean(cells):>10.0f}")
    if seeds:
        print(f"\n种子自身竖缝检出中位 {np.median(seeds):.2f}"
              f"（真人 {np.median(arts):.2f}，填充后 A3q 量到 0.15）")
        print("判读：种子高、填充后低 -> 结构是被填充淹掉的，"
              "该修填充而不是先验。\n"
              "      种子本身就低 -> 先验根本没把竖缝放对，该修先验。")


if __name__ == "__main__":
    main()
