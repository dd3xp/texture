"""结构保真度：一把能区分"有结构"和"没结构"的尺子。

为什么要造：现行的全局平坦度指标**把无种子模型排第一**
（A3o 记录：无种子 0.016，有种子 T=1.3 是 0.035），
所以它没资格裁决结构先验（A3r）。

设计原则——不量"像素统计像不像"，量"**布局描述子**像不像"：

    描述子 = [行廓线周期强度, 行廓线周期长度,
              各砖层列廓线周期强度均值, 竖缝可检出比例,
              四条边相对内部的对比度, 整体平坦度]

距离 = 生成图的描述子 与 该材质**训练瓦片描述子分布**的标准化距离
（按训练瓦片各维的 MAD 归一，避免量纲不同的维度被某一维支配）。

**验收判据（先定死再测）**：一把合格的尺子必须满足
1. 留出的真人瓦片得分**最好**——它本来就是真人画的；
2. 无种子模型得分**最差**——它确实没有结构；
3. 颗粒材质上不惩罚无种子——那些材质本来就不该有结构。
现行指标第 1、2 条都不满足。任一条不过就说明这把尺子也不能用。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_model                                      # noqa: E402
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, fill_from_seed,
                              _col_period, _edge_mask, EDGES)
from stability import split_half                                   # noqa: E402


def _acf_peak(prof: np.ndarray) -> tuple[float, float]:
    p = prof - prof.mean()
    if p.std() < 1e-9:
        return 0.0, 0.0
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag, sc = max(((l, ac[l]) for l in range(3, 9)), key=lambda t: t[1])
    return float(max(sc, 0.0)), float(lag)


def descriptor(ix: np.ndarray, bands) -> np.ndarray:
    """布局描述子。刻意不含颜色，只含空间结构。"""
    a = ix.astype(float)
    sd = a.std() + 1e-9
    row_sc, row_lag = _acf_peak(a.mean(1))
    col_scs, hits = [], 0
    for y0, y1 in (bands or [(0, a.shape[0])]):
        seg = a[y0:y1]
        if seg.size == 0:
            continue
        sc, _ = _acf_peak(seg.mean(0))
        col_scs.append(sc)
        hits += _col_period(seg)[0] is not None
    n_b = max(len(bands or [1]), 1)
    edge = [float((a[_edge_mask(a.shape, e)].mean()
                   - a[~_edge_mask(a.shape, e)].mean()) / sd) for e in EDGES]
    flat = float(np.mean(a[:, :-1] == a[:, 1:]))
    return np.array([row_sc, row_lag / 8.0, float(np.mean(col_scs or [0.0])),
                     hits / n_b, *edge, flat])


def bands_of(rows, size=16):
    out, prev = [], 0
    for r in (rows or []) + ([size] if rows and rows[-1] != size - 1 else []):
        if r > prev:
            out.append((prev, r))
        prev = r + 1
    return out or [(0, size)]


def make_scorer(train_tiles, bands):
    """用训练瓦片建该材质的描述子分布，返回打分函数（越小越像）。"""
    D = np.stack([descriptor(t, bands) for t in train_tiles])
    med = np.median(D, 0)
    mad = np.median(np.abs(D - med), 0) * 1.4826 + 1e-3
    return lambda ix: float(np.mean(np.abs(descriptor(ix, bands) - med) / mad))


def main():
    n_samp = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    ck = torch.load("experiments/model/hybrid2/best.pt", map_location="cpu",
                    weights_only=False)
    a = ck["args"]
    kw = dict(k=ck["k"], n_materials=ck["n_materials"], size=16,
              d=a["dim"], depth=a["depth"], drop=0.0)
    if ck["arch"] == "hybrid":
        kw["attn_every"] = a.get("attn_every", 1)
    net = build_model(ck["arch"], **kw).cuda().eval()
    net.load_state_dict(ck["state"])

    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat, ref = {}, {}
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        bymat.setdefault(s["material"], []).append(s)
        if s["split"] == "test" and s["material"] not in ref:
            ref[s["material"]] = s

    rows = {"真人留出": [], "先验+填充": [], "无种子模型": []}
    raw_seed, raw_none, gran = [], [], {"真人留出": [], "先验+填充": [], "无种子模型": []}
    n_struct = n_gran = 0
    for m in sorted(ref):
        tr = [s for s in bymat[m] if s["split"] == "train"]
        if len(tr) < 4 or m not in ck["mat2id"]:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        brd, egs = learn_border(raw), learn_edges(raw)
        structured = bool(pr["rows"] or brd["has_border"]
                          or any(v.get("active") for v in egs.values()))
        bands = bands_of(pr["rows"])
        score = make_scorer(raw, bands)

        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        nk = len(pal)
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)

        seeded, none = [], []
        for i in range(n_samp):
            sd = add_border_union(
                make_seed(pr, nk, rng=np.random.default_rng(8000 + i)), brd, egs, nk)
            seeded.append(score(np.clip(
                fill_from_seed(net, sd, pal, nk, ck["mat2id"][m], seed_rng=8000 + i),
                0, nk - 1)))
            none.append(score(np.clip(
                fill_from_seed(net, np.full((16, 16), -1, int), pal, nk,
                               ck["mat2id"][m], seed_rng=8000 + i), 0, nk - 1)))
        tgt = rows if structured else gran
        tgt["真人留出"].append(score(art))
        tgt["先验+填充"].append(float(np.mean(seeded)))
        tgt["无种子模型"].append(float(np.mean(none)))
        if structured:
            n_struct += 1
            raw_seed.append(seeded)
            raw_none.append(none)
        else:
            n_gran += 1

    print(f"有结构材质 {n_struct}，颗粒材质 {n_gran}，每个 {n_samp} 样本")
    print(f"\n=== 有结构材质：描述子距离（越小越像该材质）===")
    for k, v in rows.items():
        print(f"  {k:<12}{np.median(v):>9.3f}")
    print(f"\n=== 颗粒材质（对照，无种子不该被罚）===")
    for k, v in gran.items():
        print(f"  {k:<12}{np.median(v):>9.3f}")

    a_, s_, n_ = (np.array(rows[k]) for k in
                  ("真人留出", "先验+填充", "无种子模型"))
    c1 = np.median(a_) < np.median(s_)
    c2 = np.median(n_) > np.median(s_)
    g = np.array(gran["无种子模型"]) - np.array(gran["先验+填充"])
    c3 = abs(np.median(g)) < 0.15
    print(f"\n=== 验收（先定死的三条）===")
    print(f"  1 真人留出最好        {'通过' if c1 else '不通过'}"
          f"  ({np.median(a_):.3f} vs 先验 {np.median(s_):.3f})")
    print(f"  2 无种子最差          {'通过' if c2 else '不通过'}"
          f"  ({np.median(n_):.3f} vs 先验 {np.median(s_):.3f})")
    print(f"  3 颗粒材质不罚无种子  {'通过' if c3 else '不通过'}"
          f"  (差 {np.median(g):+.3f})")
    print(f"\n{split_half(raw_seed)}")
    print("三条全过才是一把能用的尺子；任一条不过就别用它裁决。")


if __name__ == "__main__":
    main()
