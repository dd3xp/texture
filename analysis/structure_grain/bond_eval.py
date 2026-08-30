"""错缝先验有没有用：直接量"生成图里竖缝可检出率"与真人的差距。

用这个量而不是颗粒统计，因为要修的就是竖缝。
对每个启用错缝先验的材质，比三组：真人 / 现行平均法 / 错缝采样法。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from model import build_model                                      # noqa: E402
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, fill_from_seed,
                              _col_period)


def bands_of(rows, size=16):
    out, prev = [], 0
    for r in rows + ([size] if rows and rows[-1] != size - 1 else []):
        if r > prev:
            out.append((prev, r))
        prev = r + 1
    return out


def joint_rate(tile: np.ndarray, bands, period: int) -> float:
    """该图有多少比例的砖层能检出目标周期的竖缝。"""
    hit = 0
    for y0, y1 in bands:
        p, _ = _col_period(tile[y0:y1].astype(float))
        hit += (p == period)
    return hit / max(len(bands), 1)


def main():
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

    print(f"{'材质':<34}{'真人':>7}{'平均法':>8}{'错缝法':>8}{'判读':>10}")
    print("-" * 70)
    win = loss = 0
    for m in sorted(bymat):
        tr = [s for s in bymat[m] if s["split"] == "train"]
        if len(tr) < 4 or m not in ck["mat2id"] or m not in ref:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr])
        bd = pr.get("bond", {})
        if not bd.get("active"):
            continue
        per = bd["period"]
        bands = bands_of(pr["rows"])
        art = np.mean([joint_rate(t, bands, per) for t in raw])

        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        nk = len(pal)
        brd = learn_border(raw)
        egs = learn_edges(raw)
        old, new = [], []
        for i in range(6):
            rng = np.random.default_rng(4000 + i)
            s_old = add_border_union(make_seed(pr, nk), brd, egs, nk)
            s_new = add_border_union(make_seed(pr, nk, rng=rng), brd, egs, nk)
            for seed, acc in ((s_old, old), (s_new, new)):
                g = fill_from_seed(net, seed, pal, nk, ck["mat2id"][m],
                                   seed_rng=4000 + i)
                acc.append(joint_rate(np.clip(g, 0, nk - 1), bands, per))
        o, n = float(np.mean(old)), float(np.mean(new))
        better = abs(n - art) < abs(o - art)
        win += better
        loss += not better
        print(f"{m:<34}{art:>7.2f}{o:>8.2f}{n:>8.2f}"
              f"{('错缝更近' if better else '平均更近'):>12}")
    print(f"\n错缝法更接近真人的材质: {win}/{win + loss}")


if __name__ == "__main__":
    main()
