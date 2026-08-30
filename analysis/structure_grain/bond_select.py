"""逐材质决定用错缝先验还是平均法。

为什么需要选择而不是一个门：候选判据（周期一致、相位不一致、检出率够）
选出 10 个材质，错缝法只在其中 4 个上更接近真人。
试过两个数据侧的门都不预测胜负（见 `learn_bond` 的注释），
所以改成量出来选。

只用**训练瓦片**算真人基准，不碰测试集，也不需要人工标注。
输出 data/tiles/bond_select.json，`learn_prior` 会读它。
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


def joint_rate(tile, bands, period):
    return sum(_col_period(tile[y0:y1].astype(float))[0] == period
               for y0, y1 in bands) / max(len(bands), 1)


def main():
    n_samp = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 1
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
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16 and s["split"] == "train":
            bymat.setdefault(s["material"], []).append(s)

    out = {}
    print(f"{'材质':<34}{'真人':>7}{'平均法':>8}{'错缝法':>8}{'选用':>8}{'重复一致':>10}")
    print("-" * 76)
    for m in sorted(bymat):
        tr = bymat[m]
        if len(tr) < 4 or m not in ck["mat2id"]:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr])
        bd = pr.get("bond", {})
        if not bd.get("active"):
            continue
        per = bd["period"]
        bands = bands_of(pr["rows"])
        art = float(np.mean([joint_rate(t, bands, per) for t in raw]))

        # 用训练瓦片自己的调色板，不碰测试集
        brd, egs = learn_border(raw), learn_edges(raw)
        # 每个重复换一块训练瓦片的调色板 + 换随机种子基
        votes, os_, ns_ = [], [], []
        for rep in range(n_rep):
            pal = np.array(tr[rep % len(tr)]["palette"], np.uint8)
            nk = len(pal)
            base = 5000 + 977 * rep
            old, new = [], []
            for i in range(n_samp):
                rng = np.random.default_rng(base + i)
                for seed, acc in (
                        (add_border_union(make_seed(pr, nk), brd, egs, nk), old),
                        (add_border_union(make_seed(pr, nk, rng=rng),
                                          brd, egs, nk), new)):
                    g = fill_from_seed(net, seed, pal, nk, ck["mat2id"][m],
                                       seed_rng=base + i)
                    acc.append(joint_rate(np.clip(g, 0, nk - 1), bands, per))
            o, n = float(np.mean(old)), float(np.mean(new))
            os_.append(o)
            ns_.append(n)
            votes.append(abs(n - art) < abs(o - art))
        agree = max(sum(votes), n_rep - sum(votes)) / n_rep
        use = sum(votes) > n_rep / 2
        out[m] = {"use_bond": bool(use), "period": int(per),
                  "artist": round(art, 4), "avg": round(float(np.mean(os_)), 4),
                  "bond": round(float(np.mean(ns_)), 4),
                  "agree": round(agree, 3), "votes": [bool(v) for v in votes],
                  "n_samples": n_samp, "n_rep": n_rep}
        print(f"{m:<34}{art:>7.2f}{np.mean(os_):>8.2f}{np.mean(ns_):>8.2f}"
              f"{('错缝' if use else '平均'):>10}{agree:>9.0%}")

    f = Path("data/tiles/bond_select.json")
    f.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    k = sum(v["use_bond"] for v in out.values())
    print(f"\n候选 {len(out)} 个，选用错缝 {k} 个 -> 写入 {f}")


if __name__ == "__main__":
    main()
