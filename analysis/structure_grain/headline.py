"""复核头条数字：先验+填充 vs 降采样基线，距真人的结构差距。

原数字是每材质 4 样本算的。A3q 发现 6–8 样本会让**逐材质**结论反转，
所以头条（跨材质聚合）也必须复核——聚合能不能把逐材质噪声平掉，
是要量的，不是假定的。

同时先验本身已变（边框并集 A3p、错缝 A3q），数字本来也要重算。

除了点估计，还做**折半稳定性**：把样本随机分两半各算一次，
两半的结论若不一致，说明样本量仍然不够。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "premise"))
from model import build_model                                      # noqa: E402
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, fill_from_seed)
from decompose import grain_stats                                  # noqa: E402
from build_testset import match_stats                              # noqa: E402
from PIL import Image                                              # noqa: E402


def main():
    n_samp = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    # prior: new=并集边框+错缝, old=周期+整圈边框（A3n 时的状态）
    # set:   new=按新先验判有无, old=只按周期或整圈边框判（A3n 时的 64 个）
    which = sys.argv[2] if len(sys.argv) > 2 else "new/new"
    p_mode, s_mode = which.split("/")
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
    pairs = json.loads(Path("data/contentdb/pairs.json").read_text())
    bymat, ref = {}, {}
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        bymat.setdefault(s["material"], []).append(s)
        if s["split"] == "test" and s["material"] not in ref:
            ref[s["material"]] = s

    art_all, base_all, mod_all, mod_halves = [], [], [], []
    n_mat = 0
    for m in sorted(ref):
        tr = [s for s in bymat[m] if s["split"] == "train"]
        if len(tr) < 4 or m not in ck["mat2id"] or m not in pairs:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr],
                         material=m if p_mode == "new" else None)
        brd = learn_border(raw)
        egs = learn_edges(raw)
        has_edge = any(v.get("active") for v in egs.values())
        # 材质集：old 口径只认周期或整圈边框（A3n 时的 64 个）
        has = bool(pr["rows"] or brd["has_border"]
                   or (has_edge if s_mode == "new" else False))
        if not has:
            continue
        if p_mode == "old":               # 旧先验：不用分边、不用错缝
            egs = {}
            pr = dict(pr, bond={"active": False})

        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        nk = len(pal)
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        art_rgb = pal[art]
        g = lambda rgb: grain_stats(rgb.astype(float), pal.astype(float))["frac_zero"]
        ta = g(art_rgb)

        base = None
        for p in pairs[m]["high"].values():
            try:
                tex = Image.open(p).convert("RGB")
            except Exception:
                continue
            x = match_stats(np.asarray(tex.resize((16, 16), Image.BOX), float),
                            art_rgb.astype(float))
            d = ((x.reshape(-1, 1, 3) - pal.astype(float).reshape(1, -1, 3)) ** 2).sum(-1)
            base = pal[d.argmin(1).reshape(16, 16)]
            break
        if base is None:
            continue

        vals = []
        for i in range(n_samp):
            sd = add_border_union(
                make_seed(pr, nk, rng=np.random.default_rng(6000 + i)), brd, egs, nk)
            gen = fill_from_seed(net, sd, pal, nk, ck["mat2id"][m], seed_rng=6000 + i)
            vals.append(g(pal[np.clip(gen, 0, nk - 1)]))
        n_mat += 1
        art_all.append(ta)
        base_all.append(g(base))
        mod_all.append(float(np.mean(vals)))
        h = n_samp // 2
        mod_halves.append((float(np.mean(vals[:h])), float(np.mean(vals[h:]))))

    art_all = np.array(art_all)
    base_all = np.array(base_all)
    mod_all = np.array(mod_all)
    db, dm = np.abs(base_all - art_all), np.abs(mod_all - art_all)
    print(f"[先验={p_mode} 材质集={s_mode}]  材质 {n_mat}，每个 {n_samp} 样本")
    print(f"{'组':<22}{'纯结构占比':>12}{'距真人中位':>12}")
    print("-" * 48)
    print(f"{'真人原生':<22}{art_all.mean():>12.3f}{'—':>12}")
    print(f"{'降采样基线':<22}{base_all.mean():>12.3f}{np.median(db):>12.3f}")
    print(f"{'先验+填充':<22}{mod_all.mean():>12.3f}{np.median(dm):>12.3f}")
    print(f"\nWilcoxon p={stats.wilcoxon(db, dm).pvalue:.3g}   "
          f"基线/模型 距离比 {np.median(db)/max(np.median(dm),1e-9):.1f}×")

    h1 = np.array([x for x, _ in mod_halves])
    h2 = np.array([y for _, y in mod_halves])
    d1, d2 = np.abs(h1 - art_all), np.abs(h2 - art_all)
    print(f"\n=== 折半稳定性（各 {n_samp//2} 样本）===")
    print(f"  上半距真人中位 {np.median(d1):.4f}   下半 {np.median(d2):.4f}")
    print(f"  两半 Wilcoxon p 分别 {stats.wilcoxon(db, d1).pvalue:.3g} / "
          f"{stats.wilcoxon(db, d2).pvalue:.3g}")
    print(f"  两半间相关 Spearman ρ={stats.spearmanr(d1, d2).correlation:+.3f}")
    print("  判读：两半结论一致且相关高 -> 样本量够，头条成立")


if __name__ == "__main__":
    main()
