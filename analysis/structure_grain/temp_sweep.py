"""重扫填充温度：双目标，且两个目标都用经过验收的量。

A3o 选 T=1.3 用的是**全局**统计（每图色数、相邻同色比例）。
A3x 证明那个量看不见目标现象：全局上看着修好（0.232 vs 真人 0.209），
面内其实完全没修好（0.314 vs 真人 0.154）——
全局统计里缝行自身的变化掩盖了面内的过度平坦。

所以重扫，两个目标：

1. **面内相邻同色**对齐真人（0.154）。种子把面拖平，温度是解药。
2. **结构描述子距离**不变差（`struct_metric` 的打分，三条判据已验收）。
   A3o 说 T≥1.6 破坏结构，但那也是旧指标判的，这里一并复查。

单一目标会走偏：只追 1 会把结构烧掉，只追 2 会退回过平的面。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model                                      # noqa: E402
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, fill_from_seed)
from struct_metric import make_scorer, bands_of                    # noqa: E402
from face_noise import face_stats                                  # noqa: E402

# (近种子温度, 远处温度)；远处为 None 表示全图同温
TEMPS = [(1.3, None), (1.9, None), (1.3, 1.9), (1.3, 2.2), (1.3, 2.6)]


def main():
    n_samp = int(sys.argv[1]) if len(sys.argv) > 1 else 8
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

    jobs = []
    for m in sorted(ref):
        tr = [s for s in bymat[m] if s["split"] == "train"]
        if len(tr) < 4 or m not in ck["mat2id"]:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        brd, egs = learn_border(raw), learn_edges(raw)
        if not (pr["rows"] or brd["has_border"]
                or any(v.get("active") for v in egs.values())):
            continue
        bands = bands_of(pr["rows"])
        if len(bands) < 2:
            continue
        jobs.append((m, tr, raw, pr, brd, egs, bands))
    print(f"参与重扫的材质 {len(jobs)} 个，每个 {n_samp} 样本 × {len(TEMPS)} 档温度",
          flush=True)

    art_face = []
    for m, tr, raw, pr, brd, egs, bands in jobs:
        s = ref[m]
        art = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
        f = face_stats(art.astype(float), bands)
        if f:
            art_face.append(f[0])
    target = float(np.median(art_face))
    print(f"真人面内相邻同色（目标）{target:.3f}\n", flush=True)

    print(f"{'温度':>10}{'面内相邻同色':>14}{'距目标':>10}{'结构距离':>10}")
    print("-" * 42)
    # 每档跑完立刻落盘并支持续跑：这些机器会静默杀掉长任务，
    # 上一版只在全部跑完时打印，被杀后一行结果都没留下。
    save = Path("experiments/temp_sweep.json")
    out = {}
    if save.exists():
        try:
            out = json.loads(save.read_text()).get("by_temp", {})
            if out:
                print(f"续跑：已有 {sorted(out)} 档", flush=True)
        except Exception:
            out = {}
    for T, TF in TEMPS:
        key0 = f"{T}" if TF is None else f"{T}->{TF}"
        if key0 in out:
            r = out[key0]
            print(f"{key0:>10}{r['face']:>14.3f}"
                  f"{abs(r['face']-target):>10.3f}{r['struct']:>10.3f}  (存档)", flush=True)
            continue
        faces, scores = [], []
        for m, tr, raw, pr, brd, egs, bands in jobs:
            s = ref[m]
            pal = np.array(s["palette"], np.uint8)
            nk = len(pal)
            sc = make_scorer(raw, bands)
            fs, ss = [], []
            for i in range(n_samp):
                sd = add_border_union(
                    make_seed(pr, nk, rng=np.random.default_rng(400 + i)),
                    brd, egs, nk)
                g = np.clip(fill_from_seed(net, sd, pal, nk, ck["mat2id"][m],
                                           seed_rng=400 + i, temperature=T,
                                           far_temperature=TF),
                            0, nk - 1)
                f = face_stats(g.astype(float), bands)
                if f:
                    fs.append(f[0])
                ss.append(sc(g))
            if fs:
                faces.append(float(np.mean(fs)))
            scores.append(float(np.mean(ss)))
        mf, msc = float(np.median(faces)), float(np.median(scores))
        key = f"{T}" if TF is None else f"{T}->{TF}"
        out[key] = {"face": mf, "struct": msc}
        print(f"{key:>10}{mf:>14.3f}{abs(mf-target):>10.3f}{msc:>10.3f}", flush=True)
        save.write_text(json.dumps({"target": target, "by_temp": out}, ensure_ascii=False))

    best = min(out, key=lambda k: abs(out[k]["face"] - target))
    print(f"\n面内最接近真人的配置：{best}")
    print("判读：选面内接近真人**且**结构距离不劣于 1.3 的那一档；"
          "两者冲突则记录冲突，不硬选")


if __name__ == "__main__":
    main()
