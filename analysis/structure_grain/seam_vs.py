"""缝档位：常数(15百分位) vs 实测。必须看图。

`seam_level.py` 量出常数把缝画得比真人暗约三倍。
但"更接近真人的档位"不等于"出图更好"——本项目多次指标动画廊不动。
所以并排出图：真人 / 常数版 / 实测版，每种 3 样本。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from model import build_model                                      # noqa: E402
from structural_prior import (learn_prior, learn_border, learn_edges,  # noqa: E402
                              make_seed, add_border_union, fill_from_seed)

MATS = ["default_brick.png", "default_stone_brick.png", "default_wood.png",
        "default_desert_stone_brick.png", "default_bookshelf.png",
        "default_diamond_block.png", "mcl_nether_nether_brick.png",
        "default_sandstone_brick.png"]
N = 3


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

    rows_out = []
    for m in MATS:
        if m not in ref or m not in ck["mat2id"]:
            continue
        tr = [s for s in bymat[m] if s["split"] == "train"]
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        brd, egs = learn_border(raw), learn_edges(raw)
        s = ref[m]
        pal = np.array(s["palette"], np.uint8)
        nk = len(pal)
        art = pal[np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)]
        cells = [art]
        pr0 = pr
        for tag in ("p15", "obs"):
            p2 = pr
            for i in range(N):
                sd = add_border_union(
                    make_seed(p2, nk, rng=np.random.default_rng(700 + i)),
                    brd, egs, nk)
                g = fill_from_seed(net, sd, pal, nk, ck["mat2id"][m],
                                   seed_rng=700 + i,
                                   temperature=1.3 if tag == "p15" else 1.9)
                cells.append(pal[np.clip(g, 0, nk - 1)])
        rows_out.append((m, pr["seam_p15"], "T1.3 vs T1.9", cells))

    C, P, LAB, HDR = 88, 4, 210, 30
    W = LAB + (1 + 2 * N) * (C + P) + P
    H = HDR + len(rows_out) * (C + P) + P
    cv = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(cv)
    heads = ["真人"] + [f"T1.3 #{i+1}" for i in range(N)] + [f"T1.9 #{i+1}" for i in range(N)]
    for ci, t in enumerate(heads):
        dr.text((LAB + ci * (C + P) + 4, 8), t, fill="black")
    for ri, (m, p15, obs, cells) in enumerate(rows_out):
        y = HDR + ri * (C + P)
        dr.text((6, y + C // 2 - 12),
                m.replace("default_", "").replace(".png", "")[:22], fill="black")
        dr.text((6, y + C // 2 + 2), str(obs), fill="#666")
        for ci, im in enumerate(cells):
            x = LAB + ci * (C + P)
            cv.paste(Image.fromarray(im.astype(np.uint8)).resize((C, C),
                                                                 Image.NEAREST), (x, y))
            dr.rectangle([x, y, x + C, y + C], outline="#bbb")
    out = Path("experiments/temp_vs.png")
    cv.save(out)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
