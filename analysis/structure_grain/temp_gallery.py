"""三档温度并排：现行 1.3 / 全局 1.9 / 定向 1.3→2.2。

定向温度的全部意义是"面升温、缝保持冷"。
全局 1.9 在两个统计量上都最好，但画面明显变噪、方块结构被打散
（`experiments/temp_vs.png`）。若定向版能拿到相近的统计量而不毁结构，
它就是对的；若它同样毁结构，说明问题不在"哪里升温"。
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

MATS = ["default_stone_brick.png", "default_desert_stone_brick.png",
        "default_diamond_block.png", "default_sandstone_brick.png",
        "default_wood.png", "default_brick.png"]
CONF = [(1.3, None, "现行1.3"), (1.3, 2.2, "定向→2.2"), (1.3, 2.6, "定向→2.6")]
N = 3          # 三档 × 3 样本，仍高于 A3f 的 n=1 教训


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

    rows = []
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
        cells = [pal[np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)]]
        for T, TF, _ in CONF:
            for i in range(N):
                sd = add_border_union(
                    make_seed(pr, nk, rng=np.random.default_rng(700 + i)),
                    brd, egs, nk)
                g = fill_from_seed(net, sd, pal, nk, ck["mat2id"][m],
                                   seed_rng=700 + i, temperature=T,
                                   far_temperature=TF)
                cells.append(pal[np.clip(g, 0, nk - 1)])
        rows.append((m, cells))

    C, P, LAB, HDR = 78, 4, 190, 30
    W = LAB + (1 + N * len(CONF)) * (C + P) + P
    H = HDR + len(rows) * (C + P) + P
    cv = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(cv)
    heads = ["真人"] + [f"{lab} #{i+1}" for _, _, lab in CONF for i in range(N)]
    for ci, t in enumerate(heads):
        dr.text((LAB + ci * (C + P) + 3, 9), t, fill="black")
    for ri, (m, cells) in enumerate(rows):
        y = HDR + ri * (C + P)
        dr.text((6, y + C // 2 - 5),
                m.replace("default_", "").replace(".png", "")[:22], fill="black")
        for ci, im in enumerate(cells):
            x = LAB + ci * (C + P)
            cv.paste(Image.fromarray(im.astype(np.uint8)).resize((C, C),
                                                                 Image.NEAREST), (x, y))
            dr.rectangle([x, y, x + C, y + C], outline="#bbb")
    out = Path("experiments/temp_directed2.png")
    cv.save(out)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
