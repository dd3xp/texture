"""把 TRD 与真人之间的分布差距拆成"结构"与"配色"两半（验证集，只用于决定下一步改哪里）。

TRD 的表示天然可拆：瓦片 = 亮度序索引网格（结构）+ 调色板（配色）。对 V-mat 每张真人瓦片，
用它的材质名、**同样的色数 k** 让 TRD 出一张，然后两两互换调色板（同 k，亮度序一一对应）：

  TRD           TRD 网格 + TRD 调色板
  结构=真人      真人网格 + TRD 调色板   → 与真人的差距只来自配色
  配色=真人      TRD 网格 + 真人调色板   → 与真人的差距只来自结构
  真人（n 对半） 参照集自身的一半 vs 另一半 → 地板

哪一行离"TRD"更近、离地板更远，差距就主要在哪一半。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from trd import sample                                         # noqa: E402
from train_trd import decode, clip_text, TEXT_TMPL, model_from_args   # noqa: E402
from colour_task import targets                                # noqa: E402
from tiles_data import load, canonicalise                      # noqa: E402
from metrics import evaluate                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--reps", type=int, default=2, help="每个目标出几张（增加 n 降方差）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.run / a.ckpt, map_location=dev)
    cb = np.load(a.run / "codebook.npy")
    model = model_from_args(ck["args"], drop=0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    T = targets("V_mat", 16)
    ref = [t["ref"] for t in T]
    # 真人：亮度序网格 + 调色板（与 tiles_data 同一规范化）
    real = []
    for t in T:
        cols, inv = np.unique(t["ref"].reshape(-1, 3), axis=0, return_inverse=True)
        g, p = canonicalise(inv.reshape(16, 16).astype(np.int64), cols.astype(np.uint8))
        real.append((g, p))
    temb = clip_text([TEXT_TMPL.format(p=t["prompt"]) for t in T], dev).to(dev)
    ks = torch.tensor([p.shape[0] for _, p in real], device=dev).clamp(max=16)

    # 检索调色板：训练集（去污染）里同色数、材质名 CLIP 文本嵌入最近的前 5 张之一的调色板
    train = load(16, "train", extra=True)
    tr_m = sorted({s["material"] for s in train})
    from train_trd import text_prompt
    tr_e = clip_text([text_prompt(m) for m in tr_m], dev).to(dev)
    tr_ix = {m: i for i, m in enumerate(tr_m)}
    by_k = {}
    for s in train:
        by_k.setdefault(s["k_used"], []).append(s)
    rng = np.random.default_rng(a.seed)

    def retrieve(i, k):
        pool = by_k.get(k) or by_k[min(by_k, key=lambda kk: abs(kk - k))]
        sims = (tr_e[[tr_ix[s["material"]] for s in pool]] @ temb[i]).cpu().numpy()
        top = np.argsort(-sims)[:5]
        return pool[top[rng.integers(len(top))]]["palette"]

    rows = {k: [] for k in ("TRD", "struct=real", "palette=real", "palette=retrieved")}
    mats = []
    for r in range(a.reps):
        for i in range(0, len(T), 32):
            sl = slice(i, i + 32)
            pal, grid = sample(model, temb[sl], ks[sl], n=16, cfg=a.cfg)
            tiles = decode(pal, grid, cb)
            for j, t in enumerate(T[sl]):
                gi = grid[j].cpu().numpy()
                tp = cb[pal[j].cpu().numpy()[:int(ks[sl][j])]]              # TRD 调色板（亮度序槽）
                rg, rp = real[i + j]
                rows["TRD"].append(tiles[j])
                rows["struct=real"].append(tp[rg].astype(np.uint8))
                rows["palette=real"].append(rp[np.clip(gi, 0, len(rp) - 1)].astype(np.uint8))
                qp = retrieve(i + j, int(ks[sl][j]))
                rows["palette=retrieved"].append(qp[np.clip(gi, 0, len(qp) - 1)].astype(np.uint8))
                mats.append(t["prompt"])
    out, cache = {}, None
    for name, tiles in rows.items():
        res, cache = evaluate(tiles, mats, ref, ref_cache=cache)
        out[name] = res
        print(f"{name:<14} " + "  ".join(f"{k}={v:.3f}" for k, v in res.items() if k != "n"), flush=True)
    # 地板：真人一半对另一半（不同目标，n 小，只作量级参考）
    half = len(ref) // 2
    res, _ = evaluate(ref[:half], [t["prompt"] for t in T[:half]], ref[half:])
    out["real_half"] = res
    print("real half vs half " + "  ".join(f"{k}={v:.3f}" for k, v in res.items() if k != "n"))
    p = ROOT / f"experiments/diag_decompose_{a.run.name}.json"
    p.write_text(json.dumps(out, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
