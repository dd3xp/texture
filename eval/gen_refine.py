"""离散 SDEdit：大模型草图提供布局，TRD 按画师的统计重画（验证集上先试）。

判官在验证集上偏好 B2（35%）：B2 一眼认得出是什么材质，TRD 常常平淡、特征少。
这里把两者接起来——SDXL 管"画的是什么"（布局），TRD 管"画师怎么画"：
1. 草图 = 某个基线的瓦片（默认 B2）；
2. 调色板：`retrieve` = 按 材质名 + 草图平均色 检索真人调色板并平移（与 retrieve_model 同一机制）；
   `draft` = 草图自己的颜色；
3. 草图亮度按检索瓦片的秩直方图做直方图匹配 → 秩网格；随机掩掉 1-keep 的格子；
4. TRD 在已知调色板 + 部分已知网格上生成其余格子（MaskGIT 天然支持）。

    python eval/gen_refine.py --run runs/trd_v2 --ckpt last.pt --set V_mat --src B2val --keep 0.5 --tag rf_b2_k50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from trd import sample, K_MAX                                              # noqa: E402
from train_trd import clip_text, TEXT_TMPL, model_from_args, encode_palette   # noqa: E402
from tiles_data import canonicalise, W_LUM                                 # noqa: E402
from prompts import load_set                                               # noqa: E402


def draft_tile(d: Path, slug):
    for name in (f"{slug}.png", f"{slug}_0.png"):
        if (d / name).exists():
            return np.asarray(Image.open(d / name).convert("RGB"))
    return None


def hist_match_ranks(tile, hist):
    """草图每个像素的亮度百分位 → 按目标秩直方图的累积分布映射成秩（保留草图的明暗布局）。"""
    lum = tile.reshape(-1, 3).astype(float) @ W_LUM
    order = np.argsort(lum, kind="stable")
    pct = np.empty(len(lum))
    pct[order] = (np.arange(len(lum)) + 0.5) / len(lum)
    cdf = np.cumsum(hist)
    return np.minimum(np.searchsorted(cdf, pct), len(hist) - 1).reshape(tile.shape[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--src", default="B2val", help="草图方法目录（experiments/baselines/<src>/<size>）")
    ap.add_argument("--keep", type=float, default=0.5, help="保留草图格子的比例，其余由 TRD 重画")
    ap.add_argument("--pal", choices=["retrieve", "draft"], default="retrieve")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ck = torch.load(a.run / a.ckpt, map_location=dev)
    cb = np.load(a.run / "codebook.npy")
    model = model_from_args(ck["args"], drop=0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    mem = None
    if a.pal == "retrieve":
        from palette_memory import PaletteMemory
        mem = PaletteMemory(dev)
    prompts, _ = load_set(a.set)
    temb = clip_text([TEXT_TMPL.format(p=e["prompt"]) for e in prompts], dev).to(dev)
    src = ROOT / "experiments/baselines" / a.src / str(a.size)
    out = ROOT / "experiments/baselines" / a.tag / str(a.size)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, e in enumerate(prompts):
        slug = e["material"].rsplit(".", 1)[0]
        t = draft_tile(src, slug)
        if t is None:
            continue
        pals, grids = [], []
        for _ in range(a.n):
            if mem is not None:
                kd = min(K_MAX, len(np.unique(t.reshape(-1, 3), axis=0)))    # 色数跟草图走
                _, j = mem.query(temb[i], kd, rng, colour=t.reshape(-1, 3).mean(0))
                p, h = mem.shifted(j, t.reshape(-1, 3).mean(0)), mem.hist[j]
                g = hist_match_ranks(t, h)
            else:
                cols, inv = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
                g, p = canonicalise(inv.reshape(t.shape[:2]).astype(np.int64), cols.astype(np.uint8))
                p = p[:K_MAX]
                g = np.minimum(g, len(p) - 1)
            g = g.copy()
            g[rng.random(g.shape) >= a.keep] = K_MAX              # GRID_MASK
            pals.append(p)
            grids.append(g)
        codes = np.full((a.n, K_MAX), len(cb) + 1, np.int64)
        for r, p in enumerate(pals):
            codes[r, :len(p)] = encode_palette(p, cb)
        ks = torch.tensor([len(p) for p in pals], device=dev)
        _, grid = sample(model, temb[i:i + 1].expand(a.n, -1), ks, n=a.size, cfg=a.cfg,
                         pal_init=torch.tensor(codes, device=dev),
                         grid_init=torch.tensor(np.stack(grids), device=dev))
        gg = grid.cpu().numpy()
        for r, p in enumerate(pals):
            Image.fromarray(np.asarray(p, np.uint8)[np.clip(gg[r], 0, len(p) - 1)]).save(out / f"{slug}_{r}.png")
        done += 1
    print(f"{done} 个材质 -> {out}")


if __name__ == "__main__":
    main()
