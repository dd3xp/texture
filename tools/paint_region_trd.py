"""任务本身（`GOAL.md`）：纯色区域 + 材质名 → 那块区域被画上 16/24/32 像素画纹理，配色跟随区域。

用新架构 TRD 生成瓦片：
- 材质名 → CLIP 文本条件（开放词表）；
- **区域颜色 → 平均色条件**（训练时学过，`[r,g,b,1]`），所以配色由模型原生跟随区域，
  不靠事后挪色；需要时 `--recolor` 再用 `recolor_to` 对齐一次；
- 环面相对位置 → 生成的瓦片**首尾天然接得上**，平铺进区域没有接缝。

区域判定、平铺相位沿用 `tools/paint_region.py` 的做法（背景 = 占据边框的颜色）。

    HF_HUB_OFFLINE=1 python tools/paint_region_trd.py in.png "oak planks" --run runs/trd_v1 -o out.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from paint_region import dominant_color, region_mask, tile_over, auto_scale   # noqa: E402
from make_texture import recolor_to                                         # noqa: E402
from trd import TRD, sample                                                 # noqa: E402
from train_trd import decode, clip_text, TEXT_TMPL                           # noqa: E402


def load_trd(run: Path, ckpt: str, dev: str):
    ck = torch.load(run / ckpt, map_location=dev)
    a = ck["args"]
    m = TRD(int(a["codes"]), d=int(a["d"]), depth=int(a["depth"]), heads=int(a["heads"]), drop=0.0)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), np.load(run / "codebook.npy")


def generate_tile(model, cb, prompt, color, size, k, n_cand, seed, dev, cfg=2.0):
    """出 n_cand 张候选，取平均色最接近区域颜色的一张。"""
    torch.manual_seed(seed)
    text = clip_text([TEXT_TMPL.format(p=prompt)], dev).to(dev).expand(n_cand, -1)
    col = torch.tensor([*(np.array(color) / 255.0), 1.0], dtype=torch.float32, device=dev)
    col = col[None].expand(n_cand, -1)
    ks = torch.full((n_cand,), k, device=dev)
    pal, grid = sample(model, text, ks, n=size, color=col, cfg=cfg)
    tiles = decode(pal, grid, cb)
    err = [np.abs(t.reshape(-1, 3).mean(0) - np.array(color)).sum() for t in tiles]
    return tiles[int(np.argmin(err))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("prompt")
    ap.add_argument("--run", type=Path, default=ROOT / "runs/trd_v1")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--size", type=int, default=16, choices=[16, 24, 32])
    ap.add_argument("--k", type=int, default=8, help="色阶数（2-16）")
    ap.add_argument("--color", help="区域颜色 RRGGBB；默认自动判定")
    ap.add_argument("--tol", type=int, default=0)
    ap.add_argument("--scale", type=int)
    ap.add_argument("--cand", type=int, default=8, help="候选数，取配色最贴区域的一张")
    ap.add_argument("--recolor", action="store_true", help="再用 recolor_to 把配色对齐到区域颜色")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    img = np.asarray(Image.open(a.image).convert("RGB"))
    color = tuple(int(a.color[i:i + 2], 16) for i in (0, 2, 4)) if a.color else dominant_color(img)
    mask = region_mask(img, color, a.tol)
    print(f"区域颜色 #{color[0]:02x}{color[1]:02x}{color[2]:02x}，覆盖 {mask.mean():.1%}")
    model, cb = load_trd(a.run, a.ckpt, dev)
    tile = generate_tile(model, cb, a.prompt, color, a.size, a.k, a.cand, a.seed, dev)
    if a.recolor:                                   # recolor_to 作用在调色板上，不是整张瓦片
        cols, inv = np.unique(tile.reshape(-1, 3), axis=0, return_inverse=True)
        newpal = np.asarray(recolor_to(cols, "%02x%02x%02x" % tuple(color)))
        tile = newpal[inv.reshape(-1)].reshape(tile.shape).astype(np.uint8)
    scale = a.scale or auto_scale(mask, a.size)
    big = tile_over(mask, tile, scale)
    out = img.copy()
    out[mask] = big[mask]
    Image.fromarray(out).save(a.out)
    Image.fromarray(tile).save(a.out.with_name(a.out.stem + "_tile.png"))
    print(f"瓦片 {a.size}x{a.size}，每纹素 {scale} 像素 -> {a.out}")


if __name__ == "__main__":
    main()
