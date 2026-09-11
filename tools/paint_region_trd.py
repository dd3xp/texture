"""任务本身（`GOAL.md`）：纯色区域 + 材质名 → 那块区域被画上 16/24/32 像素画纹理，配色跟随区域。

用新架构 TRD 生成瓦片：
- 材质名 → CLIP 文本条件（开放词表）；
- **区域颜色 → 检索增强调色板**（默认 `--pal_mode retrieve`，`model/palette_memory.py`）：
  按 材质名 + 区域颜色 从真人调色板记忆库里取一张，Lab 平移到区域颜色，作为已知条件；
  TRD 只生成结构（像素排布）。`--pal_mode model` = 旧做法（模型自己出调色板，平均色作条件）；
- 环面相对位置 → 生成的瓦片**首尾天然接得上**，平铺进区域没有接缝。

区域判定、平铺相位沿用 `tools/paint_region.py` 的做法（背景 = 占据边框的颜色）。

    HF_HUB_OFFLINE=1 python tools/paint_region_trd.py in.png "oak planks" --run runs/<run> -o out.png
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
from train_trd import decode, clip_text, TEXT_TMPL, model_from_args, encode_palette   # noqa: E402


def load_trd(run: Path, ckpt: str, dev: str):
    ck = torch.load(run / ckpt, map_location=dev)
    m = model_from_args(ck["args"], drop=0.0)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), np.load(run / "codebook.npy")


def exemplars_for(model, prompt, text, n_cand, dev, extra="train_extra.json", t16=None):
    """v7+ 模型：同材质（按文本最近）、其他画师的结构范例；给了 t16 就按跨模态相似度只留最典型的 8 张。"""
    if getattr(model, "ex_proj", None) is None:
        return None
    from exemplars import ExemplarBank
    from tiles_data import load
    from train_trd import text_prompt
    pool = load(16, "train", extra=extra)
    pm = sorted({s["material"] for s in pool})
    pe = clip_text([text_prompt(m) for m in pm], dev)
    bank = ExemplarBank(pool, {m: pe[i] for i, m in enumerate(pm)}, dev, C=64 if t16 is not None else 16)
    cand = bank.candidates([{"material": prompt}], emb=text[:1])
    if t16 is not None:
        from palette_memory import clip16_images
        pool_art = [s for s in pool if not str(s.get("pack", "")).endswith("@gen")]
        cand = bank.rank_xmodal(cand, clip16_images([s["palette"][s["idx"]] for s in pool_art], dev), t16)
    return bank.draw(cand.expand(n_cand, -1), model.n_ex, False)


def generate_tile(model, cb, prompt, color, size, k, n_cand, seed, dev, cfg=1.5, pal_mode="retrieve",
                  choice_temp=20.0, extra="train_extra.json"):
    """出 n_cand 张候选，取平均色最接近区域颜色的一张。"""
    torch.manual_seed(seed)
    text = clip_text([TEXT_TMPL.format(p=prompt)], dev).to(dev).expand(n_cand, -1)
    from palette_memory import clip16_texts
    t16 = clip16_texts([prompt], dev)                     # 跨模态检索（调色板与范例都挑该材质最典型的画法）
    ex = exemplars_for(model, prompt, text, n_cand, dev, extra, t16)
    col = torch.tensor([*(np.array(color) / 255.0), 1.0], dtype=torch.float32, device=dev)
    col = col[None].expand(n_cand, -1)
    ks = torch.full((n_cand,), k, device=dev)
    if pal_mode == "retrieve":
        from palette_memory import PaletteMemory
        from trd import K_MAX
        mem, rng = PaletteMemory(dev), np.random.default_rng(seed)
        mem.enable_xmodal(dev)
        pals, codes = [], []
        for _ in range(n_cand):
            _, i = mem.query(text[0], None, rng, colour=color, text16=t16[0])
            p = mem.shifted(i, color)
            row = np.full(K_MAX, len(cb) + 1, np.int64)
            row[:len(p)] = encode_palette(p, cb)
            pals.append(p)
            codes.append(row)
        ks = torch.tensor([len(p) for p in pals], device=dev)
        _, grid = sample(model, text, ks, n=size, color=col, cfg=cfg, choice_temp=choice_temp, ex=ex,
                         pal_init=torch.tensor(np.stack(codes), device=dev))
        g = grid.cpu().numpy()
        tiles = [p[np.clip(g[j], 0, len(p) - 1)] for j, p in enumerate(pals)]
    else:
        pal, grid = sample(model, text, ks, n=size, color=col, cfg=cfg, choice_temp=choice_temp, ex=ex)
        tiles = decode(pal, grid, cb)
    err = [np.abs(t.reshape(-1, 3).mean(0) - np.array(color)).sum() for t in tiles]
    return tiles[int(np.argmin(err))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("prompt")
    ap.add_argument("--run", type=Path, default=ROOT / "runs/trd_v8")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--pal_mode", choices=["retrieve", "model"], default="retrieve")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--choice_temp", type=float, default=20.0, help="见 eval/gen_trd.py 同名参数")
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
    ck_args = torch.load(a.run / a.ckpt, map_location="cpu")["args"]
    extra = (ck_args if isinstance(ck_args, dict) else vars(ck_args)).get("extra_file", "train_extra.json")
    tile = generate_tile(model, cb, a.prompt, color, a.size, a.k, a.cand, a.seed, dev,
                         cfg=a.cfg, pal_mode=a.pal_mode, choice_temp=a.choice_temp, extra=extra)
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
