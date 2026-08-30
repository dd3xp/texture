"""端到端：纯色图 + 材质名 + 分辨率 → 像素画纹理。

这是项目最初陈述的任务。此前所有评估都是"用真人瓦片的调色板复现该瓦片"，
**调色板是抄来的**；真实使用时只有一个纯色，调色板必须从它推出来。

流程：
    纯色 ──派生调色板──> K 色亮度阶
                          ↓
    材质名 ──查结构先验──> 种子（砖缝 / 边框）
                          ↓
    区域 mask ──────────> 只在区域内填充
                          ↓
                      模型填细节（随机顺序，T=1.3）

用法：
    python model/generate.py --color 8b5a2b --material default_wood.png --size 16
    python model/generate.py --color 8b5a2b --material default_brick.png \\
        --region assets/crate.png --out out.png
"""

import argparse
import colorsys
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model                                    # noqa: E402
from structural_prior import (learn_prior, learn_border, make_seed,  # noqa: E402
                              add_border, fill_from_seed)


def palette_from_color(hex_color: str, k: int, spread: float = 0.55) -> np.ndarray:
    """从单个纯色派生 k 档亮度阶，按亮度排序。

    模型学到的"索引 = 亮度档位"这一约定必须保持，
    所以派生的是明度阶而不是随意的 k 个颜色。
    饱和度随明度轻微下降——暗部偏灰是像素画的常见处理。
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    out = []
    for i in range(k):
        t = i / max(k - 1, 1)                      # 0=最暗 1=最亮
        v = float(np.clip(vv * (1 - spread) + spread * vv * 2 * t, 0.04, 1.0))
        s = float(np.clip(ss * (1.15 - 0.3 * t), 0.0, 1.0))
        out.append([int(round(c * 255)) for c in colorsys.hsv_to_rgb(hh, s, v)])
    pal = np.array(out, np.uint8)
    lum = pal.astype(float) @ np.array([0.299, 0.587, 0.114])
    return pal[np.argsort(lum)]


def region_mask(path: Path | None, size: int, bg=(255, 255, 255)) -> np.ndarray:
    if path is None:
        return np.ones((size, size), bool)
    im = Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
    return (np.asarray(im) != np.array(bg)).any(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", required=True, help="纯色，如 8b5a2b")
    ap.add_argument("--material", required=True, help="材质名，如 default_wood.png")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12, help="调色板色数")
    ap.add_argument("--region", type=Path, help="区域图（非白为区域）；缺省为整幅")
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--n", type=int, default=4, help="生成几个样本")
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/model/hybrid2/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/tiles/dataset_k16.json"))
    ap.add_argument("--out", type=Path, default=Path("experiments/generated.png"))
    ap.add_argument("--upscale", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    kw = dict(k=ck["k"], n_materials=ck["n_materials"], size=ck["size"],
              d=a["dim"], depth=a["depth"], drop=0.0)
    if ck["arch"] == "hybrid":
        kw["attn_every"] = a.get("attn_every", 1)
    net = build_model(ck["arch"], **kw).to(dev).eval()
    net.load_state_dict(ck["state"])

    if args.material not in ck["mat2id"]:
        raise SystemExit(f"材质 {args.material} 不在模型的材质表里")
    nk = min(args.colors, ck["k"])
    pal = palette_from_color(args.color, nk)

    # 结构先验来自训练数据里该材质的多个作者版本
    ds = json.loads(args.data.read_text())
    tr = [s for s in ds["samples"]
          if s["material"] == args.material and s["size"] == 16
          and s["split"] == "train"]
    tiles = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
             for s in tr]
    pr = learn_prior(tiles, [len(s["palette"]) for s in tr]) if tiles else \
        {"rows": [], "joints": {}, "seam": 0.0, "score": 0.0}
    bd = learn_border(tiles) if tiles else {"has_border": False}
    seed = add_border(make_seed(pr, nk, size=args.size), bd, nk)

    mask = region_mask(args.region, args.size)
    seed = np.where(mask, seed, -2)          # -2 标记区域外，稍后填背景

    print(f"材质 {args.material}  调色板 {nk} 色  区域 {int(mask.sum())}/{mask.size} 格")
    print(f"先验：横缝{pr['rows']} 边框{'有' if bd.get('has_border') else '无'} "
          f"周期得分{pr['score']:.2f}  种子 {int((seed >= 0).sum())} 格")

    outs = []
    for i in range(args.n):
        sd = np.where(seed == -2, -1, seed)
        gen = fill_from_seed(net, sd, pal, nk, ck["mat2id"][args.material],
                             device=dev, seed_rng=7000 + i,
                             temperature=args.temperature)
        rgb = pal[np.clip(gen, 0, nk - 1)]
        rgb[~mask] = 255                      # 区域外留白
        outs.append(rgb)

    u = args.upscale
    w = args.size * u
    canvas = Image.new("RGB", (w * len(outs) + 4 * (len(outs) - 1), w), "white")
    for i, o in enumerate(outs):
        canvas.paste(Image.fromarray(o).resize((w, w), Image.NEAREST), (i * (w + 4), 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"写入 {args.out}")


if __name__ == "__main__":
    main()
