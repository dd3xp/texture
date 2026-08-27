"""D2：把瓦片整理成模型可直接训练的形式，并按包划分数据集。

三个设计决定，都有依据：

1. **不按原生颜色数筛选。** 抽样看图（`experiments/dataset_colorcheck.png`）
   证明颜色数分不开"手绘像素画"和"照片缩放"——129+ 色里有明显手绘的
   TNT 和木板，0-8 色里有大量没有纹理的纯色板。
   颜色数作为元数据保留，让 D4 用实验决定，不在这里硬筛。

2. **统一量化到 K 色索引。** 模型在调色板索引空间生成，
   而任务本身也是"给定调色板去填"，所以训练样本就该是
   (K 色索引图, 调色板, 材质)。量化用中位切分。

3. **调色板按亮度排序。** 索引因此有序——相邻索引 = 相邻明度，
   模型学到的"跳几档"才有意义。

唯一的硬筛：原生颜色数 < 3 的瓦片没有纹理可学，丢掉。
这一条是自证的，不依赖未验证的假设。

划分按**包**，不按图。同一个包画风一致，按图随机划会让
验证集和训练集共享画风，把泛化能力估计得过高。
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MIN_COLORS = 3


def quantize(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """量化到 k 色，返回 (索引图, 调色板)。调色板按亮度排序。"""
    im = Image.fromarray(a.astype(np.uint8))
    p = im.convert("P", palette=Image.ADAPTIVE, colors=k)
    pal = np.array(p.getpalette()[: k * 3], dtype=np.uint8).reshape(-1, 3)
    idx = np.asarray(p, dtype=np.int64)

    used = np.unique(idx)
    pal = pal[used]
    remap = np.full(256, -1, np.int64)
    remap[used] = np.arange(len(used))
    idx = remap[idx]

    lum = pal.astype(float) @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return inv[idx], pal[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/tiles/index.json"))
    ap.add_argument("--out", type=Path, default=Path("data/tiles"))
    ap.add_argument("--k", type=int, default=16, help="统一量化的调色板色数")
    ap.add_argument("--val-packs", type=int, default=8)
    ap.add_argument("--test-packs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    idx = json.loads(args.index.read_text())
    packs = sorted(idx["packs"])
    rng = random.Random(args.seed)
    rng.shuffle(packs)
    test = set(packs[: args.test_packs])
    val = set(packs[args.test_packs: args.test_packs + args.val_packs])
    split_of = {p: ("test" if p in test else "val" if p in val else "train")
                for p in idx["packs"]}

    samples, dropped = [], Counter()
    for mat, entries in idx["materials"].items():
        for e in entries:
            try:
                a = np.asarray(Image.open(e["path"]).convert("RGB"))
            except Exception:
                dropped["读取失败"] += 1
                continue
            native = len(np.unique(a.reshape(-1, 3), axis=0))
            if native < MIN_COLORS:
                dropped["少于3色(无纹理)"] += 1
                continue
            ind, pal = quantize(a, args.k)
            samples.append({
                "material": mat, "pack": e["pack"], "size": e["size"],
                "split": split_of[e["pack"]], "native_colors": native,
                "k_used": int(len(pal)),
                "idx": ind.astype(np.uint8).tobytes().hex(),
                "palette": pal.tolist(),
            })

    by = Counter(s["split"] for s in samples)
    print(f"保留 {len(samples)} 张，丢弃 {sum(dropped.values())}")
    for k, v in dropped.items():
        print(f"  丢弃-{k}: {v}")
    print(f"\n{'划分':<8}{'瓦片':>8}{'包数':>7}{'材质':>7}")
    print("-" * 32)
    for sp in ("train", "val", "test"):
        ss = [s for s in samples if s["split"] == sp]
        print(f"{sp:<8}{len(ss):>8}{len({s['pack'] for s in ss}):>7}"
              f"{len({s['material'] for s in ss}):>7}")

    # 检查划分是否有材质泄漏之外的问题：测试集材质应基本被训练集覆盖
    tr_m = {s["material"] for s in samples if s["split"] == "train"}
    te_m = {s["material"] for s in samples if s["split"] == "test"}
    print(f"\n测试集材质被训练集覆盖: {len(te_m & tr_m)}/{len(te_m)} "
          f"= {len(te_m & tr_m)/max(len(te_m),1):.0%}")

    nc = np.array([s["native_colors"] for s in samples])
    ku = np.array([s["k_used"] for s in samples])
    print(f"\n原生颜色数  中位 {np.median(nc):.0f}  p90 {np.percentile(nc,90):.0f}")
    print(f"量化后实际用色 中位 {np.median(ku):.0f}  p10 {np.percentile(ku,10):.0f}"
          f"  (上限 K={args.k})")
    bysize = Counter(s["size"] for s in samples)
    print(f"分辨率分布: " + "  ".join(f"{s}×{s}:{bysize[s]}" for s in sorted(bysize)))

    out = args.out / f"dataset_k{args.k}.json"
    out.write_text(json.dumps({"k": args.k, "splits": {"val": sorted(val),
                                                       "test": sorted(test)},
                               "samples": samples}, ensure_ascii=False))
    print(f"\n写入 {out}  ({out.stat().st_size//1024//1024} MB)")


if __name__ == "__main__":
    main()
