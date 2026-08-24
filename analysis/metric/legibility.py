"""M2：材质可读性度量。

M2 原规格是"找一个指标把真人源和 SDXL 源分开，AUC > 0.80"。
**这个规格有问题**：M1 发现真人源并非总是更好（obsidian/mossycobble/ice/
cobble/grass 五种 SDXL 更好），所以"真人 vs SDXL"是**来源标签**而非质量标签。
去最大化那个 AUC，得到的是"判断哪条流水线做的"的检测器，不是尺子。

改为直接操作化要测的那句话——**低分辨率图还能不能被认出是什么材质**：
把图放大后交给 CLIP 在材质名集合上做零样本分类，
记录 top-1 是否正确、以及真类概率。认得出即读得出。

这样得到的是**每张图一个质量分**，不依赖来源标签，
而且直接对应任务里缺失的那条验收标准。
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from materials import load as load_materials  # noqa: E402

MODEL = "openai/clip-vit-base-patch32"
# 图片文件名去掉了 .png，键要对齐
MATERIALS = {k[:-4] if k.endswith('.png') else k: v
             for k, v in load_materials().items()}

TEMPLATE = "a texture of {}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=Path("experiments/metric/testset"))
    ap.add_argument("--out", type=Path, default=Path("experiments/metric"))
    ap.add_argument("--upscale", type=int, default=224)
    args = ap.parse_args()

    keys = list(MATERIALS)
    texts = [TEMPLATE.format(MATERIALS[k]) for k in keys]

    model = CLIPModel.from_pretrained(MODEL).eval().cuda()
    proc = CLIPProcessor.from_pretrained(MODEL)
    with torch.no_grad():
        ti = proc(text=texts, return_tensors="pt", padding=True)
        tf = model.get_text_features(**{k: v.cuda() for k, v in ti.items()})
        tf = tf / tf.norm(dim=-1, keepdim=True)

    def score(img: Image.Image) -> np.ndarray:
        # 最近邻放大：保住像素边界，不要用插值把格子糊掉
        im = img.convert("RGB").resize((args.upscale, args.upscale), Image.NEAREST)
        with torch.no_grad():
            ii = proc(images=im, return_tensors="pt")
            f = model.get_image_features(pixel_values=ii["pixel_values"].cuda())
            f = f / f.norm(dim=-1, keepdim=True)
            return (100.0 * f @ tf.T).softmax(-1)[0].cpu().numpy()

    rows = []
    for f in sorted((args.testset / "img").glob("*.png")):
        stem = f.stem                      # default_wood_16_artist
        tag = stem.rsplit("_", 1)[1]
        size = int(stem.rsplit("_", 2)[1])
        mat = stem.rsplit("_", 2)[0]
        if mat not in MATERIALS:
            continue
        p = score(Image.open(f))
        gt = keys.index(mat)
        rows.append({"material": mat, "size": size, "source": tag,
                     "correct": int(p.argmax() == gt),
                     "p_true": float(p[gt]),
                     "pred": keys[int(p.argmax())]})

    # 对照组：真人**原生**低分辨率像素画。
    # 这是关键控制——CLIP 在 224 自然图上训练，最近邻放大的 16x16 是分布外输入。
    # 若真人原生像素画也读不出，说明 CLIP 读不了这个域，尺子不成立。
    native = []
    pairs = json.loads(Path("data/contentdb/pairs.json").read_text())
    for m in MATERIALS:
        key = m + ".png"
        if key not in pairs:
            continue
        for pack, path in list(pairs[key]["low"].items())[:3]:
            try:
                im = Image.open(path)
            except Exception:
                continue
            p_ = score(im)
            native.append({"material": m, "size": 16, "source": "artist_native",
                           "correct": int(p_.argmax() == keys.index(m)),
                           "p_true": float(p_[keys.index(m)]), "pred": keys[int(p_.argmax())]})

    # 高分辨率源作参照上界
    ref = []
    for m in MATERIALS:
        f = args.testset / f"sdxl_{m}.png"
        if not f.exists():
            f = args.testset / f"sdxl_{m}"
        if f.exists():
            p = score(Image.open(f))
            ref.append({"material": m, "size": 1024, "source": "sdxl_src",
                        "correct": int(p.argmax() == keys.index(m)),
                        "p_true": float(p[keys.index(m)]), "pred": keys[int(p.argmax())]})

    print("== 材质可读性（CLIP 零样本，15 类，随机基线 6.7%）==\n")
    print(f"{'尺寸':<8}{'来源':<10}{'top-1 正确率':>13}{'真类概率中位':>14}")
    print("-" * 46)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["size"], r["source"])].append(r)
    for k in sorted(agg):
        v = agg[k]
        acc = np.mean([x["correct"] for x in v])
        pt = np.median([x["p_true"] for x in v])
        print(f"{str(k[0])+'x'+str(k[0]):<8}{k[1]:<10}{acc:>12.1%}{pt:>14.3f}")
    if native:
        acc = np.mean([x["correct"] for x in native])
        pt = np.median([x["p_true"] for x in native])
        print(f"{'16x16':<8}{'真人原生':<10}{acc:>12.1%}{pt:>14.3f}   ← 关键对照（n={len(native)}）")
    if ref:
        acc = np.mean([x["correct"] for x in ref])
        pt = np.median([x["p_true"] for x in ref])
        print(f"{'1024':<8}{'sdxl源':<10}{acc:>12.1%}{pt:>14.3f}   ← 高分辨率参照上界")

    with open(args.out / "legibility.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows + native + ref)
    print(f"\n写入 {args.out/'legibility.csv'}")


if __name__ == "__main__":
    main()
