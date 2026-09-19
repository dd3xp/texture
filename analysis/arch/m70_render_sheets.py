"""(M70) 渲染对比图页 —— 取样规则写死于预注册（docs/arch_progress.md, commit 7b89eaa）。

规则（⛔ 不许改）：
  - 方法材质 = sorted(experiments/baselines/TRD32_rr4/32/*.png 的 slug)，random.Random(0).sample(..., 8)
  - 左列 = TRD32_rr4（正式交付），右列 = B2（最强基线）
  - 真人锚点 = dataset_k16.json 中 size==32 & split=='train' 的瓦片，random.Random(0).sample(..., 8)
  - 最近邻放大 8×，不做任何滤波

本脚本只渲染，不计算任何指标、不产生任何判决。
"""
import json
import os
import random
import re
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCALE = 8
GAP = 8
OUT = os.path.join(ROOT, "experiments", "m70_sheets")


def load_png(path):
    return Image.open(path).convert("RGB")


def human_tiles():
    p = os.path.join(ROOT, "data", "tiles", "dataset_k16.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return [s for s in d["samples"] if s["size"] == 32 and s["split"] == "train"]


def decode(sample):
    n = sample["size"]
    idx = sample["idx"]
    npx = n * n
    assert len(idx) % npx == 0, (len(idx), npx)
    w = len(idx) // npx
    pal = sample["palette"]
    img = Image.new("RGB", (n, n))
    px = img.load()
    for i in range(npx):
        c = int(idx[i * w:(i + 1) * w], 16)  # 每像素 2 个十六进制位（k=16）
        assert c < len(pal), (c, len(pal))
        px[i % n, i // n] = tuple(pal[c])
    return img


def grid(images, cols, scale=SCALE, gap=GAP):
    tw = images[0].width * scale
    th = images[0].height * scale
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, rows * th + (rows - 1) * gap), (255, 255, 255))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im.resize((tw, th), Image.NEAREST), (c * (tw + gap), r * (th + gap)))
    return sheet


def main():
    os.makedirs(OUT, exist_ok=True)
    d_trd = os.path.join(ROOT, "experiments", "baselines", "TRD32_rr4", "32")
    d_b2 = os.path.join(ROOT, "experiments", "baselines", "B2", "32")
    slugs = sorted(f[:-4] for f in os.listdir(d_trd) if f.endswith(".png"))
    pick = random.Random(0).sample(slugs, 8)
    print("n_slugs=%d" % len(slugs))
    for i, s in enumerate(pick):
        print("pick[%d]=%s" % (i, s))

    # 两张对比页，各 4 对；左 = TRD32_rr4，右 = B2
    for half in (0, 1):
        ims = []
        for s in pick[half * 4:half * 4 + 4]:
            ims.append(load_png(os.path.join(d_trd, s + ".png")))
            # B2 的文件名没有 TRD 那个 "_<样本号>" 后缀；去掉后 272/272 精确一一对应
            ims.append(load_png(os.path.join(d_b2, re.sub(r"_\d+$", "", s) + ".png")))
        grid(ims, 2).save(os.path.join(OUT, "cmp_%d.png" % half))

    hs = human_tiles()
    print("n_human32_train=%d" % len(hs))
    hpick = random.Random(0).sample(hs, 8)
    for i, s in enumerate(hpick):
        print("human[%d]=%s | %s" % (i, s["pack"], s["material"]))
    grid([decode(s) for s in hpick], 2).save(os.path.join(OUT, "human.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
