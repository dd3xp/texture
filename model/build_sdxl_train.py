"""v9 的第三个数据来源：SDXL 按 B2 管线出的像素瓦片（训练集材质名，每名 1 张，16/32px）。

为什么（`docs/arch_progress.md`）：TRD 学画师，CLIP 停在 34.7（真人瓦片 34.1），B2 35.3——CLIP 奖励"字面画出名字"，
画师画的是"材质"（Minetest 的栅栏贴图是木板，画师画木板，SDXL 画栅栏）。挑图能追平 CLIP 但要牺牲分布指标。
这里把 SDXL 的字面描绘作为**带来源标记的训练数据**蒸馏进 TRD（来源 2 = sdxl），推理时用"来源引导"
（`trd.sample(dom_w=…)`：l = l_画师 + w·(l_sdxl − l_画师)）在画师画风与字面描绘之间调；推理不再需要 SDXL。

只渲染**训练包里的材质名**（`eval/prompts.py --train` → `T_all`），与评测集的真人瓦片无关。
输入：`baselines/sdxl_baselines.py --set T_all --n 1 --sizes 16 32 --out experiments/sdxl_train` 的 B2 瓦片。
输出：`data/tiles/train_sdxl.json`（样本格式同 dataset_k16，pack = "sdxl@gen"）。
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "dataset"))
sys.path.insert(0, str(ROOT / "eval"))
from prepare import MIN_COLORS, quantize     # noqa: E402
from prompts import load_set                 # noqa: E402


def main():
    items, _ = load_set("T_all")
    out, miss = [], 0
    for e in items:
        slug = e["material"].rsplit(".", 1)[0]
        for size in (16, 32):
            f = ROOT / f"experiments/sdxl_train/B2/{size}/{slug}.png"
            if not f.exists():
                miss += 1
                continue
            a = np.asarray(Image.open(f).convert("RGB"))
            native = len(np.unique(a.reshape(-1, 3), axis=0))
            if native < MIN_COLORS:
                continue
            ind, pal = quantize(a, 16)
            out.append({"material": e["material"], "pack": "sdxl@gen", "size": size, "split": "train",
                        "native_colors": native, "k_used": int(len(pal)),
                        "idx": ind.astype(np.uint8).tobytes().hex(), "palette": pal.tolist(), "extra": True})
    p = ROOT / "data/tiles/train_sdxl.json"
    p.write_text(json.dumps({"k": 16, "samples": out}))
    print(f"SDXL 来源样本 {len(out)}（缺 {miss}）-> {p}")


if __name__ == "__main__":
    main()
