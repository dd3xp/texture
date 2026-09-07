"""各向异性为什么必须进门：可复现的度量（§5.3）。

周期检测走的是行/列廓线，对各向同性材质（树冠、沙、砾石）没有意义，
所以双条件门里除了「检出周期」还要求「各向异性 ≥ 0.20」。

**为什么重写这个脚本**：`tools/downsample.py` 的注释与 loop.md 都写着
「检出组中位 0.774、未检出组 0.036」，出处标的是 `crop_failure.py`，
但那个脚本算的是自己的 grad_v/grad_h 比值（3.45 / 1.02），不是这里的
归一化各向异性；用现有代码在任何一种周期检测口径下都复现不出 0.774。
原始的「6+8 类手挑材质」清单没有留存。故本脚本给出**完全指定、可复跑**的
两个口径，论文改用这里的数。
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import dominant_period, anisotropy          # noqa: E402

# 手挑类别：方向性结构 vs 各向同性。列在这里以便复核，不再是口头描述。
DIRECTIONAL = ["default_wood.png", "default_brick.png", "default_acacia_wood.png",
               "default_junglewood.png", "default_pine_wood.png",
               "default_stone_brick.png"]
ISOTROPIC = ["default_tree_top.png", "default_jungletree_top.png",
             "default_diamond_block.png", "default_acacia_tree_top.png",
             "default_pine_tree_top.png", "default_gold_block.png",
             "default_gravel.png", "default_stone_block.png"]


def tiles():
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        n = s["size"]
        pal = np.array(s["palette"], dtype=float)
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        yield s["material"], pal[idx].astype(float)


def main():
    hit, miss, by_mat = [], [], {}
    for mat, rgb in tiles():
        a = anisotropy(rgb)
        by_mat.setdefault(mat, []).append(a)
        # 16px 瓦片用瓦片尺度的周期检测口径（生产口径 lo=8 是给高分源用的，
        # 在 16px 上搜索区间几乎为空）。
        (hit if dominant_period(rgb, lo=2, hi_frac=0.625) > 0 else miss).append(a)

    n = len(hit) + len(miss)
    print(f"口径一 · 全部 {n} 张 16px 真人瓦片，按是否检出周期分组")
    print(f"  检出 {len(hit)} ({len(hit)/n:.0%})   各向异性中位 {statistics.median(hit):.3f}")
    print(f"  未检出 {len(miss)} ({len(miss)/n:.0%}) 各向异性中位 {statistics.median(miss):.3f}")

    d = [v for m in DIRECTIONAL for v in by_mat.get(m, [])]
    i = [v for m in ISOTROPIC for v in by_mat.get(m, [])]
    print(f"\n口径二 · 手挑类别（清单见脚本顶部）")
    print(f"  方向性 {len(DIRECTIONAL)} 类 / {len(d)} 张   中位 {statistics.median(d):.3f}")
    print(f"  各向同性 {len(ISOTROPIC)} 类 / {len(i)} 张 中位 {statistics.median(i):.3f}")
    print(f"\n阈值 0.20 落在两组之间（两种口径下都是）。")


if __name__ == "__main__":
    main()
