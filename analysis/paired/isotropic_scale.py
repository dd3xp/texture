"""各向同性材质有没有同样的「单元惯例」？——先测量，再决定要不要做方法。

动机（论文限制节）：双条件门拒绝的那一半材质，基线输出接近平涂
（调色板亮度跨度中位 0.071 vs 触发组 0.262，精确置换 p=0.0018）。
那半边是更难的一半，而我们什么都没给。

`auto_crop` 的注释说颗粒材质「按特征尺度折算…在图上都不如不裁」——
那是**目视判断**，按本项目的标准不算结论；而且当时用的
`correlation_length` 走行/列**廓线**，对各向同性内容会把纹理平均掉，
量的根本不是斑块尺度。这里先把估计量换成径向的，再照 §5.2 的做法
量真人惯例：如果真人在各向同性材质上也保持大致恒定的「每图特征数」，
同一套机制就该迁移过去；如果不保持，这条路不该做。

纯 CPU，只用真人瓦片，不需要生成。
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import dominant_period, anisotropy, correlation_length  # noqa: E402

W = np.array([0.299, 0.587, 0.114])

# 与 aniso_gate.py 同一批各向同性类别，清单写死以便复核
ISOTROPIC = ["default_tree_top.png", "default_jungletree_top.png",
             "default_diamond_block.png", "default_acacia_tree_top.png",
             "default_pine_tree_top.png", "default_gold_block.png",
             "default_gravel.png", "default_stone_block.png"]


def radial_corr_length(img: np.ndarray, thresh: float = 0.5) -> float:
    """二维自相关的方位平均，首次跌破 `thresh` 的半径 = 特征尺度。

    与 `downsample.correlation_length` 的区别：那个先把图压成行/列廓线
    再做一维自相关。对**有方向性**的结构（砖缝、木纹）廓线保留了信号，
    但各向同性的斑块在廓线里互相抵消——量到的是噪声而不是斑块大小。
    这里在二维上做，再按半径平均，方向无关。
    """
    g = img @ W if img.ndim == 3 else img
    g = g - g.mean()
    if g.std() < 1e-6:
        return 0.0
    f = np.fft.rfft2(g)
    ac = np.fft.irfft2(f * np.conj(f), s=g.shape).real
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-9)
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.ogrid[:ac.shape[0], :ac.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = int(min(cy, cx))
    prof = [ac[(r >= k) & (r < k + 1)].mean() for k in range(rmax)]
    for k, v in enumerate(prof):
        if v < thresh:
            return float(k)
    return 0.0


def tiles(sizes=(16, 32, 64)):
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    for s in ds["samples"]:
        if s["size"] not in sizes:
            continue
        n = s["size"]
        pal = np.array(s["palette"], dtype=float)
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        yield s["material"], n, pal[idx].astype(float)


def main():
    by_size = {}
    iso_by_size = {}
    for mat, n, rgb in tiles():
        rl = radial_corr_length(rgb)
        old = correlation_length(rgb)
        ani = anisotropy(rgb)
        per = dominant_period(rgb, lo=2, hi_frac=0.625)
        # 只看门会拒绝的那一批：无周期，或各向异性不足
        declined = per <= 0 or ani < 0.20
        if declined and rl > 0:
            by_size.setdefault(n, []).append((rl, old))
        if mat in ISOTROPIC and rl > 0:
            iso_by_size.setdefault(n, []).append((rl, old))

    print("真人瓦片中**会被门拒绝**的那些，特征尺度与每图特征数")
    print(f"{'尺寸':>6}{'张数':>7}{'径向尺度中位':>14}{'每图特征数':>12}"
          f"{'（旧廓线口径）':>16}")
    print("-" * 60)
    for n in sorted(by_size):
        rl = [x for x, _ in by_size[n]]
        ol = [y for _, y in by_size[n]]
        m = statistics.median(rl)
        print(f"{n:>6}{len(rl):>7}{m:>14.2f}{n / m:>12.2f}"
              f"{statistics.median(ol):>16.2f}")

    print("\n手挑各向同性类别（清单见脚本顶部）")
    print(f"{'尺寸':>6}{'张数':>7}{'径向尺度中位':>14}{'每图特征数':>12}")
    print("-" * 44)
    for n in sorted(iso_by_size):
        rl = [x for x, _ in iso_by_size[n]]
        m = statistics.median(rl)
        print(f"{n:>6}{len(rl):>7}{m:>14.2f}{n / m:>12.2f}")

    print("\n判读（跑前写下）：")
    print("  若「每图特征数」在 16 与 32 上大致恒定（差 <25%）->")
    print("     各向同性材质也有单元惯例，同一机制可迁移，值得做方法；")
    print("  若不恒定 -> 真人在各向同性材质上按比例而非按数量设计，")
    print("     『裁到固定特征数』没有依据，这条路应当关闭。")


if __name__ == "__main__":
    main()
