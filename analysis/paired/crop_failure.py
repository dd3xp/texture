"""裁剪何时不触发——方法的失败条件，必须写进论文的限制一节。

端到端演示里 `wooden planks floor`(1/1.0) 与 `sandstone block wall`(1/1.4)
明显比其他偏糊，而它们恰好是自动裁剪**没触发或几乎没裁**的两例。
这不是巧合，是 `dominant_period` 的失败模式。

在**真人瓦片**上量（不依赖 SDXL，样本量大、有基准真值）：
周期检测在哪些材质上失败，失败的图有什么共同特征。
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import dominant_period, correlation_length         # noqa: E402

W = np.array([0.299, 0.587, 0.114])


def feats(rgb: np.ndarray) -> dict:
    g = rgb @ W
    return {"lum_std": float(g.std()),
            "grad_h": float(np.abs(np.diff(g, axis=1)).mean()),
            "grad_v": float(np.abs(np.diff(g, axis=0)).mean()),
            "aniso": float(abs(np.abs(np.diff(g, axis=1)).mean()
                               - np.abs(np.diff(g, axis=0)).mean())),
            "corr_len": correlation_length(rgb)}


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    hit, miss = [], []
    for s in ds["samples"]:
        n = int(s["size"])
        if n != 16:
            continue
        pal = np.array(s["palette"], np.uint8)
        rgb = pal[np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)]
        rgb = rgb.astype(float)
        p = dominant_period(rgb, lo=2, hi_frac=0.5)
        (hit if p > 0 else miss).append((s["material"], feats(rgb)))

    print(f"真人 16x16 瓦片 {len(hit)+len(miss)} 张："
          f"检出周期 {len(hit)}（{len(hit)/(len(hit)+len(miss)):.0%}），"
          f"未检出 {len(miss)}")
    keys = ["lum_std", "grad_h", "grad_v", "aniso", "corr_len"]
    print(f"\n{'特征':<12}{'检出组中位':>12}{'未检出组中位':>14}{'比值':>8}")
    print("-" * 48)
    for k in keys:
        a = np.median([f[k] for _, f in hit])
        b = np.median([f[k] for _, f in miss])
        print(f"{k:<12}{a:>12.3f}{b:>14.3f}{(b/max(a,1e-9)):>8.2f}")

    print("\n未检出组里最常见的材质（前 12）：")
    for m, c in Counter(m for m, _ in miss).most_common(12):
        print(f"  {m:<40}{c}")
    print("\n检出组里最常见的材质（前 8）：")
    for m, c in Counter(m for m, _ in hit).most_common(8):
        print(f"  {m:<40}{c}")

    print("\n判读：若未检出组的对比度/梯度明显更低 -> 失败条件是"
          "「结构对比不足」，可写成方法的适用范围；")
    print("      若两组特征接近 -> 失败与图像统计无关，是检测器本身的问题。")


if __name__ == "__main__":
    main()
