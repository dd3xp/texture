"""真人材质包的高分源，结构尺度本来就合适吗？

B2 显示：用真人高分源时，平凡降采样基线已与真人手绘不可区分——**不需要裁剪**。
而 B5 显示：用 SDXL 渲染图时不裁剪就是一团糊。

若真人高分源的结构周期本来就落在"降到 16 像素后仍可表示"的范围内，
就解释了两者的差别，也把本方法的适用范围说清楚：
**它修的是源图尺度与目标网格不匹配，而这是 SDXL 的产物，不是降采样固有的。**

判据：设源图边长 S、目标边长 n，则源图上的周期 P 降采样后占 P·n/S 个输出像素。
要能表示，需 P·n/S ≥ 2（奈奎斯特）；要"看得出结构"，B5 用的目标是 ≈ n/4.5。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import dominant_period, anisotropy                 # noqa: E402


def main():
    pairs = json.loads(Path("data/contentdb/pairs.json").read_text())
    n = 16
    rows = []
    for m, v in sorted(pairs.items()):
        for p in list(v.get("high", {}).values())[:1]:
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            a = np.asarray(im).astype(float)
            S = min(a.shape[:2])
            per = dominant_period(a, lo=2)
            if per <= 0 or anisotropy(a) < 0.20:
                continue
            rows.append((m, S, per, per * n / S))
            break
    if not rows:
        print("没有可用高分源"); return
    q = np.array([r[3] for r in rows])
    print(f"真人高分源 {len(rows)} 个（检出周期且有方向性），目标 {n}x{n}")
    print(f"  源图边长中位 {np.median([r[1] for r in rows]):.0f} px")
    print(f"  源图周期中位 {np.median([r[2] for r in rows]):.1f} px")
    print(f"  **降采样后周期占 {np.median(q):.2f} 个输出像素**（四分位 "
          f"[{np.percentile(q,25):.2f}, {np.percentile(q,75):.2f}]）")
    print(f"\n  低于奈奎斯特下限 2 像素的比例: {(q < 2).mean():.0%}")
    print(f"  低于 B5 目标 {n/4.5:.1f} 像素的比例: {(q < n/4.5).mean():.0%}")
    print("\n对照：SDXL 1024 渲染图，苔藓砖周期约 47–61px")
    for per in (47, 61):
        print(f"    周期 {per}px -> 降到 16 后占 {per*16/1024:.2f} 像素"
              f"（{'低于奈奎斯特' if per*16/1024 < 2 else '可表示'}）")
    print("\n判读：真人源若多数已达标而 SDXL 远低于下限 ->")
    print("      裁剪修的是**源图尺度不匹配**，是 SDXL 的产物，非降采样固有缺陷。")


if __name__ == "__main__":
    main()
