"""三档分辨率 16/24/32 下，按结构尺度裁剪的目标该取多少。

用户定义的任务是"纯色图 → 低分辨率像素画纹理"，分辨率取 16–32 之间几档。
B5 只在 16×16 上验过，且 `target_px=4`（一个结构单元占 4 个输出像素）
是照 Minecraft 砖材质"约 4 层"定的。

这里回答两件事：
1. 三档分辨率各自的 target_px 应该取多少；
2. 真人材质包在不同尺寸下，一个结构单元实际占几个像素——**从数据里量，不是猜**。

数据集里有 16/32/64 三种尺寸的真人瓦片，正好可以量出真人自己的取值。

**（2026-09-07）两种检测口径都要报**。本脚本原先把 `hi_frac` 写死成 0.5，
`UNITS_PER_TILE = 4.5` 就是那样量出来的；但项目后来把 `dominant_period` 的
默认值改成 **0.625**（0.5 正好等于边长一半，把"只有 2 层"的材质从定义上排除，
`default_stone_brick` 只检出 6/40），而 `auto_crop` 用的是这个默认值。
**推导常数的口径与使用它的口径因此不一致**，必须摆出来而不是只报一个。
"""

import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from downsample import dominant_period                             # noqa: E402


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    by = defaultdict(list)
    for s in ds["samples"]:
        pal = np.array(s["palette"], np.uint8)
        n = int(s["size"])
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        by[n].append((s["material"], idx, pal))

    stats = {}
    for hf, tag in [(0.625, "生产口径 hi_frac=0.625（auto_crop 实际用的）"),
                    (0.5, "旧口径 hi_frac=0.5（UNITS_PER_TILE=4.5 由此而来）")]:
        print(f"\n{tag}")
        print(f"{'尺寸':>6}{'瓦片数':>8}{'检出周期的':>12}{'周期中位':>10}"
              f"{'四分位':>16}{'每张图单元数':>14}")
        print("-" * 70)
        cur = {}
        for n in sorted(by):
            pers = []
            for m, idx, pal in by[n]:
                rgb = pal[idx].astype(float)
                p = dominant_period(rgb, lo=2, hi_frac=hf)
                if p > 0:
                    pers.append(p)
            if not pers:
                continue
            pers = np.array(pers)
            q1, q3 = np.percentile(pers, [25, 75])
            med = float(np.median(pers))
            cur[n] = med
            print(f"{n:>6}{len(by[n]):>8}{len(pers):>12}{med:>10.1f}"
                  f"{f'[{q1:.1f}, {q3:.1f}]':>16}{n/med:>14.2f}")
        if hf == 0.625:
            stats = cur

    print("\n判读：")
    print("  若「周期/边长」在三种尺寸下大致恒定 -> 真人是按**相对比例**设计的，")
    print("  那么 target_px 应随分辨率线性缩放（16→4 则 32→8）。")
    print("  若周期的**绝对像素数**恒定 -> 真人按固定粒度设计，target_px 不随分辨率变。")
    if len(stats) >= 2:
        ks = sorted(stats)
        print(f"\n  绝对周期: " + "  ".join(f"{k}px图={stats[k]:.1f}" for k in ks))
        print(f"  相对比例: " + "  ".join(f"{k}px图={stats[k]/k:.3f}" for k in ks))
        rel = [stats[k] / k for k in ks]
        ab = [stats[k] for k in ks]
        print(f"\n  相对比例的变异系数 {np.std(rel)/np.mean(rel):.3f}"
              f"   绝对周期的变异系数 {np.std(ab)/np.mean(ab):.3f}")
        print("  （变异系数小的那个才是恒定量）")


if __name__ == "__main__":
    main()
