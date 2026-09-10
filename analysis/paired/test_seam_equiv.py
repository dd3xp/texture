"""生产实现与实验实现必须给出同一个窗口。

`analysis/paired/seam_crop.py` 是**跑时原样**的记录，不能改；
`tools/downsample.py` 里的 `seam_offset` 是收进交付管线的那一份。
两份代码同时存在就会走偏，本测试把它钉住：**同图同 side 必须选中同一个位置**。

（一处已知的写法差异：实验用 `statistics.median`、生产用 `np.median`。
 dv+dh 恒为 30 个元素、偶数，两者都取中间两个的平均，结果相同——
 这条测试同时是那个等价性的检查。）

跑法：`python analysis/paired/test_seam_equiv.py`（纯 CPU，不需要 GPU）
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/paired"):
    sys.path.insert(0, str(ROOT / sub))
from downsample import seam_offset, seam_stats as prod_stats   # noqa: E402
from seam_crop import best_offset, seam_stats as exp_stats     # noqa: E402


def cases():
    rng = np.random.default_rng(3)
    H = W = 1024
    yy, xx = np.mgrid[0:H, 0:W]
    # 1) 条纹周期图
    per = 64
    a = np.zeros((H, W, 3))
    a[..., 0] = ((xx // (per // 2)) % 2) * 200 + 30
    a[..., 1] = ((yy // (per // 2)) % 2) * 180 + 40
    a[..., 2] = 120
    yield "周期条纹", a, int(per * 4.5)
    # 2) 周期图 + 一块纯平区（平坦守卫必须把它挡掉）
    b = a.copy()
    b[600:, 600:] = 128.0
    yield "周期+平坦区", b, int(per * 4.5)
    # 3) 纯噪声（无周期，守卫下限几乎人人满足）
    yield "噪声", rng.integers(0, 255, (H, W, 3)).astype(float), 288
    # 4) 小窗口（候选最多的情形）
    yield "小窗口", a, 135


def main():
    bad = 0
    for name, img, side in cases():
        py, px = seam_offset(img, side, 16)
        ey, ex, (n_all, n_adm) = best_offset(img, side, 16, 12)
        ok = (py, px) == (ey, ex)
        s1 = prod_stats(np.asarray(img[py:py + side, px:px + side][:16, :16], float))
        s2 = exp_stats(np.asarray(img[py:py + side, px:px + side][:16, :16], float))
        same_stat = np.allclose(s1, s2)
        print(f"{name:<12} side={side:4d} 生产({py},{px})  实验({ey},{ex})  "
              f"候选{n_adm}/{n_all}  位置{'一致' if ok else '**不一致**'}  "
              f"统计{'一致' if same_stat else '**不一致**'}")
        bad += (not ok) or (not same_stat)
    if bad:
        raise SystemExit(f"{bad} 个用例不一致——两份实现已走偏，必须同步")
    print("全部一致")


if __name__ == "__main__":
    main()
