"""把 B2@16 最近邻放大 2x，做成 `B2up16/32/`——"同内容、但没拿到额外输出像素"的 32px 对照。

零 GPU、零 API。产物 272 张 32x32 PNG，合计约 100KB（`/mnt/data` 只剩几 G，这个量安全）。

为什么这样就等于"把画布增益拿掉"：`analysis/arch/b2_nyquist.py` 的 (O1) 实跑证明
`auto_crop` 的裁剪边长与目标尺寸**无关**（`target_px = size/UNITS_PER_TILE` 把 size 约掉），
所以 B2@16 与 B2@32 是**同一块源区域**的两次不同密度采样。把 16px 那张最近邻放大回 32px，
内容一个字不变，只是每个结构周期仍只有 3.56 个有效像素（B2@32 是 7.11）。

⚠ 口径（写在前面）：`judge_pairs.py` 的 `panel()` 把两张都 NEAREST 放到 192x192，
所以 up2(B2@16) 在判官眼里与 B2@16 **逐像素相同**。本对照的作用是让两条臂有不同的目录名与
真实的 32x32 文件，不是为了改变判官看到的东西。
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BASE = os.path.join(ROOT, "experiments", "baselines")
SRC = os.path.join(BASE, "B2", "16")
DST = os.path.join(BASE, "B2up16", "32")


def main(argv) -> int:
    # 可选 `make_up2.py <源方法>/<尺寸> <目标方法>/<尺寸>`；不给参数则与 `557a50e` 逐字节同行为。
    global SRC, DST
    if len(argv) == 2:
        SRC, DST = (os.path.join(BASE, *p.split("/")) for p in argv)
    elif argv:
        print("[make_up2] 用法：make_up2.py [<源方法>/<尺寸> <目标方法>/<尺寸>]")
        return 1
    if not os.path.isdir(SRC):
        print(f"[make_up2] 缺 {SRC}")
        return 1
    # 尺寸从目录名取，并要求目标恰是源的 2 倍——放大倍率写死在路径里，不另给一个可以错配的参数
    s_src, s_dst = int(os.path.basename(SRC)), int(os.path.basename(DST))
    if s_dst != 2 * s_src:
        print(f"[make_up2] {s_dst} 不是 {s_src} 的 2 倍")
        return 1
    os.makedirs(DST, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(SRC)):
        if not name.endswith(".png"):
            continue
        a = np.asarray(Image.open(os.path.join(SRC, name)).convert("RGB"))
        if a.shape[:2] != (s_src, s_src):
            print(f"[make_up2] {name} 不是 {s_src}x{s_src}：{a.shape}")
            return 1
        up = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
        # 操作检验 (8)：往返必须逐像素相等，否则放大不是无损的
        if not np.array_equal(up[::2, ::2], a):
            print(f"[make_up2] 往返断言失败：{name}")
            return 1
        Image.fromarray(up).save(os.path.join(DST, name))
        n += 1
    print(f"[make_up2] {n} 张 -> {DST}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
