"""(M55) 可行性计数：有多少真人瓦片可以**无重采样、无新颜色、严格可平铺**地重铺成 24x24。

为什么问（承 (M54) 第二节）：24px 是零样本尺寸（`train_trd.py:355` 只有 `32 in a.sizes`
一条分支），而 (M53) 量到 24px 在全数据集上真人瓦片 = 0 张。要给 24px 加监督就必须造样本，
而 (M54) 第二节第 3 问写死："24x24 裁自 32x32 不可平铺，会不会把 24px 教坏？需要先有判据。"

本脚本给的是**绕开那一问**的构造：`--p_tile16` 的先例是"16px 瓦片 2x2 平铺 -> 构造上合法的
32px 可平铺纹理（周期 16）"。同一招在 24 上要求源瓦片的**精确周期 p 整除 24**；
又因为源瓦片本身是 16px / 32px 可平铺的，p 必整除 16 或 32 ->
**p 必须整除 gcd(16,24)=8 或 gcd(32,24)=8，即 p in {1,2,4,8}**。
这类瓦片重铺到 24x24 是**逐格精确**的：秩索引/调色板/色数/材质名全部不变，只是网格变大，
且新瓦片严格以 24 为周期可平铺 -> 不引入新数据来源、不裁剪、不缩放、不产生新颜色。

⛔ 本脚本**不下任何判决**、不训练、不占 GPU、零 API。它只回答一个事实问题：
**这么造能造出多少张、覆盖多少材质**，用来决定 (M55) 要不要开这条线。
产出张数若太少，这条线本身就不成立（判据写进预注册，不在这里定）。

用法：
    python analysis/arch/m55_period24_yield.py            # 只用 dataset_k16.json
    python analysis/arch/m55_period24_yield.py --extra train_extra_packs_only.json
    python analysis/arch/m55_period24_yield.py --selftest
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from tiles_data import load            # noqa: E402

OK_P = (1, 2, 4, 8)                    # 整除 24 且整除 16/32 的周期


def min_period(g: np.ndarray, axis: int) -> int:
    """网格沿 axis 的**最小精确周期**（环面意义：g 平移 p 格后逐格相同）。总能整除 n。"""
    n = g.shape[axis]
    for p in range(1, n + 1):
        if n % p:
            continue
        if np.array_equal(g, np.roll(g, p, axis=axis)):
            return p
    return n


def retile24(g: np.ndarray) -> np.ndarray:
    """周期整除 8 的网格 -> 24x24：取左上 8x8 的周期块，3x3 平铺。"""
    py, px = min_period(g, 0), min_period(g, 1)
    assert py in OK_P and px in OK_P, (py, px)
    blk = g[:8, :8]                                  # 周期整除 8 => 左上 8x8 即一个完整周期块
    return np.tile(blk, (3, 3))


def scan(rows):
    out = []
    for s in rows:
        g = s["idx"]
        py, px = min_period(g, 0), min_period(g, 1)
        out.append((s, py, px))
    return out


def report(name, rows):
    sc = scan(rows)
    good = [(s, py, px) for s, py, px in sc if py in OK_P and px in OK_P]
    # 纯平涂（k_used<2 或周期 1x1）对结构监督没有信息量，单独报
    flat = [t for t in good if t[1] == 1 and t[2] == 1]
    rich = [t for t in good if not (t[1] == 1 and t[2] == 1)]
    print(f"\n=== {name} ===")
    print(f"  总瓦片 {len(rows)}")
    print(f"  可精确重铺成 24px 的 {len(good)} 张（{len(good)/max(1,len(rows)):.1%}）"
          f"，其中纯平涂 {len(flat)}、有结构 {len(rich)}")
    print(f"  有结构那批：材质 {len({t[0]['material'] for t in rich})} 个、"
          f"包 {len({t[0]['pack'] for t in rich})} 个、k_used 中位数 "
          f"{np.median([t[0]['k_used'] for t in rich]) if rich else float('nan')}")
    print("  周期分布 (py,px) 前 8：",
          Counter((py, px) for _, py, px in sc).most_common(8))
    if rich:
        print("  有结构样例（材质/周期/色数）：",
              [(t[0]["material"], (t[1], t[2]), t[0]["k_used"]) for t in rich[:6]])
    return good, rich


def selftest() -> int:
    bad = 0
    rng = np.random.default_rng(0)
    # 1) 构造周期 4 的 16x16：min_period 必须报 4，retile24 必须给出 24x24 且以 4 为周期
    blk = rng.integers(0, 5, (4, 4))
    g = np.tile(blk, (4, 4))
    if min_period(g, 0) != 4 or min_period(g, 1) != 4:
        bad += 1
        print("【禁】周期 4 的网格没被认出来")
    t = retile24(g)
    if t.shape != (24, 24) or not np.array_equal(t, np.roll(t, 4, axis=0)) \
            or not np.array_equal(t, np.roll(t, 4, axis=1)):
        bad += 1
        print("【禁】retile24 的产物不是 24x24 或不以 4 为周期")
    # 2) 重铺必须**内容不变**：24 网格的每一格都等于源网格对应模位置
    ii, jj = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    if not np.array_equal(t, g[ii % 16, jj % 16]):
        bad += 1
        print("【禁】重铺改变了内容")
    # 3) 识别检验：无周期的随机网格必须报 16、且**不**被判为可重铺
    r = rng.integers(0, 9, (16, 16))
    if min_period(r, 0) != 16 or min_period(r, 1) != 16:
        bad += 1
        print("【禁】随机网格被误判为有小周期")
    # 4) 各向异性：py=8, px=16 -> 不合格（只有一轴能整除 8 不够）
    a = np.tile(rng.integers(0, 4, (8, 16)), (2, 1))
    if min_period(a, 0) != 8 or min_period(a, 1) != 16:
        bad += 1
        print("【禁】各向异性周期识别错")
    print(f"自检完成，失败 {bad} 项")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=None, help="data/tiles/ 下的额外训练文件名（远程才有）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    ext = a.extra if a.extra else False
    for size in (16, 32):
        rows = load(size, "train", extra=ext)
        report(f"size={size} split=train extra={a.extra}", rows)
    # val 也数一下（只为知道上限，⛔ 不训练它）
    for size in (16, 32):
        report(f"size={size} split=val", load(size, "val"))


if __name__ == "__main__":
    main()
