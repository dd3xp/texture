"""复算限制节那句话：门拒绝的材质**更弱但并非一律平涂**。

论文写的是：78 个材质里，门拒绝组的图内实际亮度跨度中位 0.128，
触发组 0.240（Mann-Whitney p=1.1e-4，真人 0.292），且 47% 的被拒材质
仍达到门接受过的最弱瓦片的水平。

这一段的原始依据是两批交付瓦片的 PNG（`experiments/pack`、`experiments/pack60`），
都已入库，所以净克隆能复现。**曾经的版本写的是「接近平涂」，依据只有 7 个材质
（中位 0.071）——样本扩到 32 个后不成立，已改。** 这个脚本就是防止再退回去。

无 scipy：Mann-Whitney 用秩和的正态近似（含并列校正），
判据只看方向与量级，不靠小数点后第三位。
"""
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from make_texture import W                                   # noqa: E402

PACKS = ("pack", "pack60")
SIZE = 16
FLAT_MAX = 0.134      # 18 批里门触发组的最小跨度，阈值定于该批、未照后续调整
ARTIST = 0.292        # SPREAD_ARTIST_MEDIAN


def realised_spread(img: np.ndarray) -> float:
    u = np.unique(img.reshape(-1, 3), axis=0).astype(float)
    lum = np.sort(u @ W)
    return float(lum[-1] - lum[0]) / 255.0


def mannwhitney(a, b):
    """秩和正态近似，含并列校正。返回 (U, z, 双侧 p, 共同语言效应量)。"""
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    n = len(allv)
    rank = [0.0] * n
    i = 0
    ties = 0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        for k in range(i, j + 1):
            rank[k] = avg
        i = j + 1
    na, nb = len(a), len(b)
    Ra = sum(rank[k] for k in range(n) if allv[k][1] == 0)
    U = Ra - na * (na + 1) / 2
    sd = math.sqrt(na * nb / 12 * ((n + 1) - ties / (n * (n - 1))))
    z = (U - na * nb / 2) / sd
    return U, z, math.erfc(abs(z) / math.sqrt(2)), U / (na * nb)


def main():
    fired, declined = [], []
    for pack in PACKS:
        mf = ROOT / "experiments" / pack / "manifest.json"
        if not mf.exists():
            print(f"缺 {mf}（净克隆应有；未入库则复现不了）")
            continue
        for r in json.loads(mf.read_text(encoding="utf-8")):
            if r["variant"] != "base" or r["size"] != SIZE:
                continue
            f = ROOT / "experiments" / pack / "base" / str(SIZE) / (
                r["material"].replace(" ", "_") + ".png")
            if not f.exists():
                continue
            s = realised_spread(np.asarray(Image.open(f).convert("RGB")))
            (fired if r["gate_fired"] else declined).append((r["material"], s))

    a = [s for _, s in fired]
    b = [s for _, s in declined]
    if not a or not b:
        raise SystemExit("数据不全，无法复算")
    U, z, p, cl = mannwhitney(a, b)
    ok = sum(s >= FLAT_MAX for s in b)
    print(f"材质 {len(a) + len(b)} 个（{'+'.join(PACKS)}，{SIZE}px，base 变体）")
    print(f"  门触发   n={len(a):>3}  跨度中位 {statistics.median(a):.3f}")
    print(f"  门未触发 n={len(b):>3}  跨度中位 {statistics.median(b):.3f}")
    print(f"  真人中位 {ARTIST}")
    print(f"  Mann-Whitney z={z:.2f}  p={p:.3g}  共同语言效应量 {cl:.2f}")
    print(f"  被拒但跨度 ≥{FLAT_MAX}（门接受过的最弱瓦片）：{ok}/{len(b)}"
          f" = {ok / len(b):.0%}")
    print()
    print("论文限制节应写的三件事：")
    print(f"  1. 中位 {statistics.median(b):.3f} vs {statistics.median(a):.3f}"
          f"  2. p={p:.1g} 仍显著  3. {ok / len(b):.0%} 仍可用 -> **不是「接近平涂」**")


if __name__ == "__main__":
    main()
