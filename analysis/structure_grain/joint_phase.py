"""诊断竖缝：周期是否跨作者一致、相位是否跨作者发散。

A3l 记录的失败是"竖缝相位跨作者不一致，平均后抵消"。
但那个诊断把周期和相位混在一起了。假设是：
**砖宽（周期）是材质属性，相位是每个作者的自由选择。**
若成立，正确做法不是平均，而是估周期、生成时采样相位。
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from structural_prior import learn_prior                    # noqa: E402


def col_period(seg: np.ndarray, min_std=0.15, min_score=0.20):
    """单个砖层内的竖缝周期与相位。返回 (周期, 相位, 得分)。"""
    c = seg.mean(0)
    c = c - c.mean()
    if c.std() < min_std:
        return None, None, 0.0
    ac = np.correlate(c, c, "full")[len(c) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag, sc = max(((l, ac[l]) for l in range(3, 9)), key=lambda t: t[1])
    if sc < min_score:
        return None, None, float(sc)
    ph = int(np.argmin([c[i::lag].mean() for i in range(lag)]))
    return lag, ph, float(sc)


def bands_of(rows, size=16):
    out, prev = [], 0
    for r in rows + ([size] if rows and rows[-1] != size - 1 else []):
        if r > prev:
            out.append((prev, r))
        prev = r + 1
    return out


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16 and s["split"] == "train":
            bymat.setdefault(s["material"], []).append(s)

    mats = sys.argv[1:] or [m for m in bymat if "brick" in m]
    agree_lag, agree_ph, n_band = 0, 0, 0
    for m in sorted(mats):
        if m not in bymat or len(bymat[m]) < 4:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in bymat[m]]
        pr = learn_prior(raw, [len(s["palette"]) for s in bymat[m]])
        if not pr["rows"]:
            continue
        cur = {b: cols for b, (_, _, cols) in pr["joints"].items()}
        print(f"=== {m}  横缝{pr['rows']}  现行平均法竖缝={cur}")
        for bi, (y0, y1) in enumerate(bands_of(pr["rows"])):
            res = [col_period(t[y0:y1].astype(float)) for t in raw]
            lags = [r[0] for r in res if r[0]]
            phs = [r[1] for r in res if r[0]]
            if len(lags) < 3:
                print(f"   层{bi} y{y0}-{y1}: 仅 {len(lags)}/{len(raw)} 作者检出，跳过")
                continue
            n_band += 1
            cl, cp = Counter(lags).most_common(), Counter(phs).most_common()
            fl, fp = cl[0][1] / len(lags), cp[0][1] / len(phs)
            agree_lag += fl >= 0.6
            agree_ph += fp >= 0.6
            print(f"   层{bi} y{y0}-{y1}: {len(lags)}/{len(raw)} 检出  "
                  f"周期{cl} 众数占{fl:.0%}  相位{cp} 众数占{fp:.0%}")
    if n_band:
        print(f"\n合计 {n_band} 个砖层：")
        print(f"  周期众数占比≥60% 的层: {agree_lag} ({agree_lag/n_band:.0%})")
        print(f"  相位众数占比≥60% 的层: {agree_ph} ({agree_ph/n_band:.0%})")
        print("  判读：周期一致率显著高于相位 -> 假设成立，应估周期+采样相位")


def _cli():
    if "--bond" in sys.argv:
        for m in [a for a in sys.argv[1:] if a != "--bond"]:
            running_bond_test(m)
    else:
        main()


def running_bond_test(mat: str) -> None:
    """区分两种可能：
    (a) 错缝砌法——每个作者选一个全局相位，相邻层固定偏移半个周期；
    (b) 每层相位各自随机。
    判据：同一作者相邻层的相位差，若集中在 period/2 就是 (a)。
    这决定生成时是"采一个相位再交替偏移"还是"每层独立采"。
    """
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    ss = [s for s in ds["samples"]
          if s["material"] == mat and s["size"] == 16 and s["split"] == "train"]
    raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16) for s in ss]
    pr = learn_prior(raw, [len(s["palette"]) for s in ss])
    bands = bands_of(pr["rows"])
    deltas, per_author = [], 0
    for t in raw:
        seq = [col_period(t[y0:y1].astype(float)) for y0, y1 in bands]
        got = [(i, p, ph) for i, (p, ph, _) in enumerate(seq) if p == 8]
        if len(got) < 2:
            continue
        per_author += 1
        for (i1, _, a), (i2, _, b) in zip(got, got[1:]):
            if i2 - i1 == 1:                       # 只看真正相邻的层
                deltas.append((b - a) % 8)
    print(f"\n=== {mat} 错缝检验：{per_author}/{len(raw)} 作者有≥2层周期为8")
    if not deltas:
        print("   相邻层配对不足")
        return
    c = Counter(deltas).most_common()
    half = sum(n for d, n in c if d in (3, 4, 5)) / len(deltas)
    zero = sum(n for d, n in c if d in (0, 1, 7)) / len(deltas)
    print(f"   相邻层相位差分布 {c}  (n={len(deltas)})")
    print(f"   差≈半周期(3-5): {half:.0%}   差≈0(0,1,7): {zero:.0%}")
    print("   判读：半周期占优 -> 错缝，采一个全局相位后交替偏移；"
          "否则每层独立采")


if __name__ == "__main__":
    _cli()
