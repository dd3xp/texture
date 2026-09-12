"""32px 缺口的第一刀：TRD 在大画布上画的结构，尺度对不对？

**为什么问这个**（`docs/arch_progress.md` 2026-09-12 03:30 那节的下一步 1）：
正式测试判官封盘的结论是 16px 胜、24px 平、**32px 显著输 B2（41%，p=0.014）**，
而 `TRD32_rr4` 已经带了重排 → 32px 的问题不在 CLIP-FD 那条汇率曲线上。
库里还躺着一条没人读的记录：`judge_pilot_REALval_vs_B2val_32_V_mat.json` —
**真人 32px 瓦片对 B2 的可解率只有 53%（地板 50%），试点没过门**，
而同一口径下真人 16px 对 B2 可解 80%、并以 66% 胜（p=0.003）。
即：32px 上判官分不出真人和 B2，却分得出 TRD 和 B2（可解 73–87%）并判 TRD 输。
→ **TRD 在 32px 上有一个看得见的毛病**，不是"风格不同"。

**本脚本的假设（先注册，后跑）**：TRD 的训练数据全是 16x16，24/32 是把 16px 的配置
直接搬上去的零样本。若模型没有尺度概念，它会在 32 的画布上画**16px 那种粒度**的结构 ——
即结构主周期（像素）不随画布放大，于是每张瓦片里的结构单元数翻倍，
正好回到本项目诊断大模型失败时的那个病（真人约 3.2 单元/瓦片，大模型约 25）。

**判读规则（跑之前写死）**，尺子 = `tools/downsample.py` 的 `dominant_period`
（lo=2、hi_frac=0.625，与 `analysis/premise/dither.py` / `eval/diag_structure.py` 同口径）：

1. 主读数 = **主周期中位数（像素）**，只在过门的瓦片上算（过门 = 检出周期 且 各向异性 >= 0.20）。
   真人是尺度的定义：真人 32px 的周期中位数应当明显大于真人 16px 的（画布翻倍、单元数不变）。
2. **判"尺度漂移"成立**的条件：TRD 的 `period32 / period16` < 1.3，**同时**真人的该比值 >= 1.5。
   两条都满足才算 —— 只看 TRD 一侧无法排除"这把尺子在 32px 上本来就给小数"。
3. 若尺度漂移不成立，则记为不成立，另看单元数与各向异性、并把 B2 作为第三条参照线
   （B2 是判官在 32px 上认可的那一方，它的数落在哪边是最直接的线索）。
4. 组间差异一律报 Mann-Whitney 双侧 p（无 scipy 实现，抄自 `analysis/paired/units_residual.py`）。

⚠ 尺子的分辨率：16px 上可取周期是整数 2..9（单元数 8, 5.3, 4, 3.2, ...），
32px 上是 2..19。**32px 能测到的单元数上限（16）比 16px（8）高一倍**，
所以"单元数"跨分辨率不可直接比，**跨分辨率只比周期（像素）**。

零 GPU、零 API、零远程磁盘写入：只读已生成的瓦片 PNG。

    python analysis/arch/scale_diag.py --root /tmp/t32
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import dominant_period, anisotropy   # noqa: E402

GATE_ANISO = 0.20
IDX = re.compile(r"_(\d+)$")


def material(stem):
    return IDX.sub("", stem)


def mannwhitney_u_p(a, b):
    """双侧 Mann-Whitney U 的正态近似 p（含并列校正）。无 scipy 依赖。"""
    a, b = list(a), list(b)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    allv = sorted(a + b)
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[allv[i]] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(ranks[v] for v in a)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    counts = {}
    for v in allv:
        counts[v] = counts.get(v, 0) + 1
    n = n1 + n2
    tie = sum(c ** 3 - c for c in counts.values())
    var = n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return u1, 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    return u1, min(1.0, math.erfc(z / math.sqrt(2)))


def measure(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    n = rgb.shape[0]
    per = dominant_period(rgb, lo=2, hi_frac=0.625)
    ani = anisotropy(rgb)
    ncol = len(np.unique(rgb.reshape(-1, 3), axis=0))
    return {"material": material(path.stem), "n": n, "period": per, "aniso": ani,
            "gated": bool(per > 0 and ani >= GATE_ANISO),
            "units": (n / per) if per > 0 else None, "ncolours": ncol}


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="含 <dir>/<size>/ *.png 的目录")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/scale_diag_32.json")
    a = ap.parse_args()

    groups = {}
    for d in sorted(a.root.iterdir()):
        if not d.is_dir():
            continue
        for sz in sorted(d.iterdir()):
            if not sz.is_dir():
                continue
            files = sorted(sz.glob("*.png"))
            if not files:
                continue
            groups[f"{d.name}/{sz.name}"] = [measure(f) for f in files]

    out = {"gate_aniso": GATE_ANISO, "groups": {}}
    print(f"{'组':<22} {'n':>4} {'过门':>6} {'周期(px)':>9} {'单元数':>7} {'各向异性':>8} {'色数':>5}")
    for name, recs in groups.items():
        g = [r for r in recs if r["gated"]]
        row = {"n": len(recs), "n_gated": len(g), "gate_rate": len(g) / len(recs),
               "median_period_gated": med([r["period"] for r in g]),
               "median_units_gated": med([r["units"] for r in g]),
               "median_aniso_all": med([r["aniso"] for r in recs]),
               "median_ncolours": med([r["ncolours"] for r in recs]),
               "periods_gated": [r["period"] for r in g],
               "materials_gated": [r["material"] for r in g]}
        out["groups"][name] = row
        print(f"{name:<22} {row['n']:>4} {row['gate_rate']:>6.0%} "
              f"{row['median_period_gated']:>9.2f} {row['median_units_gated']:>7.2f} "
              f"{row['median_aniso_all']:>8.3f} {row['median_ncolours']:>5.0f}")

    def ratio(src):
        k16, k32 = f"{src}/16", f"{src}/32"
        if k16 not in out["groups"] or k32 not in out["groups"]:
            return None
        p16 = out["groups"][k16]["median_period_gated"]
        p32 = out["groups"][k32]["median_period_gated"]
        _, p = mannwhitney_u_p(out["groups"][k32]["periods_gated"],
                               out["groups"][k16]["periods_gated"])
        return {"period16": p16, "period32": p32, "ratio": p32 / p16, "mw_p": p}

    print("\n跨分辨率的周期比（判据 2：TRD < 1.3 且 真人 >= 1.5 才算尺度漂移）")
    out["scale_ratio"] = {}
    for src in sorted({k.split("/")[0] for k in out["groups"]}):
        r = ratio(src)
        if r:
            out["scale_ratio"][src] = r
            print(f"  {src:<16} 16px {r['period16']:.2f} -> 32px {r['period32']:.2f}  "
                  f"比值 {r['ratio']:.2f}  MW p={r['mw_p']:.3g}")

    print("\n32px 上三条线互比（周期，MW 双侧）")
    out["cross32"] = {}
    keys32 = [k for k in out["groups"] if k.endswith("/32")]
    for i, ka in enumerate(keys32):
        for kb in keys32[i + 1:]:
            _, p = mannwhitney_u_p(out["groups"][ka]["periods_gated"],
                                   out["groups"][kb]["periods_gated"])
            out["cross32"][f"{ka} vs {kb}"] = p
            print(f"  {ka:<20} vs {kb:<20} p={p:.3g}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n->", a.out)


if __name__ == "__main__":
    main()
