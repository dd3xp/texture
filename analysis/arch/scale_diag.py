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

    # 跨分辨率的配对：同一条线的 16px 组与 32px 组（名字不同，显式列出）
    PAIRS = [("REALval", "REALval/16", "REALval/32"),
             ("B2val", "B2val/16", "B2val/32"),
             ("TRD(val)", "nf8_xpal/16", "v11dx_direct/32"),
             ("TRD(test)", "TRD16/16", "TRD32_rr4/32")]

    print("\n跨分辨率的周期比（判据 2：TRD < 1.3 且 真人 >= 1.5 才算尺度漂移）")
    out["scale_ratio"] = {}
    for src, k16, k32 in PAIRS:
        if k16 not in out["groups"] or k32 not in out["groups"]:
            continue
        p16 = out["groups"][k16]["median_period_gated"]
        p32 = out["groups"][k32]["median_period_gated"]
        _, p = mannwhitney_u_p(out["groups"][k32]["periods_gated"],
                               out["groups"][k16]["periods_gated"])
        out["scale_ratio"][src] = {"period16": p16, "period32": p32,
                                   "ratio": p32 / p16, "mw_p": p}
        print(f"  {src:<10} {k16:<14} {p16:.2f} -> {k32:<18} {p32:.2f}  "
              f"比值 {p32 / p16:.2f}  MW p={p:.3g}")

    print("\n32px 上三条线互比（周期，MW 双侧）")
    out["cross32"] = {}
    keys32 = [k for k in out["groups"] if k.endswith("/32")]
    for i, ka in enumerate(keys32):
        for kb in keys32[i + 1:]:
            _, p = mannwhitney_u_p(out["groups"][ka]["periods_gated"],
                                   out["groups"][kb]["periods_gated"])
            out["cross32"][f"{ka} vs {kb}"] = p
            print(f"  {ka:<20} vs {kb:<20} p={p:.3g}")

    # ---- 材质配对（必需的对照）----------------------------------------------
    # REALval/32 只有 67 个材质，是 125 个验证材质的一个子集。若那 67 个恰好来自
    # 结构性更强的包，上面 32px 的横比就全是选材质选出来的。所以把每条线都**限制到
    # 同一批材质**再比一次；并且把真人自己的 16px 也限制到这 67 个，检验
    # "真人在大画布上确实画得更有结构" 是不是同一批材质内部的事实。
    base = sorted({r["material"] for r in groups["REALval/32"]})
    print(f"\n限制到 REALval/32 的 {len(base)} 个材质（同材质对照）")
    print(f"{'组':<22} {'n':>4} {'过门':>6} {'各向异性中位':>12} {'周期(px)':>9}")
    out["matched"] = {"materials": base, "rows": {}}
    order = ["REALval/16", "REALval/32", "B2val/16", "B2val/32",
             "nf8_xpal/16", "v11dx_direct/32"]
    for name in order + [k for k in groups if k not in order]:
        if name not in groups:
            continue
        sub = [r for r in groups[name] if r["material"] in base]
        if not sub:
            continue
        g = [r for r in sub if r["gated"]]
        row = {"n": len(sub), "n_gated": len(g), "gate_rate": len(g) / len(sub),
               "median_aniso": med([r["aniso"] for r in sub]),
               "median_period_gated": med([r["period"] for r in g]),
               "anisos": [r["aniso"] for r in sub]}
        out["matched"]["rows"][name] = row
        print(f"{name:<22} {row['n']:>4} {row['gate_rate']:>6.0%} "
              f"{row['median_aniso']:>12.3f} {row['median_period_gated']:>9.2f}")

    def two_prop(k1, n1, k2, n2):
        """两比例差的双侧正态近似 p（无 scipy）。"""
        p1, p2 = k1 / n1, k2 / n2
        p = (k1 + k2) / (n1 + n2)
        se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        if se <= 0:
            return 1.0
        return min(1.0, math.erfc(abs(p1 - p2) / se / math.sqrt(2)))

    print("\n同材质下的关键对比")
    out["matched"]["tests"] = {}
    R = out["matched"]["rows"]
    tests = [("真人 32px vs 真人 16px（画布变大，真人更有结构？）", "REALval/32", "REALval/16"),
             ("TRD 32px vs 真人 32px", "v11dx_direct/32", "REALval/32"),
             ("TRD 32px vs B2 32px", "v11dx_direct/32", "B2val/32"),
             ("TRD 16px vs 真人 16px", "nf8_xpal/16", "REALval/16")]
    # 步数探针（eval/steps32_probe.sh）：主判据 ② 是各臂对 s24 的结构门差
    probe = sorted(k for k in R if k.startswith("s") and k.endswith("/32") and k != "s24/32")
    for k in probe:
        tests.append((f"步数探针 {k} vs s24/32（主判据：门 +>=10pp 且 p<0.05）", k, "s24/32"))
        tests.append((f"步数探针 {k} vs 真人 32px", k, "REALval/32"))
    # 已生成的 32px 配置横扫（零 GPU）：采样温度、以及 v8/v10/v11d 三代模型
    tests += [("采样温度 0.6 vs 1.0（v10 一代）", "v10x32_t60/32", "v10x_direct/32"),
              ("采样温度 0.6 vs 1.0（v11d 一代）", "v11dx_t60/32", "v11dx_direct/32"),
              ("v8 一代（放大法）vs v11d 一代（补了 32px 数据）", "v8x32_up/32", "v11dx_direct/32"),
              ("v10 一代 vs v11d 一代（同为 direct）", "v10x_direct/32", "v11dx_direct/32"),
              ("4 选 1 重排 vs 不重排（v11d，32px）", "v11dx100_rr4/32", "v11dx_direct/32")]
    # 检查点探针（eval/ckpt32_probe.sh）：库里所有 32px 瓦片都出自 last.pt，而两支的
    # 最佳 val 分别在 step 2000 / 1000。主判据：四对里至少一对 门 +>=10pp 且 p<0.05。
    tests += [("检查点 best vs last（v10，温度 1.0）", "v10b_direct/32", "v10x_direct/32"),
              ("检查点 best vs last（v10，温度 0.6）", "v10b_t60/32", "v10x32_t60/32"),
              ("检查点 best vs last（v11d，温度 1.0）", "v11db_direct/32", "v11dx_direct/32"),
              ("检查点 best vs last（v11d，温度 0.6）", "v11db_t60/32", "v11dx_t60/32"),
              ("best.pt 最好的一格 vs 真人 32px", "v10b_t60/32", "REALval/32")]
    for lab, ka, kb in tests:
        if ka not in R or kb not in R:
            continue
        pg = two_prop(R[ka]["n_gated"], R[ka]["n"], R[kb]["n_gated"], R[kb]["n"])
        _, pa = mannwhitney_u_p(R[ka]["anisos"], R[kb]["anisos"])
        out["matched"]["tests"][f"{ka} vs {kb}"] = {"gate_p": pg, "aniso_p": pa}
        print(f"  {lab}")
        print(f"    过门 {R[ka]['gate_rate']:.0%} vs {R[kb]['gate_rate']:.0%}  p={pg:.3g}   "
              f"各向异性 {R[ka]['median_aniso']:.3f} vs {R[kb]['median_aniso']:.3f}  MW p={pa:.3g}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n->", a.out)


if __name__ == "__main__":
    main()
