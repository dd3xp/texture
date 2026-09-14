"""32px 的"看得见的毛病"到底是什么？——**不带结构门**地重问一次尺度漂移。

零 GPU、零 API、零远程磁盘写入（只读已生成的瓦片 PNG，输出写 /tmp）。

---------------------------------------------------------------- 为什么重问
`analysis/arch/scale_diag.py`（2026-09-12）问过"TRD 在大画布上画的结构尺度对不对"，
用的尺子是 `dominant_period` + 各向异性**门**，判为"尺度漂移不成立"（周期比 16->32 = 2.00）。
那一轮之后发生了两件事，使它的判读**必须重做**：

1. **它的"真人 32px"参照是一个包**（`ref_pack_audit.py`，`1e02955`）：74 张里 66 张（89%）来自
   `ROllerozxa__mtg_tiled_32x`。判据 2 的第二个条件（真人比值 >= 1.5）就是拿这个组量的（2.5）。
2. **结构门已被项目判定不可作依据**（TRD 自家配置间不排序、温度轴上与判官反向 24pp）。
   更要命的是**覆盖率**：`scale_diag_32.json` 里 `TRD32_rr4/32` 的**过门率只有 27.6%**，
   而 `B2val/32` 是 **74.4%**。也就是说那个"周期比 2.00"是在 TRD 的**四分之一瓦片**上算的，
   且这四分之一是被一把在两组上通过率差三倍的筛子选出来的 —— **选择偏差压过了要测的量**。

所以本脚本换一把**没有门、没有阈值、覆盖 100% 瓦片**的尺子，重新问同一个问题，
并且**不使用真人 32px 作判据**（它被污染了），改用 **B2 作参照系**：
B2 是判官在 32px 上认可的那一方（TRD 输它 41%，p=0.014），它的读数落在哪边是最直接的线索。

---------------------------------------------------------------- 尺子（跑之前定死，都不带阈值）
逐瓦片，在 PNG 的 RGB 上量（这些瓦片本来就 <= 16 色）：
  - `edge`：环面上相邻格（右、下各一遍）**颜色不同**的比例。**越小越粗/越平**。  <- 主尺子
  - `flat`：最高频颜色占的格子比例。越大越平。
  - `k`   ：不同颜色数。
与 `analysis/arch/data32_content.py` 用的是同三把尺子（那边在色阶网格上、这边在 RGB 上，
定义一致），已经在 32px 训练池上用过一次。**与结构门无关**：不算周期、不算各向异性。

⚠ 尺子的一条硬性质（本脚本的全部逻辑都建立在它上面）：
若一张图的结构**按画布等比放大**（16px 的图案原样放大一倍画到 32px 上），
`edge` 大致**减半**；若结构的**像素尺度不变**（画布变大就多画细节），`edge` 大致**不变**。

---------------------------------------------------------------- 主判据（跑之前写死）
全部**按材质配对**、**符号检验**（精确二项，无 scipy）。主尺子 `edge`，主集合 = 验证集 V_mat。

  (P1) 32px：TRD 的 `edge` 是否显著**低于** B2（配对符号检验 p < 0.05 且多数材质 TRD 更低）。
  (P2) 16px：同样的比较。
  **"尺度漂移"判为成立，当且仅当 (P1) 成立而 (P2) 不成立**（16px 上 TRD 不显著更低、或反向）。
  若两档都显著更低且方向相同 -> 这是 TRD 的**风格常数**，不是尺度现象 -> 假设**推翻**。
  若两档都不显著 -> 这条线关闭，不许改判据再试第三把尺子。

  (S1) 副读数（差的差 / D-in-D）：逐材质算 `edge32 / edge16`，TRD 的比值与 B2 的比值配对比较。
       预测 TRD 的比值更小（TRD 随画布变粗得更多）。
       ⚠⚠ **这条有一个已知混杂，跑之前写下**：TRD 的 16px 与 32px 来自**不同的 run**
       （16px = `v8x100_rr4`，v8 谱系；32px = `v10x100_rr4`，v10 谱系），而 B2 两档是同一条管线。
       所以 (S1) 分不开"尺度行为"和"两个 run 不一样"。**因此 (S1) 只作旁证，主判据是 (P1)+(P2)**，
       后者在每个尺寸内部比较，不跨 run 作差。

  (S2) 测试集复现：同三把尺子在 E_mat 上再算一遍（16px `TRD16c_rr4` vs `B2`、32px `TRD32_rr4` vs `B2`）。
       判官那个 41% 就长在这个集合上，所以这里必须看一眼。
       ⚠ **描述性复现，不得据此挑任何配置**：本脚本不输出任何胜负、不对 TRD 的配置排序；
       若它提示了某个改架构的方向，那个方向必须**另行预注册、并在验证集上验**。

  (S3) 真人：`REALval/16`（125）与 `REALval/32`（67）照样报出来，**但不进任何判据** ——
       32px 那一组 89% 来自一个包（`ref_pack_audit.py`），它代表不了真人。列出来只为让读者看见
       它有多偏，免得下次又有人拿它当参照。

---------------------------------------------------------------- 方向预测（跑之前写死）
预测 **(P1) 成立、(P2) 不成立**，即尺度漂移成立。理由：`scale_diag_32.json` 里 TRD 的
中位颜色数（14-15）高于 B2（12），若 TRD 只是"更爱抖动"，它在**两档**上 `edge` 都该更高；
而 32px 上它的过门率反而从 B2 的 74% 掉到 28%、各向异性从 0.20 掉到 0.14 ——
若这不是门的选择偏差造成的，那么 16px 上 `edge` 更高、32px 上更低，正是尺度漂移的样子。

---------------------------------------------------------------- 操作检验（不过就作废）
  (O1) 每组瓦片的边长必须等于该组名义尺寸，颜色数 <= 16。
  (O2) 每个配对比较的 n 必须等于两侧材质名的交集大小，且 >= 100（V_mat）/ >= 250（E_mat）。

    python analysis/arch/scale_free_diag.py --out /tmp/scale_free_diag.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test                       # noqa: E402

IDX = re.compile(r"_(\d+)$")
RULERS = ("edge", "flat", "k")


def measure(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    n = rgb.shape[0]
    flat_cells = rgb.reshape(-1, 3)
    cols, counts = np.unique(flat_cells, axis=0, return_counts=True)
    right = (rgb != np.roll(rgb, -1, axis=1)).any(axis=2).mean()
    down = (rgb != np.roll(rgb, -1, axis=0)).any(axis=2).mean()
    return {
        "n": int(n),
        "edge": float((right + down) / 2),
        "flat": float(counts.max() / flat_cells.shape[0]),
        "k": int(cols.shape[0]),
    }


def load_group(d, size):
    """返回 {材质: {尺子: 中位数}}；同一材质多张时逐尺子取中位。"""
    d = Path(d)
    if not d.is_dir():
        return None, f"missing:{d}"
    per_mat, bad = {}, []
    for p in sorted(d.glob("*.png")):
        m = measure(p)
        if m["n"] != size or m["k"] > 16:
            bad.append((p.name, m["n"], m["k"]))
        per_mat.setdefault(IDX.sub("", p.stem), []).append(m)
    out = {}
    for mat, ms in per_mat.items():
        out[mat] = {r: float(np.median([m[r] for m in ms])) for r in RULERS}
    return out, bad


def sign_test(a, b, mats):
    """a 的值是否低于 b：返回 (n, a更低的个数, 双侧 p, 逐材质差的中位)."""
    diffs = [a[m]["edge"] - b[m]["edge"] for m in mats]
    lo = sum(1 for d in diffs if d < 0)
    hi = sum(1 for d in diffs if d > 0)
    return {
        "n_pairs": len(mats),
        "n_tied": len(mats) - lo - hi,
        "n_a_lower": lo,
        "frac_a_lower": (lo / (lo + hi)) if (lo + hi) else None,
        "p": binom_test(lo, lo + hi) if (lo + hi) else 1.0,
        "median_diff": float(np.median(diffs)),
    }


def summarise(g):
    return {r: float(np.median([v[r] for v in g.values()])) for r in RULERS} | {"n_mat": len(g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/data/kw/RoundSquisheen/texture/experiments/baselines")
    ap.add_argument("--trd32val", default="/tmp/gen32pb/v10x100_rr4/32")
    ap.add_argument("--out", type=Path, default=Path("/tmp/scale_free_diag.json"))
    a = ap.parse_args()
    B = Path(a.base)

    spec = {
        "val/TRD16": (B / "v8x100_rr4/16", 16),
        "val/B2_16": (B / "B2val/16", 16),
        "val/TRD32": (Path(a.trd32val), 32),
        "val/B2_32": (B / "B2val/32", 32),
        "val/REAL16": (B / "REALval/16", 16),
        "val/REAL32": (B / "REALval/32", 32),
        "test/TRD16": (B / "TRD16c_rr4/16", 16),
        "test/B2_16": (B / "B2/16", 16),
        "test/TRD32": (B / "TRD32_rr4/32", 32),
        "test/B2_32": (B / "B2/32", 32),
    }
    groups, bads = {}, {}
    for name, (d, size) in spec.items():
        g, bad = load_group(d, size)
        if g is None:
            bads[name] = bad
            continue
        groups[name] = g
        if bad:
            bads[name] = bad[:5]

    res = {
        "rulers": list(RULERS),
        "dirs": {k: str(v[0]) for k, v in spec.items()},
        "operational_check_O1_bad": bads,
        "summary": {k: summarise(v) for k, v in groups.items()},
    }

    # (P1)(P2)(S2)：每个尺寸内部，TRD vs B2，按材质配对
    res["per_size"] = {}
    for split in ("val", "test"):
        for size in (16, 32):
            ta, tb = f"{split}/TRD{size}", f"{split}/B2_{size}"
            if ta not in groups or tb not in groups:
                continue
            mats = sorted(set(groups[ta]) & set(groups[tb]))
            res["per_size"][f"{split}/{size}"] = sign_test(groups[ta], groups[tb], mats) | {
                "median_TRD": float(np.median([groups[ta][m]["edge"] for m in mats])),
                "median_B2": float(np.median([groups[tb][m]["edge"] for m in mats])),
                "n_mat_TRD": len(groups[ta]), "n_mat_B2": len(groups[tb]),
            }

    # (S1)：edge32/edge16 的方法间配对比较
    res["ratio_S1"] = {}
    for split in ("val", "test"):
        need = [f"{split}/TRD16", f"{split}/TRD32", f"{split}/B2_16", f"{split}/B2_32"]
        if any(k not in groups for k in need):
            continue
        mats = sorted(set.intersection(*[set(groups[k]) for k in need]))
        rt, rb = [], []
        for m in mats:
            e16t, e16b = groups[f"{split}/TRD16"][m]["edge"], groups[f"{split}/B2_16"][m]["edge"]
            if e16t <= 0 or e16b <= 0:
                continue
            rt.append(groups[f"{split}/TRD32"][m]["edge"] / e16t)
            rb.append(groups[f"{split}/B2_32"][m]["edge"] / e16b)
        d = [x - y for x, y in zip(rt, rb)]
        lo = sum(1 for v in d if v < 0)
        hi = sum(1 for v in d if v > 0)
        res["ratio_S1"][split] = {
            "n_pairs": len(d), "median_ratio_TRD": float(np.median(rt)),
            "median_ratio_B2": float(np.median(rb)),
            "n_TRD_ratio_lower": lo, "p": binom_test(lo, lo + hi) if (lo + hi) else 1.0,
            "median_diff": float(np.median(d)),
        }

    # (S3) 真人：只报，不判
    if "val/REAL16" in groups and "val/REAL32" in groups:
        res["real_S3_not_a_criterion"] = {
            "note": "REALval/32 的 89% 来自 ROllerozxa__mtg_tiled_32x 一个包，见 ref_pack_audit.py",
            "median_edge_16": float(np.median([v["edge"] for v in groups["val/REAL16"].values()])),
            "median_edge_32": float(np.median([v["edge"] for v in groups["val/REAL32"].values()])),
            "n16": len(groups["val/REAL16"]), "n32": len(groups["val/REAL32"]),
        }

    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== 各组中位（edge / flat / k, n_mat） ==")
    for k, v in res["summary"].items():
        print("  %-14s edge %.4f  flat %.3f  k %5.1f  n_mat %d"
              % (k, v["edge"], v["flat"], v["k"], v["n_mat"]))
    print("\n== 主判据 (P1)(P2)(S2)：同尺寸内 TRD vs B2 的 edge（配对符号检验） ==")
    for k, v in res["per_size"].items():
        print("  %-9s n=%3d  TRD %.4f vs B2 %.4f  TRD 更低 %d/%d = %s  p=%.3g"
              % (k, v["n_pairs"], v["median_TRD"], v["median_B2"], v["n_a_lower"],
                 v["n_a_lower"] + (v["n_pairs"] - v["n_tied"] - v["n_a_lower"]),
                 ("%.0f%%" % (100 * v["frac_a_lower"])) if v["frac_a_lower"] is not None else "-",
                 v["p"]))
    print("\n== 旁证 (S1)：edge32/edge16（跨 run，仅旁证） ==")
    for k, v in res["ratio_S1"].items():
        print("  %-5s n=%3d  TRD %.3f  B2 %.3f  TRD 更小 %d  p=%.3g"
              % (k, v["n_pairs"], v["median_ratio_TRD"], v["median_ratio_B2"],
                 v["n_TRD_ratio_lower"], v["p"]))
    if "real_S3_not_a_criterion" in res:
        r = res["real_S3_not_a_criterion"]
        print("\n== (S3) 真人，不作判据 == 16px %.4f (n=%d) / 32px %.4f (n=%d, 89%% 一个包)"
              % (r["median_edge_16"], r["n16"], r["median_edge_32"], r["n32"]))
    if bads:
        print("\n!! 操作检验 O1 有异常：", json.dumps(bads, ensure_ascii=False)[:400])
    print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
