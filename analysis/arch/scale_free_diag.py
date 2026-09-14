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

---------------------------------------------------------------- 跑完的判读（2026-09-14）
**方向预测错了，(P1) 的符号与预测相反 -> 预注册的"尺度漂移"假设被推翻。**
  - (P2) 16px：TRD 0.797 vs B2 0.803，TRD 更低 65/125 = 52%，**p=0.72**（测试集 48%，p=0.63）-> 无差异。
  - (P1) 32px：TRD 0.765 vs B2 **0.656**，TRD 更低只有 40/125 = 32%，**p=7.0e-05**
    （测试集 89/271 = 33%，**p=1.7e-08**）-> TRD 在 32px 上显著**更花**，不是更粗。
  判据要的是 "TRD 更低"，实测是 "TRD 更高" -> 尺度漂移**不成立**，且这次不是"没测到"，是**反向显著**。
  (S1) 旁证同向：edge32/edge16 的中位，TRD **0.954**、B2 **0.840**（配对 p=4.2e-10；测试集 0.961/0.821）。
  -> 画布翻倍时**变粗的是 B2，不是 TRD**；TRD 几乎原样保持 16px 的逐像素密度。
  （`scale_diag.py` 用周期给的是相反的读数：TRD 比值 2.00 > B2 1.75。两者矛盾，
   而那把尺子在 TRD 的 32px 上只覆盖 27.6% 的瓦片、在 B2 上覆盖 74.4% -> **以本脚本为准**。）

**因此换来的那句可查的事实**（它不是"尺度漂移"，是它的反面）：
> TRD 与 B2 在 16px 上逐像素密度**完全相同**；画布到 32px 时 B2 降到 0.656 而 TRD 只降到 0.765。
> 判官在 16px 上选 TRD、在 32px 上选 B2。**这是目前唯一一个"随尺寸改变符号"的已测内容差异。**
⚠ **"更花 = 更差"没有被证明**，本轮只测到相关（两个尺寸、一条轴）。不许拿 `edge` 当优化目标、
   当挑配置的依据，或当任何臂的判据——本项目已经因为"拿结构门当依据"栽过一次。

⛔⛔ **上面那句"随尺寸改变符号"已于 2026-09-14 被否**（`analysis/arch/edge_vs_judge.py`，
   预注册 `8fde6ed`，零 GPU 零 API）。补上第三个点、并在组内直接问了一次：
     - **24px**（`TRD24_rr4` 与 `TRD32_rr4` **同一个 v10 检查点**，没有 (S1) 那个跨 run 混杂）：
       `edge` 差 **+0.062**、TRD 更花占 **63.6%**（p=8.5e-06）—— 已经和 32px（+0.097 / 67.3%）**相当**，
       而判官在 24px 上是 **94/188 = 50.0%（平）**。**大的 `edge` 差与"不输"完全相容。**
     - **组内**（就在产生 41% 的那条臂里，按材质分组）：TRD **输**掉的材质 `edge` 差中位 **+0.069**、
       **赢**的 **+0.117**，AUC **0.414**、**p=0.038** —— **方向与"更花=更差"相反**；
       16px 上同向更强（输 −0.031 / 赢 +0.035，AUC 0.377，**p=0.003**）。
   -> `edge` 差**随画布单调增长**（0.004 -> 0.062 -> 0.097）而**判官不随**（57% -> 50% -> 41% 不是一条能被它解释的线），
      它更像**画布尺寸的产物**（B2 的降采样倍率随画布变）。**"32px 那个看得见的毛病"这句话作废**，
      本节剩下的只是一句中性描述：TRD 的逐像素密度不随画布下降，B2 的会。
⚠ 参照线的口径：同材质的真人 16px（`REALval/16`）是 **0.766**，比两个模型都低；真人 32px
   **没有可用参照**（`REALval/32` 89% 来自一个包，见 (S3)）。所以"0.656 和 0.765 哪个对"**无从判定**。

---------------------------------------------------------------- 给 arch_nod32 的机制探针（预注册）
写于 `arch_nod32` **一步都没训**的时候（`/tmp/nod32.txt` 0 字节，已排队 1h43m）。
**不改 `eval/val_nod32.sh` 的任何判据**，这只是一条零 API 的旁读，用来解释臂的结果：

  臂跑完后，对两臂各自的 32px 瓦片跑
      python analysis/arch/scale_free_diag.py \
        --extra32 nod32=/tmp/gen32nod/nod32x100_rr4/32 \
        --extra32 ctrl0=/tmp/gen32nod/ctrl0x100_rr4/32
  **方向预测**：A（撤掉 32px 数据）的 `edge` **低于** B（对照）。
  依据：上面那句事实 + 上一轮 `data32_content.py` 量到的 32px 训练池比 16px 池**更花**
  （`flat` 更低 p=1.5e-08、`k_used` 更高 p=5.5e-08）-> 若 TRD@32 的高密度是**学来的**，
  撤掉这批数据就该降下来；若撤掉后 `edge` 不动，那这个密度是**架构给的**，与数据无关。
  ⚠ 这条**不判臂的胜负**，也**不是**臂的判据：判官怎么判以 `val_nod32.sh` 为准。
     它只区分"数据造成的"和"架构造成的"，这两种情况的下一步完全不同。
  ⛔ **2026-09-14 补充（`edge_vs_judge.py`）：这条探针照跑，但读数里"低"不等于"好"。**
     那时写"依据：上面那句事实"，而那句事实已被否——`edge` 差与判官的判读不同向，组内还反向。
     所以探针只剩它本来的那一半用途：**区分密度是数据给的还是架构给的**。
     若 A 的 `edge` 降下来，**不许**就此说 A 更好；胜负一律以 `val_nod32.sh` 的判官判据为准。
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
    ap.add_argument("--extra32", action="append", default=[], metavar="名字=目录",
                    help="再量一个 32px 生成目录，并按材质与 val/B2_32 配对。"
                         "用途见文件末尾『给 arch_nod32 的机制探针』。")
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
    for e in a.extra32:
        name, _, d = e.partition("=")
        spec[f"extra/{name}"] = (Path(d), 32)
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

    for e in a.extra32:
        name = e.partition("=")[0]
        k = f"extra/{name}"
        if k in groups and "val/B2_32" in groups:
            mats = sorted(set(groups[k]) & set(groups["val/B2_32"]))
            res["per_size"][f"extra/{name} vs val/B2_32"] = \
                sign_test(groups[k], groups["val/B2_32"], mats) | {
                    "median_TRD": float(np.median([groups[k][m]["edge"] for m in mats])),
                    "median_B2": float(np.median([groups["val/B2_32"][m]["edge"] for m in mats])),
                    "n_mat_TRD": len(groups[k]), "n_mat_B2": len(groups["val/B2_32"]),
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
