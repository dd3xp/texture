"""上一轮量到的 `edge` 差，**能不能解释判官在 32px 上的那个 41%**？

零 GPU、零 API、零远程磁盘写入（只读已生成的瓦片 PNG 和**已封盘**的判定 JSON，输出写 /tmp）。

---------------------------------------------------------------- 为什么必须先问这一条
`analysis/arch/scale_free_diag.py`（`0c691e7`）拿到了目前**唯一一个"随尺寸改变符号"的已测内容差异**：

    16px：TRD 0.793 vs B2 0.785，TRD 更低 130/269 = 48%，p=0.63   —— 判官 114/199 = 57.3%（胜）
    32px：TRD 0.751 vs B2 0.628，TRD 更低  89/271 = 33%，p=1.7e-08 —— 判官  83/202 = 41.1%（输，p=0.014）

于是账本把它写成了"32px 那个看得见的毛病的第一个量化描述"，`arch_nod32` 的机制探针也已经架在它上面。
**但"TRD 在 32px 上更花"和"判官在 32px 上判 TRD 输"目前只是两句同时成立的话**——
两个尺寸、两个点，任何单调的量都能连上。在拿它当下一步改架构的依据之前，本脚本用**两条互相独立、
都零成本**的检验去打它：

  (A) **第三个点**：24px。`eval/final_test.sh` 里 `TRD24_rr4` 与 `TRD32_rr4` **出自同一个检查点
      `runs/trd_v10`**（只差 `--ret_nname` 300/100 与画布大小），B2 三档同一条管线
      -> **(S1) 那个"16px 来自另一个 run"的混杂在 24 vs 32 之间不存在**。
      判官在 24px 上是 94/188 = **50.0%（平）**、在 32px 上是 **41.1%（输）**。
      若"更花"是输的原因，24px 的 `edge` 差就该明显小于 32px。
  (B) **组内相关**：就在产生那个 41% 的**同一条臂、同一个尺寸**里，把 271 个材质按判官的逐材质
      判定分成"TRD 赢"和"TRD 输"两组，看**逐材质的 `edge` 差**在两组间有没有差别。
      若"更花 -> 判官不喜欢"，TRD 输掉的材质应当是 `edge` 差更大的那些。
      这一条完全不跨尺寸、不跨 run，是最直接的一问。

---------------------------------------------------------------- 尺子与数据（跑之前定死）
`edge`（环面上相邻格颜色不同的比例）沿用 `scale_free_diag.measure`，**一个字不改、不加第二把尺子**。
逐材质的量 = `gap = edge(TRD) - edge(B2)`（正 = TRD 更花）。
判定来自已入库、**已封盘**的三份 JSON（`recheck_judge.py` 每轮复核它们）：

    judge_full_TRD16c_rr4_vs_B2_16.json   (114, 199)
    judge_full_TRD24_rr4_vs_B2_24.json    ( 94, 188)
    judge_full_TRD32_rr4_vs_B2_32.json    ( 83, 202)

瓦片目录与判官当时读的**必须是同一批**：`eval/judge_pairs.py:110` 取 `first_tile`（`{slug}_0.png`
或 `{slug}.png`），本脚本按同一顺序解析文件名，并用 `first_tile` 亲自读一遍、**逐像素**核对
自己量的那张图就是判官看过的那张；材质顺序按 `eval/prompts.load_set("E_mat")` 重建，
**操作检验 (O1)** 还要逐条核对 `records[i]["material"] == pairs[i][0]`、以及三份 JSON 的胜负数
与账本一致，任何一条不过就退 1。

---------------------------------------------------------------- 判据 (A)：24px 那个第三点
按材质配对、符号检验（精确二项，无 scipy）。记 `f24` = TRD 更花的材质占比，`d24` = 逐材质差的中位。
锚点是**已经量到的两端**（测试集）：16px `f=0.517 / d=+0.0020`，32px `f=0.672 / d=+0.0967`。

  (A-跟随)  `f24 <= 0.57` 且 `d24 <= 0.03`（贴近 16px 那一端）
            -> `edge` 差与判官的判读同向单调（16 胜/无差、24 平/小差、32 输/大差），
               它作为"32px 毛病的描述"**得以存活**（仍然只是描述，见下面的禁令）。
  (A-不跟随) `f24 >= 0.62` 且 `d24 >= 0.06`（贴近 32px 那一端）
            -> **判官判平的那一档上，`edge` 差已经和判输那一档一样大**
               -> 大的 `edge` 差与"不输"是相容的 -> 它**不足以**解释那个 41%。
               账本里"32px 那个看得见的毛病"这句话必须降级为"TRD 在大画布上的一个属性"。
  (A-不明)  其余 -> 只记录数字，**不下结论**，也不许换阈值再判一次。

---------------------------------------------------------------- 判据 (B)：32px 组内相关
判定里 `A` = TRD 赢、`B` = B2 赢，`inconsistent`/null 按既有协议弃用。
两组的 `gap` 做 Mann-Whitney（正态近似 + 并列校正，n 约 200，标准库实现），报
`AUC = P(gap[TRD输] > gap[TRD赢])`。

  **方向预测（跑之前写死）：AUC > 0.5**，即 TRD 输掉的材质 `gap` 更大。
  (B-支持) AUC > 0.5 且 p < 0.05 -> 机制在组内得到支持。
  (B-否)   p >= 0.05            -> 在产生 41% 的那条臂内部，`edge` 差**预测不了**输在哪些材质上。
  (B-反向) AUC < 0.5 且 p < 0.05 -> 反向，机制被推翻。
  16px 与 24px 同样算一遍，作**描述性对照**（不进 (B) 的判读）。

  ⚠⚠ **(B) 的不对称性，跑之前写下**：它只能测到**逐材质变化**的效应。
  若"更花"是一个对所有材质**齐步**起作用的整体风格惩罚，它会把所有材质的 gap 一起抬高、
  在组间比较里**完全不可见**。所以 **(B-否) 不等于"`edge` 与判官无关"**，只等于
  "`edge` 差解释不了 TRD 输在**哪些**材质上"。(A) 没有这个盲区，两条要合起来读。

---------------------------------------------------------------- 禁令（与结果无关，一律生效）
- `edge` **仍然不许**当优化目标、不许当挑配置的依据、不许当任何臂的判据（`0c691e7` 已写死）。
  本脚本**不输出任何配置排序、不改任何臂的判据**，`eval/val_nod32.sh` 一个字不动。
- 本脚本把**已发表、已封盘**的判定当作**被解释项**来读，**不是**拿它去选配置、也不重跑任何判定。
  三份 JSON 的胜负数字由 `analysis/arch/recheck_judge.py` 守着，本脚本不写它们。
- (A) 的一个已知混杂，跑之前写下：B2 在 24px 与 32px 上是从同一张 1024 渲染按**不同倍率**降采样来的，
  所以 `edge` 差本来就可能随画布单调变化，**与判官无关**。这正是 (A) 要测的东西——
  若它随画布单调变化而判官不随，那它就是个画布尺寸的产物。

    python analysis/arch/edge_vs_judge.py
"""
import argparse
import json
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "arch"))
sys.path.insert(0, str(ROOT / "eval"))
from exact import binom_test                              # noqa: E402
from scale_free_diag import measure                       # noqa: E402

# 尺寸 -> (TRD 目录名, B2 目录名, 判定 JSON 名, 账本里的 (A胜, 有效))
ARMS = {
    16: ("TRD16c_rr4", "B2", "judge_full_TRD16c_rr4_vs_B2_16.json", (114, 199)),
    24: ("TRD24_rr4", "B2", "judge_full_TRD24_rr4_vs_B2_24.json", (94, 188)),
    32: ("TRD32_rr4", "B2", "judge_full_TRD32_rr4_vs_B2_32.json", (83, 202)),
}
# (A) 的阈值，锚在测试集已量到的两端：16px f=0.517/d=+0.0020、32px f=0.672/d=+0.0967
A_FOLLOW = (0.57, 0.03)      # f24 <= 且 d24 <= -> 跟随
A_NOFOLLOW = (0.62, 0.06)    # f24 >= 且 d24 >= -> 不跟随


def tile_path(d: Path, slug):
    """与 `eval/judge_pairs.first_tile` 同一个文件名解析顺序，但返回路径（measure 要路径）。"""
    for name in (f"{slug}_0.png", f"{slug}.png"):
        if (d / name).exists():
            return d / name
    return None


def mann_whitney(x, y):
    """P(x > y) 的 AUC + 正态近似双尾 p（含并列校正）。x、y 为一维序列。"""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return {"n_x": nx, "n_y": ny, "auc": None, "p": None}
    allv = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float)
    i = 0
    tie_term = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    rx = ranks[:nx].sum()
    ux = rx - nx * (nx + 1) / 2.0            # #{x > y} + 0.5 * #{并列}
    n = nx + ny
    mu = nx * ny / 2.0
    var = nx * ny * (n + 1) / 12.0 - nx * ny * tie_term / (12.0 * n * (n - 1))
    z = (ux - mu) / sqrt(var) if var > 0 else 0.0
    return {"n_x": nx, "n_y": ny, "auc": float(ux / (nx * ny)),
            "z": float(z), "p": float(erfc(abs(z) / sqrt(2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/data/kw/RoundSquisheen/texture/experiments/baselines")
    ap.add_argument("--judge_dir", type=Path, default=ROOT / "experiments")
    ap.add_argument("--out", type=Path, default=Path("/tmp/edge_vs_judge.json"))
    a = ap.parse_args()
    B = Path(a.base)

    from judge_pairs import first_tile                     # noqa: E402  判官当时取图的同一个函数
    from prompts import load_set                           # noqa: E402
    items = load_set("E_mat")[0]

    res, o1 = {"arms": {}}, []
    for size, (ta_name, tb_name, jname, expect) in ARMS.items():
        da, db = B / ta_name / str(size), B / tb_name / str(size)
        pairs = []
        for e in items:
            slug = e["material"].rsplit(".", 1)[0]
            pa, pb = tile_path(da, slug), tile_path(db, slug)
            if pa is not None and pb is not None:
                pairs.append((e["prompt"], pa, pb, slug))

        d = json.loads((a.judge_dir / jname).read_text(encoding="utf-8"))
        recs = d["records"]
        # (O1) 操作检验：本脚本重建的配对顺序必须与判定 JSON 完全一致
        if len(recs) != len(pairs):
            o1.append(f"{size}px：配对数 {len(pairs)} != 判定条数 {len(recs)}")
        else:
            for r in recs:
                if pairs[r["pair"]][0] != r["material"]:
                    o1.append(f"{size}px：pair {r['pair']} 材质对不上 "
                              f"({pairs[r['pair']][0]!r} vs {r['material']!r})")
                    break
        got = (sum(x["verdict"] == "A" for x in recs),
               sum(x["verdict"] in ("A", "B") for x in recs))
        if got != expect:
            o1.append(f"{size}px：判定 {got} != 账本 {expect}")

        gaps, verdicts, bad_n, bad_px = [], [], [], []
        for r in recs:
            _, pa, pb, slug = pairs[r["pair"]]
            ma, mb = measure(pa), measure(pb)
            if ma["n"] != size or mb["n"] != size:
                bad_n.append((r["pair"], ma["n"], mb["n"]))
            # 逐像素核对：自己量的这两张就是判官看过的那两张
            for d_, p_ in ((da, pa), (db, pb)):
                if not np.array_equal(first_tile(d_, slug),
                                      np.asarray(Image.open(p_).convert("RGB"))):
                    bad_px.append(str(p_))
            gaps.append(ma["edge"] - mb["edge"])
            verdicts.append(r["verdict"])
        if bad_n:
            o1.append(f"{size}px：{len(bad_n)} 张边长不是 {size}，例 {bad_n[:3]}")
        if bad_px:
            o1.append(f"{size}px：{len(bad_px)} 张与判官读到的像素不一致，例 {bad_px[:3]}")
        gaps = np.asarray(gaps, float)

        hi = int((gaps > 0).sum())
        lo = int((gaps < 0).sum())
        won = gaps[[v == "A" for v in verdicts]]           # TRD 赢的材质
        lost = gaps[[v == "B" for v in verdicts]]          # TRD 输的材质
        mw = mann_whitney(lost, won)                       # AUC = P(输组 gap > 赢组 gap)

        res["arms"][str(size)] = {
            "dirs": [str(da), str(db)], "judge": jname,
            "judge_a_wins": got[0], "judge_decided": got[1],
            "judge_rate": got[0] / got[1] if got[1] else None,
            "n_mat": len(gaps),
            "frac_TRD_busier": hi / (hi + lo) if (hi + lo) else None,
            "median_gap": float(np.median(gaps)),
            "p_sign": binom_test(hi, hi + lo) if (hi + lo) else 1.0,
            "B_within": mw | {
                "median_gap_TRD_lost": float(np.median(lost)) if len(lost) else None,
                "median_gap_TRD_won": float(np.median(won)) if len(won) else None},
        }

    r24 = res["arms"]["24"]
    f24, d24 = r24["frac_TRD_busier"], r24["median_gap"]
    if f24 <= A_FOLLOW[0] and d24 <= A_FOLLOW[1]:
        res["verdict_A"] = "跟随：24px 的 edge 差贴近 16px 那一端，与判官同向单调"
    elif f24 >= A_NOFOLLOW[0] and d24 >= A_NOFOLLOW[1]:
        res["verdict_A"] = ("不跟随：判官判平的 24px 上 edge 差已与判输的 32px 相当"
                            " -> edge 差不足以解释那个 41%")
    else:
        res["verdict_A"] = "不明：不下结论，且不许换阈值再判"
    b32 = res["arms"]["32"]["B_within"]
    if b32["p"] is None:
        res["verdict_B"] = "不可判"
    elif b32["p"] >= 0.05:
        res["verdict_B"] = "(B-否)：组内 edge 差预测不了 TRD 输在哪些材质（注意齐步效应的盲区）"
    elif b32["auc"] > 0.5:
        res["verdict_B"] = "(B-支持)：TRD 输掉的材质 edge 差显著更大"
    else:
        res["verdict_B"] = "(B-反向)：TRD 输掉的材质 edge 差显著更小"
    res["operational_check_O1"] = o1
    res["thresholds"] = {"A_follow": A_FOLLOW, "A_nofollow": A_NOFOLLOW}

    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("== (A) 跨尺寸：edge 差 vs 判官 ==")
    print("  %-5s %-24s %s" % ("尺寸", "判官（TRD 胜率）", "edge：TRD 更花的材质 / 差的中位"))
    for s in ("16", "24", "32"):
        v = res["arms"][s]
        print("  %-5s %3d/%3d = %.1f%%            %.3f (%d 材质)  中位 %+.4f  p=%.3g"
              % (s + "px", v["judge_a_wins"], v["judge_decided"], 100 * v["judge_rate"],
                 v["frac_TRD_busier"], v["n_mat"], v["median_gap"], v["p_sign"]))
    print("  -> " + res["verdict_A"])
    print("\n== (B) 组内：输/赢两组的 edge 差（Mann-Whitney） ==")
    for s in ("16", "24", "32"):
        v = res["arms"][s]["B_within"]
        print("  %-5s 输 n=%3d 中位 %+.4f | 赢 n=%3d 中位 %+.4f | AUC=%.3f  p=%.3g"
              % (s + "px", v["n_x"], v["median_gap_TRD_lost"], v["n_y"],
                 v["median_gap_TRD_won"], v["auc"], v["p"]))
    print("  -> " + res["verdict_B"] + "（判读只看 32px 那一行）")
    if o1:
        print("\n!! 操作检验 O1 未通过：", json.dumps(o1, ensure_ascii=False)[:600])
    print("\n-> %s" % a.out)
    return 1 if o1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
