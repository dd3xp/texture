"""前提检查：管线瓦片「比真人更不空间相干」这条线索，去掉调色板混淆后还在不在。

来源是抖动那一轮（`dither.py`）末尾**明确标为「线索，不是发现」**的一段：
同一个 checker_rate 统计量上，管线瓦片的富集是 box 0.67x / point 0.54x，
真人（门拒绝组）是 0.34x，即**管线瓦片比真人瓦片更不空间相干，且与换哪个
降采样器无关**。那一段自己写着「这是跨样本比较：n=29 vs 2971、材质集不同、
调色板 12 色 vs 中位 16 色，三个都影响这个统计量。要当假设用必须单独预注册」。

本脚本就是那份单独预注册。**它只能去掉三个混淆里的一个**（调色板大小），
外加一个新的特异性对照。做不到的部分在下面「灵敏度与外部效度」里写死。

纯本地、不用 GPU、不占服务器盘。

--- 统计量（不新造，整套复用 `dither.py`）---

逐瓦片 `d = checker_rate(观测) - 自己的格内洗牌均值`，即 dither.py 主判据用的
那一个。checker_rate = 不跨边的 2x2 窗里，「恰好含 2 个不同下标」的窗中对角
交替所占比例。d 越大 = 越像棋盘 = **越不空间相干**。
`MIN_WINDOWS`、`SHUFFLES` 都沿用 dither.py 的值。

--- 四组（跑之前固定）---

  P_iso  管线、门拒绝：`experiments/iso_point/16/*_box.png` 的 29 张
         （B23 的门拒绝材质，box 降采样，调色板恰好 12 色 —— 已核对 29/29）
  A_iso  真人、门拒绝：`data/tiles/dataset_k16.json` 里 size==16、过同一个门
         判为**不触发**、且**用色数落在 [10,14]** 的瓦片
  P_per  管线、门触发：`experiments/bestof/*_best.png` 中 `bestof_units.json`
         记为 `best_eff=True` 的那些（门真的触发且真的裁了）
  A_per  真人、门触发：同 A_iso 但门**触发**

用色数窗口 [10,14] 是**对称套在管线的 12 上**，跑之前只看过用色数的直方图
（真人 size==16 瓦片：12 色 77 张、10-14 色合计 463 张），**没有看过任何
checker_rate 的数**。窗口不许事后调整。

--- 主判据 ---

P_iso 与 A_iso 的 d 做 Mann-Whitney 双侧（`decline_spread.mannwhitney`，含并列校正）：
  **成立**：P_iso 中位 > A_iso 中位 **且** p<0.05
        -> 调色板匹配后差距仍在，「管线瓦片更不相干」可以当假设用，
           第六条路（在量化里做空间相干性修复）**可以立项**，目标值取 A_iso 中位。
  **证伪**：p>=0.05，或方向反过来（P_iso <= A_iso）
        -> 那条线索是调色板混淆，**不立项**，如实记进 loop.md 并把线索划掉。

--- 特异性对照（描述性，不改主判据的判读，但改结论怎么写）---

对 P_per vs A_per 做同一个检验。若门触发那半的差距**不小于**门拒绝这半
（即 `median(P_per)-median(A_per) >= median(P_iso)-median(A_iso)`），
那么这就是一条**管线 vs 真人的通病**，不是各向同性那半特有的，
结论必须这么写，且第六条路**不许**被说成「修各向同性那半的东西」。

--- 灵敏度与外部效度（预注册，事后不许缩小）---

1. 只测 50% 棋盘签名，与 dither.py 同一个上限：25%/75% 有序抖动在 2x2 里
   与色块边缘同为 3+1，不可分。
2. **只控住了三个混淆里的一个**。材质集不同（B23 的 29 条提示词 vs contentdb
   包）与来源不同（SDXL 渲染降采样 vs 手绘）**仍然没控**，本轮控不了。
   所以即使主判据成立，也只能说「差距不是调色板大小造成的」，
   **不能**说「差距一定来自降采样管线」。
3. 管线侧 n=29 / n<=42，小样本；MW 用正态近似。
4. 真人瓦片来自 16x16 contentdb 包、中位用满 16 色；限制到 10-14 色这一层
   本身就是个**子集**，不代表全体真人惯例。
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/premise"))
sys.path.insert(0, str(ROOT / "analysis/paired"))
from downsample import dominant_period, anisotropy            # noqa: E402
from dither import (MIN_WINDOWS, as_index_map, checker_rate,   # noqa: E402
                    shuffled_rates, tiles)
from decline_spread import mannwhitney                         # noqa: E402

COLOR_LO, COLOR_HI = 10, 14      # 用色数窗口，对称套在管线的 12 上（预注册）
OUT = ROOT / "experiments/coherence_premise.json"


def stat(idx: np.ndarray, seed: int):
    """逐瓦片 d = 观测 checker_rate - 自己的洗牌均值。不合格返回 None。"""
    obs, n2 = checker_rate(idx)
    if obs is None or n2 < MIN_WINDOWS:
        return None
    nulls = shuffled_rates(idx, seed)
    if not nulls:
        return None
    return {"obs": float(obs), "null": float(np.mean(nulls)),
            "d": float(obs - np.mean(nulls)), "n2": n2,
            "ncolors": int(len(np.unique(idx)))}


def tile_gate(rgb: np.ndarray) -> bool:
    """瓦片层的门——与真人组用的是同一行判断（见 `artist_groups`）。"""
    return (dominant_period(rgb, lo=2, hi_frac=0.625) > 0
            and anisotropy(rgb) >= 0.20)


def pipeline_group(paths, seed0: int, source: str = ""):
    recs = []
    for k, p in enumerate(paths):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        idx = as_index_map(rgb)
        s = stat(idx, seed0 + k)
        if s:
            s["material"] = p.name.rsplit("_", 1)[0].replace("_", " ")
            s["source"] = source
            # 只记录、不参与预注册主判据；供 --tile-gate 的敏感性分析重新分组。
            s["tile_gated"] = bool(tile_gate(rgb.astype(float)))
            recs.append(s)
    return recs


def artist_groups():
    """真人瓦片按管线同一个门分两组，只留用色数在窗口内的。"""
    iso, per = [], []
    for k, (mat, idx, pal) in enumerate(tiles()):
        nc = len(np.unique(idx))
        if not (COLOR_LO <= nc <= COLOR_HI):
            continue
        rgb = pal[idx].astype(float)
        gated = (dominant_period(rgb, lo=2, hi_frac=0.625) > 0
                 and anisotropy(rgb) >= 0.20)
        s = stat(idx, 500000 + k)
        if not s:
            continue
        s["material"] = mat
        (per if gated else iso).append(s)
    return iso, per


def _guard():
    if OUT.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUT.name} 已存在。重跑会覆盖这份一手数据，"
                         "确认要重跑就加 --force。")


def describe(tag, recs):
    d = [r["d"] for r in recs]
    print(f"  {tag:<6} n={len(recs):<5} d 中位 {statistics.median(d):+.4f}"
          f"   观测中位 {statistics.median([r['obs'] for r in recs]):.3f}"
          f"   洗牌中位 {statistics.median([r['null'] for r in recs]):.3f}")
    return statistics.median(d)


def tile_gate_mode(P_iso, P_per, A_iso, A_per):
    """敏感性分析：把**瓦片层的门**也套到管线组上，两边用同一个定义。

    不是预注册的一部分。另一会话在**看结果之前**指出（`2b11231`）：真人组的
    "门拒绝" 判在 16x16 瓦片上，管线组却继承自 1024 渲染图的判定，
    29 张里 5 张按瓦片层的门反而算触发——**分组方式本身**是个混淆，
    与已列的"材质集不同"不是一回事。修法不用 GPU（瓦片都在本地），
    所以这里补做，主判据的数**照原样保留在上面**，两种口径并排看。
    """
    pool = P_iso + P_per
    p_iso2 = [r for r in pool if not r["tile_gated"]]
    p_per2 = [r for r in pool if r["tile_gated"]]
    print("\n=== 敏感性分析（非预注册）：两组都用瓦片层的门 ===")
    for tag, g in (("P_iso'", p_iso2), ("P_per'", p_per2)):
        src = {}
        for r in g:
            src[r["source"]] = src.get(r["source"], 0) + 1
        print(f"  {tag} 来源构成：" + "、".join(f"{k} {v}" for k, v in sorted(src.items())))
    if not p_iso2 or not p_per2:
        print("  某一组为空，无法比较")
        return None
    m_pi2 = describe("P_iso'", p_iso2)
    m_ai = statistics.median([r["d"] for r in A_iso])
    m_pp2 = describe("P_per'", p_per2)
    m_ap = statistics.median([r["d"] for r in A_per])
    _, z1, q1, c1 = mannwhitney([r["d"] for r in p_iso2], [r["d"] for r in A_iso])
    _, z2, q2, c2 = mannwhitney([r["d"] for r in p_per2], [r["d"] for r in A_per])
    print(f"  P_iso' vs A_iso：中位差 {m_pi2 - m_ai:+.4f}"
          f"   MW z={z1:+.2f} p={q1:.3g}   共同语言 {c1:.0%}")
    print(f"  P_per' vs A_per：中位差 {m_pp2 - m_ap:+.4f}"
          f"   MW z={z2:+.2f} p={q2:.3g}   共同语言 {c2:.0%}")
    print("  -> 与上面预注册口径的判读一致则结论稳；不一致则**两种都要报**。")
    return {"n": {"P_iso": len(p_iso2), "P_per": len(p_per2)},
            "iso": {"gap": m_pi2 - m_ai, "z": z1, "p": q1, "cles": c1},
            "per": {"gap": m_pp2 - m_ap, "z": z2, "p": q2, "cles": c2},
            "moved": sorted(r["material"] for r in P_iso if r["tile_gated"])}


def main():
    unknown = [a for a in sys.argv[1:] if a not in ("--force", "--tile-gate")]
    if unknown:
        raise SystemExit(f"不认识的参数：{' '.join(unknown)}")
    _guard()

    iso_dir = ROOT / "experiments/iso_point/16"
    p_iso_paths = sorted(iso_dir.glob("*_box.png"))
    if not p_iso_paths:
        raise SystemExit(f"缺 {iso_dir}（B23 的门拒绝瓦片，已入库，净克隆应有）")

    units = json.loads((ROOT / "experiments/bestof_units.json")
                       .read_text(encoding="utf-8"))
    eff = {r["prompt"].replace(" ", "_") for r in units if r["best_eff"]}
    bestof = ROOT / "experiments/bestof"
    p_per_paths = sorted(p for p in bestof.glob("*_best.png")
                         if p.name[:-len("_best.png")] in eff)
    if not p_per_paths:
        raise SystemExit(f"缺 {bestof} 里 best_eff 的瓦片（需从服务器取回）")

    P_iso = pipeline_group(p_iso_paths, 100, "iso_point/box")
    P_per = pipeline_group(p_per_paths, 200, "bestof/best")
    A_iso, A_per = artist_groups()

    print(f"用色数窗口 [{COLOR_LO},{COLOR_HI}]（管线侧实测全为 12 色）\n")
    print("门拒绝（各向同性）")
    m_pi = describe("P_iso", P_iso)
    m_ai = describe("A_iso", A_iso)
    print("门触发（周期）· 特异性对照")
    m_pp = describe("P_per", P_per)
    m_ap = describe("A_per", A_per)

    _, z1, p1, cl1 = mannwhitney([r["d"] for r in P_iso], [r["d"] for r in A_iso])
    _, z2, p2, cl2 = mannwhitney([r["d"] for r in P_per], [r["d"] for r in A_per])
    gap_iso, gap_per = m_pi - m_ai, m_pp - m_ap

    print(f"\n主判据  P_iso vs A_iso：中位差 {gap_iso:+.4f}"
          f"   MW z={z1:+.2f} p={p1:.3g}   共同语言 {cl1:.0%}")
    print(f"对照    P_per vs A_per：中位差 {gap_per:+.4f}"
          f"   MW z={z2:+.2f} p={p2:.3g}   共同语言 {cl2:.0%}")

    overlap = ({r["material"] for r in P_iso} | {r["material"] for r in P_per}) \
        & ({r["material"] for r in A_iso} | {r["material"] for r in A_per})
    print(f"\n描述性：管线材质名与真人材质名的交集 {len(overlap)} 个"
          f"{'：' + '、'.join(sorted(overlap)) if overlap else '（材质集不重叠，此混淆未控）'}")

    res = {"color_window": [COLOR_LO, COLOR_HI],
           "median_d": {"P_iso": m_pi, "A_iso": m_ai, "P_per": m_pp, "A_per": m_ap},
           "n": {"P_iso": len(P_iso), "A_iso": len(A_iso),
                 "P_per": len(P_per), "A_per": len(A_per)},
           "iso": {"gap": gap_iso, "z": z1, "p": p1, "cles": cl1},
           "per": {"gap": gap_per, "z": z2, "p": p2, "cles": cl2},
           "material_overlap": sorted(overlap),
           "tiles": {"P_iso": P_iso, "P_per": P_per, "A_iso": A_iso, "A_per": A_per}}
    if "--tile-gate" in sys.argv:
        res["tile_gate_sensitivity"] = tile_gate_mode(P_iso, P_per, A_iso, A_per)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n--- 主判据（预注册）---")
    ok = gap_iso > 0 and p1 < 0.05
    print(f"  P_iso 中位 > A_iso: {gap_iso > 0}   p<0.05: {p1 < 0.05}")
    if ok:
        print("  -> **前提成立**：调色板匹配后差距仍在。第六条路"
              "（量化里做空间相干性修复）可以立项，目标值 A_iso 中位"
              f" d={m_ai:+.4f}。")
        if gap_per >= gap_iso:
            print("  ⚠ 特异性对照**没通过**：门触发那半的差距不小于这半"
                  f"（{gap_per:+.4f} >= {gap_iso:+.4f}），")
            print("     这是管线 vs 真人的**通病**，不是各向同性特有的；"
                  "第六条路不许说成「修各向同性那半」。")
        else:
            print(f"  特异性对照通过：门触发那半差距更小（{gap_per:+.4f}"
                  f" < {gap_iso:+.4f}），差距在各向同性这半更突出。")
    else:
        print("  -> **前提不成立**：调色板匹配后差距不显著或方向反了。")
        print("     那条线索是混淆，不立项；loop.md 里把它划掉。")
    print("  注意预注册的外部效度限制：只控住了调色板大小这一个混淆，"
          "材质集与来源仍未控——")
    print("  即使成立也只能说「差距不是调色板造成的」，"
          "不能说「差距来自降采样管线」。")


if __name__ == "__main__":
    main()
