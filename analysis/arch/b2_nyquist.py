"""三档（16 胜 / 24 平 / 32 输）的**分母**是怎么随画布变的？——直接从 B2 的构造里读，不靠判官。

零 GPU、零 API、零远程写入：只跑本机合成图 + 读两份已在库的 JSON，结果写 `experiments/b2_nyquist.json`。

---------------------------------------------------------------- 为什么现在问这一条
`8ec5446` 把三档拆成两个互斥假设，一年来所有 32px 杠杆都按前者设计、全灭：

    (H_TRD) TRD 随画布变差。      -> 账本的准入条件成立，缺口在我们这边。
    (H_B2)  B2 随画布变好。        -> "32px 缺口"这个提法本身要改写。

`eval/tier_anchor.sh` 想用经验方法把两者分开（拿 B1 当降采样族的正对照、B7 当原生生成的锚），
结果 B1@24 那条腿卡在 n=15 的试点门上，整对梯度作废（`d0345de`），**两个假设仍未分开**。

但 (H_B2) 的那一半**根本不需要判官**：B2 是我们自己的管线，它随画布怎么变是**代码里写死的**。
tier_anchor 的预注册第 18-20 行把机制写成了一个似乎合理的推断——
"把 1024px 渲染降到 32px 比降到 16px **必然**保留更多结构"——**但从来没有人去查过**。
本项目已经两次栽在"没查就当前提"上（`edge` 那条解释、`613bdbb` 的真人 32px 参照包），
所以这次先查前提，再谈假设。

---------------------------------------------------------------- 尺子（跑之前定死）
**不引入任何新尺子**。`edge`、结构门、各向异性一概不碰（它们已被禁止当判据）。
本脚本唯一的量是 `tools/downsample.py` 自己的两个量：裁剪边长 `side`，以及由它决定的
**每个结构周期分到的输出像素数** `size * per / side`——这不是一把用来给方法排序的尺子，
而是 B2 这条管线的**构造参数**，读它等于读代码。

---------------------------------------------------------------- 预注册判据（跑之前写死）
 (O1) **实跑检验，不许只读代码**：合成三张周期分别为 16/32/64 px、有方向性（竖条纹）的 1024 图，
      对每张调 `auto_crop(img, size)`，size 取 16/24/32。
        - 若三个 size 给出的 `side` **逐字节相同** -> "B2 三档看的是同一块源内容"成立。
        - 若不同 -> 本脚本的整个前提作废，**立刻停**，并把 tier_anchor 第 18-20 行标为未经检验。
 (O2) **覆盖率先于效应**（`scale_diag.py` 作废那次的教训）：机制只对**走了裁剪路径**的材质成立。
      从 `experiments/baselines/manifest_B1_B2.json` 取 B2 实际用的那一张（`b2_pick`）的 `frac`，
      分三类：下钳位（frac <= 0.0801，per < 18.2px）、正常裁剪（0.0801 < frac < 0.999）、
      未裁剪（frac >= 0.999，**两种情况混在一起**：无周期 或 周期大到 4.5 个放不下，
      见 `downsample.py:321-325`，两者不可分）。
      报三类在 272 材质上的占比。**若"正常裁剪"< 50%，机制只覆盖少数材质，结论降级为局部。**
 (O3) **材质级 2x2 的功效预检（写在看结果之前）**：本想问"24->32 的 18 个翻转材质是不是集中在
      走了裁剪路径的那批"。但机制对所有裁剪材质给出的是**同一个**密度跳变（3.56 -> 7.11，
      与 frac 无关，见 (O1)），因此它**预测不出材质间的差异**；而且若 (O2) 的裁剪占比高，
      2x2 根本没有对比度。**预注册的处理**：先算共同判出的 118 个材质上的裁剪占比 b。
        - b > 0.80 -> **宣布该 2x2 无功效，只报计数、不算 p、不作任何解读**。
        - b <= 0.80 -> 算 Fisher 精确检验（`math.comb`，无 scipy），仍只作事后描述。
      无论哪种情况，**它都不足以开新臂**。
 (O4) **本脚本不授权任何配置变更**。`UNITS_PER_TILE` 一个字不改——`downsample.py:294` 写着
      "改之前必须重跑盲比"，而所有已报结果都出自 4.5 的流水线。这是诊断，不是选型。
 (O5) **它能证明什么、不能证明什么（写在前面，防止事后放大）**：
      能证明的只有"**B2 的分母随画布单调变好**是构造性的、与我们的模型无关"。
      **不能**证明的是这个增益在**判官口径下有多大**——那要另一条臂去量（见脚本末尾的建议）。
      所以本脚本**不判 (H_TRD) 死**，只把 (H_B2) 从"假设"降格为"分母那半边已是事实"。
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP = os.path.join(ROOT, "experiments")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import downsample as D  # noqa: E402

SIZES = (16, 24, 32)
TIER_FILES = {
    "16": "judge_full_TRD16c_rr4_vs_B2_16.json",
    "24": "judge_full_TRD24_rr4_vs_B2_24.json",
    "32": "judge_full_TRD32_rr4_vs_B2_32.json",
}


def stripes(period: int, n: int = 1024) -> np.ndarray:
    """周期 `period` px 的竖条纹：有周期、有方向性，两个门都过。"""
    x = np.arange(n)
    band = ((x // (period // 2)) % 2) * 200.0 + 20.0
    return np.repeat(np.tile(band, (n, 1))[:, :, None], 3, axis=2)


def o1_side_is_size_independent() -> dict:
    out = {}
    for per in (16, 32, 64):
        img = stripes(per)
        sides = {}
        for s in SIZES:
            crop, frac = D.auto_crop(img, s)
            sides[str(s)] = {"side": int(crop.shape[0]), "frac": round(float(frac), 6)}
        vals = {v["side"] for v in sides.values()}
        out[str(per)] = {"per_sizes": sides, "identical": len(vals) == 1,
                         "side": sides["16"]["side"]}
    return out


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """2x2 单侧 Fisher（[[a,b],[c,d]]，问 a 是否偏大）。math.comb 精确，无 scipy。"""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    p = 0.0
    lo = max(0, c1 - b)
    hi = min(r1, c1)
    for k in range(a, hi + 1):
        p += math.comb(r1, k) * math.comb(n - r1, c1 - k)
    return p / math.comb(n, c1)


def main() -> int:
    res = {}

    # ---- (O1) 实跑：side 与目标尺寸无关吗
    o1 = o1_side_is_size_independent()
    res["O1_side_size_independent"] = o1
    ok = all(v["identical"] for v in o1.values())
    res["O1_pass"] = ok
    print("== (O1) 裁剪边长是否与目标尺寸无关（合成竖条纹，1024px 源）")
    for per, v in o1.items():
        marks = "  ".join(f"{s}->side {v['per_sizes'][s]['side']}" for s in ("16", "24", "32"))
        print(f"  周期 {per:>2}px:  {marks}   4.5*per={4.5 * int(per):.0f}   "
              f"{'一致' if v['identical'] else '**不一致**'}")
    if not ok:
        print("  -> (O1) 不通过：前提作废，按预注册立刻停。")
        json.dump(res, open(os.path.join(EXP, "b2_nyquist.json"), "w"), indent=1)
        return 1
    print("  -> 三档看的是**同一块源内容**。原因：target_px = size/UNITS_PER_TILE，")
    print(f"     side = per*size/target_px = per*{D.UNITS_PER_TILE}，size 被**约掉**。")

    # 每个结构周期分到的输出像素数（正常裁剪路径下与材质无关，是个常数）
    ppp = {str(s): round(s / D.UNITS_PER_TILE, 3) for s in SIZES}
    res["px_per_period"] = ppp
    print("\n== 每个结构周期分到的输出像素数（正常裁剪路径，与材质无关）")
    for s in SIZES:
        v = s / D.UNITS_PER_TILE
        print(f"  {s}px 画布: {v:.2f} px/周期   距奈奎斯特下限(2.0) {v / 2:.2f} 倍")
    print("  -> 同一块源内容，采样密度 16->32 恰好翻倍。**这一边的改善是构造性的。**")

    # ---- (O2) 覆盖率
    man = json.load(open(os.path.join(EXP, "baselines", "manifest_B1_B2.json"), encoding="utf-8"))
    cls = {}
    for slug, e in man.items():
        f = e["samples"][e["b2_pick"]]["frac"]
        if f >= 0.999:
            c = "uncropped"
        elif f <= 0.0801:
            c = "lower_clipped"
        else:
            c = "cropped"
        cls[slug] = {"frac": f, "class": c}
    counts = {k: sum(1 for v in cls.values() if v["class"] == k)
              for k in ("cropped", "lower_clipped", "uncropped")}
    res["O2_classes"] = counts
    res["O2_n"] = len(cls)
    frac_cropped = counts["cropped"] / len(cls)
    res["O2_frac_cropped"] = round(frac_cropped, 4)
    print("\n== (O2) 机制覆盖率：B2 实际用的那一张走了哪条路径（272 材质）")
    for k, lab in (("cropped", "正常裁剪 (side=4.5*per)"),
                   ("lower_clipped", "下钳位 (per<18.2px，更糊)"),
                   ("uncropped", "未裁剪 (无周期 或 周期过大，两者不可分)")):
        print(f"  {lab:<38} {counts[k]:>4}  {counts[k] / len(cls):6.1%}")
    print(f"  -> 机制覆盖 {frac_cropped:.0%} 的测试材质"
          f"{'（>50%，结论适用于整体）' if frac_cropped > 0.5 else '（<=50%，结论降级为局部）'}")

    # ---- (O3) 材质级 2x2 的功效预检
    decided = {}
    for tier, fn in TIER_FILES.items():
        d = json.load(open(os.path.join(EXP, fn), encoding="utf-8"))
        decided[tier] = {r["material"]: r["verdict"] for r in d["records"]
                         if r["verdict"] in ("A", "B")}
    common = sorted(set(decided["24"]) & set(decided["32"]) & set(decided["16"]))
    res["O3_common_n"] = len(common)

    # 判官记的 `material` 是 `e["prompt"]`（judge_pairs.py:112,124），manifest 也存了 prompt
    # （sdxl_baselines.py:78），所以用 prompt 做**精确**映射，别按名字猜下划线。
    by_prompt = {}
    for slug, e in man.items():
        by_prompt.setdefault(e["prompt"], []).append(slug)
    ambiguous = {p: s for p, s in by_prompt.items()
                 if len({cls[x]["class"] for x in s}) > 1}

    def slug_of(m):
        s = by_prompt.get(m)
        return s[0] if s and m not in ambiguous else None

    miss = [m for m in common if slug_of(m) is None]
    res["O3_unmapped"] = miss
    res["O3_ambiguous_prompts"] = len(ambiguous)
    mapped = [m for m in common if slug_of(m) is not None]
    b = sum(1 for m in mapped if cls[slug_of(m)]["class"] == "cropped") / max(len(mapped), 1)
    res["O3_base_rate_cropped"] = round(b, 4)
    flips = [m for m in mapped if decided["24"][m] == "A" and decided["32"][m] == "B"]
    rflips = [m for m in mapped if decided["24"][m] == "B" and decided["32"][m] == "A"]
    fc = sum(1 for m in flips if cls[slug_of(m)]["class"] == "cropped")
    rc = sum(1 for m in rflips if cls[slug_of(m)]["class"] == "cropped")
    res["O3"] = {"n_mapped": len(mapped), "flips_24win_32lose": len(flips),
                 "flips_cropped": fc, "reverse_flips": len(rflips), "reverse_cropped": rc}
    print(f"\n== (O3) 材质级 2x2 的功效预检（三档共同判出 {len(common)} 个，"
          f"能对上 manifest 的 {len(mapped)} 个）")
    print(f"  共同判出集里走裁剪路径的底率 b = {b:.1%}")
    print(f"  24 赢/32 输的翻转 {len(flips)} 个，其中裁剪 {fc}；反向翻转 {len(rflips)} 个，其中裁剪 {rc}")
    if b > 0.80:
        res["O3_read"] = "no_power"
        print("  -> b > 80%：按预注册 (O3)，**该 2x2 无功效，只报计数、不算 p、不作解读**。")
        print("     机制对所有裁剪材质给的是同一个密度跳变，本来就预测不出材质间差异。")
    else:
        rest = len(mapped) - len(flips)
        rest_c = sum(1 for m in mapped if m not in flips
                     and cls[slug_of(m)]["class"] == "cropped")
        p = fisher_exact_greater(fc, len(flips) - fc, rest_c, rest - rest_c)
        res["O3_read"] = "fisher"
        res["O3_p"] = p
        print(f"  -> Fisher 单侧 p = {p:.4g}（**仅事后描述，不足以开臂**）")

    json.dump(res, open(os.path.join(EXP, "b2_nyquist.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n-> experiments/b2_nyquist.json")
    print("\n== (O5) 本脚本**没有**测的东西")
    print("  这个构造性增益在**判官口径下值多少**，仍未量过。能量它的最便宜的臂：")
    print("  把 B2@16 最近邻放大 2x 当作「同内容、但没拿到额外输出像素」的 32px 对照，")
    print("  判 B2@32 vs up2(B2@16)。零 GPU（只是 PIL 放大已在盘的瓦片）。")
    print("  注意：那条臂要**另行预注册**，且按 `d0345de` 的教训把试点门加在「对」上而不是每条腿。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
