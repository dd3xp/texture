"""32px 训练池的**内容**盘点：它真的比 16px 池"平"吗？（零 GPU、零 API、只读已入库 JSON）

**为什么问这个，以及为什么现在问**：`eval/val_nod32.sh`（预注册 `7e96e2c`）起的那条臂假设
"这 641 张 32px 训练数据是净负的"，并给了一句**机制猜想**：

    "641 张几乎全来自两三个近乎平涂的大包 -> 32px 那半个训练预算学的是'这几个包长什么样'"

臂本身只判"符号"（撤掉数据后 32px 变好还是变坏），**判不了这句机制**。而这句话此前
**从未被量过**：`analysis/arch/data32_supply.py` 量的是**来源集中度**（HHI / 有效包数），
`analysis/arch/train_pool_structure.py` 量过内容但用的是**结构门**（周期 + 各向异性）——
那把尺子在本项目已被判定**不可信作依据**（TRD 自家配置间不排序、温度轴上与判官反向 24pp、
它的"真人 32px"参照被证明是一个包，见 `ref_pack_audit.py`）。

所以本脚本用**另一把尺子**、并且**按材质配对**，在臂出结果之前把这句机制断言钉死成事实或推翻。
**跑在结果之前**：写这个脚本时 `arch_nod32` 还在排队（`/tmp/nod32.txt` 0 字节，一步都没训）。

---------------------------------------------------------------- 尺子（跑之前定死，三个都不带阈值）
逐瓦片，在**归一化后的色阶网格**（`tiles_data.load` 给的 `idx`，值 = 按亮度排的秩）上量：
  - `k_used`：用了几种颜色。
  - `flat`  ：最高频色阶占的格子比例（1.0 = 纯平涂）。**越大越平**。
  - `edge`  ：环面上相邻格（右、下各一遍）**颜色不同**的比例。**越小越平**。
这三个都是平铺纹理的朴素内容统计，**没有阈值、没有挑选**，与结构门无关（不算周期、不算各向异性）。

⚠ **本脚本不是"挑池子/挑配置"的判据**，一个胜负数字都不看，也不产出任何可拿去调参的排序。
它只回答"32px 池的内容比 16px 池更平吗"，用来**解释**臂的结果，不用来**替代**它。

---------------------------------------------------------------- 主判读（配对，跑之前写死）
项目吃过"跨集合中位数不能说成从 A 降到 B"的亏（记忆里的口径陷阱，全项目扫查过三处）。
所以主读数是**按材质配对**：只取 16px 池与 32px 池**都有**的材质，每个材质在各自尺寸上
取该材质所有瓦片的中位数，再逐材质作差，用**符号检验**（精确二项，无 scipy）。

  H_flat（机制断言成立）：配对下 32px 侧**更平**——`edge` 显著更低（符号检验 p<0.05）
                          且中位差的方向一致。
  H_flat 被推翻        ：`edge` 无显著差异，或方向相反（32px 反而更"花"）。
  另报 `flat` 与 `k_used` 两个尺子作旁证；三个尺子不一致就如实说不一致。

⚠ 配对的已知限制（跑之前写下）：同一材质在两个尺寸上**可能来自不同的包**，配对控住的是
"画的是什么材质"，控不住"谁画的"。所以脚本同时报一个把包也控住的子集。
  ⚠⚠ **这个副控制第一版写坏了，已修（结果公布前）**：原先要求同材质两侧的**包集合完全相等**，
  而 16px 一个材质常横跨十来个包、32px 常只有一两个 -> 子集恒为空（实跑 n=0），什么都没控住。
  （它不是"两个池子的包不相交"：24 个 32px 包里 21 个也出现在 16px 池里。）
  改成按 **(材质, 包)** 配对：同一个包画的同一个材质，16px 与 32px 各取一次。这**更严**，
  连画师都控住了。**主判据（全部配对材质上的 `edge`）一个字没改**，副控制的修正不影响它。
⚠ 另一个限制：32px 瓦片有 4 倍的格子，`edge` 与 `flat` 都是**按格数归一化**的比例，
可比；但"同一个材质在更大画布上画得更细"本身就是画师的选择，本脚本量的是**画出来的结果**，
不是"该不该更细"。

---------------------------------------------------------------- 顺带回答"两三个大包"
另按包给 32px 池的分布（张数、占比、各尺子的中位数），看排头那几个包是不是真的比池子整体更平。

    python analysis/arch/data32_content.py --out /tmp/data32_content.json

extra 文件只在服务器上；净克隆缺它时自动退回"不含 extra"的原生池并在输出里标 missing。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "model"))
from exact import binom_test                       # noqa: E402
from tiles_data import load                        # noqa: E402

PACKS = "train_extra_packs_only.json"
RULERS = ("edge", "flat", "k_used")


def measure(s):
    g = s["idx"]
    n = g.shape[0]
    diff = int((g != np.roll(g, -1, axis=1)).sum() + (g != np.roll(g, -1, axis=0)).sum())
    counts = np.bincount(g.ravel())
    return {"edge": diff / (2 * n * n),
            "flat": float(counts.max()) / (n * n),
            "k_used": float(s["k_used"]),
            "material": s["material"], "pack": s["pack"]}


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def quarts(xs):
    if not len(xs):
        return [float("nan")] * 3
    return [float(np.percentile(xs, q)) for q in (25, 50, 75)]


def by_material(recs):
    out = {}
    for r in recs:
        out.setdefault(r["material"], []).append(r)
    return out


def by_key(recs):
    out = {}
    for r in recs:
        out.setdefault((r["material"], r["pack"]), []).append(r)
    return out


def sign_test(pairs):
    """pairs = [(x16, x32)]；返回 (32 更小的个数, 有效对数, p, 配对差的中位数)。

    差为 0 的对按惯例弃用（不计入 n），并在输出里报出来。"""
    d = [b - a for a, b in pairs]
    nz = [x for x in d if x != 0]
    lower = sum(1 for x in nz if x < 0)
    p = binom_test(lower, len(nz)) if nz else float("nan")
    return lower, len(nz), p, med(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/data32_content.json"))
    a = ap.parse_args()

    have = (ROOT / "data/tiles" / PACKS).exists()
    extra = PACKS if have else False
    missing = [] if have else [PACKS]
    p16 = [measure(s) for s in load(16, "train", extra=extra)]
    p32 = [measure(s) for s in load(32, "train", extra=extra)]
    out = {"extra_file": PACKS if have else None, "missing_files": missing,
           "n16": len(p16), "n32": len(p32), "rulers": list(RULERS)}
    print(f"训练池（extra={'有' if have else '无（净克隆口径）'}）：16px {len(p16)} 张、32px {len(p32)} 张")

    # ---- 池级（跨集合，只作背景，不许说成"从 A 降到 B"）-----------------------
    print(f"\n池级四分位（跨集合口径，仅背景）  {'25%':>8}{'中位':>8}{'75%':>8}")
    out["pool"] = {}
    for r in RULERS:
        q16, q32 = quarts([x[r] for x in p16]), quarts([x[r] for x in p32])
        out["pool"][r] = {"p16": q16, "p32": q32}
        print(f"  {r:<7} 16px  {q16[0]:>8.3f}{q16[1]:>8.3f}{q16[2]:>8.3f}")
        print(f"  {r:<7} 32px  {q32[0]:>8.3f}{q32[1]:>8.3f}{q32[2]:>8.3f}")

    # ---- 主读数：按材质配对 ---------------------------------------------------
    m16, m32 = by_material(p16), by_material(p32)
    both = sorted(set(m16) & set(m32))
    # 副控制（见文件头）：按 (材质, 包) 配对，连画师也控住
    k16, k32 = by_key(p16), by_key(p32)
    both_mp = sorted(set(k16) & set(k32))
    out["n_materials"] = {"p16": len(m16), "p32": len(m32), "paired": len(both),
                          "paired_material_pack": len(both_mp)}
    print(f"\n配对：16px 池 {len(m16)} 个材质、32px 池 {len(m32)} 个；两侧都有的 {len(both)} 个"
          f"；按 (材质, 包) 两侧都有的 {len(both_mp)} 组")

    out["paired"] = {}
    for subset, keys, src16, src32 in (("全部配对材质", both, m16, m32),
                                       ("同材质同包子集", both_mp, k16, k32)):
        print(f"\n  [{subset}] n={len(keys)}")
        out["paired"][subset] = {}
        for r in RULERS:
            pairs = [(med([x[r] for x in src16[m]]), med([x[r] for x in src32[m]])) for m in keys]
            lower, n, p, dmed = sign_test(pairs)
            a16, a32 = med([x for x, _ in pairs]), med([y for _, y in pairs])
            out["paired"][subset][r] = {"n_pairs": len(pairs), "n_nonzero": n,
                                        "n_32_lower": lower, "p": p,
                                        "median_paired_diff": dmed,
                                        "median_16": a16, "median_32": a32}
            print(f"    {r:<7} 16px 中位 {a16:.3f} / 32px 中位 {a32:.3f}；"
                  f"配对差中位 {dmed:+.3f}；32px 更小 {lower}/{n} = "
                  f"{(lower / n if n else float('nan')):.0%}  p={p:.3g}")

    # ---- 机制断言里的"两三个大包" --------------------------------------------
    packs = {}
    for x in p32:
        packs.setdefault(x["pack"], []).append(x)
    top = sorted(packs.items(), key=lambda kv: -len(kv[1]))[:8]
    out["packs32"] = {p: {"n": len(v), "share": len(v) / len(p32),
                          "edge": med([x["edge"] for x in v]),
                          "flat": med([x["flat"] for x in v]),
                          "k_used": med([x["k_used"] for x in v])} for p, v in packs.items()}
    pool_edge = med([x["edge"] for x in p32])
    print(f"\n32px 池按包（前八；池子整体 edge 中位 {pool_edge:.3f}）")
    print(f"  {'张数':>5} {'占比':>6} {'edge':>7} {'flat':>7} {'k':>5}  包")
    for p, v in top:
        row = out["packs32"][p]
        print(f"  {row['n']:>5} {row['share']:>6.0%} {row['edge']:>7.3f} "
              f"{row['flat']:>7.3f} {row['k_used']:>5.1f}  {p}")

    # ---- 机械判读（按上面写死的判据）------------------------------------------
    e = out["paired"]["全部配对材质"]["edge"]
    if e["n_nonzero"] and e["p"] < 0.05 and e["median_paired_diff"] < 0:
        verdict = ("H_flat 成立：同材质下 32px 侧显著更平（edge 更低）"
                   " -> 预注册里那句机制猜想得到独立支持")
    elif e["n_nonzero"] and e["p"] < 0.05 and e["median_paired_diff"] > 0:
        verdict = ("H_flat 反向：同材质下 32px 侧显著**更花**（edge 更高）"
                   " -> 那句机制猜想与数据相反，臂即使赢也不能用它解释")
    else:
        verdict = ("H_flat 未获支持：同材质下两个尺寸的 edge 测不出差别"
                   " -> 那句机制猜想没有证据，臂的结果只能读成'符号'本身")
    out["verdict"] = verdict
    print(f"\n判读 => {verdict}")
    print("提醒：本脚本是描述性的，不构成任何换配置/换池子的依据；32px 那条臂的判读"
          "一律以 eval/val_nod32.sh 里预注册的判据为准。")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", a.out)


if __name__ == "__main__":
    main()
