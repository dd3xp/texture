"""32px 缺口的第二刀：模型画不出结构，还是它的 32px **训练池**本来就没结构？

**为什么问这个**（`docs/arch_progress.md` 2026-09-12 的下一步）：
上一轮（`613bdbb`，`analysis/arch/scale_diag.py`）把 32px 的缺口定位成"大多数瓦片根本没有结构"：
同 67 材质下 TRD 32px 过门 28%、各向异性中位 **0.130**，真人 32px 过门 64%、**0.534**；
而 16px 那一档 TRD 与真人分不出（0.326 vs 0.287，p=0.29）。六条配置杠杆全否（`36f151a`）。

上一轮的下一步原本写的是"验环面偏置带限假说"。**本轮开工先查了三支 run 的 config.json，
那个假说的前提是错的**：`scale_diag.py` 的 docstring 说"训练数据全是 16x16，32px 是零样本"，
但 `runs/trd_v8|v10|v11d/config.json` 全是 `sizes:[16,32]`，`p32` 分别 **0.3 / 0.5 / 0.7** ——
**32px 一直在训练，而且占了三到七成的批次**。于是问题变成：训了这么多，为什么还是没结构？

最便宜的答案候选：**它训的那批 32px 瓦片本身就没结构**。两条具体理由：
1. 原生 32px 真人瓦片只有 **517 张 train / 156 张 val**（16px 是 3395 / 291）；
2. v11d 额外并入的 `train_64to32.json` 是**把 64px 真人图降采样到 32**，而本项目自己的
   B23 早已量过：**五种降采样器全是格内归约、一律毁方差**（`analysis/arch` 外的
   `experiments/point_sample_iso.json`；跨度 0.109 vs 真人 0.292）。
   若这批占了 v11d 32px 训练池的大头，模型学到的"32px 长什么样"就是被抹平过的。
   这还会**顺带解释已判定的事实**：v11d（p32=0.7，带 64to32）在判官下反而不如 v10（61% vs 39%，p=0.064）。

**跑完之后（2026-09-14 补记，判读规则一个字没改）**：预注册判据判**不定**（判据 3）——
B 组 0.107 < 0.33 但对 val32 的 MW p=0.112，且 16px 那一档对照（判据 4）自己也响了
（0.143 vs 0.197，p=0.0125）。判据 5 成立：A（v10 池）显著好于 B（v11d 池），
0.149 vs 0.107 MW p=0.0027、门 36% vs 23% p=2.3e-09，祸首是 64to32 那 1236 张（门 17%）。
⚠⚠ **下面引作参照的"真人 32px 门 64% / 0.534"，事后查出是一个包**：那个参照组 n=74，
其中 66 张（89%）来自 `ROllerozxa__mtg_tiled_32x`；整份 val32 是 46% / 0.132。
**判据里的 0.33 这条线因此是照着一个包画的**（原样留着不改，改了就是事后改判据），
但别再把 0.534 当"真人 32px"引用。见 `analysis/arch/ref_pack_audit.py`（零 GPU 零 API 可复算）
与 `docs/arch_progress.md` 的「参照组是一个包」那节。

**判读规则（跑之前写死，见 git 提交顺序）**，尺子与 `scale_diag.py` 完全一致：
`tools/downsample.dominant_period(lo=2, hi_frac=0.625)` + `anisotropy`，过门 = 周期>0 且 各向异性>=0.20。
参照数（上一轮同一把尺子）：TRD 32px 门 28% / aniso 0.130；真人 32px 门 64% / aniso 0.534。

1. **H 成立**（病在数据）：B 组 = v11d 实际训练池，其 aniso 中位 **< 0.33**
   （0.130 与 0.534 的中点）**且**对 val32 的 Mann-Whitney p<0.05。
   -> 下一步是修 32px 训练池（剔除/降权 64->32 那部分、或补原生 32px 数据），**不是改架构**。
2. **H 否决**（病在架构/优化）：B 组 aniso 中位 **>= 0.33 且过门率 >= 50%**。
   -> 训练池有结构而模型画不出，下一步仍在架构上。
3. 两条都不满足 -> 记为**不定**，不据此换方向。
4. **对照是必须的**：同时报 16px 的 train vs val。若 16px 那一档 train/val 也差很多，
   说明这是划分本身的性质、不是 32px 专有，**判据 1 当场作废**。
5. 另报 A 组（v10 口径，不含 64to32）与 64to32 子集本身。A 明显好于 B 才算给
   "加 32px 数据反而更差"找到机制；A 与 B 一样差则那条数据线不是主因。

零 GPU、零 API、零磁盘写入（输出 JSON 可指到 /tmp）。extra 文件只在服务器上，
净克隆缺它们时对应组会跳过并在输出里标 missing。

    python analysis/arch/train_pool_structure.py --out /tmp/train_pool_structure.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "model"))
from downsample import dominant_period, anisotropy   # noqa: E402
from tiles_data import load                          # noqa: E402

GATE_ANISO = 0.20
MID = 0.33            # 判据 1/2 的分界：TRD 0.130 与真人 0.534 的中点
PACKS = "train_extra_packs_only.json"
DOWN32 = "train_64to32.json"


def mannwhitney_u_p(a, b):
    """双侧 Mann-Whitney U 的正态近似 p（含并列校正）。抄自 analysis/arch/scale_diag.py。"""
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


def two_prop(k1, n1, k2, n2):
    """两比例差的双侧正态近似 p（无 scipy）。"""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se <= 0:
        return 1.0
    return min(1.0, math.erfc(abs(p1 - p2) / se / math.sqrt(2)))


def measure(sample):
    """瓦片 = 调色板[索引]，与 scale_diag.py 量 PNG 的是同一个数组。"""
    rgb = sample["palette"][sample["idx"]].astype(float)
    per = dominant_period(rgb, lo=2, hi_frac=0.625)
    ani = anisotropy(rgb)
    return {"period": per, "aniso": ani, "gated": bool(per > 0 and ani >= GATE_ANISO)}


def med(xs):
    return float(np.median(xs)) if len(xs) else float("nan")


def summarise(recs):
    g = [r for r in recs if r["gated"]]
    return {"n": len(recs), "n_gated": len(g),
            "gate_rate": (len(g) / len(recs)) if recs else float("nan"),
            "median_aniso": med([r["aniso"] for r in recs]),
            "median_period_gated": med([r["period"] for r in g]),
            "anisos": [r["aniso"] for r in recs]}


def have(fn):
    return (ROOT / "data/tiles" / fn).exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/train_pool_structure.json"))
    a = ap.parse_args()

    groups, missing = {}, []

    def add(name, samples):
        groups[name] = [measure(s) for s in samples]

    base32_train = load(32, "train")
    add("A0 train32 原生(不含extra)", base32_train)
    add("C  val32 真人参照", load(32, "val"))
    add("E  val16 真人参照", load(16, "val"))
    add("D0 train16 原生(不含extra)", load(16, "train"))

    if have(PACKS):
        add("A  train32 池(v10口径: 原生+packs)", load(32, "train", extra=PACKS))
        add("D  train16 池(v10/v11d口径)", load(16, "train", extra=PACKS))
    else:
        missing.append(PACKS)

    if have(DOWN32) and have(PACKS):
        pool_b = load(32, "train", extra=f"{PACKS}+{DOWN32}")
        add("B  train32 池(v11d口径: +64to32)", pool_b)
        # 隔离 64->32 子集：load() 按 rows = 基础 + extra 文件 顺序追加并保序，
        # 所以 v11d 池的前缀就是 v10 池，尾巴就是 64to32 那批。前缀相等要断言，别默认。
        pool_a = load(32, "train", extra=PACKS)
        assert len(pool_b) > len(pool_a), "v11d 池不比 v10 池大，前缀切片的前提不成立"
        for x, y in zip(pool_a, pool_b):
            assert x["material"] == y["material"] and np.array_equal(x["idx"], y["idx"]), \
                "前缀不相等：load() 的顺序假设不成立，不能用切片隔离 64to32"
        add("B- 仅 64to32 子集", pool_b[len(pool_a):])
    elif not have(DOWN32):
        missing.append(DOWN32)

    out = {"gate_aniso": GATE_ANISO, "midpoint": MID, "missing_files": missing,
           "reference_last_round": {"TRD32_gate": 0.28, "TRD32_aniso": 0.130,
                                    "REAL32_gate": 0.64, "REAL32_aniso": 0.534},
           "groups": {}}
    print(f"{'组':<34} {'n':>5} {'过门':>6} {'aniso中位':>10} {'周期(px)':>9}")
    for name, recs in groups.items():
        row = summarise(recs)
        out["groups"][name] = row
        print(f"{name:<34} {row['n']:>5} {row['gate_rate']:>6.0%} "
              f"{row['median_aniso']:>10.3f} {row['median_period_gated']:>9.2f}")

    G = out["groups"]
    tests = [("T1 判据1/2 的主读数: B(v11d池) vs C(val32)", "B  train32 池(v11d口径: +64to32)", "C  val32 真人参照"),
             ("T2 A(v10池) vs C(val32)", "A  train32 池(v10口径: 原生+packs)", "C  val32 真人参照"),
             ("T3 机制: 仅 64to32 vs C(val32)", "B- 仅 64to32 子集", "C  val32 真人参照"),
             ("T4 对照(16px 这一档是成立的): D vs E", "D  train16 池(v10/v11d口径)", "E  val16 真人参照"),
             ("T5 原生 train32 vs val32（同一次划分）", "A0 train32 原生(不含extra)", "C  val32 真人参照")]
    print("\n对比（各向异性 Mann-Whitney 双侧；过门率两比例）")
    out["tests"] = {}
    for lab, ka, kb in tests:
        if ka not in G or kb not in G:
            continue
        _, pa = mannwhitney_u_p(G[ka]["anisos"], G[kb]["anisos"])
        pg = two_prop(G[ka]["n_gated"], G[ka]["n"], G[kb]["n_gated"], G[kb]["n"])
        out["tests"][f"{ka} vs {kb}"] = {"aniso_p": pa, "gate_p": pg}
        print(f"  {lab}")
        print(f"    aniso {G[ka]['median_aniso']:.3f} vs {G[kb]['median_aniso']:.3f}  MW p={pa:.3g}   "
              f"过门 {G[ka]['gate_rate']:.0%} vs {G[kb]['gate_rate']:.0%}  p={pg:.3g}")

    # ---- 预注册判据的机械判读（不许跑完再改）---------------------------------
    print("\n预注册判读")
    kB, kC = "B  train32 池(v11d口径: +64to32)", "C  val32 真人参照"
    kD, kE = "D  train16 池(v10/v11d口径)", "E  val16 真人参照"
    verdict = "无法判读（缺组）"
    if kB in G and kC in G and kD in G and kE in G:
        pctrl = out["tests"][f"{kD} vs {kE}"]["aniso_p"]
        ctrl_gap = abs(G[kD]["median_aniso"] - G[kE]["median_aniso"])
        pB = out["tests"][f"{kB} vs {kC}"]["aniso_p"]
        aB, gB = G[kB]["median_aniso"], G[kB]["gate_rate"]
        print(f"  判据4 对照: 16px train/val aniso {G[kD]['median_aniso']:.3f} vs "
              f"{G[kE]['median_aniso']:.3f}（差 {ctrl_gap:.3f}, p={pctrl:.3g}）")
        if aB < MID and pB < 0.05:
            verdict = "判据1 成立：病在 32px 训练池（数据），下一步修数据不是改架构"
        elif aB >= MID and gB >= 0.50:
            verdict = "判据2 成立：训练池有结构而模型画不出，病在架构/优化"
        else:
            verdict = "判据3：不定，不据此换方向"
        print(f"  B 组 aniso 中位 {aB:.3f}（线 {MID}）、过门 {gB:.0%}、对 val32 p={pB:.3g}")
    print(f"  => {verdict}")
    out["verdict"] = verdict

    # ---- 事后（**不是**预注册的）：按材质包看 train/val 的风格错位 ------------
    # 判据 3 判成"不定"之后才做的，写清楚免得日后被当成预注册结论。
    # 起因：参照组 REALval/32 只有 67 个材质（74 张瓦片）、aniso 中位 0.524，而整份 val32
    # （156 张）只有 0.132。查下来 74 张里 66 张（89%）来自**同一个包**。
    # ⚠ 下面的 ref_pack_counts_Vmat 需要 eval/prompt_sets_val.json；缺它时为 None，
    #   参照组那一侧改由 analysis/arch/ref_pack_audit.py 从已入库数据单独复算。
    print("\n事后（非预注册）：按材质包")
    out["posthoc_packs"] = posthoc_packs()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n->", a.out)


def posthoc_packs():
    """每个材质包的 aniso 中位；并问：val 那一侧的包落在 train 那一侧包的范围内吗？"""
    res = {}
    vmat = None
    ps = ROOT / "eval/prompt_sets_val.json"
    if ps.exists():
        vmat = {p["material"] for p in json.loads(ps.read_text())["V_mat"]}

    def profile(size, split, extra=False):
        by = {}
        for s in load(size, split, extra=extra):
            rgb = s["palette"][s["idx"]].astype(float)
            by.setdefault(s["pack"], []).append(anisotropy(rgb))
        return {p: {"n": len(v), "median_aniso": med(v)} for p, v in by.items()}

    for size in (16, 32):
        tr = profile(size, "train", extra=PACKS if have(PACKS) else False)
        va = profile(size, "val")
        res[f"{size}px"] = {"train_packs": tr, "val_packs": va,
                            "disjoint": sorted(set(tr) & set(va)) == []}
        tr_med = sorted(v["median_aniso"] for v in tr.values())
        print(f"  {size}px: train {len(tr)} 包 / val {len(va)} 包，包名不相交 "
              f"{sorted(set(tr) & set(va)) == []}")
        print(f"    train 包 aniso 中位的范围 [{tr_med[0]:.3f}, {tr_med[-1]:.3f}]")
        for p, v in sorted(va.items(), key=lambda kv: -kv[1]["n"]):
            above = sum(x > v["median_aniso"] for x in tr_med)
            print(f"    val 包 {p:<34} n={v['n']:<4} {v['median_aniso']:.3f}  "
                  f"（train 里有 {above}/{len(tr_med)} 个包比它更高）")
            res[f"{size}px"]["val_packs"][p]["n_train_packs_above"] = above
    if vmat:
        v32 = load(32, "val")
        cnt = {}
        for s in v32:
            if s["material"] in vmat:
                cnt[s["pack"]] = cnt.get(s["pack"], 0) + 1
        res["ref_pack_counts_Vmat"] = cnt
        print(f"  参照组（val32 与 V_mat 的交集）的来源包: {cnt}")
    return res


if __name__ == "__main__":
    main()
