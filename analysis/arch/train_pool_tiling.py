#!/usr/bin/env python
"""(M23) 预注册：**训练集** 32px 池是不是也由"16px 内容平铺两遍"主导？零 GPU、零 API、零判官。

**为什么问这个**（上一轮 `515efa5` 写死的"下一步 1"，原文照抄）：
V_mat 的 32px **验证参照**里 83/113 来自 `ROllerozxa__mtg_tiled_32x`，逐像素恰是 16px 平铺两遍
（判决 `REF_PERIODIC_FRAGILE`，只能引方向）。那一条只动**验证参照**。真正要紧的下一问是：
**训练侧**的 32px 池呢？若训练料本身大多不含 32px 尺度的信息，
「32px 为什么偏偏坏」就有了一个**数据侧的候选答案**；若训练侧干净，则上一条只是验证参照的问题，
该线就此关闭。⚠ 本脚本一个胜负数字都不看。

口径 = 训练脚本真正吃的那份：`load(32, "train", extra="train_extra_packs_only.json+train_64to32.json")`
（v11d 及其后；decontam 默认开）。⚠ 这两个 extra 文件**只在服务器上**（gitignore）→ 本脚本必须上远程跑。

仪器 = 上一轮 `tile2_degeneracy.py` 那一把，一个字没改：
  - `exact` —— 整张图与"左上 16×16 平铺两遍"**逐像素完全相同**的比例；
  - `mad`   —— 同一比较下**每通道绝对差的均值**（0–255），报中位数。
    ⚠ 项目规矩：量图像差异只许用这个，不许数"多少像素不相等"。

================================ 判据（跑之前写死） ================================

**门沿用上一轮那两个数，不许重挑**：MAD 中位 ≤ 0.69 = 平铺侧；≥ 9.47 = 干净侧。

主判据（对"实际训练池"这一个合计读数）：
  - **P_TRAIN_PERIODIC**：exact_rate ≥ 50%  **且** mad_median ≤ 0.69
  - **P_TRAIN_CLEAN**   ：exact_rate ≤ 10%  **且** mad_median ≥ 9.47
  - 其余一律 **UNDECIDED**。⛔ 不许挑合上的那一半讲、⛔ 不许事后改门。

操作检验（**先读、不过就作废主判据**）：
  (OP1) 尺子非平凡：同一把尺子问 **16px 训练池**"是不是 8px 平铺两遍"。
        若该 exact_rate ≥ 20% → 这把尺子在本语料上普遍触发、32px 的读数不可解读
        → 判 `RULER_TRIVIAL`，本轮作废。
  (OP2) 样本量：合计 n ≥ 300 才下主判据（现池约 1877 张，留足余量）。
        单个源 n < 30 的不单独报读数，只记入合计。

聚集检验（(M16) 纪律，**主判据之后必跑**）：
  (C) 按**包**报逐包读数 + HHI；再做 LOPO（逐个去掉一个包重算合计判决）。
      若存在任一包，去掉它就翻转主判决 → 判决加后缀 **`_FRAGILE`**，
      **只能引方向、⛔ 不许引合计百分比当"训练池"的一般结论**。

判决之后能做什么（也跑之前写死）：
  - `P_TRAIN_PERIODIC` → 记一条**候选**：32px 流的训练料本身大多不含 32px 尺度信息。
    ⛔ 仍**不等于** 32px 缺口被解释；32px 准入条件（`6c64796`）一个字不松；
    ⛔ 不授权任何新臂、不改 `final_test.sh`／任何默认值。
  - `P_TRAIN_CLEAN`    → 上一轮那条只是**验证参照**的问题，"数据平铺"这条线关闭、不再提。
  - `UNDECIDED`        → 只记账。⛔ 不许读成任何一侧。

用法（远程）：
    python analysis/arch/train_pool_tiling.py --out /tmp/train_pool_tiling.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from tiles_data import load  # noqa: E402

EXTRA = "train_extra_packs_only.json+train_64to32.json"
GATE_PERIODIC_MAD = 0.69      # 上一轮 tile2 上界组的 MAD 中位门（`515efa5`）
GATE_CLEAN_MAD = 9.47         # 上一轮控制组 v11dx_direct 的 MAD 中位门
GATE_PERIODIC_EXACT = 0.50
GATE_CLEAN_EXACT = 0.10
OP1_TRIVIAL_EXACT = 0.20
OP2_MIN_N = 300


def readout(samples):
    """每张：与"左上半边平铺两遍"比。返回逐张 (exact, mad)。"""
    ex, mads = [], []
    for s in samples:
        a = s["palette"][s["idx"]].astype(np.int16)      # [n,n,3]，逐像素等于原图
        n = a.shape[0]
        h = n // 2
        t = np.tile(a[:h, :h], (2, 2, 1))
        ex.append(bool(np.array_equal(a, t)))
        mads.append(float(np.abs(a - t).mean()))
    return np.array(ex), np.array(mads)


def summarise(ex, mads):
    return {"n": int(ex.size),
            "exact": int(ex.sum()),
            "exact_rate": float(ex.mean()) if ex.size else None,
            "mad_median": float(np.median(mads)) if mads.size else None,
            "mad_mean": float(mads.mean()) if mads.size else None}


def verdict(d):
    """主判据。返回 P_TRAIN_PERIODIC / P_TRAIN_CLEAN / UNDECIDED。"""
    if d["n"] == 0:
        return "UNDECIDED"
    if d["exact_rate"] >= GATE_PERIODIC_EXACT and d["mad_median"] <= GATE_PERIODIC_MAD:
        return "P_TRAIN_PERIODIC"
    if d["exact_rate"] <= GATE_CLEAN_EXACT and d["mad_median"] >= GATE_CLEAN_MAD:
        return "P_TRAIN_CLEAN"
    return "UNDECIDED"


def hhi(counts):
    tot = sum(counts)
    return float(sum((c / tot) ** 2 for c in counts)) if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=EXTRA)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    res = {"extra": a.extra, "gates": {
        "periodic_exact": GATE_PERIODIC_EXACT, "periodic_mad": GATE_PERIODIC_MAD,
        "clean_exact": GATE_CLEAN_EXACT, "clean_mad": GATE_CLEAN_MAD,
        "op1_trivial_exact": OP1_TRIVIAL_EXACT, "op2_min_n": OP2_MIN_N}}

    # ---------------- (OP1) 尺子非平凡：16px 训练池问"是不是 8px 平铺两遍" ----------------
    s16 = load(16, "train", extra=a.extra)
    e16, m16 = readout(s16)
    op1 = summarise(e16, m16)
    op1["pass"] = bool(op1["exact_rate"] < OP1_TRIVIAL_EXACT)
    res["op1_ruler_16to8"] = op1
    print(f"(OP1) 16px 训练池 vs 8px 平铺两遍：n={op1['n']}  逐像素全同 {op1['exact']} "
          f"= {op1['exact_rate']:.1%}  MAD 中位 {op1['mad_median']:.2f}  "
          f"-> {'过' if op1['pass'] else '不过 => RULER_TRIVIAL，本轮作废'}", flush=True)

    # ---------------- 主读数：实际训练用的 32px 池 ----------------
    s32 = load(32, "train", extra=a.extra)
    e32, m32 = readout(s32)
    pooled = summarise(e32, m32)
    res["pool32_train"] = pooled
    op2 = bool(pooled["n"] >= OP2_MIN_N)
    res["op2_pass"] = op2
    print(f"\n32px 训练池（extra={a.extra}）：n={pooled['n']}  逐像素全同 {pooled['exact']} "
          f"= {pooled['exact_rate']:.1%}  MAD 中位 {pooled['mad_median']:.2f} "
          f"均值 {pooled['mad_mean']:.2f}")
    print(f"(OP2) n>={OP2_MIN_N}? {'过' if op2 else '不过'}", flush=True)

    v = verdict(pooled)
    if not (op1["pass"] and op2):
        v = "RULER_TRIVIAL" if not op1["pass"] else "UNDERPOWERED"
    res["verdict_main"] = v

    # ---------------- 分源（描述性；64to32 是 64px 降采样，本来就该另算） ----------------
    # 源的判别：base = dataset_k16.json 本体；其余两份 extra 只能按包名回推 → 直接按文件重读一遍。
    by_src = {}
    base_ids = {(s["pack"], s["material"]) for s in load(32, "train", extra=False)}
    for name, keep in (("dataset_k16(base)", lambda s: (s["pack"], s["material"]) in base_ids),
                       ("extra(packs_only+64to32)", lambda s: (s["pack"], s["material"]) not in base_ids)):
        sub = [s for s in s32 if keep(s)]
        if not sub:
            continue
        ee, mm = readout(sub)
        by_src[name] = summarise(ee, mm)
    res["by_source"] = by_src
    print("\n分源（描述性，不作判据）")
    for k, d in by_src.items():
        print(f"  {k:<26} n={d['n']:>5}  全同 {d['exact_rate']:>6.1%}  MAD 中位 {d['mad_median']:>7.2f}")

    # ---------------- (C) 聚集：逐包 + HHI + LOPO ----------------
    idx_by_pack = defaultdict(list)
    for i, s in enumerate(s32):
        idx_by_pack[s["pack"]].append(i)
    packs = []
    for p, ids in sorted(idx_by_pack.items(), key=lambda kv: -len(kv[1])):
        ii = np.array(ids)
        packs.append({"pack": p, **summarise(e32[ii], m32[ii])})
    res["by_pack"] = packs
    res["pack_hhi"] = hhi([p["n"] for p in packs])
    res["n_packs"] = len(packs)
    print(f"\n(C) 逐包：{len(packs)} 个包，HHI={res['pack_hhi']:.3f}（前 8）")
    for p in packs[:8]:
        print(f"  {str(p['pack'])[:44]:<44} n={p['n']:>5}  全同 {p['exact_rate']:>6.1%}  "
              f"MAD 中位 {p['mad_median']:>7.2f}")

    flips = []
    for p, ids in idx_by_pack.items():
        mask = np.ones(e32.size, bool)
        mask[np.array(ids)] = False
        if mask.sum() == 0:
            continue
        if verdict(summarise(e32[mask], m32[mask])) != verdict(pooled):
            flips.append(p)
    res["lopo_flips"] = flips
    fragile = bool(flips)
    res["verdict"] = v + ("_FRAGILE" if fragile and v in
                          ("P_TRAIN_PERIODIC", "P_TRAIN_CLEAN") else "")
    print(f"\nLOPO：去掉任一包会翻转主判决的包 = {flips if flips else '无'}"
          f"  -> {'FRAGILE，只能引方向' if fragile else '稳健'}")
    print(f"\n判决：{res['verdict']}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
