#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M16) 真人 32px 语料的有效自由度只有 ~2.7 个包——那些 16-vs-32 的 CI 还算数吗？
（预注册：本文件先提交再跑）

## 为什么问这个

本项目所有"真人瓦片按画布怎么变"的结论，都把**瓦片**当独立单位重采样：
`gen_scale.py:146` 的 C（Δ 的瓦片级自助，403 张 32px）、`scale_prior.py:155`（同上）、
`repeat_count.py` 的 (T1)（材质级精确二项，192 个材质）。

跑本文件之前先数了一遍包（**这是描述性的、已经看过了，不是本轮的判据**）：

| | 瓦片数 | 包数 | HHI | 1/HHI |
|---|---|---|---|---|
| 16px train | 2732 | 47 | 0.048 | ~21 |
| **32px train** | **403** | **19** | **0.370** | **~2.7** |

32px 那一侧 **85% 来自两个包**（`ROllerozxa__macrotex` 175 + `sirrobzeroone__dungeonsoup` 168）。
同一个包里的瓦片出自同一个作者、同一套惯例 → **不是独立样本**。
把 403 张当 403 个独立单位，CI 会窄成假的；而 16px 那侧（47 包、HHI 0.048）几乎没有这个问题
→ **偏差不对称，专坑"16 比 32"这一类比较**。

这解释得了一件已经发生的事：(M15) 的 (T1) 在 train 上 p=1.1e-09、在 val 上 p=0.80
（`08cedc2` 第四节），而 val 的 32px 只有 4 个包、train 只有 19 个（其中 2 个占 85%）。
账本当时写的是"那多半是 train 那批包的性质"——本文件把这句话从**推测**变成**可检验的量**。

**零 GPU、零 API、零判官、零新臂、零配置改动。** 只读 `data/tiles/dataset_k16.json`
的 train split（`extra=False`，与 (M9)/(M14)/(M15) 口径一致）。**不碰 test split。**

## 主判据 (A)：给 C 换重采样单位

选 **C（`gen_scale.py` 的 `C_real_16_32`）** 当主对象，理由跑前写死：它是目前**唯一**
不含相关长度 L 的尺子（(M12)/`5042ca5` 已禁 L 与 R 的数值和 CI），也是 (M14) 唯一的正向收获
——"(M9) 方向多一把不含 L 的尺子"。它若站不住，(M9) 的方向就只剩被禁引数值的那把尺子。

统计量 **Δ = E_canvas − E_cell** 与 `gen_scale.delta` **逐字复用**（import，不复制）。
已发表值：Δ=+0.012291、瓦片级 95%CI [+0.003924, +0.018256]、判 `G2_cell`。

- **(A1) 包级自助**：两档**各自**按包有放回重采样（抽 n_packs 个包，取其全部瓦片），
  B=2000、seed=0，取 Δ 的 95% 百分位区间。
- **(A2) 留一包法 (LOPO)**：对 32px train 的每个包，把该包的瓦片**从两档同时**删掉，
  重算 Δ。只看**符号**，不看幅度（理由见混杂 3）。

**判据（跑前写死，不许改）**

- **(V1) 稳健**：(A1) CI 下界 > 0 **且** (A2) 所有合格 fold 的 Δ > 0
  → C 不是包级产物，(M9) 的方向保住它的第二把尺子，本轮什么都不改。
- **(V2) 降级**：(A1) CI 跨 0 **或** 任一合格 fold 的 Δ ≤ 0
  → C 的**已发表 CI 不可用**；今后引用 C 必须同时引用本轮。
  - 若 (A2) 所有 fold 仍同号、只是 (A1) CI 跨 0 → 记 **(V2a) 方向稳健、证据不足**。
  - 若有 fold 翻号 → 记 **(V2b) 连方向都是包级的**。
- ⚠ (V2) **不等于 "C 是假的"**，只等于"证据强度被高估"。**不许**把 (V2) 读成反向证据。

## 次判据 (B)：同样两件事施加到 (M15) 的 (T1) train

(M15) 的判决已经是 `UNDECIDED`，**本轮不修改它**；(B) 只回答"p=1.1e-09 这个数字有多虚"。

配对材质按 **32px 侧的众数包**归属（同数并列取字典序最小；这是跑前定死的口径，
也是唯一的自由选择，照实披露）。

- **(B1) 包×桶独立性置换检验**：列联表 pack × {cell, canvas, other}，统计量 = Pearson χ²，
  **置换桶标签**（B=20000、seed=0，行列边际自动固定）。p<0.05 → 桶归属与包**不独立**
  → 材质不是独立单位。
- **(B2) 按包自助的 w**：抽包（有放回）→ 取其全部材质 → 重算 w = n_cell/(n_cell+n_canvas)，
  B=2000、seed=0，95% 百分位 CI 是否仍不含 0.5。

## 操作检验（任一不过 → 对应判决作废，照实写"没测到"）

- **(OP1)** 全量复算必须**精确复现** `experiments/gen_scale.json` 的 C：
  `delta` / `E_cell` / `E_canvas` 三个数按 1e-12 相对容差相同（证明尺子是逐字复用的）。
- **(OP2)** **瓦片级自助锚点**：用 seed=0、B=2000 的**瓦片级**自助复现已发表 CI
  [+0.003924, +0.018256]（同一 RNG 调用序列 → 应精确相同）。
  这一条证明 (A1) 与已发表数字之间**只有重采样单位变了**。
- **(OP3)** 每张瓦片恰属一个包，且各包瓦片数之和 = 瓦片总数。
- **(OP4)** 有效自助样本 ≥ 1900（(A1)、(B2) 各自）。
- **(OP5)** LOPO 的某个 fold 若剩余瓦片 < 100（任一档），该 fold 记为 `skipped`，
  **不参与 (V1)/(V2) 判据**；若合格 fold < 2 → (A2) 作废。

## 跑之前就写明的混杂（结果是哪个方向都不许事后改口径）

1. **本轮不是盲设计**：包计数（175/168/34/…）与 HHI=0.370 我**跑前已经看过**，
   正是它触发了本轮。没看过的是 (A1)/(A2)/(B1)/(B2) 的**任何输出**。
2. 只有 19 个包、其中 17 个 ≤6 张 → 包级自助的 CI **必然变宽**。
   **"变宽"本身不是发现**；判据看的只是**是否仍不含 0**。
3. 删掉 macrotex 或 dungeonsoup 各砍掉 32px 的 43%/42% → 单 fold 噪声大，
   所以 (A2) **只看符号**。
4. 16px 侧（47 包、HHI 0.048）本来就不太需要包级重采样，但为对称仍照做；
   这会让 (A1) 比"只改 32px 侧"更保守，**这是跑前的选择，不许事后换成单侧**。
5. (B) 的众数包归属：材质名可能跨包出现，众数是唯一的自由选择，见上。
6. `extra=False`，与 (M9)/(M14)/(M15) 一致；(M11)（`ca77839`）已证派生包与原生包同尺。

## 授权（跑前写死）

- **(V1)** → 什么都不授权，只把 C 的地位钉牢。
- **(V2)** → **不授权任何新臂、不授权训练、不授权判官**。只授权两件纸面上的事：
  ①在账本与记忆里给 C 加引用禁令；②把"**真人语料的统计以包为重采样单位**"
  写成今后这类分析的默认规矩。
- 无论判决如何：**不训练、不调判官、不出胜率、不改 `final_test.sh` / `UNITS_PER_TILE` /
  `judge_pairs.py` / 任何默认值。**
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import tiles_data  # noqa: E402
from analysis.arch.scale_prior import tile_curves  # noqa: E402  尺子逐字复用
from analysis.arch.gen_scale import delta  # noqa: E402  Δ 逐字复用
from analysis.arch.repeat_count import mat_fstar  # noqa: E402  f* 逐字复用

B_BOOT = 2000
B_PERM = 20000
MIN_BOOT = 1900
MIN_TILES = 100
PUBLISHED = {"delta": 0.01229125960644857,
             "E_cell": 0.021077871890815454,
             "E_canvas": 0.033369131497264025,
             "ci": [0.00392442643812525, 0.018256039277634993]}


def load(size):
    return [x for x in tiles_data.load(size=size, split="train") if x["k_used"] >= 3]


def pack_groups(rows):
    """-> (包名有序列表, 每个包的行下标数组)"""
    idx = {}
    for i, r in enumerate(rows):
        idx.setdefault(r["pack"], []).append(i)
    names = sorted(idx)
    return names, [np.asarray(idx[p]) for p in names]


def cluster_boot(cur_m, cur_n, gm, gn, m, n, seed=0, b=B_BOOT):
    """两档各自按包有放回重采样 -> Δ 的自助分布。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        sm = np.concatenate([gm[i] for i in rng.integers(0, len(gm), len(gm))])
        sn = np.concatenate([gn[i] for i in rng.integers(0, len(gn), len(gn))])
        out.append(delta(cur_m[sm].mean(0), cur_n[sn].mean(0), m, n)[0])
    return np.asarray(out)


def tile_boot(cur_m, cur_n, m, n, seed=0, b=B_BOOT):
    """(OP2) 与 gen_scale.compare 逐调用相同的瓦片级自助。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        bm = cur_m[rng.integers(0, len(cur_m), len(cur_m))].mean(0)
        bn = cur_n[rng.integers(0, len(cur_n), len(cur_n))].mean(0)
        out.append(delta(bm, bn, m, n)[0])
    return np.asarray(out)


def chi2(pack_idx, bucket, n_pack, n_buck=3):
    t = np.bincount(pack_idx * n_buck + bucket,
                    minlength=n_pack * n_buck).reshape(n_pack, n_buck).astype(float)
    tot = t.sum()
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / tot
    ok = e > 0
    return float((((t - e) ** 2)[ok] / e[ok]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/pack_cluster.json")
    a = ap.parse_args()
    out = {}

    r16, r32 = load(16), load(32)
    c16, c32 = tile_curves(r16, 16), tile_curves(r32, 32)
    n16, g16 = pack_groups(r16)
    n32, g32 = pack_groups(r32)

    # ---- 描述（非判据） ----
    desc = {}
    for lbl, rows, names in (("16", r16, n16), ("32", r32, n32)):
        c = Counter(x["pack"] for x in rows)
        tot = sum(c.values())
        hhi = sum((v / tot) ** 2 for v in c.values())
        desc[lbl] = {"n_tiles": tot, "n_packs": len(c), "hhi": hhi,
                     "eff_packs": 1.0 / hhi, "top3": c.most_common(3)}
        print(f"[desc] {lbl}px train: tiles={tot} packs={len(c)} "
              f"HHI={hhi:.3f} eff={1/hhi:.1f}")
    out["desc"] = desc

    # ---- (OP3) ----
    op3 = (sum(len(g) for g in g16) == len(r16) and sum(len(g) for g in g32) == len(r32)
           and all(x.get("pack") for x in r16 + r32))
    print(f"(OP3) pack partition ok: {op3}")

    # ---- (OP1) 全量复算 ----
    d_full, e_cell, e_canvas = delta(c16.mean(0), c32.mean(0), 16, 32)
    op1 = all(abs(v - PUBLISHED[k]) <= 1e-12 * max(1.0, abs(PUBLISHED[k]))
              for k, v in (("delta", d_full), ("E_cell", e_cell), ("E_canvas", e_canvas)))
    print(f"(OP1) reproduce published point: {op1}  Δ={d_full:+.6f}")

    # ---- (OP2) 瓦片级自助锚点 ----
    tb = tile_boot(c16, c32, 16, 32)
    tlo, thi = (float(x) for x in np.percentile(tb, [2.5, 97.5]))
    op2 = (abs(tlo - PUBLISHED["ci"][0]) <= 1e-12
           and abs(thi - PUBLISHED["ci"][1]) <= 1e-12)
    print(f"(OP2) reproduce published tile-CI: {op2}  [{tlo:+.6f}, {thi:+.6f}]")

    # ---- (A1) 包级自助 ----
    cb = cluster_boot(c16, c32, g16, g32, 16, 32)
    lo, hi = (float(x) for x in np.percentile(cb, [2.5, 97.5]))
    op4a = len(cb) >= MIN_BOOT
    print(f"[A1] cluster 95%CI [{lo:+.6f}, {hi:+.6f}]  (tile CI was "
          f"[{tlo:+.6f}, {thi:+.6f}])")

    # ---- (A2) LOPO ----
    lopo = []
    for p in n32:
        k16 = np.array([i for i, r in enumerate(r16) if r["pack"] != p])
        k32 = np.array([i for i, r in enumerate(r32) if r["pack"] != p])
        if len(k16) < MIN_TILES or len(k32) < MIN_TILES:
            lopo.append({"pack": p, "status": "skipped",
                         "n16": int(len(k16)), "n32": int(len(k32))})
            continue
        d = delta(c16[k16].mean(0), c32[k32].mean(0), 16, 32)[0]
        lopo.append({"pack": p, "status": "ok", "n16": int(len(k16)),
                     "n32": int(len(k32)), "delta": d})
    kept = [f for f in lopo if f["status"] == "ok"]
    for f in sorted(kept, key=lambda x: x["n32"])[:4]:
        print(f"[A2] drop {f['pack'][:34]:34s} n32={f['n32']:4d} Δ={f['delta']:+.6f}")
    op5 = len(kept) >= 2
    all_pos = all(f["delta"] > 0 for f in kept)
    print(f"(OP5) usable folds={len(kept)}/{len(lopo)}  all Δ>0: {all_pos}")

    if not (op1 and op2 and op3 and op4a and op5):
        verdict = "OP_FAIL"
    elif lo > 0 and all_pos:
        verdict = "V1_robust"
    else:
        verdict = "V2a_direction_only" if all_pos else "V2b_pack_level"
    print(f"主判决 (A) = {verdict}")

    out["A"] = {"published": PUBLISHED, "delta_full": d_full,
                "E_cell": e_cell, "E_canvas": e_canvas,
                "tile_ci": [tlo, thi], "cluster_ci": [lo, hi],
                "n_boot": int(len(cb)), "lopo": lopo,
                "n_folds_ok": len(kept), "all_folds_positive": bool(all_pos)}
    out["ops"] = {"op1": bool(op1), "op2": bool(op2), "op3": bool(op3),
                  "op4a": bool(op4a), "op5": bool(op5)}
    out["verdict"] = verdict

    # ---- 次判据 (B)：(T1) train 的包级重读 ----
    f16, f32 = mat_fstar(r16, 16), mat_fstar(r32, 32)
    mats = sorted(set(f16) & set(f32))
    mode_pack = {}
    for m in mats:
        c = Counter(x["pack"] for x in r32 if x["material"] == m)
        mode_pack[m] = min(sorted(c), key=lambda p: (-c[p], p))
    packs = sorted(set(mode_pack.values()))
    pi = np.array([packs.index(mode_pack[m]) for m in mats])
    bu = np.array([0 if f32[m] == 2 * f16[m] else (1 if f32[m] == f16[m] else 2)
                   for m in mats])
    obs = chi2(pi, bu, len(packs))
    rng = np.random.default_rng(0)
    ge = sum(1 for _ in range(B_PERM)
             if chi2(pi, rng.permutation(bu), len(packs)) >= obs)
    p_perm = (ge + 1) / (B_PERM + 1)
    print(f"[B1] mats={len(mats)} packs={len(packs)} chi2={obs:.2f} "
          f"perm p={p_perm:.4g}")

    by_pack = [np.where(pi == j)[0] for j in range(len(packs))]
    rng = np.random.default_rng(0)
    ws = []
    for _ in range(B_BOOT):
        s = np.concatenate([by_pack[j] for j in rng.integers(0, len(by_pack),
                                                            len(by_pack))])
        nc = int((bu[s] == 0).sum())
        nv = int((bu[s] == 1).sum())
        if nc + nv:
            ws.append(nc / (nc + nv))
    wlo, whi = (float(x) for x in np.percentile(ws, [2.5, 97.5]))
    w_obs = float((bu == 0).sum() / max(1, (bu != 2).sum()))
    print(f"[B2] w={w_obs:.3f} cluster 95%CI [{wlo:.3f}, {whi:.3f}] "
          f"(excludes 0.5: {not (wlo <= 0.5 <= whi)})  n_boot={len(ws)}")
    out["B"] = {"n_mats": len(mats), "n_packs": len(packs), "chi2": obs,
                "perm_p": p_perm, "b_perm": B_PERM, "w": w_obs,
                "cluster_ci": [wlo, whi], "n_boot": len(ws),
                "op4b": len(ws) >= MIN_BOOT,
                "excludes_half": bool(not (wlo <= 0.5 <= whi))}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
