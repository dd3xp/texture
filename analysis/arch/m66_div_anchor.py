#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M66) 判读器：多样性（LPIPS_div）这把**参照无关**的尺子，人的锚点在哪儿？它分得开"无关的一对"吗？

判据逐字照抄已提交的预注册（`docs/arch_progress.md` 的 (M66) 节，commit 63faf80），
**盲写**：写这个文件时没有算过 H_div、没有算过 R_div、没有算过任何配对距离、
没有数过同材质跨包配对有多少。已披露的偏倚：B1/B4/B7 的 `LPIPS_div`（0.2492/0.1478/0.3329）
与 16px 的 B5＝0.3816 本来就在已入库的 `experiments/final_E_mat_{16,32}.json` 里，开工前就看到了。

零训练、零 API、零判官、零活件改动。只读：
  - `data/tiles/dataset_k16.json`（经 `model/tiles_data.load`，与 (M53)(M64)(M65) 同一个调用口径）
  - `experiments/baselines/<方法>/32/*.png`（正式测试那批产物）
  - `experiments/final_E_mat_32.json`（只用于 (OP1) 对账）

`lpips_diversity` / `lpips_net` / `upscale` / `load_method` 四个函数**直接用活件那一份**
（import，⛔ 没有重写）。⚠ 要 torch ⇒ 主流程只能在 emnlp 上跑；`--selftest` 不碰 torch，本机可跑。

用法：
    python analysis/arch/m66_div_anchor.py --selftest
    python analysis/arch/m66_div_anchor.py --out /tmp/m66_div_anchor.json
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

METHODS = ["B1", "B2", "B4", "B7", "TRD32", "TRD32_rr4"]
BASES = ["B1", "B4", "B7"]              # (V1)/(V2) 的对照臂候选
TRT = "TRD32"                           # 重排前，每材质 4 张（TRD32_rr4 只有 1 张 ⇒ 构造上无读数）
B_BOOT, SEED = 2000, 0
PAIR_CAP_PER_MAT, PAIR_CAP_TOTAL = 50, 20000
MIN_MATS, MIN_PACKS = 20, 8             # (V-1)
TOL_OP1, TOL_OP2 = 1e-3, 1e-6
EXPECT_COUNT = {("32", "train"): 403, ("32", "val"): 156}      # (OP4)
MIN_GROUPS = 50                                                # (OP5)


# ---------------------------------------------------------------- 活件（import，不重写）
def pair_dists(tiles_a, tiles_b, bs=256):
    """逐配对 LPIPS，用活件的 lpips_net/upscale，口径与 lpips_diversity 一致（side=64）。"""
    import torch
    from metrics import DEV, lpips_net, upscale
    net = lpips_net()
    out = []
    for i in range(0, len(tiles_a), bs):
        xa = upscale(tiles_a[i:i + bs], 64).to(DEV)
        xb = upscale(tiles_b[i:i + bs], 64).to(DEV)
        with torch.no_grad():
            out.extend(float(v) for v in net(xa, xb))
    return out


# ---------------------------------------------------------------- 聚合与自助（预注册第三节）
def mean_over_materials(pairs, dists, weights=None):
    """pairs: list[(mat, pack_i, pack_j)]；先对每个材质求配对均值，再对材质取均值。"""
    num, den = {}, {}
    for (m, _, _), d, w in zip(pairs, dists, weights if weights is not None else [1.0] * len(pairs)):
        if w <= 0:
            continue
        num[m] = num.get(m, 0.0) + w * d
        den[m] = den.get(m, 0.0) + w
    vals = [num[m] / den[m] for m in num if den[m] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def pack_bootstrap_ci(pairs, dists, packs, b=B_BOOT, seed=SEED):
    """按包有放回重采样；配对 (p,q) 的权重 = c_p*c_q（p!=q ⇒ 不会出现同包自配对）。"""
    packs = sorted(packs)
    rng = random.Random(seed)
    vals = []
    for _ in range(b):
        c = {p: 0 for p in packs}
        for _ in range(len(packs)):
            c[packs[rng.randrange(len(packs))]] += 1
        w = [c[pi] * c[pj] for (_, pi, pj) in pairs]
        v = mean_over_materials(pairs, dists, w)
        if math.isfinite(v):
            vals.append(v)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return lo, hi


# ---------------------------------------------------------------- 判据（逐字照抄预注册第四节）
def classify(n_mats, n_packs, h, lo, hi, r_div, d_trt, d_base):
    """返回 (判决名, 理由)。顺序：(V-1) -> (V0) -> (V1)/(V2)/(V3)。"""
    if n_mats < MIN_MATS or n_packs < MIN_PACKS:
        return "DIV_ANCHOR_TOO_THIN", "n_mats=%d (<%d) or n_packs=%d (<%d)" % (
            n_mats, MIN_MATS, n_packs, MIN_PACKS)
    if lo <= r_div <= hi:
        return "VOID_RULER_BLIND", "R_div inside human CI"
    t_in, b_in = (lo <= d_trt <= hi), (lo <= d_base <= hi)
    if abs(d_trt - h) < abs(d_base - h) and t_in and not b_in:
        return "DIV_ANCHOR_FAVORS_TRD", "TRD closer and inside, base outside"
    if abs(d_base - h) < abs(d_trt - h) and b_in and not t_in:
        return "DIV_ANCHOR_FAVORS_BASE", "base closer and inside, TRD outside"
    return "DIV_ANCHOR_UNDECIDED", "ruler does not separate the two"


# ---------------------------------------------------------------- 料
def human_rows(size, split):
    from tiles_data import load                   # noqa: E402  与 (M53)(M64)(M65) 同一调用口径
    return [(s["palette"][s["idx"]], s.get("pack"), s["material"]) for s in load(size, split)]


def prompt_slugs(setname="E_mat"):
    from prompts import load_set                  # noqa: E402
    prompts, _ = load_set(setname)
    return [e["material"].rsplit(".", 1)[0] for e in prompts]


def build_pairs(rows, seed=SEED):
    """同材质**跨包**配对（同包配对全部剔除并计数），按预注册的两道预算上限截断。"""
    by_mat = {}
    for k, (img, pack, mat) in enumerate(rows):
        by_mat.setdefault(mat, []).append(k)
    rng = random.Random(seed)
    kept, dropped_same_pack = [], 0
    for mat in sorted(by_mat):
        idx = by_mat[mat]
        cand = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if rows[i][1] == rows[j][1]:
                    dropped_same_pack += 1
                else:
                    cand.append((mat, i, j))
        if len(cand) > PAIR_CAP_PER_MAT:
            cand = rng.sample(cand, PAIR_CAP_PER_MAT)
        kept.extend(cand)
    if len(kept) > PAIR_CAP_TOTAL:
        kept = rng.sample(kept, PAIR_CAP_TOTAL)
    return kept, dropped_same_pack, by_mat


def build_control(rows, kept, by_mat, seed=SEED):
    """对每个材质抽同样多的「异材质跨包」配对（第一张来自该材质）。"""
    rng = random.Random(seed + 1)
    n_per = {}
    for (m, _, _) in kept:
        n_per[m] = n_per.get(m, 0) + 1
    all_idx = list(range(len(rows)))
    ctrl = []
    for mat in sorted(n_per):
        for _ in range(n_per[mat]):
            i = rng.choice(by_mat[mat])
            for _ in range(200):
                j = rng.choice(all_idx)
                if rows[j][2] != mat and rows[j][1] != rows[i][1]:
                    ctrl.append((mat, i, j))
                    break
    return ctrl


def as_pack_pairs(rows, pairs):
    return [(m, rows[i][1], rows[j][1]) for (m, i, j) in pairs]


# ---------------------------------------------------------------- 主流程
def run(root, base_dir, out_path):
    from metrics import lpips_diversity            # noqa: E402  活件
    from run_eval import load_method               # noqa: E402  活件
    from pathlib import Path

    res = {"note": "(M66) LPIPS diversity anchor; criteria pre-registered in 63faf80", "op": {}}

    # ---- 真人锚点
    rows, counts = [], {}
    for size, split in [(32, "train"), (32, "val")]:
        r = human_rows(size, split)
        counts["%d_%s" % (size, split)] = len(r)
        rows.extend(r)
    kept, dropped_same_pack, by_mat = build_pairs(rows)
    ctrl = build_control(rows, kept, by_mat)
    n_mats = len(set(m for (m, _, _) in kept))
    packs_h = sorted(set(rows[i][1] for (_, i, j) in kept) | set(rows[j][1] for (_, i, j) in kept))

    d_h = pair_dists([rows[i][0] for (_, i, _) in kept], [rows[j][0] for (_, _, j) in kept])
    d_c = pair_dists([rows[i][0] for (_, i, _) in ctrl], [rows[j][0] for (_, _, j) in ctrl])
    pp_h, pp_c = as_pack_pairs(rows, kept), as_pack_pairs(rows, ctrl)
    h = mean_over_materials(pp_h, d_h)
    r_div = mean_over_materials(pp_c, d_c)
    lo, hi = pack_bootstrap_ci(pp_h, d_h, packs_h)

    # ---- 方法（全部走活件）
    slugs = prompt_slugs("E_mat")
    per_method = {}
    for m in METHODS:
        d = os.path.join(root, base_dir, m, "32")
        if not os.path.isdir(d):
            per_method[m] = {"error": "missing dir %s" % d}
            continue
        first, groups = load_method(Path(d), slugs)
        ok = [i for i, t in enumerate(first) if t is not None]
        g = [groups[i] for i in ok if len(groups[i]) >= 2]
        per_method[m] = {"n": len(ok), "n_groups_ge2": len(g),
                         "LPIPS_div": lpips_diversity(g) if g else float("nan")}

    # ---- 操作检验
    with open(os.path.join(root, "experiments", "final_E_mat_32.json"), encoding="utf-8") as f:
        official = json.load(f)
    op1 = {}
    for m in BASES:
        got = per_method[m].get("LPIPS_div")
        want = official.get(m, {}).get("LPIPS_div")
        op1[m] = {"recomputed": got, "official": want,
                  "ok": got is not None and want is not None and abs(got - want) < TOL_OP1}

    mats_ge2 = [m for m in sorted(by_mat) if len(by_mat[m]) >= 2][:5]
    gtest = [[rows[k][0] for k in by_mat[m][:4]] for m in mats_ge2]
    mine = []
    for g in gtest:
        ii, jj = np.triu_indices(len(g), 1)
        dd = pair_dists([g[a] for a in ii], [g[b] for b in jj])
        mine.append(float(np.mean(dd)))
    op2_mine, op2_active = float(np.mean(mine)) if mine else float("nan"), lpips_diversity(gtest)

    same = pair_dists([rows[0][0]], [rows[0][0]])[0]
    rng = np.random.default_rng(SEED)
    a_list, jit_list, unrel_list = [], [], []
    for _ in range(50):
        i, j = int(rng.integers(len(rows))), int(rng.integers(len(rows)))
        t = np.asarray(rows[i][0], np.uint8)
        a_list.append(t)
        jit_list.append(np.clip(t.astype(np.int16) + rng.integers(-1, 2, t.shape), 0, 255).astype(np.uint8))
        unrel_list.append(np.asarray(rows[j][0], np.uint8))
    d_jit = float(np.mean(pair_dists(a_list, jit_list)))
    d_unrel = float(np.mean(pair_dists(a_list, unrel_list)))

    op5 = {m: per_method[m].get("n_groups_ge2") for m in METHODS}
    res["op"] = {
        "OP1_recompute_matches_official": {"ok": all(v["ok"] for v in op1.values()), "detail": op1},
        "OP2_pair_machinery_matches_active": {"mine": op2_mine, "active": op2_active,
                                              "ok": abs(op2_mine - op2_active) < TOL_OP2},
        "OP3_scale_sentinels": {"self": same, "jitter": d_jit, "unrelated": d_unrel,
                                "ok": same < 1e-6 and d_jit < d_unrel},
        "OP4_human_counts": {"ok": all(counts["%s_%s" % k] == v for k, v in EXPECT_COUNT.items()),
                             "counts": counts},
        "OP5_group_structure": {"detail": op5,
                                "ok": op5.get("B2") == 0 and op5.get("TRD32_rr4") == 0
                                and all((op5.get(m) or 0) >= MIN_GROUPS for m in BASES + [TRT])},
    }
    ops_ok = all(v["ok"] for v in res["op"].values())

    def div_of(m):
        v = per_method.get(m, {}).get("LPIPS_div")
        return float("nan") if v is None else float(v)

    d_trt = div_of(TRT)
    cands = [(abs(div_of(m) - h), m) for m in BASES if math.isfinite(div_of(m))]
    best_base = min(cands)[1] if cands else None
    d_base = div_of(best_base) if best_base else float("nan")
    verdict, why = classify(n_mats, len(packs_h), h, lo, hi, r_div, d_trt, d_base)
    if not ops_ok:
        verdict, why = "VOID_OP", "operation check failed"

    res.update({
        "human_anchor_H_div": h, "human_ci95_pack_bootstrap": [lo, hi],
        "n_pairs": len(kept), "n_materials_with_cross_pack_pair": n_mats,
        "n_packs_contributing": len(packs_h), "dropped_same_pack_pairs": dropped_same_pack,
        "R_div_control": r_div, "n_control_pairs": len(ctrl),
        "methods": per_method, "best_base": best_base,
        "dist_TRD32_to_H": abs(d_trt - h) if math.isfinite(d_trt) else None,
        "dist_base_to_H": abs(d_base - h) if math.isfinite(d_base) else None,
        "verdict": verdict, "why": why,
    })
    if out_path:                                    # 落盘一律在打印之前
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("human anchor H_div = %.4f   pack-bootstrap 95%%CI [%.4f, %.4f]" % (h, lo, hi))
    print("  pairs=%d  materials=%d  packs=%d  (dropped same-pack pairs: %d)"
          % (len(kept), n_mats, len(packs_h), dropped_same_pack))
    print("cross-material control R_div = %.4f  (n=%d)" % (r_div, len(ctrl)))
    for m in METHODS:
        r = per_method[m]
        if "error" in r:
            print("  %-12s %s" % (m, r["error"]))
        else:
            print("  %-12s n=%3d  groups>=2: %3d  LPIPS_div=%.4f"
                  % (m, r["n"], r["n_groups_ge2"], r["LPIPS_div"]))
    print("  best_base = %s" % best_base)
    for k, v in res["op"].items():
        print("  [%s] %s" % ("ok " if v["ok"] else "FAIL", k))
    print("VERDICT: %s   (%s)" % (verdict, why))
    if out_path:
        print("wrote %s" % out_path)
    return res


# ---------------------------------------------------------------- selftest（不碰 torch）
def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    # 聚合：先材质内均值、再对材质均值（⇒ 不是逐配对均值）
    pairs = [("a", "p", "q"), ("a", "p", "r"), ("b", "p", "q")]
    chk(abs(mean_over_materials(pairs, [0.0, 1.0, 1.0]) - 0.75) < 1e-12, "mean over materials")
    chk(abs(mean_over_materials(pairs, [0.0, 1.0, 1.0], [1, 1, 0]) - 0.5) < 1e-12, "zero weight drops material")
    chk(math.isnan(mean_over_materials(pairs, [1.0, 1.0, 1.0], [0, 0, 0])), "all-zero weight -> nan")

    # 自助：单包时所有跨包权重恒为 c*c>0，区间退化到点估计
    lo, hi = pack_bootstrap_ci(pairs, [0.0, 1.0, 1.0], ["p", "q", "r"], b=200)
    chk(lo <= 0.75 <= hi, "bootstrap covers point estimate")
    chk(lo < hi, "bootstrap interval non-degenerate")

    # 配对构造：同包配对必须被剔除并计数；每材质上限生效
    rows = [(None, "P1", "m"), (None, "P1", "m"), (None, "P2", "m"), (None, "P2", "x")]
    kept, dropped, by_mat = build_pairs(rows)
    chk(dropped == 1, "same-pack pair dropped")
    chk(len(kept) == 2 and all(rows[i][1] != rows[j][1] for (_, i, j) in kept), "kept are cross-pack")
    chk(set(m for (m, _, _) in kept) == {"m"}, "single-tile material has no pair")
    big = [(None, "P%d" % p, "m") for p in range(40)]
    kept2, _, _ = build_pairs(big)
    chk(len(kept2) == PAIR_CAP_PER_MAT, "per-material cap")

    # 对照：数量与主配对逐材质相同、且异材质跨包
    rows3 = [(None, "P1", "m"), (None, "P2", "m"), (None, "P3", "x"), (None, "P4", "y")]
    kept3, _, bm3 = build_pairs(rows3)
    ctrl = build_control(rows3, kept3, bm3)
    chk(len(ctrl) == len(kept3), "control same count")
    chk(all(rows3[j][2] != m and rows3[i][1] != rows3[j][1] for (m, i, j) in ctrl), "control cross-material cross-pack")

    # 判据分支（逐条照抄预注册第四节）
    chk(classify(19, 20, .4, .3, .5, .9, .4, .2)[0] == "DIV_ANCHOR_TOO_THIN", "V-1 materials")
    chk(classify(30, 7, .4, .3, .5, .9, .4, .2)[0] == "DIV_ANCHOR_TOO_THIN", "V-1 packs")
    chk(classify(30, 20, .4, .3, .5, .45, .4, .2)[0] == "VOID_RULER_BLIND", "V0")
    chk(classify(30, 20, .4, .3, .5, .9, .4, .2)[0] == "DIV_ANCHOR_FAVORS_TRD", "V1")
    chk(classify(30, 20, .4, .3, .5, .9, .2, .4)[0] == "DIV_ANCHOR_FAVORS_BASE", "V2")
    chk(classify(30, 20, .4, .3, .5, .9, .35, .45)[0] == "DIV_ANCHOR_UNDECIDED", "V3 both in")
    chk(classify(30, 20, .4, .3, .5, .9, .1, .2)[0] == "DIV_ANCHOR_UNDECIDED", "V3 both out")
    print("selftest OK %d/%d" % (ok, 18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--base_dir", default="experiments/baselines")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a.root, a.base_dir, a.out)


if __name__ == "__main__":
    main()
