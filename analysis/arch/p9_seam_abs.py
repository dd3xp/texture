#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P9) 判读器：接缝尺子的汇总统计量换成**绝对偏离**之后，它还瞎不瞎？我们交付的 16px 输出在不在真人范围里？

判据逐字照抄已提交的预注册（`docs/arch_progress.md` 的 (P9) 节，commit c12132c），**盲写**：
写这个文件时，我没有算过任何绝对口径的读数 —— 没算过真人锚点 `H_abs`、没算过 `R_crop`、
没算过任何方法的 `D_m`、没跑过本文件。已披露的偏倚全部列在预注册第五节
（＝已入库 JSON 里的**签名**列，以及 (P8) 那四条**别的**臂的绝对口径）。

零 GPU、零 API、零判官、零活件改动。只读：
  - `data/tiles/dataset_k16.json`（经 `model/tiles_data.load`，与 (M53)(M64)(M65) 同一调用口径）
  - `remote_tmp/p9/<方法>/*.png`（16px 正式测试产物）、`remote_tmp/m65/<方法>/*.png`（32px，(M65) 已拉回）
  - `experiments/final_E_mat_{16,32}.json`（只用于 (OP1) 对账）

⚑ `tile_seam_ratio` / `load_method` 等活件**不重写**：直接 import 冻结的 `m65_seam_anchor`，
复用它的 `activepieces` / `median` / `finite` / `flat_rate` / `pack_bootstrap_ci` /
`human_tiles` / `prompt_slugs`。⛔ 没改 `m65_seam_anchor.py` 一个字节。

用法：
    python analysis/arch/p9_seam_abs.py --selftest
    python analysis/arch/p9_seam_abs.py --out experiments/p9_seam_abs.json
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m65_seam_anchor as M65                                      # noqa: E402  冻结的 (M65) 判读器

ROOT = M65.ROOT
B_BOOT, SEED = M65.B_BOOT, M65.SEED
FLAT_MAX, FLAT_GAP = 0.10, 0.05          # (V1) 门槛，与 (M65) 同值
NOISE_N, NOISE_TOL = 200, 0.10           # (OP2)
OP5_SIGNED_TOL, OP5_ABS_MIN = 0.05, 0.20  # (OP5) 抵消机制的阳性对照
OP6_TOL = 0.02                           # (OP6) 自助稳健

TRT16 = "TRD16c_rr4"                     # 主判据读的那个方法＝正式交付的 16px 系统
SIZES = {
    16: {"dirs": "remote_tmp/p9", "json": "final_E_mat_16.json",
         "methods": ["B1", "B2", "B4", "B5", "B7", "TRD16", "TRD16c_rr4"]},
    32: {"dirs": "remote_tmp/m65", "json": "final_E_mat_32.json",
         "methods": ["B1", "B2", "B4", "B7", "TRD32_rr4"]},
}
EXPECT_COUNT = {("16", "train"): 2732, ("16", "val"): 291,          # (OP4)
                ("32", "train"): 403, ("32", "val"): 156, ("64", "train"): 335}


# ---------------------------------------------------------------- 绝对口径
def dev(r):
    """d(t) = |tile_seam_ratio(t) - 1|；非有限（内部平涂）原样传下去，由 finite() 丢弃。"""
    return abs(r - 1.0) if isinstance(r, float) and math.isfinite(r) else float("nan")


def summarize(ratios):
    """同一批瓦片上并列报三种汇总：已发表口径（签名中位数）/ 绝对偏离中位数 / ratio>1 的比例。"""
    fin = M65.finite(ratios)
    return {
        "n": len(ratios),
        "n_finite": len(fin),
        "flat_rate": M65.flat_rate(ratios),
        "median_signed": M65.median(fin) if fin else float("nan"),
        "median_abs_dev": M65.median([dev(x) for x in fin]) if fin else float("nan"),
        "frac_above_1": (sum(1 for x in fin if x > 1.0) / len(fin)) if fin else float("nan"),
    }


# ---------------------------------------------------------------- 判据（逐字照抄预注册第三节）
def classify16(lo, hi, r_crop, d_trt, flat_h, flat_t):
    """顺序：(V0) -> (V1) -> 主判据 (C1)。返回 (判决名, 理由)。"""
    if lo <= r_crop <= hi:
        return "VOID_RULER_BLIND_ABS_16", "R_crop inside human CI"
    if flat_h >= FLAT_MAX or flat_t >= FLAT_MAX:
        return "VOID_FLAT_SELECTION_16", "flat rate >= %.2f" % FLAT_MAX
    if abs(flat_h - flat_t) >= FLAT_GAP:
        return "VOID_FLAT_SELECTION_16", "flat rate gap >= %.2f" % FLAT_GAP
    if d_trt > hi:
        return "SEAM16_WORSE_THAN_HUMAN", "D above human CI"
    if d_trt < lo:
        return "SEAM16_TIGHTER_THAN_HUMAN", "D below human CI (over-smooth, NOT better)"
    return "SEAM16_HUMAN_LEVEL", "D inside human CI"


def classify32(lo, hi, r_crop):
    """判据② 只登记：换绝对汇总之后 32px 这把尺子还瞎不瞎。"""
    if lo <= r_crop <= hi:
        return "ABS_RULER_STILL_BLIND_32", "R_crop inside human CI"
    return "ABS_RULER_LIVES_32", "R_crop outside human CI"


# ---------------------------------------------------------------- (OP5) 两侧发散的合成瓦片
def two_sided_tiles(n_each=25):
    """自造已知两侧发散的瓦片：一半 ratio~1.30、一半 ratio~0.70。

    构造：8x8x3，每行相同（=> 纵向那两项恰好为 0），行向量 v=[0,100,0,100,0,100,0,g]。
    此时 ratio = |v[7]-v[0]| * 7 / (6*100 + |g|)。g=137 -> ~1.301；g=67 -> ~0.703。
    两簇等量 => 签名中位数贴 1（抵消），而绝对偏离中位数 ~0.30。
    """
    out = []
    for g in (137, 67):
        row = np.array([0, 100, 0, 100, 0, 100, 0, g], np.uint8)
        t = np.tile(row[None, :, None], (8, 1, 3))
        out.extend([t] * n_each)
    return out


# ---------------------------------------------------------------- 一个尺寸的全部读数
def read_size(ns, size, cfg, root):
    from pathlib import Path
    seam = ns["tile_seam_ratio"]
    rep = {"size": size}

    # 真人锚点：train + val 合并，按包收 d 值
    counts, by_pack, all_r = {}, {}, []
    for split in ("train", "val"):
        rows = M65.human_tiles(size, split)
        counts["%d_%s" % (size, split)] = len(rows)
        for img, pack in rows:
            v = seam(img)
            all_r.append(v)
            if math.isfinite(v):
                by_pack.setdefault(pack, []).append(dev(v))
    rep["human"] = summarize(all_r)
    lo, hi, n_packs = M65.pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED)
    lo1, hi1, _ = M65.pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED + 1)
    rep["human_abs_ci95"] = [lo, hi]
    rep["n_packs"] = n_packs
    rep["op6_ci_seed1"] = [lo1, hi1]
    rep["op6_ok"] = abs(lo - lo1) < OP6_TOL and abs(hi - hi1) < OP6_TOL

    # 不可平铺对照：真人 2S px 瓦片随机裁 SxS
    big = M65.human_tiles(2 * size, "train")
    counts["%d_train" % (2 * size)] = len(big)
    rng = random.Random(SEED)
    crops = []
    for img, _ in big:
        i, j = rng.randint(0, size), rng.randint(0, size)
        crops.append(seam(np.asarray(img)[i:i + size, j:j + size]))
    rep["R_crop"] = summarize(crops)
    rep["counts"] = counts

    # 方法：已发表那条口径（每材质第一张、RGB、E_mat 272 slug）
    slugs = M65.prompt_slugs("E_mat")
    with open(os.path.join(root, "experiments", cfg["json"]), encoding="utf-8") as f:
        official = json.load(f)
    methods, op1 = {}, {}
    for m in cfg["methods"]:
        d = os.path.join(root, cfg["dirs"], m)
        if not os.path.isdir(d):
            methods[m] = {"error": "missing dir %s" % d}
            op1[m] = {"ok": False, "why": "missing dir"}
            continue
        first, _ = ns["load_method"](Path(d), slugs)
        ratios = [seam(np.asarray(t, np.float64)) for t in first if t is not None]
        s = summarize(ratios)
        methods[m] = s
        want = official.get(m, {}).get("tile_seam_ratio")
        op1[m] = {"recomputed_signed": s["median_signed"], "official": want,
                  "ok": want is not None and s["median_signed"] == want}
    rep["methods"] = methods
    # (OP1)：B5 走 run_eval 的 retrieval() 而不是方法目录 => 复现不了就排除并披露
    excluded = [m for m, v in op1.items() if not v["ok"]]
    rep["op1"] = {"detail": op1, "excluded_from_C3": excluded,
                  "ok": op1.get(TRT16 if size == 16 else "TRD32_rr4", {}).get("ok", False)}
    return rep


def run(root, out_path):
    ns = M65.activepieces()
    seam = ns["tile_seam_ratio"]
    res = {"note": "(P9) absolute-deviation seam ruler; criteria pre-registered in c12132c"}

    sizes = {}
    for size, cfg in SIZES.items():
        sizes[str(size)] = read_size(ns, size, cfg, root)
    res["sizes"] = sizes

    # ---- 操作检验
    nrng = np.random.default_rng(SEED)
    noise = [seam(nrng.integers(0, 256, (16, 16, 3)).astype(np.uint8)) for _ in range(NOISE_N)]
    ns_sum = summarize(noise)
    op2 = (abs(ns_sum["median_signed"] - 1.0) <= NOISE_TOL
           and ns_sum["median_abs_dev"] <= NOISE_TOL)
    op3 = not math.isfinite(seam(np.full((16, 16, 3), 7, np.uint8)))
    got_counts = {}
    for s in sizes.values():
        got_counts.update(s["counts"])
    # ⚠ 键必须是 str：tuple 键会让 json.dump 崩在落盘那一步（首跑踩到，只改序列化、判据一字未动）
    op4 = {"%s_%s" % k: got_counts.get("%s_%s" % k) == v for k, v in EXPECT_COUNT.items()}
    ts = summarize([seam(t) for t in two_sided_tiles()])
    op5 = (abs(ts["median_signed"] - 1.0) <= OP5_SIGNED_TOL
           and ts["median_abs_dev"] >= OP5_ABS_MIN)
    op6 = all(s["op6_ok"] for s in sizes.values())
    op1_16 = sizes["16"]["op1"]["ok"]
    res["op"] = {
        "OP1_TRT_signed_reproduces_official": {"ok": bool(op1_16),
                                               "detail": {k: v["op1"]["detail"] for k, v in sizes.items()}},
        "OP2_noise_neutral": {"ok": bool(op2), "detail": ns_sum},
        "OP3_flat_sentinel_fires": {"ok": bool(op3)},
        "OP4_human_counts": {"ok": all(op4.values()), "detail": op4, "counts": got_counts},
        "OP5_two_sided_cancellation": {"ok": bool(op5), "detail": ts},
        "OP6_bootstrap_seed_stable": {"ok": bool(op6)},
    }
    ops_ok = all(v["ok"] for v in res["op"].values())

    s16 = sizes["16"]
    lo, hi = s16["human_abs_ci95"]
    d_trt = s16["methods"].get(TRT16, {}).get("median_abs_dev", float("nan"))
    verdict, why = classify16(lo, hi, s16["R_crop"]["median_abs_dev"], d_trt,
                              s16["human"]["flat_rate"],
                              s16["methods"].get(TRT16, {}).get("flat_rate", 1.0))
    if not ops_ok:
        verdict, why = "VOID_OP", "operation check failed"
    s32 = sizes["32"]
    c2, c2why = classify32(s32["human_abs_ci95"][0], s32["human_abs_ci95"][1],
                           s32["R_crop"]["median_abs_dev"])
    res.update({"verdict": verdict, "why": why, "C1_D_TRT16": d_trt,
                "C2_registered_only": c2, "C2_why": c2why})

    if out_path:                                     # 落盘一律在打印之前
        p = out_path if os.path.isabs(out_path) else os.path.join(root, out_path)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    for size in (16, 32):
        s = sizes[str(size)]
        lo_, hi_ = s["human_abs_ci95"]
        print("=== %dpx ===" % size)
        print("  human anchor H_abs = %.4f   pack-bootstrap 95%%CI [%.4f, %.4f]  (%d packs, %d tiles, flat %.3f)"
              % (s["human"]["median_abs_dev"], lo_, hi_, s["n_packs"],
                 s["human"]["n"], s["human"]["flat_rate"]))
        print("  non-tileable R_crop: abs=%.4f  signed=%.4f  (n=%d)"
              % (s["R_crop"]["median_abs_dev"], s["R_crop"]["median_signed"], s["R_crop"]["n"]))
        print("  %-12s %5s %9s %9s %9s %7s" % ("method", "n", "signed", "abs_dev", "frac>1", "flat"))
        for m, v in s["methods"].items():
            if "error" in v:
                print("  %-12s %s" % (m, v["error"]))
                continue
            print("  %-12s %5d %9.4f %9.4f %9.3f %7.3f"
                  % (m, v["n"], v["median_signed"], v["median_abs_dev"],
                     v["frac_above_1"], v["flat_rate"]))
        print("  OP1 excluded (signed median did not reproduce official): %s"
              % (s["op1"]["excluded_from_C3"] or "none"))
    for k, v in res["op"].items():
        print("  [%s] %s" % ("ok " if v["ok"] else "FAIL", k))
    print("C2 (registered only): %s  (%s)" % (c2, c2why))
    print("VERDICT: %s   (%s)" % (verdict, why))
    if out_path:
        print("wrote %s" % out_path)
    return res


# ---------------------------------------------------------------- selftest
def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    ns = M65.activepieces()
    seam = ns["tile_seam_ratio"]
    chk(callable(seam) and callable(ns["load_method"]), "borrowed live pieces")
    chk(M65.__file__.endswith("m65_seam_anchor.py"), "frozen reader imported")

    # dev()
    chk(dev(1.0) == 0.0 and abs(dev(1.3) - 0.3) < 1e-12 and abs(dev(0.7) - 0.3) < 1e-12, "dev abs")
    chk(not math.isfinite(dev(float("nan"))), "dev of nan stays nan")

    # summarize()：同一批数上三种汇总互不相同，且绝对口径不会被符号抵消骗到
    s = summarize([0.7, 1.3, float("nan")])
    chk(s["n"] == 3 and s["n_finite"] == 2, "summarize counts")
    chk(abs(s["flat_rate"] - 1 / 3) < 1e-12, "summarize flat rate")
    chk(abs(s["median_signed"] - 1.0) < 1e-12, "signed median cancels")
    chk(abs(s["median_abs_dev"] - 0.3) < 1e-12, "abs median does not cancel")
    chk(abs(s["frac_above_1"] - 0.5) < 1e-12, "frac above 1")

    # (OP5) 的合成瓦片：构造必须真的做到「签名贴 1、绝对 ~0.30」
    ts = summarize([seam(t) for t in two_sided_tiles()])
    chk(abs(ts["median_signed"] - 1.0) <= OP5_SIGNED_TOL, "OP5 signed near 1")
    chk(ts["median_abs_dev"] >= OP5_ABS_MIN, "OP5 abs dev large")
    r_hi = seam(two_sided_tiles(1)[0])
    r_lo = seam(two_sided_tiles(1)[1])
    chk(abs(r_hi - 1.301) < 0.01, "OP5 high cluster ratio")
    chk(abs(r_lo - 0.703) < 0.01, "OP5 low cluster ratio")

    # 活件方向（与 (M65) selftest 同款，确认借来的确实是那一份）
    chk(not math.isfinite(seam(np.zeros((8, 8, 3), np.uint8))), "flat -> nan")
    ramp = np.tile(np.linspace(0, 255, 8, dtype=np.uint8)[None, :, None], (8, 1, 3))
    chk(dev(seam(ramp)) > 2.0, "ramp large abs dev")

    # 判据分支（逐条照抄预注册第三节）
    chk(classify16(0.1, 0.3, 0.2, 0.2, 0.0, 0.0)[0] == "VOID_RULER_BLIND_ABS_16", "V0")
    chk(classify16(0.1, 0.3, 0.9, 0.2, 0.2, 0.0)[0] == "VOID_FLAT_SELECTION_16", "V1 human rate")
    chk(classify16(0.1, 0.3, 0.9, 0.2, 0.0, 0.2)[0] == "VOID_FLAT_SELECTION_16", "V1 trt rate")
    chk(classify16(0.1, 0.3, 0.9, 0.2, 0.0, 0.06)[0] == "VOID_FLAT_SELECTION_16", "V1 gap")
    chk(classify16(0.1, 0.3, 0.9, 0.4, 0.0, 0.0)[0] == "SEAM16_WORSE_THAN_HUMAN", "C1 worse")
    chk(classify16(0.1, 0.3, 0.9, 0.05, 0.0, 0.0)[0] == "SEAM16_TIGHTER_THAN_HUMAN", "C1 tighter")
    chk(classify16(0.1, 0.3, 0.9, 0.2, 0.0, 0.0)[0] == "SEAM16_HUMAN_LEVEL", "C1 human level")
    # 边界：恰好落在端点上算「在区间内」（(M14) 差 0.0002 也是没过，双向都按闭区间）
    chk(classify16(0.1, 0.3, 0.9, 0.3, 0.0, 0.0)[0] == "SEAM16_HUMAN_LEVEL", "C1 hi boundary")
    chk(classify16(0.1, 0.3, 0.9, 0.1, 0.0, 0.0)[0] == "SEAM16_HUMAN_LEVEL", "C1 lo boundary")
    chk(classify32(0.1, 0.3, 0.2)[0] == "ABS_RULER_STILL_BLIND_32", "C2 blind")
    chk(classify32(0.1, 0.3, 0.9)[0] == "ABS_RULER_LIVES_32", "C2 lives")

    # 自助：单包时区间退化（冻结件的行为，确认我喂进去的是 d 值不是 ratio）
    lo, hi, npk = M65.pack_bootstrap_ci({"p": [0.1, 0.2, 0.3]}, b=50)
    chk(lo == hi == 0.2 and npk == 1, "bootstrap on d values")

    print("selftest OK %d/%d" % (ok, 27))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a.root, a.out)


if __name__ == "__main__":
    main()
