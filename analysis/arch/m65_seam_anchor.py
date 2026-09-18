#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M65) 判读器：可平铺性这把**参照无关**的尺子，人的锚点在哪儿？TRD 与 B2 谁更靠近它？

判据逐字照抄已提交的预注册（`docs/arch_progress.md` 的 (M65) 节，commit dbc9af0），
**盲写**：写这个文件时没有算过真人锚点 H、没有算过任何逐张分布、没有算过平坦率、
没有算过不可平铺对照 R_crop。已披露的偏倚：两个方法的中位数（0.9694 / 0.6361）
本来就在已入库的 `experiments/final_E_mat_32.json` 里，开工前就看到了。

零 GPU、零 API、零判官、零活件改动。只读：
  - `data/tiles/dataset_k16.json`（经 `model/tiles_data.load`，与 (M53)(M64) 同一个调用口径）
  - `remote_tmp/m65/<方法>/*.png`（正式测试那批产物，scp 自远程 `experiments/baselines/<方法>/32/`）
  - `experiments/final_E_mat_32.json`（只用于 (OP1) 对账）

`tile_seam_ratio` / `tileability` / `load_method` 三个函数**逐字复用**活件源码
（从 AST 里取出函数体执行），⛔ 没有重写、⛔ 没有 import torch。

用法：
    python analysis/arch/m65_seam_anchor.py --selftest
    python analysis/arch/m65_seam_anchor.py --out experiments/m65_seam_anchor.json
"""
import argparse
import ast
import json
import math
import os
import random
import re
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "eval"))

METHODS = ["B1", "B2", "B4", "B7", "TRD32_rr4"]
TRT, CTRL = "TRD32_rr4", "B2"          # 主判据的两个方法
B_BOOT, SEED = 2000, 0
FLAT_MAX, FLAT_GAP = 0.10, 0.05        # (V4) 门槛
NOISE_N, NOISE_TOL = 200, 0.10         # (OP2) 中性点校准
EXPECT_COUNT = {("32", "train"): 403, ("32", "val"): 156, ("64", "train"): 335}   # (OP4)


# ---------------------------------------------------------------- 逐字复用活件
def borrow(path, names, ns):
    """从源文件的 AST 里取出这些函数，逐字执行进 ns。⛔ 不 import 整个模块（那会拖 torch 进来）。"""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    want = {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef,)) and n.name in names}
    missing = [n for n in names if n not in want]
    if missing:
        raise RuntimeError("source changed, missing %s in %s" % (missing, path))
    mod = ast.Module(body=[want[n] for n in names], type_ignores=[])
    exec(compile(mod, path, "exec"), ns)          # noqa: S102  只跑我们自己的仓库源码
    return ns


def activepieces():
    from pathlib import Path
    ns = {"np": np, "re": re, "Image": Image, "Path": Path}
    borrow(os.path.join(ROOT, "eval", "metrics.py"), ["tile_seam_ratio", "tileability"], ns)
    borrow(os.path.join(ROOT, "eval", "run_eval.py"), ["load_method"], ns)
    return ns


# ---------------------------------------------------------------- 统计小工具
def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return float(s[n // 2]) if n % 2 else float((s[n // 2 - 1] + s[n // 2]) / 2.0)


def finite(xs):
    return [x for x in xs if isinstance(x, float) and math.isfinite(x)]


def flat_rate(xs):
    return 1.0 - len(finite(xs)) / len(xs) if xs else float("nan")


def pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED):
    """按包有放回重采样（(M14)-(M16) 纪律），返回中位数的 95% 百分位区间。"""
    packs = sorted(by_pack)
    rng = random.Random(seed)
    meds = []
    for _ in range(b):
        pool = []
        for _ in range(len(packs)):
            pool.extend(by_pack[packs[rng.randrange(len(packs))]])
        v = finite(pool)
        if v:
            meds.append(median(v))
    meds.sort()
    lo = meds[int(0.025 * len(meds))]
    hi = meds[min(len(meds) - 1, int(0.975 * len(meds)))]
    return lo, hi, len(packs)


# ---------------------------------------------------------------- 判据（逐字照抄预注册）
def classify(h, lo, hi, r_crop, t, b, flat_t, flat_b):
    """返回 (判决名, 理由)。顺序：(V0) -> (V4) -> (V1)/(V2)/(V3)。"""
    if lo <= r_crop <= hi:
        return "VOID_RULER_BLIND", "R_crop median inside human CI"
    if flat_t >= FLAT_MAX or flat_b >= FLAT_MAX:
        return "VOID_FLAT_SELECTION", "flat rate >= %.2f" % FLAT_MAX
    if abs(flat_t - flat_b) >= FLAT_GAP:
        return "VOID_FLAT_SELECTION", "flat rate gap >= %.2f" % FLAT_GAP
    t_in, b_in = (lo <= t <= hi), (lo <= b <= hi)
    if abs(t - h) < abs(b - h) and t_in and not b_in:
        return "SEAM_ANCHOR_FAVORS_TRD", "TRD closer and inside, B2 outside"
    if abs(b - h) < abs(t - h) and b_in and not t_in:
        return "SEAM_ANCHOR_FAVORS_B2", "B2 closer and inside, TRD outside"
    return "SEAM_ANCHOR_UNDECIDED", "ruler does not separate the two"


# ---------------------------------------------------------------- 料
def human_tiles(size, split):
    from tiles_data import load                     # noqa: E402  与 (M53)(M64) 同一调用口径
    rows = load(size, split)
    return [(s["palette"][s["idx"]], s.get("pack")) for s in rows]


def prompt_slugs(setname="E_mat"):
    from prompts import load_set                    # noqa: E402
    prompts, _ = load_set(setname)
    return [e["material"].rsplit(".", 1)[0] for e in prompts]


def method_ratios(ns, d, slugs):
    from pathlib import Path
    first, _ = ns["load_method"](Path(d), slugs)
    ok = [t for t in first if t is not None]
    return [ns["tile_seam_ratio"](t) for t in ok], len(ok)


def run(root, dirs, out_path):
    ns = activepieces()
    seam, tileab = ns["tile_seam_ratio"], ns["tileability"]
    res = {"note": "(M65) tileability anchor; criteria pre-registered in dbc9af0", "op": {}}

    # ---- 真人锚点
    counts, by_pack, all_h = {}, {}, []
    for size, split in [(32, "train"), (32, "val")]:
        rows = human_tiles(size, split)
        counts["%d_%s" % (size, split)] = len(rows)
        for img, pack in rows:
            v = seam(img)
            all_h.append(v)
            by_pack.setdefault(pack, []).append(v)
    h64 = human_tiles(64, "train")
    counts["64_train"] = len(h64)
    h = median(finite(all_h))
    lo, hi, n_packs = pack_bootstrap_ci(by_pack)

    # ---- 不可平铺对照：64px 真人瓦片随机裁 32x32
    rng = random.Random(SEED)
    crops = []
    for img, _ in h64:
        i, j = rng.randint(0, 32), rng.randint(0, 32)
        crops.append(seam(np.asarray(img)[i:i + 32, j:j + 32]))
    r_crop = median(finite(crops))

    # ---- 方法
    slugs = prompt_slugs("E_mat")
    per_method = {}
    for m in METHODS:
        d = os.path.join(root, dirs, m)
        if not os.path.isdir(d):
            per_method[m] = {"error": "missing dir %s" % d}
            continue
        vals, n = method_ratios(ns, d, slugs)
        per_method[m] = {"n": n, "median": median(finite(vals)), "flat_rate": flat_rate(vals)}

    # ---- 操作检验
    with open(os.path.join(root, "experiments", "final_E_mat_32.json"), encoding="utf-8") as f:
        official = json.load(f)
    op1 = {}
    for m in METHODS:
        got = per_method[m].get("median")
        want = official.get(m, {}).get("tile_seam_ratio")
        op1[m] = {"recomputed": got, "official": want,
                  "ok": got is not None and want is not None and abs(got - want) < 1e-9}
    nrng = np.random.default_rng(SEED)
    noise = [seam(nrng.integers(0, 256, (32, 32, 3)).astype(np.uint8)) for _ in range(NOISE_N)]
    op2_med = median(finite(noise))
    op3 = not math.isfinite(seam(np.full((32, 32, 3), 7, np.uint8)))
    op4 = {k: counts["%s_%s" % k] == v for k, v in EXPECT_COUNT.items()}
    res["op"] = {
        "OP1_recompute_matches_official": {"ok": all(v["ok"] for v in op1.values()), "detail": op1},
        "OP2_noise_neutral_point": {"median": op2_med, "ok": abs(op2_med - 1.0) <= NOISE_TOL},
        "OP3_flat_sentinel_fires": {"ok": bool(op3)},
        "OP4_human_counts": {"ok": all(op4.values()), "counts": counts},
    }
    ops_ok = all(v["ok"] for v in res["op"].values())

    t = per_method[TRT].get("median")
    b = per_method[CTRL].get("median")
    ft = per_method[TRT].get("flat_rate")
    fb = per_method[CTRL].get("flat_rate")
    verdict, why = classify(h, lo, hi, r_crop, t, b, ft, fb)
    if not ops_ok:
        verdict, why = "VOID_OP", "operation check failed"

    res.update({
        "human_anchor_H": h, "human_ci95_pack_bootstrap": [lo, hi], "n_packs": n_packs,
        "human_n": len(all_h), "human_flat_rate": flat_rate(all_h),
        "R_crop_median": r_crop, "R_crop_n": len(crops), "R_crop_flat_rate": flat_rate(crops),
        "methods": per_method, "verdict": verdict, "why": why,
        "dist_TRD_to_H": abs(t - h) if t is not None else None,
        "dist_B2_to_H": abs(b - h) if b is not None else None,
    })
    if out_path:                                    # 落盘一律在打印之前
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("human anchor H = %.4f   pack-bootstrap 95%%CI [%.4f, %.4f]  (%d packs, %d tiles)"
          % (h, lo, hi, n_packs, len(all_h)))
    print("non-tileable control R_crop = %.4f  (n=%d)" % (r_crop, len(crops)))
    for m in METHODS:
        r = per_method[m]
        if "error" in r:
            print("  %-12s %s" % (m, r["error"]))
        else:
            print("  %-12s n=%3d  median=%.4f  flat=%.3f" % (m, r["n"], r["median"], r["flat_rate"]))
    for k, v in res["op"].items():
        print("  [%s] %s" % ("ok " if v["ok"] else "FAIL", k))
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

    ns = activepieces()
    seam = ns["tile_seam_ratio"]
    # 借来的函数确实是活件那一份
    chk(callable(seam) and callable(ns["tileability"]) and callable(ns["load_method"]), "borrowed")
    # 平坦瓦片 -> nan（(OP3) 的逻辑）
    chk(not math.isfinite(seam(np.zeros((8, 8, 3), np.uint8))), "flat -> nan")
    # 横向梯度（不可平铺）-> 接缝远大于内部
    ramp = np.tile(np.linspace(0, 255, 8, dtype=np.uint8)[None, :, None], (8, 1, 3))
    chk(seam(ramp) > 3.0, "ramp seam >> inner")
    # 严格周期（可平铺）-> 比值 ~1 量级
    rng = np.random.default_rng(0)
    per = np.tile(rng.integers(0, 256, (4, 4, 3)).astype(np.uint8), (8, 8, 1))
    chk(0.3 < seam(per) < 3.0, "periodic ratio ~1")
    # 中位数与平坦率
    chk(median([3.0, 1.0, 2.0]) == 2.0, "median odd")
    chk(median([1.0, 2.0, 3.0, 4.0]) == 2.5, "median even")
    chk(abs(flat_rate([1.0, float("nan"), 2.0, float("nan")]) - 0.5) < 1e-12, "flat rate")
    # 包自助：单包时区间退化到该包的中位数
    lo, hi, npk = pack_bootstrap_ci({"p": [1.0, 2.0, 3.0]}, b=50)
    chk(lo == hi == 2.0 and npk == 1, "bootstrap single pack")
    # 判据分支（逐条照抄预注册）
    chk(classify(1.0, 0.9, 1.1, 1.0, 1.0, 0.5, 0.0, 0.0)[0] == "VOID_RULER_BLIND", "V0")
    chk(classify(1.0, 0.9, 1.1, 2.0, 1.0, 0.5, 0.2, 0.0)[0] == "VOID_FLAT_SELECTION", "V4 rate")
    chk(classify(1.0, 0.9, 1.1, 2.0, 1.0, 0.5, 0.06, 0.0)[0] == "VOID_FLAT_SELECTION", "V4 gap")
    chk(classify(1.0, 0.9, 1.1, 2.0, 1.0, 0.5, 0.0, 0.0)[0] == "SEAM_ANCHOR_FAVORS_TRD", "V1")
    chk(classify(1.0, 0.9, 1.1, 2.0, 0.5, 1.0, 0.0, 0.0)[0] == "SEAM_ANCHOR_FAVORS_B2", "V2")
    chk(classify(1.0, 0.9, 1.1, 2.0, 0.95, 1.05, 0.0, 0.0)[0] == "SEAM_ANCHOR_UNDECIDED", "V3 both in")
    chk(classify(1.0, 0.9, 1.1, 2.0, 0.5, 0.3, 0.0, 0.0)[0] == "SEAM_ANCHOR_UNDECIDED", "V3 both out")
    print("selftest OK %d/%d" % (ok, 15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--dirs", default="remote_tmp/m65")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a.root, a.dirs, a.out)


if __name__ == "__main__":
    main()
