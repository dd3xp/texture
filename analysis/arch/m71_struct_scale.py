#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M71) 判读器：结构尺度这把**参照无关**的尺子 —— 人的锚点在哪儿，我们交付的输出在不在里面？

判据逐字照抄已提交的预注册（`docs/arch_progress.md` 的 (M71) 节，commit ec16dd3），**盲写**：
写这个文件时我**一个 LF 读数都没算过** —— 没算过真人锚点、没算过 CI、没算过 `R_shuf`、
没算过任何方法的 LF，也没跑过本文件（除 `--selftest`，那只吃自造的合成瓦片）。
已披露的偏倚全部列在预注册第六节（核心一条：0.25 周/像素这个截止频率是**看过 (M70) 那三张图之后**定的）。

零 GPU、零 API、零判官、零活件改动。只读：
  - `data/tiles/dataset_k16.json`（经 `model/tiles_data.load`，与 (M53)(M64)(M65)(P9) 同一调用口径）
  - `remote_tmp/p9/<方法>/*.png`（16px）、`remote_tmp/m65/<方法>/*.png`（32px）
  - `experiments/final_E_mat_{16,32}.json`（只用于 (OP1) 出处核对）

⚑ 活件**不重写**：直接 import 冻结的 `m65_seam_anchor` / `p9_seam_abs`，复用它们的
`activepieces`（从 AST 借 `tile_seam_ratio`/`load_method`）/ `median` / `finite` /
`pack_bootstrap_ci` / `human_tiles` / `prompt_slugs` / `SIZES`。
⛔ 没改 `eval/*` 一个字节、⛔ 没改两个冻结判读器一个字节、⛔ 没 import torch。

用法：
    python analysis/arch/m71_struct_scale.py --selftest
    python analysis/arch/m71_struct_scale.py --out experiments/m71_struct_scale.json
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
import p9_seam_abs as P9                                           # noqa: E402  冻结的 (P9) 判读器

ROOT = M65.ROOT
B_BOOT, SEED = M65.B_BOOT, M65.SEED          # 2000 / 0，与 (M65)(P9) 同值
LF_CUT = 0.25                                # 截止频率：周/像素（⇔ 波长 ≥ 4px）
FLAT_MAX, FLAT_GAP = 0.10, 0.05              # (OP6) 门槛，与 (M65)(P9) 同值
NOISE_N, NOISE_TOL = 200, 0.05               # (OP2) 白噪声中性点
OP3_SINE_MIN, OP3_SQUARE_MIN, OP3_CHECKER_MAX = 0.95, 0.80, 0.05   # (OP3) 结构哨兵，见预注册补注
OP5_TOL = 0.02                               # (OP5) 自助稳健

SIZES = P9.SIZES                             # 16 -> remote_tmp/p9、32 -> remote_tmp/m65（冻结件里那份）
TRT = {16: "TRD16c_rr4", 32: "TRD32_rr4"}    # 每个尺寸的"正式交付"方法
EXPECT_COUNT = {("16", "train"): 2732, ("16", "val"): 291,          # (OP4)
                ("32", "train"): 403, ("32", "val"): 156}


# ---------------------------------------------------------------- 统计量（预注册第二节逐条）
def lowfreq_mask(shape, cut=LF_CUT):
    """频率格子 (u,v) 的径向频率 <= cut 的布尔掩码；DC 格子恒为 False。"""
    n, m = shape
    u = np.arange(n)
    v = np.arange(m)
    fu = np.minimum(u, n - u) / float(n)
    fv = np.minimum(v, m - v) / float(m)
    f = np.sqrt(fu[:, None] ** 2 + fv[None, :] ** 2)
    mask = f <= cut
    mask[0, 0] = False
    return mask


def neutral_point(shape, cut=LF_CUT):
    """白噪声的期望 LF ＝ 低频格子数 / 非 DC 格子数（纯几何常数，与任何数据无关）。"""
    mask = lowfreq_mask(shape, cut)
    return float(mask.sum()) / float(shape[0] * shape[1] - 1)


def lf(t, cut=LF_CUT):
    """LF(t)：去均值亮度的二维功率谱中，径向频率 <= cut 的能量占比；平涂瓦片返回 nan。"""
    a = np.asarray(t, np.float64)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    y = y - y.mean()
    p = np.abs(np.fft.fft2(y)) ** 2
    p[0, 0] = 0.0                                   # 丢掉 DC
    tot = float(p.sum())
    if not (tot > 0.0):
        return float("nan")
    return float(p[lowfreq_mask(y.shape, cut)].sum() / tot)


def shuffle_pixels(t, rng):
    """逐像素随机置换：颜色多重集逐位不变、结构全毁（(V0) 的对照）。"""
    a = np.asarray(t)
    flat = a.reshape(-1, a.shape[-1])
    idx = list(range(flat.shape[0]))
    rng.shuffle(idx)
    return flat[idx].reshape(a.shape)


def summarize(vals):
    fin = M65.finite(vals)
    return {"n": len(vals), "n_finite": len(fin),
            "flat_rate": M65.flat_rate(vals),
            "median_LF": M65.median(fin) if fin else float("nan")}


# ---------------------------------------------------------------- 判据（预注册第四节逐条）
def classify(lo, hi, val, below, inside, above):
    if val is None or not math.isfinite(val):
        return "VOID_NO_DATA", "method LF not available"
    if val < lo:
        return below, "LF below human CI"
    if val > hi:
        return above, "LF above human CI"
    return inside, "LF inside human CI"


def verdict_for_size(size, lo, hi, r_shuf, l_trt):
    """(V0) 分辨力先行 -> 主判据 (C1)/(C2)。"""
    if lo <= r_shuf <= hi:
        return "VOID_STRUCT_RULER_BLIND", "R_shuf inside human CI"
    if size == 32:
        return classify(lo, hi, l_trt, "STRUCT32_BELOW_HUMAN",
                        "STRUCT32_IN_HUMAN_RANGE", "STRUCT32_ABOVE_HUMAN")
    return classify(lo, hi, l_trt, "STRUCT16_BELOW_HUMAN",
                    "STRUCT16_IN_HUMAN_RANGE", "STRUCT16_ABOVE_HUMAN")


# ---------------------------------------------------------------- (OP3) 结构哨兵
def stripe_tile(n=32, period=8, square=True):
    """周期 8px 的横条纹。

    square=False（正弦）：全部能量在 f=1/8 <= 0.25 => LF 解析上恰为 1。
    square=True （方波）：三次谐波 f=3/8=0.375 落在截止**之外**、带走 1/9 功率
                          => LF 解析上只有 ~0.854（见预注册补注，跑真料之前算出的）。
    """
    x = np.arange(n, dtype=np.float64)
    prof = (((x % period) < period / 2.0) * 200.0 if square
            else 100.0 + 90.0 * np.sin(2 * np.pi * x / period))
    return np.repeat(prof[:, None, None], n, axis=1).repeat(3, axis=2).astype(np.uint8)


def checker_tile(n=32):
    """1px 棋盘：全部能量落在 Nyquist => LF 必须 ~0。"""
    g = ((np.arange(n)[:, None] + np.arange(n)[None, :]) % 2) * 200
    return np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)


# ---------------------------------------------------------------- 一个尺寸的全部读数
def read_size(ns, size, cfg, root):
    from pathlib import Path
    rep = {"size": size}

    # ---- 真人锚点（train + val，⛔ 不碰 test），按包收 LF
    counts, by_pack, all_h, tiles_h = {}, {}, [], []
    for split in ("train", "val"):
        rows = M65.human_tiles(size, split)
        counts["%d_%s" % (size, split)] = len(rows)
        for img, pack in rows:
            v = lf(img)
            all_h.append(v)
            tiles_h.append(img)
            if math.isfinite(v):
                by_pack.setdefault(pack, []).append(v)
    rep["human"] = summarize(all_h)
    rep["counts"] = counts
    lo, hi, n_packs = M65.pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED)
    lo1, hi1, _ = M65.pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED + 1)
    rep["human_ci95"] = [lo, hi]
    rep["n_packs"] = n_packs
    rep["op5_ci_seed1"] = [lo1, hi1]
    rep["op5_ok"] = abs(lo - lo1) < OP5_TOL and abs(hi - hi1) < OP5_TOL

    # ---- (V0) 对照：逐像素打乱的真人瓦片（配色完全相同）
    rng = random.Random(SEED)
    rep["R_shuf"] = summarize([lf(shuffle_pixels(t, rng)) for t in tiles_h])

    # ---- 方法：已发表那条口径（E_mat 272 slug、每材质第一张）
    slugs = M65.prompt_slugs("E_mat")
    with open(os.path.join(root, "experiments", cfg["json"]), encoding="utf-8") as f:
        official = json.load(f)
    seam = ns["tile_seam_ratio"]
    methods, op1 = {}, {}
    for m in cfg["methods"]:
        d = os.path.join(root, cfg["dirs"], m)
        if not os.path.isdir(d):
            methods[m] = {"error": "missing dir %s" % d}
            op1[m] = {"ok": False, "why": "missing dir"}
            continue
        first, _ = ns["load_method"](Path(d), slugs)
        got = [t for t in first if t is not None]
        methods[m] = summarize([lf(t) for t in got])
        # (OP1) 出处核对：同一批瓦片上重算活件的签名接缝比，必须与已发表列逐位相同
        sig = M65.median(M65.finite([seam(np.asarray(t, np.float64)) for t in got]))
        want = official.get(m, {}).get("tile_seam_ratio")
        op1[m] = {"recomputed_seam": sig, "official_seam": want,
                  "ok": want is not None and sig == want}
    rep["methods"] = methods
    rep["op1"] = {"detail": op1, "n_mismatch": sum(1 for v in op1.values() if not v["ok"]),
                  "ok": all(v["ok"] for v in op1.values())}

    # ---- 判决
    trt = TRT[size]
    l_trt = methods.get(trt, {}).get("median_LF")
    f_trt = methods.get(trt, {}).get("flat_rate")
    f_hum = rep["human"]["flat_rate"]
    rep["op6_ok"] = (f_trt is not None and f_trt < FLAT_MAX and f_hum < FLAT_MAX
                     and abs(f_trt - f_hum) < FLAT_GAP)
    v, why = verdict_for_size(size, lo, hi, rep["R_shuf"]["median_LF"], l_trt)
    rep["trt"] = trt
    rep["trt_LF"] = l_trt
    rep["verdict"] = v
    rep["why"] = why
    return rep


def run(root, out_path):
    ns = M65.activepieces()
    res = {"note": "(M71) structure-scale ruler LF; criteria pre-registered in ec16dd3",
           "LF_cut_cyc_per_px": LF_CUT}

    sizes = {}
    for size, cfg in SIZES.items():
        sizes[str(size)] = read_size(ns, size, cfg, root)
    res["sizes"] = sizes

    # ---- 操作检验（与尺寸无关的几条）
    op2 = {}
    nrng = np.random.default_rng(SEED)
    for size in sorted(SIZES):
        vals = [lf(nrng.integers(0, 256, (size, size, 3)).astype(np.uint8))
                for _ in range(NOISE_N)]
        med = M65.median(M65.finite(vals))
        expect = neutral_point((size, size))
        op2["%d" % size] = {"median_LF": med, "expected": expect,
                            "ok": abs(med - expect) <= NOISE_TOL}
    lf_sine = lf(stripe_tile(square=False))
    lf_square = lf(stripe_tile(square=True))
    lf_checker = lf(checker_tile())
    got_counts = {}
    for s in sizes.values():
        got_counts.update(s["counts"])
    op4 = {"%s_%s" % k: got_counts.get("%s_%s" % k) == v for k, v in EXPECT_COUNT.items()}

    res["op"] = {
        "OP1_provenance_seam_matches_official": {
            "ok": all(s["op1"]["ok"] for s in sizes.values()),
            "n_mismatch": {k: s["op1"]["n_mismatch"] for k, s in sizes.items()}},
        "OP2_white_noise_neutral_point": {"ok": all(v["ok"] for v in op2.values()), "detail": op2},
        "OP3_structure_sentinels": {"ok": bool(lf_sine >= OP3_SINE_MIN
                                               and lf_square >= OP3_SQUARE_MIN
                                               and lf_checker <= OP3_CHECKER_MAX),
                                    "sine_stripe_LF": lf_sine, "square_stripe_LF": lf_square,
                                    "checker_LF": lf_checker},
        "OP4_human_counts": {"ok": all(op4.values()), "detail": op4, "counts": got_counts},
        "OP5_bootstrap_stable": {"ok": all(s["op5_ok"] for s in sizes.values()),
                                 "ci_seed1": {k: s["op5_ci_seed1"] for k, s in sizes.items()}},
        "OP6_flat_rate": {"ok": all(s["op6_ok"] for s in sizes.values()),
                          "detail": {k: {"human": s["human"]["flat_rate"],
                                         "trt": s["methods"].get(s["trt"], {}).get("flat_rate")}
                                     for k, s in sizes.items()}},
    }
    ops_ok = all(v["ok"] for v in res["op"].values())
    if not ops_ok:
        for k in sizes:
            sizes[k]["verdict"], sizes[k]["why"] = "VOID_OP", "operation check failed"
    res["verdict_32"] = sizes["32"]["verdict"]
    res["verdict_16"] = sizes["16"]["verdict"]

    if out_path:                                    # 落盘一律在打印之前
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    for k in ("32", "16"):
        s = sizes[k]
        print("=== %spx  (%d packs, %d human tiles)" % (k, s["n_packs"], s["human"]["n"]))
        print("  human anchor LF = %.4f   pack-bootstrap 95%%CI [%.4f, %.4f]"
              % (s["human"]["median_LF"], s["human_ci95"][0], s["human_ci95"][1]))
        print("  control R_shuf  = %.4f   (n=%d)" % (s["R_shuf"]["median_LF"], s["R_shuf"]["n"]))
        for m, r in s["methods"].items():
            if "error" in r:
                print("    %-12s %s" % (m, r["error"]))
            else:
                print("    %-12s n=%3d  LF=%.4f  flat=%.3f%s"
                      % (m, r["n"], r["median_LF"], r["flat_rate"],
                         "   <= delivered" if m == s["trt"] else ""))
        print("  VERDICT[%spx]: %s   (%s)" % (k, s["verdict"], s["why"]))
    for k, v in res["op"].items():
        print("  [%s] %s" % ("ok " if v["ok"] else "FAIL", k))
    if out_path:
        print("wrote %s" % out_path)
    return res


# ---------------------------------------------------------------- selftest（只吃自造的合成瓦片）
def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    ns = M65.activepieces()
    chk(callable(ns["tile_seam_ratio"]) and callable(ns["load_method"]), "borrowed活件")
    chk(SIZES[16]["dirs"] == "remote_tmp/p9" and SIZES[32]["dirs"] == "remote_tmp/m65", "冻结 SIZES")

    # ---- 掩码几何
    mk = lowfreq_mask((32, 32))
    chk(not mk[0, 0], "DC 被排除")
    chk(mk[1, 0] and mk[0, 1], "最低频在掩码内")
    chk(not mk[16, 16], "Nyquist 角不在掩码内")
    chk(mk[8, 0] and not mk[9, 0], "cut=0.25 的边界恰在 u=8")
    chk(lowfreq_mask((16, 32)).shape == (16, 32), "非方形瓦片也能算")
    np_ = neutral_point((32, 32))
    chk(0.0 < np_ < 1.0, "中性点在 (0,1)")
    chk(abs(neutral_point((16, 16)) - lowfreq_mask((16, 16)).sum() / 255.0) < 1e-12, "中性点定义")

    # ---- LF 的基本性质
    chk(math.isnan(lf(np.full((16, 16, 3), 7, np.uint8))), "平涂 -> nan")
    chk(lf(stripe_tile(square=False)) >= OP3_SINE_MIN, "(OP3) 正弦条纹哨兵")
    chk(lf(stripe_tile(square=True)) >= OP3_SQUARE_MIN, "(OP3) 方波条纹哨兵")
    chk(lf(stripe_tile(square=True)) < 0.90, "方波确实被三次谐波拉低（补注那条算术）")
    chk(lf(checker_tile()) <= OP3_CHECKER_MAX, "(OP3) 棋盘哨兵")
    sine = np.zeros((32, 32, 3), np.float64)
    sine += (np.sin(2 * np.pi * np.arange(32) / 32.0)[:, None, None] * 50.0 + 100.0)
    chk(abs(lf(sine) - 1.0) < 1e-9, "单一低频 -> LF=1")
    chk(abs(lf(sine) - lf(sine * 3.0)) < 1e-9, "对比度缩放不变")
    chk(abs(lf(sine) - lf(sine + 17.0)) < 1e-9, "加常数不变")
    v = lf(stripe_tile() // 2 + checker_tile() // 2)
    chk(0.0 <= v <= 1.0, "LF 落在 [0,1]")
    rng0 = np.random.default_rng(0)
    noise = rng0.integers(0, 256, (32, 32, 3)).astype(np.uint8)
    chk(abs(lf(noise) - neutral_point((32, 32))) < 0.15, "白噪声贴中性点")

    # ---- 打乱像素
    r = random.Random(0)
    st = stripe_tile()
    sh = shuffle_pixels(st, r)
    chk(sorted(map(tuple, sh.reshape(-1, 3).tolist()))
        == sorted(map(tuple, st.reshape(-1, 3).tolist())), "打乱保持颜色多重集")
    chk(lf(sh) < lf(st), "打乱把 LF 压下去")
    chk(shuffle_pixels(st, random.Random(0)).tobytes() == sh.tobytes(), "打乱可复现")

    # ---- 判据分支（逐条照抄预注册）
    chk(verdict_for_size(32, 0.4, 0.8, 0.1, 0.2)[0] == "STRUCT32_BELOW_HUMAN", "C1 below")
    chk(verdict_for_size(32, 0.4, 0.8, 0.1, 0.6)[0] == "STRUCT32_IN_HUMAN_RANGE", "C1 in")
    chk(verdict_for_size(32, 0.4, 0.8, 0.1, 0.9)[0] == "STRUCT32_ABOVE_HUMAN", "C1 above")
    chk(verdict_for_size(16, 0.4, 0.8, 0.1, 0.2)[0] == "STRUCT16_BELOW_HUMAN", "C2 below")
    chk(verdict_for_size(16, 0.4, 0.8, 0.1, 0.6)[0] == "STRUCT16_IN_HUMAN_RANGE", "C2 in")
    chk(verdict_for_size(32, 0.4, 0.8, 0.5, 0.2)[0] == "VOID_STRUCT_RULER_BLIND", "V0 先行")
    chk(verdict_for_size(32, 0.4, 0.8, 0.1, float("nan"))[0] == "VOID_NO_DATA", "缺数据")

    # ---- 汇总
    s = summarize([0.2, float("nan"), 0.4, 0.6])
    chk(s["n"] == 4 and s["n_finite"] == 3 and abs(s["flat_rate"] - 0.25) < 1e-12, "summarize 计数")
    chk(abs(s["median_LF"] - 0.4) < 1e-12, "summarize 中位数")
    lo, hi, npk = M65.pack_bootstrap_ci({"p": [0.1, 0.2, 0.3]}, b=50)
    chk(lo == hi == 0.2 and npk == 1, "单包自助退化")
    print("selftest OK %d/%d" % (ok, 32))


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
