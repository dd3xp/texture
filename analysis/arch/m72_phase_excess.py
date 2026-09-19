#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M72) 判读器：`LF` 的**盲区**那一维 —— 相位/组织。人的锚点在哪儿，我们交付的输出在不在里面？

判据逐字照抄已提交的预注册（`docs/arch_progress.md` 的 (M72) 节，commit a533d6c），**盲写**：
写这个文件时我**一个 `GS` / `PX` 读数都没算过** —— 没算过真人锚点、没算过 CI、没算过
`R_phase`/`R_shuf`、没算过任何方法，也没跑过本文件（除 `--selftest`，那只吃自造的合成瓦片）。
已披露的偏倚全部列在预注册第六节（核心一条：我看过 (M70) 那三张图、也看过 (M71) 的全部读数）。

零 GPU、零 API、零判官、零活件改动。只读：
  - `data/tiles/dataset_k16.json`（经 `model/tiles_data.load`，与 (M53)(M64)(M65)(P9)(M71) 同口径）
  - `remote_tmp/p9/<方法>/*.png`（16px）、`remote_tmp/m65/<方法>/*.png`（32px）
  - `experiments/final_E_mat_{16,32}.json`（只用于 (OP1) 出处核对）

⚑ 活件**不重写**：直接 import 冻结的 `m65_seam_anchor` / `p9_seam_abs` / `m71_struct_scale`，
复用它们的 `activepieces`（从 AST 借 `tile_seam_ratio`/`load_method`）/ `median` / `finite` /
`flat_rate` / `pack_bootstrap_ci` / `human_tiles` / `prompt_slugs` / `SIZES` / `lf` / `shuffle_pixels`。
⛔ 没改 `eval/*` 一个字节、⛔ 没改三个冻结判读器一个字节、⛔ 没 import torch。

用法：
    python analysis/arch/m72_phase_excess.py --selftest
    python analysis/arch/m72_phase_excess.py --out experiments/m72_phase_excess.json
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
import m71_struct_scale as M71                                     # noqa: E402  冻结的 (M71) 判读器

ROOT = M65.ROOT
B_BOOT, SEED = M65.B_BOOT, M65.SEED          # 2000 / 0，与 (M65)(P9)(M71) 同值
REPS = 2                                     # `PX` 里控制项的独立抽相位次数（预注册第二节）
FLAT_MAX, FLAT_GAP = 0.10, 0.05              # (OP6) 门槛，与 (M65)(P9)(M71) 同值
NOISE_N = 200                                # (OP2) 白噪声批量
NOISE_LO, NOISE_HI = 0.30, 0.40              # (OP2) 解析区间：三角 1/3 ~ 高斯 1-2/pi=0.3634
OP3_SAMPLE, OP3_IMAG_MAX, OP3_LF_TOL = 32, 1e-6, 1e-9          # (OP3) 精确检验
OP5_TOL = 0.02                               # (OP5) 自助稳健
SQUARE_GS, CHECKER_GS, OP7_TOL = 0.875, 0.0, 1e-9              # (OP7a)(OP7b) 解析值
SINE_GS = (6.0 - math.sqrt(2.0)) / 8.0       # 0.5732233…（只登记，⛔ 不设门槛）

SIZES = P9.SIZES                             # 16 -> remote_tmp/p9、32 -> remote_tmp/m65（冻结件里那份）
TRT = {16: "TRD16c_rr4", 32: "TRD32_rr4"}    # 每个尺寸的"正式交付"方法
EXPECT_COUNT = {("16", "train"): 2732, ("16", "val"): 291,          # (OP4)
                ("32", "train"): 403, ("32", "val"): 156}


# ---------------------------------------------------------------- 第 1 层：GS（预注册第二节逐条）
def rgb(t):
    """统一成 H×W×3（RGBA 只取前三通道）。"""
    return np.asarray(t)[..., :3]


def gs(t):
    """GS(t) = 1 - (mean|g|)^2 / mean(g^2)，g ＝环面一阶差分、两方向合并；平涂瓦片返回 nan。"""
    a = np.asarray(rgb(t), np.float64)
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    g = np.concatenate([(y - np.roll(y, -1, axis=0)).ravel(),
                        (y - np.roll(y, -1, axis=1)).ravel()])
    m2 = float((g * g).mean())
    if not (m2 > 0.0):
        return float("nan")
    m1 = float(np.abs(g).mean())
    return float(1.0 - m1 * m1 / m2)


# ---------------------------------------------------------------- 第 2 层：相位随机化 + 重量化
def herm_phase(shape, rng):
    """厄米反对称随机相位场：psi[(-u)%H,(-v)%W] == -psi[u,v]，自共轭格子（含 DC/Nyquist）恒为 0。"""
    n, m = shape
    u = np.arange(n)[:, None]
    v = np.arange(m)[None, :]
    cu = (-u) % n
    cv = (-v) % m
    a = rng.uniform(-np.pi, np.pi, (n, m))
    first = (u < cu) | ((u == cu) & (v <= cv))          # 每对共轭格子里恰好挑中一个
    psi = np.where(first, a, -a[cu, cv])
    return np.where((u == cu) & (v == cv), 0.0, psi)


def phase_randomize_float(t, psi):
    """逐通道用**同一个** psi 换相位（DC 不动 ⇒ 均色不变）。返回 (浮点结果, 最大虚部)。"""
    a = np.asarray(rgb(t), np.float64)
    e = np.exp(1j * psi)
    out = np.empty(a.shape, np.float64)
    imax = 0.0
    for c in range(a.shape[2]):
        z = np.fft.ifft2(np.fft.fft2(a[..., c]) * e)
        imax = max(imax, float(np.abs(z.imag).max()))
        out[..., c] = z.real
    return out, imax


def palette_of(t):
    """瓦片自己的唯一颜色集合（K×3, uint8）。"""
    a = rgb(t)
    return np.unique(a.reshape(-1, 3), axis=0)


def quantize_to_palette(arr, pal):
    """每个像素取调色板里 RGB 欧氏距离最近的那一个。"""
    flat = np.asarray(arr, np.float64).reshape(-1, 3)
    pf = np.asarray(pal, np.float64)
    d = ((flat[:, None, :] - pf[None, :, :]) ** 2).sum(-1)
    return pal[d.argmin(1)].reshape(np.asarray(arr).shape[:2] + (3,))


def phi_twin(t, rng):
    """Phi(t)：幅度谱逐位不变、调色板逐色不变、组织全毁的那张"孪生瓦片"。"""
    a = rgb(t)
    psi = herm_phase(a.shape[:2], rng)
    f, _ = phase_randomize_float(a, psi)
    return quantize_to_palette(f, palette_of(a))


def px(t, rng, reps=REPS):
    """PX(t) = GS(t) - mean_r GS(Phi_r(t))（主统计量，逐张配对）。"""
    g0 = gs(t)
    if not math.isfinite(g0):
        return float("nan")
    vals = [gs(phi_twin(t, rng)) for _ in range(reps)]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("nan")
    return float(g0 - sum(vals) / len(vals))


def summarize(pxs, gss, ncol):
    fp, fg = M65.finite(pxs), M65.finite(gss)
    return {"n": len(pxs), "n_finite": len(fp),
            "flat_rate": M65.flat_rate(pxs),
            "median_PX": M65.median(fp) if fp else float("nan"),
            "median_GS": M65.median(fg) if fg else float("nan"),
            "median_ncolors": M65.median([float(x) for x in ncol]) if ncol else float("nan")}


# ---------------------------------------------------------------- 判据（预注册第四节逐条）
def classify(lo, hi, val, below, inside, above):
    if val is None or not math.isfinite(val):
        return "VOID_NO_DATA", "method PX not available"
    if val < lo:
        return below, "PX below human CI"
    if val > hi:
        return above, "PX above human CI"
    return inside, "PX inside human CI"


def verdict_for_size(size, lo, hi, r_phase, p_trt):
    """(V0) 分辨力先行 -> 主判据 (C1)/(C2)。"""
    if lo <= r_phase <= hi:
        return "VOID_PHASE_RULER_BLIND", "R_phase inside human CI"
    if size == 32:
        return classify(lo, hi, p_trt, "PHASE32_BELOW_HUMAN",
                        "PHASE32_IN_HUMAN_RANGE", "PHASE32_ABOVE_HUMAN")
    return classify(lo, hi, p_trt, "PHASE16_BELOW_HUMAN",
                    "PHASE16_IN_HUMAN_RANGE", "PHASE16_ABOVE_HUMAN")


# ---------------------------------------------------------------- 合成哨兵（(OP7)）
def float_sine_tile(n=32, period=8):
    """浮点正弦条纹：解析上 GS = (6-sqrt2)/8（⛔ 不取整，取整会破坏那个闭式）。"""
    prof = 100.0 + 90.0 * np.sin(2 * np.pi * np.arange(n) / float(period))
    return np.repeat(prof[:, None, None], n, axis=1).repeat(3, axis=2)


def bands16_tile(n=32, h=2, seed=7):
    """(OP7d) 的阳性刺激（见 `docs/arch_progress.md` 的 (M72) 补注）：
    16 条 2px 高的随机色横带 ＝ 能量摊在很多频率上、而这些相位**对齐**成了 16 条干净的边。
    ⛔ 换掉的只是刺激：原来的方波条纹 `PX` **恒等于 0**（单一主频 ⇒ 阈值后转折点数与相位无关）。"""
    pal = np.random.default_rng(seed).integers(0, 256, (n // h, 3)).astype(np.uint8)
    return np.repeat(pal, h, axis=0)[:, None, :].repeat(n, axis=1)


# ---------------------------------------------------------------- 一个尺寸的全部读数
def read_size(ns, size, cfg, root):
    from pathlib import Path
    rep = {"size": size}

    # ---- 真人锚点（train + val，⛔ 不碰 test），按包收 PX
    counts, by_pack, pxs, gss, ncol, tiles_h = {}, {}, [], [], [], []
    rng = np.random.default_rng(SEED)
    for split in ("train", "val"):
        rows = M65.human_tiles(size, split)
        counts["%d_%s" % (size, split)] = len(rows)
        for img, pack in rows:
            v = px(img, rng)
            pxs.append(v)
            gss.append(gs(img))
            ncol.append(len(palette_of(img)))
            tiles_h.append(rgb(img))
            if math.isfinite(v):
                by_pack.setdefault(pack, []).append(v)
    rep["human"] = summarize(pxs, gss, ncol)
    rep["counts"] = counts
    lo, hi, n_packs = M65.pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED)
    lo1, hi1, _ = M65.pack_bootstrap_ci(by_pack, b=B_BOOT, seed=SEED + 1)
    rep["human_ci95"] = [lo, hi]
    rep["n_packs"] = n_packs
    rep["op5_ci_seed1"] = [lo1, hi1]
    rep["op5_ok"] = abs(lo - lo1) < OP5_TOL and abs(hi - hi1) < OP5_TOL

    # ---- (V0) 对照 R_phase：先把真人瓦片相位随机化一次，再对它算 PX（真值应 ~0）
    rngc = np.random.default_rng(SEED + 100)
    twins = [phi_twin(t, rngc) for t in tiles_h]
    rep["R_phase"] = summarize([px(t, rngc) for t in twins],
                               [gs(t) for t in twins], [len(palette_of(t)) for t in twins])

    # ---- 只登记的第二条对照 R_shuf：逐像素打乱（配色多重集逐位不变）
    rs = random.Random(SEED)
    shuf = [M71.shuffle_pixels(t, rs) for t in tiles_h]
    rngs = np.random.default_rng(SEED + 200)
    rep["R_shuf"] = summarize([px(t, rngs) for t in shuf],
                              [gs(t) for t in shuf], [len(palette_of(t)) for t in shuf])

    # ---- (OP3) 相位对照确实是 LF 的盲区：重量化之前那一步，逐位精确
    step = max(1, len(tiles_h) // OP3_SAMPLE)
    op3_imag, op3_dlf = 0.0, 0.0
    rng3 = np.random.default_rng(SEED + 300)
    for t in tiles_h[::step][:OP3_SAMPLE]:
        f, imax = phase_randomize_float(t, herm_phase(np.asarray(t).shape[:2], rng3))
        op3_imag = max(op3_imag, imax)
        a, b = M71.lf(t), M71.lf(f)
        if math.isfinite(a) and math.isfinite(b):
            op3_dlf = max(op3_dlf, abs(a - b))
    rep["op3"] = {"max_imag": op3_imag, "max_dLF": op3_dlf,
                  "n_checked": len(tiles_h[::step][:OP3_SAMPLE]),
                  "ok": op3_imag < OP3_IMAG_MAX and op3_dlf < OP3_LF_TOL}

    # ---- 方法：已发表那条口径（E_mat 272 slug、每材质第一张）
    slugs = M65.prompt_slugs("E_mat")
    with open(os.path.join(root, "experiments", cfg["json"]), encoding="utf-8") as f:
        official = json.load(f)
    seam = ns["tile_seam_ratio"]
    methods, op1 = {}, {}
    rngm = np.random.default_rng(SEED + 400)
    for m in cfg["methods"]:
        d = os.path.join(root, cfg["dirs"], m)
        if not os.path.isdir(d):
            methods[m] = {"error": "missing dir %s" % d}
            op1[m] = {"ok": False, "why": "missing dir"}
            continue
        first, _ = ns["load_method"](Path(d), slugs)
        got = [t for t in first if t is not None]
        methods[m] = summarize([px(t, rngm) for t in got], [gs(t) for t in got],
                               [len(palette_of(t)) for t in got])
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
    p_trt = methods.get(trt, {}).get("median_PX")
    f_trt = methods.get(trt, {}).get("flat_rate")
    f_hum = rep["human"]["flat_rate"]
    rep["op6_ok"] = (f_trt is not None and f_trt < FLAT_MAX and f_hum < FLAT_MAX
                     and abs(f_trt - f_hum) < FLAT_GAP)
    v, why = verdict_for_size(size, lo, hi, rep["R_phase"]["median_PX"], p_trt)
    rep["trt"] = trt
    rep["trt_PX"] = p_trt
    rep["verdict"] = v
    rep["why"] = why
    return rep


def run(root, out_path):
    ns = M65.activepieces()
    res = {"note": "(M72) phase-excess ruler PX; criteria pre-registered in a533d6c", "reps": REPS}

    sizes = {}
    for size, cfg in SIZES.items():
        sizes[str(size)] = read_size(ns, size, cfg, root)
    res["sizes"] = sizes

    # ---- 与尺寸无关的操作检验
    op2 = {}
    nrng = np.random.default_rng(SEED)
    for size in sorted(SIZES):
        vals = [gs(nrng.integers(0, 256, (size, size, 3)).astype(np.uint8))
                for _ in range(NOISE_N)]
        med = M65.median(M65.finite(vals))
        op2["%d" % size] = {"median_GS": med, "ok": NOISE_LO <= med <= NOISE_HI}

    sq, ck, bd = M71.stripe_tile(square=True), M71.checker_tile(), bands16_tile()
    gs_sq, gs_ck, gs_sine = gs(sq), gs(ck), gs(float_sine_tile())
    px_bd = px(bd, np.random.default_rng(SEED))
    op7 = {"square_GS": gs_sq, "checker_GS": gs_ck, "sine_GS_float": gs_sine,
           "sine_GS_analytic": SINE_GS, "flat_is_nan": not math.isfinite(gs(np.full((16, 16, 3), 7,
                                                                                    np.uint8))),
           "bands16_PX": px_bd, "bands16_GS": gs(bd),
           "square_PX_is_zero_by_construction": px(sq, np.random.default_rng(SEED))}
    op7["ok"] = bool(abs(gs_sq - SQUARE_GS) < OP7_TOL and abs(gs_ck - CHECKER_GS) < OP7_TOL
                     and op7["flat_is_nan"] and math.isfinite(px_bd) and px_bd > 0.0)

    got_counts = {}
    for s in sizes.values():
        got_counts.update(s["counts"])
    op4 = {"%s_%s" % k: got_counts.get("%s_%s" % k) == v for k, v in EXPECT_COUNT.items()}

    res["op"] = {
        "OP1_provenance_seam_matches_official": {
            "ok": all(s["op1"]["ok"] for s in sizes.values()),
            "n_mismatch": {k: s["op1"]["n_mismatch"] for k, s in sizes.items()}},
        "OP2_white_noise_bracket": {"ok": all(v["ok"] for v in op2.values()), "detail": op2,
                                    "bracket": [NOISE_LO, NOISE_HI]},
        "OP3_phase_control_is_LF_blindspot": {"ok": all(s["op3"]["ok"] for s in sizes.values()),
                                              "detail": {k: s["op3"] for k, s in sizes.items()}},
        "OP4_human_counts": {"ok": all(op4.values()), "detail": op4, "counts": got_counts},
        "OP5_bootstrap_stable": {"ok": all(s["op5_ok"] for s in sizes.values()),
                                 "ci_seed1": {k: s["op5_ci_seed1"] for k, s in sizes.items()}},
        "OP6_flat_rate": {"ok": all(s["op6_ok"] for s in sizes.values()),
                          "detail": {k: {"human": s["human"]["flat_rate"],
                                         "trt": s["methods"].get(s["trt"], {}).get("flat_rate")}
                                     for k, s in sizes.items()}},
        "OP7_synthetic_sentinels": op7,
    }
    if not all(v["ok"] for v in res["op"].values()):
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
        print("  human anchor PX = %.4f   pack-bootstrap 95%%CI [%.4f, %.4f]   (raw GS %.4f)"
              % (s["human"]["median_PX"], s["human_ci95"][0], s["human_ci95"][1],
                 s["human"]["median_GS"]))
        print("  control R_phase = %.4f   R_shuf = %.4f"
              % (s["R_phase"]["median_PX"], s["R_shuf"]["median_PX"]))
        for m, r in s["methods"].items():
            if "error" in r:
                print("    %-12s %s" % (m, r["error"]))
            else:
                print("    %-12s n=%3d  PX=%+.4f  GS=%.4f  ncol=%5.1f  flat=%.3f%s"
                      % (m, r["n"], r["median_PX"], r["median_GS"], r["median_ncolors"],
                         r["flat_rate"], "   <= delivered" if m == s["trt"] else ""))
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
    chk(callable(ns["tile_seam_ratio"]) and callable(ns["load_method"]), "borrowed 活件")
    chk(SIZES[16]["dirs"] == "remote_tmp/p9" and SIZES[32]["dirs"] == "remote_tmp/m65", "冻结 SIZES")
    chk(abs(M71.LF_CUT - 0.25) < 1e-12 and callable(M71.lf), "冻结的 (M71) LF")

    # ---- GS 的解析哨兵
    sq, ck = M71.stripe_tile(square=True), M71.checker_tile()
    chk(abs(gs(sq) - SQUARE_GS) < OP7_TOL, "(OP7a) 方波条纹 GS=0.875")
    chk(abs(gs(ck) - CHECKER_GS) < OP7_TOL, "(OP7b) 棋盘 GS=0")
    chk(not math.isfinite(gs(np.full((16, 16, 3), 7, np.uint8))), "(OP7c) 平涂 -> nan")
    chk(abs(gs(float_sine_tile()) - SINE_GS) < 1e-9, "正弦条纹 GS=(6-sqrt2)/8")
    chk(gs(sq) > gs(float_sine_tile()), "硬边比渐变 GS 高（第二节那条限制）")
    chk(abs(gs(sq) - gs(np.asarray(sq, np.float64) * 3.0)) < 1e-9, "对比度缩放不变")
    chk(abs(gs(sq) - gs(np.asarray(sq, np.float64) + 17.0)) < 1e-9, "加常数不变")
    chk(0.0 <= gs(np.random.default_rng(0).integers(0, 256, (32, 32, 3))) <= 1.0, "GS 落在 [0,1]")
    noise = [gs(np.random.default_rng(i).integers(0, 256, (32, 32, 3)).astype(np.uint8))
             for i in range(20)]
    chk(NOISE_LO <= M65.median(noise) <= NOISE_HI, "(OP2) 白噪声落在解析区间")

    # ---- 厄米随机相位
    rng = np.random.default_rng(0)
    psi = herm_phase((32, 32), rng)
    u = np.arange(32)
    cj = psi[(-u) % 32][:, (-u) % 32]
    chk(np.abs(psi + cj).max() < 1e-12, "psi 厄米反对称")
    chk(psi[0, 0] == 0.0 and psi[16, 0] == 0.0 and psi[0, 16] == 0.0 and psi[16, 16] == 0.0,
        "自共轭格子 psi=0")
    chk(np.abs(psi).max() <= np.pi + 1e-12, "psi 落在 [-pi,pi]")
    chk(herm_phase((16, 32), np.random.default_rng(0)).shape == (16, 32), "非方形也能画")

    # ---- 相位随机化：虚部为零、幅度谱逐位不变
    f, imax = phase_randomize_float(sq, psi)
    chk(imax < OP3_IMAG_MAX, "(OP3) 虚部数值为零")
    chk(abs(M71.lf(sq) - M71.lf(f)) < OP3_LF_TOL, "(OP3) LF 逐位不变＝确实是盲区")
    chk(abs(np.asarray(sq, np.float64).mean() - f.mean()) < 1e-9, "DC 不动 ⇒ 均色不变")
    chk(gs(f) != gs(sq), "相位换了 ⇒ GS 变了")

    # ---- 调色板与重量化
    pal = palette_of(ck)
    chk(pal.shape == (2, 3), "棋盘只有两种颜色")
    q = quantize_to_palette(np.asarray(ck, np.float64), pal)
    chk(q.tobytes() == rgb(ck).tobytes(), "已在板上 ⇒ 重量化是恒等")
    tw = phi_twin(sq, np.random.default_rng(1))
    chk(set(map(tuple, palette_of(tw).tolist())) <= set(map(tuple, palette_of(sq).tolist())),
        "孪生瓦片的颜色是原板的子集")
    chk(phi_twin(sq, np.random.default_rng(1)).tobytes() == tw.tobytes(), "孪生瓦片可复现")

    # ---- PX
    bd = bands16_tile()
    chk(px(bd, np.random.default_rng(SEED)) > 0.0, "(OP7d) 16 色横带 PX>0")
    chk(abs(px(sq, np.random.default_rng(SEED))) < OP7_TOL, "单频方波 PX 恒 0（补注那条限制）")
    chk(not math.isfinite(px(np.full((16, 16, 3), 7, np.uint8), np.random.default_rng(0))),
        "平涂 -> PX=nan")
    nz = np.random.default_rng(3).integers(0, 256, (32, 32, 3)).astype(np.uint8)
    chk(abs(px(nz, np.random.default_rng(4))) < 0.10, "白噪声 PX 贴 0")
    chk(px(bd, np.random.default_rng(5)) > px(nz, np.random.default_rng(5)), "有组织的 PX 更大")

    # ---- 判据分支（逐条照抄预注册）
    chk(verdict_for_size(32, 0.2, 0.6, 0.0, 0.1)[0] == "PHASE32_BELOW_HUMAN", "C1 below")
    chk(verdict_for_size(32, 0.2, 0.6, 0.0, 0.4)[0] == "PHASE32_IN_HUMAN_RANGE", "C1 in")
    chk(verdict_for_size(32, 0.2, 0.6, 0.0, 0.9)[0] == "PHASE32_ABOVE_HUMAN", "C1 above")
    chk(verdict_for_size(16, 0.2, 0.6, 0.0, 0.1)[0] == "PHASE16_BELOW_HUMAN", "C2 below")
    chk(verdict_for_size(16, 0.2, 0.6, 0.0, 0.4)[0] == "PHASE16_IN_HUMAN_RANGE", "C2 in")
    chk(verdict_for_size(32, 0.2, 0.6, 0.3, 0.1)[0] == "VOID_PHASE_RULER_BLIND", "V0 先行")
    chk(verdict_for_size(32, 0.2, 0.6, 0.0, float("nan"))[0] == "VOID_NO_DATA", "缺数据")

    # ---- 汇总
    s = summarize([0.2, float("nan"), 0.4, 0.6], [0.5, 0.5, 0.5, 0.5], [16, 16, 16, 16])
    chk(s["n"] == 4 and s["n_finite"] == 3 and abs(s["flat_rate"] - 0.25) < 1e-12, "summarize 计数")
    chk(abs(s["median_PX"] - 0.4) < 1e-12 and abs(s["median_ncolors"] - 16.0) < 1e-12, "summarize 值")
    lo, hi, npk = M65.pack_bootstrap_ci({"p": [0.1, 0.2, 0.3]}, b=50)
    chk(lo == hi == 0.2 and npk == 1, "单包自助退化")
    st = M71.shuffle_pixels(sq, random.Random(0))
    chk(sorted(map(tuple, st.reshape(-1, 3).tolist()))
        == sorted(map(tuple, rgb(sq).reshape(-1, 3).tolist())), "冻结的打乱保持颜色多重集")
    print("selftest OK %d/%d" % (ok, 40))


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
