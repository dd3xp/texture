# -*- coding: utf-8 -*-
"""(M89) 等料轮：给 (M84) 判据的**统计量那一侧**标一次抽样精度。

⚠⚠ 本文件**不下任何判决、不改任何判据**。(M84) 的门槛 0.1250、两个标签
(`FROMSCRATCH_TOO_BLUNT` / `FROMSCRATCH_USABLE`)、(M85) 的分叉表，一个字未动。
本轮产出的全部是**只登记项**，与 (M73) 的 `f`（"只决定跑不跑、不决定报什么"）同款，
这里更弱一层：**它连跑不跑都不决定，只决定判决文本里要不要带一句"这次是贴线的"。**

## 立项（盲期写死，⛔ (M84) 处理臂读数 0/28 时提交）

(M84) 的主判据是一条**硬切换**：`MDE_retrain >= 0.1250` 判 `FROMSCRATCH_TOO_BLUNT`，
否则 `FROMSCRATCH_USABLE`，而两个落点在 (M85) 分叉表里指向**不同的下一轮**。
但 `MDE_retrain = 2.8016 * rms(D) / sqrt(67)` 本身是个**估计量**：它由 67 个材质上的
配对差 D 算出来，换一批材质就会抖。本项目从没量过这个抖动有多大。
⇒ 若 `MDE_retrain` 落在 0.1250 附近，判决照机械规则下，但"证据有多强"是另一回事
（(M82)「判决稳 ≠ 数稳」、(M19)「证据被高估 ≠ 结论被推翻」同族）。

## 判决日的规则（盲期写死，⛔ 读数出来后不许改）

料齐、(M84) 判决**落盘之后**，用本文件的冻结函数 `mde_ci()` 对 (M84) 自己的 D 做
**按材质**自助（B=20000、seed 0、percentile 法）⇒ 得 `MDE_retrain` 的 95% 区间。

- 该区间**盖住** 0.1250 ⇒ 登记 `NEAR_LINE`：判决标签不变，但账本与摘要引用该判决时
  **必须同引"这次是贴线的"**。
- 不盖住 ⇒ 登记 `CLEAR_OF_LINE`。
- ⛔ 两种落点都**不改判决标签**、**不改 (M85) 分叉表的下一轮动作**。

## 本轮做的事：在**已发表**的三条臂上标定这个区间会有多宽

自助宽度事先算不出（依赖 D 的形状），但可以在同口径的已发表臂上量：

    CAL = M60(K=17) / M62(K=28) / M75(K=28)  —— 同一台仪器、同一个统计单位 n_mat=67

对每条臂报 `rel_halfwidth = ((hi-lo)/2) / point`。**聚合规则先写死＝取三条的 max**
（⚑ 取 max 是保守方向：带越宽越容易判 `NEAR_LINE` ＝ 越不容易高估自己的判决）。
由它定出**贴线带** `[THR/(1+h), THR*(1+h)]`，判决日若 `mde_ci()` 还没跑就先用这个带看。

⚑ 按材质自助的合法性来自 (M69)：`D_m` 是「提示词文本 vs 我们自己生成的图」的两臂之差，
**一张真人瓦片都不读** ⇒ 包级 deff 实测 0.334 < 1 ⇒ 重采样单位就是材质、⛔ 不是包。

## 门槛那一侧（(B) 组，**只登记、⛔ 不进贴线带**）

0.1250 ＝ (M60) 的 `mean_D` 点估计，其已发表 95%CI ＝ [+0.02596, +0.22403]
（`experiments/m62_mde_audit.json`）⇒ 相对半宽 ~79%。
⚠⚠ **我们*选择*不把它放进贴线带**，理由：在 (M84) 里 0.1250 是一个**冻结的目标定义**
（"要能测到 (M60) 那种量级的效应"），不是本轮在估计的量。
⛔ 这是选择、不是"它不成立" —— 若把 (B) 也算进去，几乎任何落点都会被判贴线，
那等于让判据失去内容。本条必须原样写进账本，防止将来有人拿 (B) 去软化不喜欢的判决。

## 操作检验（全部在本文件 selftest + 真料上机械执行）

- (OP1) 三条臂的 `mde_point` 必须复现已发表值（M60/M62 对 `m62_mde_audit.json` 逐位；
        M75 对 `m84_read_retrain.MDE_M75`=0.2027 到 5e-4）。
- (OP2) 阴性对照：常数 D 向量的自助宽度必须**恰好 0**。
- (OP3) 自助种子稳定性：seed 0 与 seed 1 的两个端点相对差 < 1%。
- (OP4) 三条臂的 `n_mat` 必须都 ＝ 67。
- (OP5) 阳性对照：把 D 向量平铺 4 份（n=268）⇒ 相对半宽必须缩 ~2 倍（比值落在 [1.7,2.3]）。

⛔ 本文件**不读** (M84) 的任何读数（`m84_w384b_s*.json` 一个字没碰）。
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 冻结件：一律 import，⛔ 不重写 ----
from m60_read_scale import N_MAT_EXPECT, Z80_TWOSIDED, load_arm      # noqa: E402
from m62_mde_audit import diff_over, rms                              # noqa: E402
from m84_read_retrain import MDE_M75, THR_MDE                         # noqa: E402

B_BOOT = 20000            # 自助重抽次数（冻结）
BOOT_SEED = 0             # 冻结
BOOT_SEED_ALT = 1         # (OP3) 稳定性用
CI_LO, CI_HI = 2.5, 97.5  # percentile 法（冻结）

# (轮次, 目录, 控制 tag, 处理 tag, K, 已发表 MDE, 容差)
CAL = [
    ("M60", "remote_tmp/m60_read", "m60_ctrl", "m60_more", 17, 0.1415590725441918, 1e-9),
    ("M62", "remote_tmp/m62", "m62_ctrl", "m62_dbl", 28, 0.15254921467667584, 1e-9),
    ("M75", "remote_tmp", "m75_w384", "m75_w512", 28, MDE_M75, 5e-4),
]


def mde_of(d):
    """(M62) 实测口径：MDE = 2.8016 * rms(D) / sqrt(n)。⛔ 不是 σ1 公式。"""
    return Z80_TWOSIDED * rms(d) / math.sqrt(len(d))


def percentile(sorted_xs, q):
    """线性插值分位数（无 numpy）。"""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (q / 100.0) * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def mde_ci(d, b=B_BOOT, seed=BOOT_SEED):
    """按材质自助求 MDE 的 95% 区间。返回 dict（判决日直接调这一个函数）。"""
    n = len(d)
    point = mde_of(d)
    rng = random.Random(seed)
    boots = []
    for _ in range(b):
        s = 0.0
        for _ in range(n):
            x = d[rng.randrange(n)]
            s += x * x
        boots.append(Z80_TWOSIDED * math.sqrt(s / n) / math.sqrt(n))
    boots.sort()
    lo = percentile(boots, CI_LO)
    hi = percentile(boots, CI_HI)
    return {"n": n, "point": point, "lo": lo, "hi": hi, "b": b, "seed": seed,
            "rel_halfwidth": ((hi - lo) / 2.0) / point if point else float("nan")}


def band(thr, h):
    """贴线带：MDE 落在此区间内 ⇒ 登记 NEAR_LINE（⛔ 不改判决标签）。"""
    return [thr / (1.0 + h), thr * (1.0 + h)]


def load_d(root, d, ctag, ttag, k):
    """取一条已发表臂的逐材质 D 向量（走冻结的 load_arm / diff_over）。"""
    dirp = os.path.join(root, d)
    ctrl, _, cprob = load_arm(dirp, ctag, k)
    trt, _, tprob = load_arm(dirp, ttag, k)
    if cprob or tprob or len(ctrl) != k or len(trt) != k:
        return None, {"cprob": cprob, "tprob": tprob,
                      "n_ctrl": len(ctrl), "n_trt": len(trt)}
    mats = sorted(set(ctrl[sorted(ctrl)[0]]))
    return diff_over(ctrl, trt, mats, sorted(ctrl), sorted(trt)), None


def selftest():
    ok = 0
    # 1) mde_of 就是冻结口径
    assert abs(mde_of([0.2] * 67) - Z80_TWOSIDED * 0.2 / math.sqrt(67)) < 1e-12; ok += 1
    # 1b) 与 (M75) 已发表值对账：rms 0.5921 -> 0.2027
    assert abs(mde_of([0.5921] * 67) - MDE_M75) < 5e-4; ok += 1
    # 1c) 门槛没被本文件碰过
    assert THR_MDE == 0.1250; ok += 1

    # 2) percentile 的边界与插值
    assert abs(percentile([1.0, 2.0, 3.0], 50.0) - 2.0) < 1e-12; ok += 1
    assert abs(percentile([1.0, 2.0, 3.0], 0.0) - 1.0) < 1e-12; ok += 1
    assert abs(percentile([1.0, 2.0, 3.0], 100.0) - 3.0) < 1e-12; ok += 1
    assert abs(percentile([0.0, 1.0], 25.0) - 0.25) < 1e-12; ok += 1

    # 3) (OP2) 阴性对照：常数向量 -> 自助宽度恰 0
    #    ⚠ selftest 在见真料前改的断言（⛔ 没改式子）：`hi == lo == point` 写得太紧 ——
    #    点估计走 rms()、自助走累加，末位差 1 ulp。(OP2) 要问的是**宽度**恰零，
    #    它原样成立；点估计只要到 1e-12。
    c = mde_ci([0.3] * 67, b=500, seed=0)
    assert c["hi"] == c["lo"] and c["rel_halfwidth"] == 0.0; ok += 1
    assert abs(c["point"] - c["lo"]) < 1e-12; ok += 1

    # 4) 高斯样本的相对半宽应接近理论 1.96/sqrt(2n)
    rng = random.Random(12345)
    g = [rng.gauss(0.0, 1.0) for _ in range(67)]
    cg = mde_ci(g, b=4000, seed=0)
    theo = 1.96 / math.sqrt(2.0 * 67)
    assert 0.5 * theo < cg["rel_halfwidth"] < 2.0 * theo, cg["rel_halfwidth"]; ok += 1
    assert cg["lo"] < cg["point"] < cg["hi"]; ok += 1

    # 5) (OP5) 阳性对照：平铺 4 份 -> n 变 4 倍 -> 相对半宽缩 ~2 倍
    c4 = mde_ci(g * 4, b=4000, seed=0)
    ratio = cg["rel_halfwidth"] / c4["rel_halfwidth"]
    assert 1.7 < ratio < 2.3, ratio; ok += 1
    # 5b) 平铺不改点估计（rms 不变）
    assert abs(c4["point"] - Z80_TWOSIDED * rms(g) / math.sqrt(268)) < 1e-12; ok += 1

    # 6) 确定性：同 seed 逐位可复现；换 seed 会变
    assert mde_ci(g, b=800, seed=0) == mde_ci(g, b=800, seed=0); ok += 1
    assert mde_ci(g, b=800, seed=1)["lo"] != mde_ci(g, b=800, seed=0)["lo"]; ok += 1

    # 7) band 的形状与边界
    lo, hi = band(0.1250, 0.2)
    assert abs(hi - 0.15) < 1e-12 and abs(lo - 0.1250 / 1.2) < 1e-12; ok += 1
    assert band(0.1250, 0.0) == [0.1250, 0.1250]; ok += 1
    # 7b) h 越大带越宽（保守方向 = 取 max）
    a0, a1 = band(0.1250, 0.1), band(0.1250, 0.3)
    assert a1[0] < a0[0] and a1[1] > a0[1]; ok += 1

    # 8) 贴线判定：门槛在区间内 <=> NEAR_LINE（⛔ 与判决标签无关）
    assert near_line({"lo": 0.10, "hi": 0.15}, 0.1250) == "NEAR_LINE"; ok += 1
    assert near_line({"lo": 0.13, "hi": 0.20}, 0.1250) == "CLEAR_OF_LINE"; ok += 1
    assert near_line({"lo": 0.1250, "hi": 0.20}, 0.1250) == "NEAR_LINE"; ok += 1   # 闭区间
    assert near_line({"lo": 0.05, "hi": 0.1250}, 0.1250) == "NEAR_LINE"; ok += 1

    # 9) ⚠ 反向断言：本文件不许碰 (M84) 的处理臂读数
    #    ⚠ token 必须拼出来，否则断言自己就是一条匹配（selftest 第二次改断言）。
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    tok = "m84_" + "w384b"
    assert src.count(tok) == 1 and (tok + "_s*.json") in src; ok += 1   # 仅 docstring 那一处
    assert "m84_read_retrain" in src; ok += 1      # 但门槛必须是 import 来的

    # 10) CAL 的形状：三条臂、n_mat 期望 67、K 与已发表登记一致
    assert len(CAL) == 3 and [c[0] for c in CAL] == ["M60", "M62", "M75"]; ok += 1
    assert [c[4] for c in CAL] == [17, 28, 28]; ok += 1
    assert N_MAT_EXPECT == 67; ok += 1

    # 11) 聚合规则＝max（写死；⚠ 若写成 min 会把带调窄＝不保守）
    hs = [0.1, 0.25, 0.17]
    assert aggregate_h(hs) == 0.25; ok += 1
    assert band(THR_MDE, aggregate_h(hs))[1] > band(THR_MDE, sorted(hs)[0])[1]; ok += 1

    print("m89 selftest OK (%d checks)" % ok)


def near_line(ci, thr=THR_MDE):
    """⚠ 只登记：门槛落在 MDE 的 95% 区间内（闭区间）⇒ NEAR_LINE。"""
    return "NEAR_LINE" if ci["lo"] <= thr <= ci["hi"] else "CLEAR_OF_LINE"


def aggregate_h(hs):
    """聚合规则冻结＝max（保守：带越宽越不容易高估判决）。"""
    return max(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    res = {"note": "registration only; no verdict, no criterion changed",
           "threshold_M84": THR_MDE, "b_boot": B_BOOT, "boot_seed": BOOT_SEED,
           "arms": [], "ops": {}}
    hs, op1, op3, op4, op5 = [], [], [], [], []

    for name, d, ctag, ttag, k, published, tol in CAL:
        dv, prob = load_d(a.root, d, ctag, ttag, k)
        if dv is None:
            res["arms"].append({"round": name, "ok": False, "problems": prob})
            continue
        ci = mde_ci(dv)
        alt = mde_ci(dv, seed=BOOT_SEED_ALT)
        tiled = mde_ci(dv * 4, b=4000)
        row = {"round": name, "ok": True, "k": k, "n_mat": ci["n"],
               "rms_D": rms(dv), "mde_point": ci["point"],
               "mde_published": published,
               "mde_ci95": [ci["lo"], ci["hi"]],
               "rel_halfwidth": ci["rel_halfwidth"],
               "near_line_vs_0.1250": near_line(ci),
               "mde_over_threshold": ci["point"] / THR_MDE}
        res["arms"].append(row)
        hs.append(ci["rel_halfwidth"])
        op1.append([name, abs(ci["point"] - published) < tol,
                    abs(ci["point"] - published)])
        op3.append([name,
                    abs(alt["lo"] - ci["lo"]) / ci["point"] < 0.01 and
                    abs(alt["hi"] - ci["hi"]) / ci["point"] < 0.01,
                    max(abs(alt["lo"] - ci["lo"]), abs(alt["hi"] - ci["hi"])) / ci["point"]])
        op4.append([name, ci["n"] == N_MAT_EXPECT, ci["n"]])
        sc = ci["rel_halfwidth"] / tiled["rel_halfwidth"] if tiled["rel_halfwidth"] else 0.0
        op5.append([name, 1.7 < sc < 2.3, sc])

    # (OP2) 阴性对照
    neg = mde_ci([0.3] * N_MAT_EXPECT, b=2000)
    res["ops"] = {
        "OP1_reproduce_published": op1,
        "OP2_constant_vector_zero_width": [neg["rel_halfwidth"] == 0.0, neg["rel_halfwidth"]],
        "OP3_seed_stability_lt_1pct": op3,
        "OP4_n_mat_is_67": op4,
        "OP5_tile4x_halves_width": op5,
    }
    res["ops_all_pass"] = (all(x[1] for x in op1) and res["ops"]["OP2_constant_vector_zero_width"][0]
                           and all(x[1] for x in op3) and all(x[1] for x in op4)
                           and all(x[1] for x in op5))

    if hs:
        h = aggregate_h(hs)
        res["h_rel_halfwidth_max"] = h
        res["h_rel_halfwidth_median"] = sorted(hs)[len(hs) // 2]
        res["near_line_band"] = band(THR_MDE, h)
    # (B) 组：门槛那一侧，只登记、⛔ 不进带
    res["threshold_side_registered_only"] = {
        "source": "experiments/m62_mde_audit.json M60 ci95",
        "mean_D": 0.12499542587990321,
        "ci95": [0.025960171496630324, 0.2240306802631761],
        "rel_halfwidth": (0.2240306802631761 - 0.025960171496630324) / 2 / 0.12499542587990321,
        "why_excluded": "0.1250 在 (M84) 里是冻结的目标定义、不是本轮在估计的量；"
                        "把它算进带会让几乎任何落点都判贴线 = 判据失去内容。"
                        "这是选择、不是它不成立。",
    }

    # ⚠ 落盘一律在打印之前（(M33) 纪律：非 GBK 字形会崩在写 JSON 之前）
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    for r in res["arms"]:
        if not r.get("ok"):
            print("%s: NOT OK %s" % (r["round"], r["problems"]))
            continue
        print("== %s  K=%d  n_mat=%d  rms(D)=%.4f" % (r["round"], r["k"], r["n_mat"], r["rms_D"]))
        print("   MDE=%.4f (published %.4f)  95%%CI=[%.4f, %.4f]  rel_halfwidth=%.1f%%"
              % (r["mde_point"], r["mde_published"], r["mde_ci95"][0], r["mde_ci95"][1],
                 100 * r["rel_halfwidth"]))
        print("   vs 0.1250: x%.2f  -> %s" % (r["mde_over_threshold"], r["near_line_vs_0.1250"]))
    if "h_rel_halfwidth_max" in res:
        print("h(max over arms) = %.1f%%   -> near-line band = [%.4f, %.4f]"
              % (100 * res["h_rel_halfwidth_max"],
                 res["near_line_band"][0], res["near_line_band"][1]))
    print("ops_all_pass = %s" % res["ops_all_pass"])
    for kk, vv in res["ops"].items():
        print("  %-28s %s" % (kk, vv))
    print("threshold-side (B, registered only): rel_halfwidth = %.1f%%"
          % (100 * res["threshold_side_registered_only"]["rel_halfwidth"]))
    print("  【禁】本轮不下判决、不改 (M84) 判据/门槛/分叉表; "
          "NEAR_LINE 只改判决文本要不要带一句, 不改标签")
    if a.out:
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
