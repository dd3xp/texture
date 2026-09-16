#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M34) 执行 (M31) 盲写的那条规则: 用同配置重训漂移 Dmax 结算账本 3(a) 那道硬门。

3(a) 原文: 下一个架构候选必须先解释 "(M29) 里 16px 为何被带着动" (+2.118)。
规则在 docs/arch_progress.md:6182 一节, 写于三条漂移臂的 16px KID 均不存在时(盲写):

    Dmax >= 2.118          -> CLEARED      结清
    1.0 <= Dmax <  2.118   -> PARTIAL      部分结清
    Dmax <  1.0            -> NOT_CLEARED  仍未结清, 下一个候选还得回答它

本脚本只做算术, 零 GPU/零 API。判决标签与阈值一个字不改。
落盘在打印之前 (打印崩了 JSON 也保得住); 打印只用 GBK 安全字形。
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (M31) 盲写的三档阈值, 一个字不改
GATE_CLEARED = 2.118      # >= 此值 -> 结清
GATE_PARTIAL = 1.0        # >= 此值 -> 部分结清

ARMS = ("seed0", "seed1", "seed2")
CTRL_KEY = "v11dx"        # (M29) 的控制臂
TREAT_KEY = "scondx"      # (M29) 的处理臂 (bias_size_cond)


def verdict_of(dmax):
    if dmax >= GATE_CLEARED:
        return "CLEARED"
    if dmax >= GATE_PARTIAL:
        return "PARTIAL"
    return "NOT_CLEARED"


def t_pvalue_df2(t):
    """df=2 的 t 分布双侧 p 值, 闭式解 (本机无 scipy 也能算)。

    F(t) = 1/2 + t / (2*sqrt(2+t^2))  =>  双侧 p = 1 - |t|/sqrt(2+t^2)
    """
    a = abs(t)
    return 1.0 - a / math.sqrt(2.0 + a * a)


T_CRIT_DF2_05 = 4.302652729911275  # 双侧 alpha=0.05, df=2


def analyze(drift, scond):
    """drift = m31_drift.json 的内容; scond = eval_scond_Vmat_16.json 的内容。"""
    out = {"gate_cleared": GATE_CLEARED, "gate_partial": GATE_PARTIAL, "ops": {}}
    ops = out["ops"]

    # ---- 操作检验 (每项另报"已查几项", 免得空集合冒充通过) ----
    # (OP1) 两个读数必须来自同一把尺子: 键名 KID_x1e3 在两个 JSON 上都真实存在
    missing = []
    for k in (CTRL_KEY, TREAT_KEY):
        if k not in scond:
            missing.append("scond:" + k)
        elif "KID_x1e3" not in scond[k]:
            missing.append("scond:%s.KID_x1e3" % k)
    if "kid16" not in drift:
        missing.append("drift:kid16")
    else:
        for a in ARMS:
            if a not in drift["kid16"]:
                missing.append("drift:kid16." + a)
    ops["op1"] = not missing
    ops["op1_missing"] = missing
    ops["op1_checked"] = 2 + 1 + len(ARMS)

    # (OP2) 分母相同: run_eval.py:100 的 ok 是每方法各算各的, 比较前必须查 n/materials
    ns = {}
    bad_n = []
    for k in (CTRL_KEY, TREAT_KEY):
        if k in scond:
            ns[k] = (scond[k].get("n"), scond[k].get("materials"))
    if len(set(ns.values())) != 1:
        bad_n.append(ns)
    ops["op2"] = not bad_n
    ops["op2_n"] = {k: list(v) for k, v in ns.items()}
    ops["op2_problems"] = bad_n
    ops["op2_checked"] = len(ns)

    # (OP3) 控制臂同一性: (M29) 的 v11dx 必须就是漂移标定里的 seed0, 否则不是同一把尺子
    ctrl = scond.get(CTRL_KEY, {}).get("KID_x1e3")
    seed0 = drift.get("kid16", {}).get("seed0")
    same = (ctrl is not None and seed0 is not None and abs(ctrl - seed0) < 1e-9)
    ops["op3"] = bool(same)
    ops["op3_ctrl"] = ctrl
    ops["op3_seed0"] = seed0
    ops["op3_absdiff"] = None if (ctrl is None or seed0 is None) else abs(ctrl - seed0)
    ops["op3_checked"] = 1

    out["ops_all_pass"] = bool(ops["op1"] and ops["op2"] and ops["op3"])
    if not out["ops_all_pass"]:
        out["verdict"] = "OPS_FAILED"
        return out

    # ---- 主判据: 照 (M31) 盲写的规则查表 ----
    treat = scond[TREAT_KEY]["KID_x1e3"]
    delta = treat - ctrl
    dmax = drift["Dmax"]
    out["m29_delta_kid16"] = delta
    out["m29_ctrl"] = ctrl
    out["m29_treat"] = treat
    out["drift_Dmax"] = dmax
    out["drift_pairwise"] = drift.get("pairwise_abs_diff_kid16")
    out["verdict"] = verdict_of(dmax)

    # ---- 报告项 (事后选择的统计量, 不是预注册判据, 只用来给下一轮算成本) ----
    vals = [drift["kid16"][a] for a in ARMS]
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    se = sd * math.sqrt(1.0 + 1.0 / n)          # 单个新观测 vs n 个臂的组
    tstat = delta / se if se > 0 else float("inf")
    out["report_only"] = {
        "note": "post-hoc estimator, NOT pre-registered; df=2 => very low power",
        "arm_values": dict(zip(ARMS, vals)),
        "arm_mean": mean,
        "arm_sd": sd,
        "se_new_obs": se,
        "t": tstat,
        "df": n - 1,
        "p_two_sided": t_pvalue_df2(tstat),
        "min_detectable_delta_at_n3": T_CRIT_DF2_05 * se,
        "detected_at_alpha_05": bool(abs(tstat) > T_CRIT_DF2_05),
    }
    return out


def selftest():
    """跑前自测: 三档规则各一例 + 操作检验各一例失败。"""
    def mk(ctrl, treat, kid16, dmax):
        drift = {"kid16": kid16, "Dmax": dmax,
                 "pairwise_abs_diff_kid16": {}}
        scond = {CTRL_KEY: {"KID_x1e3": ctrl, "n": 125, "materials": 125},
                 TREAT_KEY: {"KID_x1e3": treat, "n": 125, "materials": 125}}
        return drift, scond

    base = {"seed0": 6.0, "seed1": 5.5, "seed2": 5.0}
    # 三档边界
    for dmax, want in ((2.118, "CLEARED"), (2.5, "CLEARED"),
                       (1.0, "PARTIAL"), (1.087, "PARTIAL"), (2.1179, "PARTIAL"),
                       (0.999, "NOT_CLEARED"), (0.0, "NOT_CLEARED")):
        d, s = mk(6.0, 8.118, base, dmax)
        r = analyze(d, s)
        assert r["verdict"] == want, (dmax, r["verdict"], want)
    # OP2: 分母不等必须判 OPS_FAILED
    d, s = mk(6.0, 8.118, base, 1.5)
    s[TREAT_KEY]["n"] = 120
    assert analyze(d, s)["verdict"] == "OPS_FAILED"
    # OP3: 控制臂不是 seed0 必须判 OPS_FAILED
    d, s = mk(6.5, 8.118, base, 1.5)
    assert analyze(d, s)["verdict"] == "OPS_FAILED"
    # OP1: 键名写错必须判 OPS_FAILED (KID vs KID_x1e3 那个老坑)
    d, s = mk(6.0, 8.118, base, 1.5)
    s[TREAT_KEY] = {"KID": 8.118, "n": 125, "materials": 125}
    assert analyze(d, s)["verdict"] == "OPS_FAILED"
    # 算术: delta 与 sd
    d, s = mk(6.0, 8.118, base, 1.087)
    r = analyze(d, s)
    assert abs(r["m29_delta_kid16"] - 2.118) < 1e-9
    assert abs(r["report_only"]["arm_sd"] - 0.5) < 1e-9
    # df=2 闭式 p 值对得上教科书临界值
    assert abs(t_pvalue_df2(T_CRIT_DF2_05) - 0.05) < 1e-9
    assert abs(t_pvalue_df2(0.0) - 1.0) < 1e-12
    print("selftest ok (rule 7 cases + ops 3 cases + arith 3 cases)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", default=os.path.join(ROOT, "experiments", "m31_drift.json"))
    ap.add_argument("--scond", default=os.path.join(ROOT, "experiments", "eval_scond_Vmat_16.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "m34_door3a.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    with open(a.drift, "r", encoding="utf-8") as f:
        drift = json.load(f)
    with open(a.scond, "r", encoding="utf-8") as f:
        scond = json.load(f)
    res = analyze(drift, scond)
    res["inputs"] = {"drift": os.path.basename(a.drift), "scond": os.path.basename(a.scond)}

    # 落盘一律在打印之前
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    o = res["ops"]
    print("(OP1) 键名在真 JSON 上验过: %s (已查 %d 项)" % (o["op1"], o["op1_checked"]))
    print("(OP2) 两臂分母相同: %s  %s" % (o["op2"], o["op2_n"]))
    print("(OP3) 控制臂 == 漂移 seed0: %s (|diff|=%s)" % (o["op3"], o["op3_absdiff"]))
    if res["verdict"] == "OPS_FAILED":
        print("判决: OPS_FAILED  -> 【禁】不下判决")
        return
    print("")
    print("(M29) 16px KID: %.6f -> %.6f, delta = %+.6f" %
          (res["m29_ctrl"], res["m29_treat"], res["m29_delta_kid16"]))
    print("同配置重训漂移 Dmax = %.6f  (阈: >=%.3f 结清 / >=%.1f 部分结清)" %
          (res["drift_Dmax"], GATE_CLEARED, GATE_PARTIAL))
    print("判决(照 (M31) 盲写规则查表): %s" % res["verdict"])
    r = res["report_only"]
    print("")
    print("[报告项, 事后统计量, 非判据] 三臂 sd=%.4f, t=%.3f, df=%d, p=%.4f" %
          (r["arm_sd"], r["t"], r["df"], r["p_two_sided"]))
    print("[报告项] n=3 只能测出 delta >= %.3f; 实测 %.3f -> alpha=0.05 下未测到差异"
          % (r["min_detectable_delta_at_n3"], res["m29_delta_kid16"]))
    print("")
    print("写入 %s" % a.out)


if __name__ == "__main__":
    main()
