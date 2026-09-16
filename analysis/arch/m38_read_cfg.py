#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M38) 判读器：按 `eval/cfg16_ab.sh` 预注册判据 (1)(2)(4) 逐字映射。**盲写**（料未出）。

判据不是本文件定的，全文在 `eval/cfg16_ab.sh` 头部（`bb39c38`，写在第一张图生成之前）：
  (1) 可解率对地板 21/118 检验；不过 -> VOID_UNRESOLVABLE
  (2) 主判据 = 判官 A=cfg25_rr4 的**族级 (U1) CI**（裸二项 p 只当下界）
  (4) 下界>0.5 -> CFG_HIGH_WINS；上界<0.5 -> CFG_LOW_WINS；含 0.5 -> CFG_NULL

族级统计量**不重新实现**：直接 import (M17)/(M18)/(M19) 冻结的 `judge_cluster.analyse`。
⛔ 本文件不碰 `judge_cluster.py`、不重跑 `judge_cluster_sweep.py`（(M19) 的一次性预注册产物）。

⚠ 打印里只用 ASCII 与常见汉字（Windows 控制台 GBK），`⛔/⚠/⚑` 只许出现在注释里；落盘一律在打印之前。
用法：`python analysis/arch/m38_read_cfg.py --selftest` 先自测；料到了再不带参数跑。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARM = "judge_full_cfg25_rr4_vs_cfg15_rr4_16_V_mat"
OUT = Path(__file__).resolve().parents[2] / "experiments" / "m38_cfg_cluster.json"
FLOOR = 21 / 118


def decide(resolvable, cluster_verdict, lo, hi):
    """判据 (4) 的四种读法，逐字映射；事后不许再编第五种。"""
    if not resolvable:
        return "VOID_UNRESOLVABLE"
    if cluster_verdict.startswith("VOID"):
        return "VOID_" + cluster_verdict
    if lo > 0.5:
        return "CFG_HIGH_WINS"
    if hi < 0.5:
        return "CFG_LOW_WINS"
    return "CFG_NULL"


def selftest():
    cases = [
        ((False, "W1_robust", 0.60, 0.80), "VOID_UNRESOLVABLE"),
        ((True, "VOID_no_data", 0.60, 0.80), "VOID_VOID_no_data"),
        ((True, "W1_robust", 0.55, 0.70), "CFG_HIGH_WINS"),
        ((True, "W1_robust", 0.22, 0.44), "CFG_LOW_WINS"),
        ((True, "W2a", 0.45, 0.62), "CFG_NULL"),
        ((True, "W1_robust", 0.50, 0.70), "CFG_NULL"),   # 下界恰等于 0.5 = 含 0.5，不是胜
        ((True, "W1_robust", 0.30, 0.50), "CFG_NULL"),   # 上界恰等于 0.5 同理
    ]
    bad = []
    for args, want in cases:
        got = decide(*args)
        if got != want:
            bad.append((args, want, got))
    print("[selftest] 已查 %d 例，problems=%d %s" % (len(cases), len(bad), bad))
    # 料不在时必须报"没查到"，不能拿空集冒充没事（(M31) 那条坑）
    raw_p = OUT.parent / (ARM + ".json")
    print("[selftest] 判官 JSON %s: %s" % (raw_p, "在" if raw_p.exists() else "【禁】还没到，不能判"))
    return 1 if bad else 0


def main():
    from judge_cluster import analyse

    raw_p = OUT.parent / (ARM + ".json")
    if not raw_p.exists():
        print("【禁】判官 JSON 不存在: %s -> 已查 0 项，什么也没测到" % raw_p)
        return 2
    out, decided, recs = analyse("M38_cfg16", ARM,
                                 "(M38) v8 last.pt CFG 2.5 vs CFG 1.5（TRD16c 配方）")
    raw = json.loads(raw_p.read_text(encoding="utf-8"))

    # 判据 (1)：可解率对地板
    rr, pf = raw.get("resolve_rate"), raw.get("p_vs_floor")
    resolvable = rr is not None and pf is not None and pf < 0.05 and rr > FLOOR

    g = out["grains"]["U1_drop_last"]
    lo, hi = g["ci"]
    verdict = decide(resolvable, out["verdict"], lo, hi)

    checks = {
        "OP1_recount_exact": out["OP1_recount_exact"],
        "OP2_degenerate_pass": out["OP2_degenerate"]["pass"],
        "OP3_pass": g["OP3_pass"],
        "OP4_boot_pass": g["OP4_pass"],
        "api_fail_zero": raw["api_fail"] == 0,
        "lofo_valid": g["lofo"]["valid"],
    }
    problems = [k for k, v in checks.items() if not v]

    res = {"arm": ARM, "verdict": verdict, "resolvable": resolvable,
           "resolve_rate": rr, "p_vs_floor": pf, "floor": FLOOR,
           "binomial": {"a_wins": raw["a_wins"], "decided": raw["decided"],
                        "rate": raw["rate"], "p": raw["p"], "jeffreys": raw["jeffreys"],
                        "inconsistent": raw["inconsistent"], "api_fail": raw["api_fail"]},
           "family_U1": {"ci": [lo, hi], "n_families": g["n_families"],
                         "eff_families": g["eff_families"], "max_family": g["max_family"],
                         "width": g["width"], "widening": g["widening"],
                         "contains_half": g["contains_half"],
                         "lofo_n_folds": g["lofo"]["n_folds"],
                         "lofo_all_same_side": g["lofo"]["all_same_side"],
                         "lofo_flipped": len(g["lofo"]["flipped"])},
           "family_U2_report_only": {"ci": out["grains"]["U2_first"]["ci"],
                                     "contains_half": out["grains"]["U2_first"]["contains_half"]},
           "cluster_label": out["verdict"],
           "n_checked": len(checks), "problems": problems}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("-> %s" % OUT)
    print("可解率 %.1f%% vs 地板 %.1f%% p=%.3g -> %s"
          % (rr * 100, FLOOR * 100, pf, "有分辨力" if resolvable else "【禁】不可解读"))
    print("裸二项（只当下界）: A(cfg2.5) 胜 %d/%d = %.1f%%  p=%.4g  Jeffreys [%.3f,%.3f]"
          % (raw["a_wins"], raw["decided"], raw["rate"] * 100, raw["p"],
             raw["jeffreys"][0], raw["jeffreys"][1]))
    print("族级 (U1) CI = [%.4f, %.4f]  族数 %d（有效 %.1f，最大族 %d）拓宽 %.2fx  含 0.5: %s"
          % (lo, hi, g["n_families"], g["eff_families"], g["max_family"],
             g["widening"], g["contains_half"]))
    print("LOFO %d fold，全同侧 %s，翻侧 %d  -> 聚类标签 %s"
          % (g["lofo"]["n_folds"], g["lofo"]["all_same_side"],
             len(g["lofo"]["flipped"]), out["verdict"]))
    print("操作检验：已查 %d 项，problems=%d %s" % (len(checks), len(problems), problems))
    print("判决 = %s" % verdict)
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
