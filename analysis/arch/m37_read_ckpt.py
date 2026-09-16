#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M37) 判读器：按 `eval/ckpt16_ab.sh` 预注册判据 (1)(2)(4) 逐字映射。

判据不是本文件定的，全文在 `eval/ckpt16_ab.sh` 头部（`a704b31`，写在第一张图生成之前）：
  (1) 可解率对地板 21/118 检验；不过 -> VOID_UNRESOLVABLE
  (2) 主判据 = 判官 A=ck3k_rr4 的**族级 (U1) CI**（裸二项 p 只当下界）
  (4) 下界>0.5 -> CKPT_EARLY_WINS；上界<0.5 -> CKPT_LATE_WINS；含 0.5 -> CKPT_NULL

族级统计量**不重新实现**：直接 import (M17)/(M18)/(M19) 冻结的 `judge_cluster.analyse`
（同一条代码路径、同一个种子）。⛔ 本文件不碰 `judge_cluster.py`、不重跑
`judge_cluster_sweep.py`（那是 (M19) 对 28 条臂的一次性预注册产物，加一条臂会动它的 (B)(C)）。

⚠ 打印里只用 ASCII 与常见汉字（Windows 控制台 GBK）；落盘一律在打印之前。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_cluster import analyse                      # noqa: E402

ARM = "judge_full_ck3k_rr4_vs_ck20k_rr4_16_V_mat"
OUT = Path(__file__).resolve().parents[2] / "experiments" / "m37_ckpt_cluster.json"
FLOOR = 21 / 118


def main():
    out, decided, recs = analyse("M37_ckpt16", ARM, "(M37) v8 best.pt(step3000) vs last.pt(step20000)")
    raw = json.loads((OUT.parent / (ARM + ".json")).read_text(encoding="utf-8"))

    # 判据 (1)：可解率对地板
    rr, pf = raw.get("resolve_rate"), raw.get("p_vs_floor")
    resolvable = rr is not None and pf is not None and pf < 0.05 and rr > FLOOR

    g = out["grains"]["U1_drop_last"]
    lo, hi = g["ci"]
    # 判据 (4)：四种读法逐字映射
    if not resolvable:
        verdict = "VOID_UNRESOLVABLE"
    elif out["verdict"].startswith("VOID"):
        verdict = "VOID_" + out["verdict"]
    elif lo > 0.5:
        verdict = "CKPT_EARLY_WINS"
    elif hi < 0.5:
        verdict = "CKPT_LATE_WINS"
    else:
        verdict = "CKPT_NULL"

    # 操作检验：另报"已查几项"，不拿空集冒充没事（(M31) 那条坑）
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
    print("裸二项（只当下界）: A(ck3k=step3000) 胜 %d/%d = %.1f%%  p=%.4g  Jeffreys [%.3f,%.3f]"
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


if __name__ == "__main__":
    main()
