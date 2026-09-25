#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(h16) 人工盲比判读：判据与作废条件**逐字抄自** `analysis/annotate/build_study_h32.py` 的头部
（16px 版由 `build_study_h16.py` 继承，只换臂名与尺寸），⛔ 一个字不许放宽。

主判据：在**两序一致（decided）**的对里，`TRD16c_rr4` 的胜率对 0.5 做二项检验。
  >50% 且 p<0.05 -> `HUMAN16_TRD_WINS`；<50% 且 p<0.05 -> `HUMAN16_B2_LEADS`；其余 -> `HUMAN16_TIE`
  （⛔ `TIE` 不许读成"打平"，只能读成"在这个样本量上没测到"）。
作废条件（任一触发 ⇒ **该份整份作废**，⛔ 不许只删出错的那几题）：
  (V1) 注意力检查错 >1 个（`check` 应选清晰那张；`check_same` 应按"分不出"）。
  (V2) decided 对数 < 60（本轮为分工设计，按**合并后**的 decided 计，见下）。
  (V3) `check_same` 上按"分不出"的比例 < 50%。
本轮口径补注（**分工版**，预注册见 `analysis/annotate/split_study_h16.py`，切页早于任何数据）：
  五份各自是 26 个材质，(V1)(V3) 逐份判；(V2) 的 60 对门槛按**合并后**判，
  一致性只在共同块的 10 个材质上算。

    python analysis/annotate/read_h16_study.py --csv <目录或文件...>
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
from exact import binom_test, jeffreys        # noqa: E402

ARM_A, ARM_B = "TRD16c_rr4", "B2"


def load(paths):
    out = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            out[Path(p).stem] = list(csv.DictReader(f))
    return out


def judge_one(rows):
    """逐份：注意力检查、同图题、两序一致的真题。"""
    chk = [r for r in rows if r["kind"] == "check"]
    same = [r for r in rows if r["kind"] == "check_same"]
    # check：blur 在哪边就该选另一边；CSV 的 chosen 记的是被选中那侧的臂名
    chk_wrong = sum(1 for r in chk if r["chosen"] != "good")
    same_tie = sum(1 for r in same if r["choice"] == "tie")
    real = [r for r in rows if r["kind"] == "real"]
    by_pair = defaultdict(list)
    for r in real:
        by_pair[r["pair"]].append(r)
    decided, inconsistent, ties = {}, 0, 0
    for pr, rs in by_pair.items():
        if len(rs) != 2:
            continue
        ch = {r["chosen"] for r in rs}
        if "tie" in {r["choice"] for r in rs}:
            ties += 1
        elif len(ch) == 1:
            decided[pr] = (ch.pop(), rs[0]["material"])
        else:
            inconsistent += 1
    v1 = chk_wrong > 1
    v3 = (same_tie / len(same) < 0.5) if same else True
    return {"n_check": len(chk), "check_wrong": chk_wrong,
            "n_same": len(same), "same_tie": same_tie,
            "n_pairs": len(by_pair), "decided": len(decided),
            "inconsistent": inconsistent, "ties": ties,
            "V1_failed_checks": v1, "V3_side_picking_on_identical": v3,
            "voided": bool(v1 or v3), "decided_map": decided}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/annotate/h16_verdict.json")
    a = ap.parse_args()
    paths = []
    for c in a.csv:
        p = Path(c)
        paths += sorted(p.glob("*.csv")) if p.is_dir() else [p]
    sets = load(paths)

    per, alive = {}, {}
    for name, rows in sorted(sets.items()):
        r = judge_one(rows)
        per[name] = {k: v for k, v in r.items() if k != "decided_map"}
        if not r["voided"]:
            alive[name] = r["decided_map"]

    pooled = {}
    for name, dm in alive.items():
        for pr, (chosen, mat) in dm.items():
            pooled[(name, pr)] = chosen
    wins = sum(1 for v in pooled.values() if v == ARM_A)
    tot = len(pooled)
    p = binom_test(wins, tot) if tot else float("nan")
    lo, hi = jeffreys(wins, tot) if tot else (float("nan"), float("nan"))
    v2 = tot < 60
    verdict = ("VOID_ALL_SETS" if not alive else
               "VOID_V2_UNDERPOWERED" if v2 else
               "HUMAN16_TRD_WINS" if wins / tot > 0.5 and p < 0.05 else
               "HUMAN16_B2_LEADS" if wins / tot < 0.5 and p < 0.05 else "HUMAN16_TIE")

    out = {"arm_a": ARM_A, "arm_b": ARM_B, "per_set": per,
           "sets_alive": sorted(alive), "pooled_decided": tot, "wins_a": wins,
           "rate": (wins / tot) if tot else None, "p": p, "jeffreys": [lo, hi],
           "V2_underpowered": v2, "verdict": verdict}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'份':<22}{'检查错':>7}{'同图按分不出':>14}{'对数':>6}{'一致':>6}{'不一致':>7}{'平局':>6}{'作废':>7}")
    for name, r in per.items():
        print(f"{name[:22]:<22}{r['check_wrong']:>4}/{r['n_check']:<3}{r['same_tie']:>6}/{r['n_same']:<7}"
              f"{r['n_pairs']:>6}{r['decided']:>6}{r['inconsistent']:>7}{r['ties']:>6}"
              f"{('是' if r['voided'] else '否'):>7}")
    print(f"\n未作废的份：{sorted(alive) or '无'}")
    if tot:
        print(f"合并 decided {tot} 对，{ARM_A} 胜 {wins} = {wins/tot:.1%}  p={p:.3g}  [{lo:.0%},{hi:.0%}]")
    print(f"判决：**{verdict}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
