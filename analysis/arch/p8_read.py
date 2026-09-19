#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P8) 判读器：判据逐字抄自已提交的预注册 `scripts/p8_bias_ablation.sh` 头部，⛔ 一个字不许放宽。

判据① 主判据（性质）：16px 产物的 |ratio−1|（`tile_seam_ratio`，理想 1，两侧都差）。
  `BIAS_IS_LOAD_BEARING` 需 **A1/A2/A3 三条全部** 满足：中位 |ratio−1| > C 的中位 + 0.05
  **且** 逐材质配对符号检验 p < 0.05；部分满足 → `BIAS_PARTIAL`（报出是哪几条）；一条都不满足 → `BIAS_NULL`。
判据② 质量守门（读数，不单独授权）：16px KID 差值对着 **max(重训漂移 1.087, 采样下限 4.76) = 4.76** 读；
  ⛔ 小于 4.76 一律记"什么也没测到"。
判据③ 操作检验：四臂 `config.json` 只允许 `bias_wrap`/`bias_off` 不同；四臂都有 `last.pt`；每臂 125 材质齐。

    python analysis/arch/p8_read.py --stamp 09190900 --eval /tmp/p8_eval_Vmat_16.json --out /tmp/p8_read.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "eval"))
from exact import binom_test                      # noqa: E402
from prompts import load_set                      # noqa: E402

ARMS = {"C": "p8Cx", "A1": "p8A1x", "A2": "p8A2x", "A3": "p8A3x"}
DELTA, KID_FLOOR = 0.05, 4.76      # 判据①的 0.05 与判据②的地板，跑前写死
ALLOWED_DIFF = {"bias_wrap", "bias_off", "out"}


def dev(t):
    a = np.asarray(t, np.float64)
    seam = (np.abs(a[:, -1] - a[:, 0]).mean() + np.abs(a[-1, :] - a[0, :]).mean()) / 2
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    return abs(seam / inner - 1) if inner > 1e-6 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="09190900")
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")   # gen_trd 不给 --out 时就写这里（V_mat 也一样）
    ap.add_argument("--runs", type=Path, default=Path("/tmp/runs"))
    ap.add_argument("--eval", type=Path, default=Path("/tmp/p8_eval_Vmat_16.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/p8_read.json"))
    a = ap.parse_args()
    slugs = [e["material"].rsplit(".", 1)[0] for e in load_set("V_mat")[0]]
    bad = []

    # ---- 判据③ 操作检验 ----
    cfgs = {}
    for k in ARMS:
        p = a.runs / f"trd_p8{k}_{a.stamp}" / "config.json"
        if not p.exists() or not (p.parent / "last.pt").exists():
            bad.append(f"OP2：{k} 缺 config.json 或 last.pt")
            continue
        cfgs[k] = json.loads(p.read_text(encoding="utf-8"))
    if "C" in cfgs:
        for k, c in cfgs.items():
            if k == "C":
                continue
            diff = {x for x in set(c) | set(cfgs["C"]) if c.get(x) != cfgs["C"].get(x)}
            if diff - ALLOWED_DIFF:
                bad.append(f"OP1：{k} 与 C 的配方差异超出允许集 {sorted(diff - ALLOWED_DIFF)}")

    # ---- 逐材质 |ratio−1| ----
    per = {}
    for k, tag in ARMS.items():
        d = a.root / tag / "16"
        vals = {}
        for s in slugs:
            p = d / f"{s}_0.png"
            if p.exists():
                vals[s] = dev(np.asarray(Image.open(p).convert("RGB")))
        per[k] = vals
        if len(vals) < 125:
            bad.append(f"OP3：{k} 只有 {len(vals)}/125 个材质")

    med = {k: float(np.nanmedian(list(v.values()))) for k, v in per.items()}
    rows, passed = {}, []
    for k in ("A1", "A2", "A3"):
        common = [s for s in per[k] if s in per["C"]
                  and math.isfinite(per[k][s]) and math.isfinite(per["C"][s])]
        worse = sum(per[k][s] > per["C"][s] for s in common)
        p_sign = binom_test(worse, len(common)) if common else float("nan")
        ok_gap = med[k] > med["C"] + DELTA
        ok_sign = p_sign < 0.05 and worse > len(common) / 2
        rows[k] = {"median_dev": med[k], "gap_vs_C": med[k] - med["C"], "gap_ok": bool(ok_gap),
                   "worse_than_C": worse, "n_paired": len(common), "p_sign": p_sign,
                   "sign_ok": bool(ok_sign), "meets_criterion": bool(ok_gap and ok_sign)}
        if ok_gap and ok_sign:
            passed.append(k)
    verdict = ("BIAS_IS_LOAD_BEARING" if len(passed) == 3 else
               "BIAS_PARTIAL" if passed else "BIAS_NULL")

    # ---- 判据② 质量读数 ----
    ev = json.loads(a.eval.read_text(encoding="utf-8"))
    kid = {k: ev[t]["KID_x1e3"] for k, t in ARMS.items() if t in ev}
    q = {k: {"kid": kid[k], "delta_vs_C": kid[k] - kid["C"],
             "above_floor": bool(abs(kid[k] - kid["C"]) > KID_FLOOR)}
         for k in kid if k != "C"}

    out = {"verdict_criterion1": verdict, "passed_arms": passed, "delta": DELTA,
           "median_dev": med, "rows": rows, "kid_floor": KID_FLOOR, "kid": kid,
           "quality": q, "ops_failed": bad}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"C 的中位 |ratio-1| = {med['C']:.4f}（判据①门槛 = C + {DELTA}）")
    for k, r in rows.items():
        print(f"  {k}: 中位 {r['median_dev']:.4f}（差 {r['gap_vs_C']:+.4f}，过门={r['gap_ok']}）  "
              f"逐材质更差 {r['worse_than_C']}/{r['n_paired']}  符号检验 p={r['p_sign']:.3g}  "
              f"满足判据①={r['meets_criterion']}")
    print(f"\n判据① 判决：**{verdict}**（满足的臂：{passed or '无'}）")
    print(f"判据② KID（地板 {KID_FLOOR}）：C={kid.get('C', float('nan')):.2f}  " +
          "  ".join(f"{k}={v['kid']:.2f}（{v['delta_vs_C']:+.2f}，超地板={v['above_floor']}）"
                    for k, v in q.items()))
    if bad:
        print("⚠ 操作检验未过：" + "；".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
