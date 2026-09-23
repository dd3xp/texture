#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P8b) 判读器：判据逐字抄自已提交的预注册 `scripts/p8b_2x2.sh` 头部，⛔ 一个字不许放宽。

worse(X,Y) := 中位 |ratio−1| X > Y + 0.05 且 逐材质配对符号检验 p<0.05（X 更差占多数）。
  K1 = worse(N0, NR)；K2 = worse(N0, T0)；K3 = not worse(T0, TR)。
  K2∧K3 -> BY_CONSTRUCTION；K2∧¬K3 -> TORUS_PARTIAL；¬K2∧K1 -> AUG_CARRIES；¬K1∧¬K2 -> RULER_BLIND。
质量读数（不参与判决）：16px V_mat KID 对地板 4.76。
操作检验：四臂 config.json 只允许 bias_wrap / no_roll / out 不同；四臂都有 last.pt；每臂 125 材质齐。

    python analysis/arch/p8b_read.py --stamp 09240430 --eval /tmp/p8b_eval_Vmat_16.json --out /tmp/p8b_read.json
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
sys.path.insert(0, str(ROOT / "analysis/arch"))
sys.path.insert(0, str(ROOT / "eval"))
from exact import binom_test                      # noqa: E402
from p8_read import dev                           # noqa: E402  (P8) 同一把尺子
from prompts import load_set                      # noqa: E402

ARMS = {"TR": "p8bTRx", "T0": "p8bT0x", "NR": "p8bNRx", "N0": "p8bN0x"}
DELTA, KID_FLOOR = 0.05, 4.76
ALLOWED_DIFF = {"bias_wrap", "no_roll", "out"}


def worse(per, med, x, y):
    common = [s for s in per[x] if s in per[y] and math.isfinite(per[x][s]) and math.isfinite(per[y][s])]
    n_worse = int(sum(per[x][s] > per[y][s] for s in common))
    p = binom_test(n_worse, len(common)) if common else float("nan")
    gap_ok = med[x] > med[y] + DELTA
    sign_ok = p < 0.05 and n_worse > len(common) / 2
    return {"x": x, "y": y, "gap": med[x] - med[y], "gap_ok": bool(gap_ok), "worse": n_worse,
            "n": len(common), "p_sign": p, "sign_ok": bool(sign_ok), "holds": bool(gap_ok and sign_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="09240430")
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--runs", type=Path, default=Path("/tmp/runs"))
    ap.add_argument("--eval", type=Path, default=Path("/tmp/p8b_eval_Vmat_16.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/p8b_read.json"))
    a = ap.parse_args()
    slugs = [e["material"].rsplit(".", 1)[0] for e in load_set("V_mat")[0]]
    bad = []

    cfgs = {}
    for k in ARMS:
        p = a.runs / f"trd_p8b{k}_{a.stamp}" / "config.json"
        if not p.exists() or not (p.parent / "last.pt").exists():
            bad.append(f"OP2：{k} 缺 config.json 或 last.pt")
            continue
        cfgs[k] = json.loads(p.read_text(encoding="utf-8"))
    if "TR" in cfgs:
        for k, c in cfgs.items():
            diff = {x for x in set(c) | set(cfgs["TR"]) if c.get(x) != cfgs["TR"].get(x)}
            if diff - ALLOWED_DIFF:
                bad.append(f"OP1：{k} 与 TR 的配方差异超出允许集 {sorted(diff - ALLOWED_DIFF)}")
        for k in cfgs:
            if cfgs[k].get("init_from"):
                bad.append(f"OP1：{k} 不是从头训练（init_from={cfgs[k]['init_from']}）")

    per = {}
    for k, tag in ARMS.items():
        d = a.root / tag / "16"
        per[k] = {s: dev(np.asarray(Image.open(d / f"{s}_0.png").convert("RGB")))
                  for s in slugs if (d / f"{s}_0.png").exists()}
        if len(per[k]) < 125:
            bad.append(f"OP3：{k} 只有 {len(per[k])}/125 个材质")
    med = {k: float(np.nanmedian(list(v.values()))) for k, v in per.items()}

    k1, k2, t0_vs_tr = worse(per, med, "N0", "NR"), worse(per, med, "N0", "T0"), worse(per, med, "T0", "TR")
    K1, K2, K3 = k1["holds"], k2["holds"], not t0_vs_tr["holds"]
    verdict = ("BY_CONSTRUCTION" if K2 and K3 else "TORUS_PARTIAL" if K2 else
               "AUG_CARRIES" if K1 else "RULER_BLIND")

    ev = json.loads(a.eval.read_text(encoding="utf-8"))
    kid = {k: ev[t]["KID_x1e3"] for k, t in ARMS.items() if t in ev}
    q = {k: {"kid": kid[k], "delta_vs_TR": kid[k] - kid["TR"],
             "above_floor": bool(abs(kid[k] - kid["TR"]) > KID_FLOOR)} for k in kid if k != "TR"}

    out = {"verdict": verdict, "K1": K1, "K2": K2, "K3": K3, "tests": [k1, k2, t0_vs_tr],
           "median_dev": med, "delta": DELTA, "kid_floor": KID_FLOOR, "kid": kid, "quality": q,
           "ops_failed": bad}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("中位 |ratio-1|：" + "  ".join(f"{k}={v:.4f}" for k, v in med.items()))
    for t in (k1, k2, t0_vs_tr):
        print(f"  worse({t['x']},{t['y']})：差 {t['gap']:+.4f}（过门={t['gap_ok']}）  "
              f"逐材质更差 {t['worse']}/{t['n']}  p={t['p_sign']:.3g}  成立={t['holds']}")
    print(f"K1={K1} K2={K2} K3={K3}  ⇒ 判决 **{verdict}**")
    print(f"KID（地板 {KID_FLOOR}）：TR={kid.get('TR', float('nan')):.2f}  " +
          "  ".join(f"{k}={v['kid']:.2f}（{v['delta_vs_TR']:+.2f}，超地板={v['above_floor']}）" for k, v in q.items()))
    if bad:
        print("⚠ 操作检验未过：" + "；".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
