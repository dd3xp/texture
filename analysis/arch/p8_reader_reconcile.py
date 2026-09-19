#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P8) 两个判读器的对账 —— 【诊断，不是判决】。

背景：仓库里现在有两份 (P8) 读数，判决同为 `BIAS_NULL`，但逐材质中位 |ratio-1| 对不上：

    experiments/p8_bias.json （analysis/arch/p8_read_bias.py）  C 0.22613  A1 0.13126  A2 0.18031  A3 0.26331
    experiments/p8_read.json （analysis/arch/p8_read.py）       C 0.22538  A1 0.09289  A2 0.14815  A3 0.21874

而 docs/paper_plan.md 第十二节与 docs/paper_draft.md 的摘要引的是**后一份**，
docs/arch_progress.md 的 (P8) 正文引的是**前一份**。本脚本只回答一个问题：
**差别从哪来、哪一种口径与已发表列同口径**，⛔ 不重读判决、不改任何判据、不产生新结论。

两个判读器在两条轴上不同（各自都有文档依据，谁也没写错）：
  (1) 通道：p8_read_bias 走**灰度** `t.mean(axis=2)`；p8_read 走 **RGB**（与 `evaluate()` 一致）。
      `analysis/arch/p8_seam_twoside.py:22` 已把这条差异写在案。
  (2) 样本：p8_read_bias 每材质取**两张的中位**；p8_read 只取 `<slug>_0.png`＝**第一张**
      （`run_eval.py:109` 不带 --all_samples 时的口径，(P8) 判后诊断 OP-D1 查出来的那条）。

所以本脚本把 2x2 四格都算出来，并且用**已发表列**（experiments/p8_eval_Vmat_16.json 的
`tile_seam_ratio`，＝带符号比值的中位数）当阳性对照：只有与它逐位相同的那一格，才配叫
"与已发表列同口径"。

    python analysis/arch/p8_reader_reconcile.py --selftest
    python analysis/arch/p8_reader_reconcile.py --dir remote_tmp/p8 --out experiments/p8_reader_reconcile.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis", "arch"))

import p8_read_bias as R                      # noqa: E402  冻结判读器，只 import 不改

ARMS = ["C", "A1", "A2", "A3"]
CELLS = [("rgb", "first"), ("rgb", "med2"), ("gray", "first"), ("gray", "med2")]
MARGIN = R.MARGIN                              # 0.05，抄冻结件、不在这里重写
PUBLISHED = os.path.join(ROOT, "experiments", "p8_eval_Vmat_16.json")


def ratios(ns, imgs, channel):
    """一个材质的若干张图 -> 逐张带符号 ratio。"""
    out = []
    for im in imgs:
        t = np.asarray(im, np.float64)
        out.append(ns["tile_seam_ratio"](t.mean(axis=2) if channel == "gray" else t))
    return out


def collect(base, ns, slugs):
    """{arm: {cell: {slug: 带符号 ratio 列表}}}，一次读盘四格共用。"""
    per = {}
    for arm in ARMS:
        d = R.arm_dir(base, arm)
        _, groups = ns["load_method"](Path(d), slugs)
        per[arm] = {"rgb": {}, "gray": {}}
        for s, imgs in zip(slugs, groups):
            if not imgs:
                continue
            for ch in ("rgb", "gray"):
                per[arm][ch][s] = ratios(ns, imgs, ch)
    return per


def cell_dev(per_arm, channel, sample):
    """一格的 {材质: |ratio-1|}。sample=first 只用第一张，med2 取所有张的中位。"""
    out = {}
    for s, rs in per_arm[channel].items():
        use = rs[:1] if sample == "first" else rs
        devs = R.finite([abs(r - 1.0) for r in use])
        if devs:
            out[s] = R.median(devs)
    return out


def signed_median_first(per_arm, channel):
    """已发表列的口径：每材质第一张的带符号 ratio，跨材质取中位。"""
    vals = R.finite([rs[0] for rs in per_arm[channel].values() if rs])
    return R.median(vals)


def analyse(base, out_path=None):
    ns = R.activepieces()
    slugs = sorted(R.slugs_of(R.arm_dir(base, "C")))
    per = collect(base, ns, slugs)

    pub = json.load(open(PUBLISHED, encoding="utf-8"))
    op1 = {}
    for arm in ARMS:
        tag = "p8%sx" % arm
        got = {ch: signed_median_first(per[arm], ch) for ch in ("rgb", "gray")}
        want = pub[tag]["tile_seam_ratio"]
        op1[arm] = {"published": want, "rgb_first": got["rgb"], "gray_first": got["gray"],
                    "rgb_matches": got["rgb"] == want, "gray_matches": got["gray"] == want}
    same_cal = all(v["rgb_matches"] for v in op1.values())

    grid = {}
    for ch, smp in CELLS:
        name = "%s_%s" % (ch, smp)
        dev = {arm: cell_dev(per[arm], ch, smp) for arm in ARMS}
        med = {arm: R.median(list(dev[arm].values())) for arm in ARMS}
        rows = {}
        for a in ("A1", "A2", "A3"):
            common = sorted(set(dev[a]) & set(dev["C"]))
            worse = sum(1 for s in common if dev[a][s] > dev["C"][s])
            p = R.binom_two_sided(worse, len(common))
            rows[a] = {"median_dev": med[a], "gap_vs_C": med[a] - med["C"],
                       "gap_over_margin": bool(med[a] > med["C"] + MARGIN),
                       "worse_than_C": worse, "n_paired": len(common), "p_sign": p,
                       "meets_criterion1": bool(med[a] > med["C"] + MARGIN and p < R.ALPHA
                                                and worse > len(common) / 2)}
        grid[name] = {"median_C": med["C"], "rows": rows,
                      "verdict_if_this_cell": "BIAS_NULL" if not any(
                          r["meets_criterion1"] for r in rows.values()) else "NOT_NULL"}

    res = {"dir": base, "n_materials": len(slugs),
           "op1_published_column": op1, "rgb_first_is_published_calibration": same_cal,
           "grid": grid,
           "verdict_same_in_all_cells": len({g["verdict_if_this_cell"] for g in grid.values()}) == 1,
           "note": "diagnostic only; the published (P8) verdict BIAS_NULL is not re-read here"}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def report(res):
    print("材质数 %d   已发表列可复现(RGB+第一张)=%s"
          % (res["n_materials"], res["rgb_first_is_published_calibration"]))
    for arm, v in res["op1_published_column"].items():
        print("  %-3s 已发表 %.10f | RGB %.10f %s | 灰度 %.10f %s"
              % (arm, v["published"], v["rgb_first"], "OK" if v["rgb_matches"] else "差",
                 v["gray_first"], "OK" if v["gray_matches"] else "差"))
    print("\n逐材质 |ratio-1| 中位（门槛 C+%.2f）：" % MARGIN)
    for name, g in res["grid"].items():
        print("  [%s] C=%.4f  " % (name, g["median_C"])
              + "  ".join("%s=%.4f(差%+.4f, 更差%d/%d, p=%.2e)"
                          % (a, r["median_dev"], r["gap_vs_C"], r["worse_than_C"],
                             r["n_paired"], r["p_sign"]) for a, r in g["rows"].items())
              + "  -> " + g["verdict_if_this_cell"])
    print("\n四格判决一致 = %s" % res["verdict_same_in_all_cells"])


def selftest():
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  %s %s" % ("PASS" if cond else "FAIL", name))

    ns = R.activepieces()
    chk("借到活件 tile_seam_ratio/load_method",
        callable(ns["tile_seam_ratio"]) and callable(ns["load_method"]))
    chk("门槛抄自冻结件", MARGIN == 0.05 and R.ALPHA == 0.05)

    # 构造一张 RGB 与灰度必然给出不同 ratio 的瓦片：两通道反号，灰度里抵消。
    t = np.zeros((4, 4, 3), np.float64)
    t[:, 0, 0], t[:, -1, 0] = 0.0, 40.0
    t[:, 0, 1], t[:, -1, 1] = 40.0, 0.0
    t[0, :, 2], t[-1, :, 2] = 10.0, 30.0
    r_rgb = ns["tile_seam_ratio"](t)
    r_gray = ns["tile_seam_ratio"](t.mean(axis=2))
    chk("RGB 与灰度确实不是同一个量 (%.4f vs %.4f)" % (r_rgb, r_gray),
        math.isfinite(r_rgb) and math.isfinite(r_gray) and abs(r_rgb - r_gray) > 1e-6)

    fake = {"rgb": {"m1": [1.30, 1.10], "m2": [0.70, 0.90]}, "gray": {"m1": [1.0], "m2": [1.0]}}
    d_first = cell_dev(fake, "rgb", "first")
    d_med = cell_dev(fake, "rgb", "med2")
    chk("first 只用第一张", abs(d_first["m1"] - 0.30) < 1e-12 and abs(d_first["m2"] - 0.30) < 1e-12)
    chk("med2 用两张的中位", abs(d_med["m1"] - 0.20) < 1e-12 and abs(d_med["m2"] - 0.20) < 1e-12)
    chk("两种取样真的给出不同的数", d_first["m1"] != d_med["m1"])
    chk("带符号中位只看第一张", abs(signed_median_first(fake, "rgb") - 1.0) < 1e-12)

    # 阴性对照：全部非有限时该格为空，不许冒充"量过没事"
    chk("非有限被丢弃", cell_dev({"rgb": {"m": [float("nan")]}}, "rgb", "first") == {})

    print("selftest %d/%d" % (ok[0], ok[1]))
    return ok[0] == ok[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "remote_tmp", "p8"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    report(analyse(a.dir, a.out))


if __name__ == "__main__":
    main()
