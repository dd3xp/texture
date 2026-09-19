#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P8) **判决之后**才跑的诊断：已发表的 `tile_seam_ratio` 一列为什么和主判据方向相反？

⚠ 本文件写在 `experiments/p8_bias.json` 落盘之后，**不是判据、不改判决**（(M62) 那条
「判决落盘之后才跑的诊断」的做法）。⛔ 不许拿它重读 (P8) 的 `BIAS_NULL`。

要解释的矛盾（同一批 PNG、同一个活件 `tile_seam_ratio`）：
  - `run_eval` 那一列（`tileability` ＝ **250 张 ratio 的中位数**）：C 1.0016 最靠近 1，A1 0.9938；
  - (P8) 主判据（**逐材质 |ratio−1| 的中位数**）：C 0.2261，A1 0.1313 ＝ A1 明显更紧。
两个方向相反。假设：`tileability` 取的是**带符号**量的中位数 ⇒ 两侧偏离互相抵消
⇒ 它可以在「典型瓦片偏离 ±0.23」时照样读出 1.00。

本脚本在同一批瓦片上同时算两种统计量并报符号分布。零 GPU、零 API、零活件改动
（`tile_seam_ratio` / `load_method` 仍从 AST 借；⚑ 直接 import 冻结的 `p8_read_bias`）。

(OP-D1) 复现检验：用 `evaluate()` 的**同一条**口径必须逐位复现 `p8_eval_Vmat_16.json` 的
        `tile_seam_ratio` 一列。不逐位相同 ⇒ 说明我复现的不是那一列，整个诊断作废。
        ⚑ 这条检验第一次跑就**响了**：我原先拿 250 张全算，而 `run_eval.py:109` 不带
        `--all_samples` 时 `tiles = [first[i] ...]` ＝**每材质只取第一张**（JSON 里 `n=125`
        早已写明）⇒ 已发表那一列是 **125 张**的中位数。修的是我的复现口径、⛔ 不是检验本身。
⚠ 判读器走的是**灰度**（`.mean(axis=2)`）路径，`evaluate()` 走 RGB ⇒ 两列差异不止符号一项，
  故本脚本把灰度/RGB 两种都算出来，⛔ 不许把全部差异都归给符号抵消。

用法：
    python analysis/arch/p8_seam_twoside.py --dir remote_tmp/p8 --out experiments/p8_seam_twoside.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p8_read_bias as R                                            # noqa: E402  冻结的判读器


def stats(ratios):
    r = [x for x in ratios if math.isfinite(x)]
    if not r:
        return {"n": 0}
    return {
        "n": len(r),
        "median_signed": R.median(r),                 # ＝ 活件 tileability 的口径
        "median_abs_dev": R.median([abs(x - 1.0) for x in r]),
        "frac_above_1": sum(1 for x in r if x > 1.0) / len(r),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp/p8")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from pathlib import Path
    ns = R.activepieces()
    base = a.dir if os.path.isabs(a.dir) else os.path.join(R.ROOT, a.dir)

    ev = R.load_json(os.path.join(base, "p8_eval_Vmat_16.json"))
    if ev is None:
        raise SystemExit("缺 p8_eval_Vmat_16.json，拒绝出诊断（⛔ 不许拿空集冒充量过）")

    slugs = sorted(set.intersection(*[R.slugs_of(R.arm_dir(base, m)) for m in R.ARMS]))
    rep = {"dir": a.dir, "n_materials": len(slugs), "arms": {}}
    for arm in R.ARMS:
        first, groups = ns["load_method"](Path(R.arm_dir(base, arm)), slugs)
        # 已发表那一列的口径：每材质只取第一张、RGB 直接进活件（run_eval.py:109 + metrics.py:251）
        pub = [ns["tile_seam_ratio"](np.asarray(t, np.float64)) for t in first if t is not None]
        allr, gray = [], []
        for imgs in groups:
            for im in imgs:
                t = np.asarray(im, np.float64)
                allr.append(ns["tile_seam_ratio"](t))
                gray.append(ns["tile_seam_ratio"](t.mean(axis=2)))
        published = ev.get("p8%sx" % arm, {}).get("tile_seam_ratio")
        s_pub = stats(pub)
        rep["arms"][arm] = {
            "published_caliber_first_tile_rgb": s_pub,   # ＝已发表那一列的口径
            "all_tiles_rgb": stats(allr),
            "all_tiles_gray": stats(gray),               # ＝判读器主判据用的灰度路径
            "published_tile_seam_ratio": published,
            "OP_D1_reproduces_published": (
                published is not None and s_pub.get("n")
                and s_pub["median_signed"] == published),
        }
    rep["OP_D1_all_pass"] = all(v["OP_D1_reproduces_published"] for v in rep["arms"].values())

    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(R.ROOT, a.out)
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
    print(txt)
    return 0 if rep["OP_D1_all_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
