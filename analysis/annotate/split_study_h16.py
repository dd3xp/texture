#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""把已验钥的 16px 人工盲比页面按**材质**切成 5 份（方案 A），每人约 56 题。

预注册：本文件先提交再切页，**切页发生在任何标注数据产生之前** ⇒ 改仪器合法；
⛔ 切完不再改分配（分配表随判读一起存档）。

## 为什么切 & 怎么切（判据写死）

预算固定：5 人 × 约 50 题 = 125 份"材质-评分"。三种花法里只有本方案还有功效
（预期效应 57–60%，90 个材质约剩 60–70 对可判 ⇒ 只测得动 ≥65%；材质砍到 62 或 41 就必然"没测到"）。

- **共同块 `SHARED`**：随机抽 **10** 个材质，**5 个人全标** → 用来算标注者间一致性（Fleiss κ）。
- **独有块**：其余 **80** 个材质随机均分 5 份，每人 16 个。
- 每人 = 26 材质 × 2 种呈现顺序 = **52 题** + 全部 **4 道检查题**（同图对 + 显然可辨对各取若干）。
- ⛔ **分配只用随机种子，与任何产物的好坏无关** —— 严禁按"效果好"挑材质（那等于先看结果再选题）。
- ⚠ 同一材质的**两种顺序必须分到同一个人**（两序去偏的前提是同一人前后判断可比）。

## 不重新生成图

直接**重切已验钥的那份页面的 ITEMS**，图的 base64 一个字节没动 ⇒ 像素键天然仍然成立；
切完仍对每一份跑 `audit_h16_pixels.py` 复验（⛔ 不复验不许发出去）。

## 留给判读的口径（跑前写死）

主判据仍在**合并后的全部两序一致对**上做二项检验（与建页时的预注册逐字相同）；
另按**每人各自**报一遍；一致性只在 `SHARED` 的 10 个材质上算，⛔ 不许把独有块的单人判定当成共识。

    python analysis/annotate/split_study_h16.py --page experiments/annotate/study_h16_v2_zh.html --n 5
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED, N_SHARED = 1618, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=Path, default=ROOT / "experiments/annotate/study_h16_v2_zh.html")
    ap.add_argument("--n", type=int, default=5, help="标注者人数")
    ap.add_argument("--outdir", type=Path, default=ROOT / "experiments/annotate")
    a = ap.parse_args()
    s = a.page.read_text(encoding="utf-8")
    m = re.search(r"const ITEMS = (\[.*?\]);", s, re.S)
    items = json.loads(m.group(1))

    real = [it for it in items if it.get("kind") == "real"]
    checks = [it for it in items if it.get("kind") != "real"]
    by_mat = defaultdict(list)
    for it in real:
        by_mat[it["material"]].append(it)
    mats = sorted(by_mat)
    assert all(len(v) == 2 for v in by_mat.values()), "每个材质应当正反序各一次"

    rng = np.random.default_rng(SEED)
    order = list(rng.permutation(mats))
    shared = sorted(order[:N_SHARED])
    rest = order[N_SHARED:]
    blocks = [sorted(rest[i::a.n]) for i in range(a.n)]        # 随机均分，与产物好坏无关

    plan = {"seed": SEED, "page": a.page.name, "n_annotators": a.n,
            "shared": shared, "blocks": {f"P{i+1}": b for i, b in enumerate(blocks)}}
    (a.outdir / "study_h16_split_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    for i, blk in enumerate(blocks, 1):
        mine = shared + blk
        sub = [it for mt in mine for it in by_mat[mt]]
        rng2 = np.random.default_rng(SEED + i)
        sub = [sub[j] for j in rng2.permutation(len(sub))]      # 打乱题序，但两序仍在同一人手里
        out_items = sub + checks
        out_items = [out_items[j] for j in np.random.default_rng(SEED + 100 + i).permutation(len(out_items))]
        dst = a.outdir / f"study_h16_p{i}.html"
        dst.write_text(s[:m.start(1)] + json.dumps(out_items, ensure_ascii=False) + s[m.end(1):],
                       encoding="utf-8")
        print(f"P{i}: 材质 {len(mine)}（共同 {len(shared)} + 独有 {len(blk)}）"
              f" -> 题 {len(out_items)}（真题 {len(sub)} + 检查 {len(checks)}）  {dst.name}"
              f"  {dst.stat().st_size/1e6:.2f} MB")
    print(f"\n分配表 -> {a.outdir / 'study_h16_split_plan.json'}")
    print("⚠ 发出去之前每一份都要跑 analysis/annotate/audit_h16_pixels.py 复验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
