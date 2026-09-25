#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""按**像素**验钥 study_h16_v2.html：页面里标成某条臂的那块图，是不是真的来自那条臂的瓦片。

为什么另写一个：`build_study_h32.py::audit` 比的是 **base64 字符串的 sha256**，
所以只要建页与验钥两边的 PNG 编码器有一丁点差别（PIL 版本、压缩级别），就会 **360/360 全不一致** ——
那是假警报，不是键错位。本项目的纪律是**验钥要用定义本身**：解码回像素、逐像素比。

判据（跑前写死）：
  面板 = 左半 192×192 的最近邻放大图 + 下方 3×3 平铺，与 `build_study_h32.panel()` 同构。
  只取**左上角那块 192×192**，缩回 16×16（每个源像素恰好 12×12，取块内众数），
  与 `experiments/baselines/<臂>/16/<slug>` 的瓦片逐像素比。
  `MATCH` = 逐像素全等；`MISMATCH` = 有差异；另报把左右臂**对调**后是否反而全等（`SWAPPED`）
  —— 那才是真正会让结论整体倒转的那种错。
结论：
  全部 MATCH                  -> `KEY_OK`，页面可以发给标注者
  存在 SWAPPED                -> `KEY_SWAPPED`，⛔ 绝对不许发
  其余                        -> `KEY_MISMATCH`，⛔ 不许发，需查建页脚本

    python analysis/annotate/audit_h16_pixels.py --page experiments/annotate/study_h16_v2.html
"""
import argparse
import base64
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from prompts import load_set                     # noqa: E402


def panel_to_tile(b64, size=16, box=192):
    """把页面里的面板图还原成 size×size 瓦片：取左上 box×box，按块取众数（放大是最近邻，块内应当同色）。"""
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    a = np.asarray(im)[:box, :box]
    k = box // size
    out = np.zeros((size, size, 3), np.uint8)
    for y in range(size):
        for x in range(size):
            blk = a[y * k:(y + 1) * k, x * k:(x + 1) * k].reshape(-1, 3)
            out[y, x] = Counter(map(tuple, blk)).most_common(1)[0][0]
    return out


def first_tile(d, slug):
    for name in (f"{slug}_0.png", f"{slug}.png"):
        p = d / name
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", type=Path, default=ROOT / "experiments/annotate/study_h16_v2.html")
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    a = ap.parse_args()
    items = json.loads(re.search(r"const ITEMS = (\[.*?\]);",
                                 a.page.read_text(encoding="utf-8"), re.S).group(1))
    slug = {e["prompt"]: e["material"].rsplit(".", 1)[0] for e in load_set(a.set)[0]}

    n = ok = mism = swapped = missing = 0
    bad_examples = []
    for it in items:
        if it.get("kind") != "real":
            continue
        s = slug.get(it["material"])
        if s is None:
            missing += 1
            continue
        for side, key in (("left", "limg"), ("right", "rimg")):
            n += 1
            arm = it[side]
            got = panel_to_tile(it[key], a.size)
            want = first_tile(a.root / arm / str(a.size), s)
            other_arm = it["right" if side == "left" else "left"]
            other = first_tile(a.root / other_arm / str(a.size), s)
            if want is not None and np.array_equal(got, want):
                ok += 1
            elif other is not None and np.array_equal(got, other):
                swapped += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"SWAPPED {it['material']} {side}={arm}")
            else:
                mism += 1
                if len(bad_examples) < 5:
                    bad_examples.append(f"MISMATCH {it['material']} {side}={arm}")

    verdict = ("KEY_SWAPPED" if swapped else "KEY_OK" if mism == 0 and ok == n else "KEY_MISMATCH")
    print(f"面板 {n} 块：逐像素相符 {ok}，左右对调才相符 {swapped}，都不符 {mism}，材质缺失 {missing}")
    for b in bad_examples:
        print("   ", b)
    print(f"\n判决：**{verdict}**  " +
          {"KEY_OK": "页面可以发给标注者",
           "KEY_SWAPPED": "⛔ 键错位，绝对不许发（结论会整体倒转）",
           "KEY_MISMATCH": "⛔ 不许发，需查建页脚本"}[verdict])
    return 0 if verdict == "KEY_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
