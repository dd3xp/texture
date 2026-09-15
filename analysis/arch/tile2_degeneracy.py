#!/usr/bin/env python
"""退化检验：生成的 32px 图是不是"16px 复制两遍"。零 GPU、零 API、零判官。

`scripts/trd_tile16_train_eval.sh` 判据③的仪器。--p_tile16 把"16px 真人瓦片 2×2 平铺"
喂进 32px 那一路，风险是模型学到"32px = 周期 16"，而 e4 已经证明原生 32px 优于放大的 16px
（136/189 = 72%），塌成周期 16 就是退化。

两个读数：
  - `exact`（判据③用的那个）：整张图与"左上 16×16 平铺两遍"**逐像素完全相同**的比例。
  - `mad`（描述性）：同一比较下**每通道绝对差的均值**（0–255）。⚠ 项目规矩：量图像差异只许用
    这个，不许数"多少像素不相等"（四条边都会报 100%，什么也看不见）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def scan(d: Path):
    ex, mads = 0, []
    files = sorted(d.glob("*.png"))
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"), dtype=np.int16)
        n = a.shape[0]
        if n % 2 or a.shape[1] != n:
            raise SystemExit(f"{f}: 尺寸不是偶数方阵 {a.shape}")
        h = n // 2
        t = np.tile(a[:h, :h], (2, 2, 1))          # 左上象限平铺两遍
        ex += int(np.array_equal(a, t))
        mads.append(float(np.abs(a - t).mean()))
    return {"dir": str(d), "n": len(files), "exact": ex,
            "exact_rate": ex / len(files) if files else None,
            "mad_mean": float(np.mean(mads)) if mads else None,
            "mad_median": float(np.median(mads)) if mads else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = []
    for tag in a.tags:
        d = a.root / tag / str(a.size)
        if not d.is_dir():
            print(f"[skip] 无此目录：{d}")
            continue
        r = scan(d)
        r["tag"] = tag
        rows.append(r)
        print(f"{tag:18s} n={r['n']:4d}  四象限全同 {r['exact']:4d} = {r['exact_rate']:.1%}"
              f"   每通道绝对差 均值 {r['mad_mean']:.2f} 中位 {r['mad_median']:.2f}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)
        print("->", a.out)


if __name__ == "__main__":
    main()
