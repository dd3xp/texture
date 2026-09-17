"""把 B5（检索基线）的瓦片落盘，供判官与区域颜色任务使用。

为什么需要：`eval/run_eval.py` 的 B5 是**现场构造**的（`retrieval()` 返回内存里的瓦片），
所以 `experiments/baselines/` 下从来没有 B5 目录 —— 而 `eval/judge_pairs.py` 与
`eval/build_colour_dirs.py` 都只认落盘文件。于是 44 条判官臂里**没有一条对 B5**，
而 B5 在 16px 分布指标上是最强的（KID 2.69 / FID 43.0 / FD 45.3，因为它返回的就是真人原图）。

⚠ 本脚本**不重新实现检索**：直接 import `run_eval.retrieval`，与指标表里那个 B5 是同一段代码、
同一个数据池（`load(size,"train",extra=True)`）、同一个 `rng = default_rng(0)` → 落盘的图与
已发表指标表里的 B5 逐像素相同（`--verify` 会把这件事查一遍）。

    python eval/build_b5.py --set E_mat --size 16 --n 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "model"))
from prompts import load_set                  # noqa: E402
from run_eval import retrieval                # noqa: E402  同一段检索代码，别另写一份


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--n", type=int, default=4, help="每材质落盘几张（与其他基线的 4 张对齐）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify", action="store_true",
                    help="再跑一次 retrieval，逐像素比对已落盘的图（确认可复现）")
    a = ap.parse_args()
    prompts = load_set(a.set)[0]
    d = a.out or (ROOT / ("experiments/baselines_val" if a.set.startswith("V_") else "experiments/baselines")
                  / "B5" / str(a.size))
    d.mkdir(parents=True, exist_ok=True)
    first, groups = retrieval(prompts, a.n, a.size)
    n_files = n_bad = 0
    for e, g in zip(prompts, groups):
        slug = e["material"].rsplit(".", 1)[0]
        for j, t in enumerate(g[:a.n]):
            p = d / f"{slug}_{j}.png"
            arr = np.asarray(t, np.uint8)
            if a.verify and p.exists():
                old = np.asarray(Image.open(p).convert("RGB"))
                n_bad += int(not np.array_equal(old, arr))
            else:
                Image.fromarray(arr).save(p)
            n_files += 1
    print(f"{a.set} {a.size}px：{len(prompts)} 个材质 -> {n_files} 张 {'已比对' if a.verify else '已写入'} {d}")
    if a.verify:
        print(f"逐像素不一致：{n_bad} 张" + ("（可复现）" if n_bad == 0 else "  ⚠ 不可复现，别用"))
        return 1 if n_bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
