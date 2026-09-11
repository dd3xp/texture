"""把所有方法放进同一张指标表（M2/M4）。

方法目录约定：`experiments/baselines/<方法>/<size>/<slug>_<k>.png`（每材质 n 张）
或 `<slug>.png`（每材质 1 张，如 B2）。B5 检索基线在本脚本里现场构造。

公平性口径（写死）：
- FID / KID / FD-DINOv2 / CLIP / 平铺性：**每个方法每个材质只取第 0 张**，所以各方法 n 相同
  （FID 对 n 敏感，n 不同不可比）。
- LPIPS 多样性：用同一材质的全部样本（≥2 张的方法才报）。
- 参照集：E-mat 对应的测试集真人瓦片（857 张）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "model"))
from metrics import evaluate, lpips_diversity           # noqa: E402
from prompts import is_material, prompt_words, load_set  # noqa: E402
from tiles_data import load                             # noqa: E402


def load_method(d: Path, slugs):
    """返回 (第0张列表, 分组列表) —— 缺图的材质记 None。"""
    first, groups = [], []
    for s in slugs:
        if (d / f"{s}.png").exists():
            many = [d / f"{s}.png"]
        else:                       # 先按"材质名_数字"严格过滤，再排序（前缀相同的材质名会误中 glob）
            many = [p for p in d.glob(f"{s}_*.png") if re.fullmatch(re.escape(s) + r"_\d+", p.stem)]
            many.sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in many]
        first.append(imgs[0] if imgs else None)
        groups.append(imgs)
    return first, groups


def retrieval(prompts, n, size=16):
    """B5：训练集里与提示词共享词最多的材质，取其真人瓦片（最多 n 张）。"""
    train = load(size, "train", extra=True)          # 与 TRD 的调色板记忆库同一个数据池
    by_mat = {}
    for s in train:
        by_mat.setdefault(s["material"], []).append(s["palette"][s["idx"]])
    words = {m: set(prompt_words(m)) for m in by_mat}
    rng = np.random.default_rng(0)
    first, groups = [], []
    for e in prompts:
        q = set(e["prompt"].split())
        score = {m: len(q & w) / max(1, len(q | w)) for m, w in words.items()}
        best = max(score.values())
        cands = sorted(m for m, v in score.items() if v == best)
        m = cands[rng.integers(len(cands))]
        tiles = by_mat[m][:n]
        first.append(tiles[0])
        groups.append(tiles)
    return first, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--methods", nargs="+", default=["B1", "B2", "B4", "B5"])
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--all_samples", action="store_true",
                    help="分布指标用每材质全部样本（只用于验证集调参降方差；正式测试表不用，"
                         "因为 B2 每材质只有 1 张，n 不同 FID 不可比）")
    a = ap.parse_args()

    prompts, ref_split = load_set(a.set)            # E_* -> test 参照；V_* -> val 参照（仅调参用）
    slugs = [e["material"].rsplit(".", 1)[0] for e in prompts]
    test = load(a.size, ref_split)
    ref = [s["palette"][s["idx"]] for s in test
           if a.set.endswith("_all") or is_material(s["material"])]
    print(f"{a.set}: {len(prompts)} 个材质；参照 {len(ref)} 张真人 {a.size}px", flush=True)

    rows, cache = {}, None
    for m in a.methods:
        if m == "B5":
            if a.size not in (16, 32):
                print(f"  B5: 训练集没有 {a.size}px 瓦片，跳过")
                continue
            first, groups = retrieval(prompts, 4, a.size)
        else:
            d = a.root / m / str(a.size)
            if not d.exists():
                print(f"  {m}: 无目录 {d}，跳过")
                continue
            first, groups = load_method(d, slugs)
        ok = [i for i, t in enumerate(first) if t is not None]
        if len(ok) < len(prompts):
            print(f"  {m}: 只有 {len(ok)}/{len(prompts)} 个材质有图（仍在生成？）")
        if len(ok) < 50:
            continue
        if a.all_samples:
            tiles = [t for i in ok for t in groups[i]]
            mats = [prompts[i]["prompt"] for i in ok for _ in groups[i]]
        else:
            tiles, mats = [first[i] for i in ok], [prompts[i]["prompt"] for i in ok]
        r, cache = evaluate(tiles, mats, ref, ref_cache=cache)
        g = [groups[i] for i in ok if len(groups[i]) >= 2]
        r["LPIPS_div"] = lpips_diversity(g) if g else float("nan")
        r["materials"] = len(ok)
        rows[m] = r
        print(f"  {m:<28} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()
                                          if k not in ("n", "materials")), flush=True)

    out = a.out or ROOT / f"experiments/eval_{a.set}_{a.size}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    cols = ["KID_x1e3", "FID", "FD_DINOv2", "CLIP", "LPIPS_div", "tile_seam_ratio", "materials"]
    print("\n| 方法 | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for m, r in rows.items():
        print(f"| {m} | " + " | ".join(f"{r.get(c, float('nan')):.3f}" if c != "materials"
                                        else str(r.get(c)) for c in cols) + " |")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
