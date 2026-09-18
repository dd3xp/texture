"""(M64) 记账：64px 真人瓦片在各 split 的供给，以及降到 32px 之后还剩多少。
零 GPU、零 API、纯计数，⛔ 不产生任何判决。

为什么问（(M63) 第六节写死的"下一轮第一件事"）：
32px 上现在没有任何一把能"认证领先基线"的尺子 —— CLIP 天花板在 B2 之下（(M63)）、
KID/FID/FD 在测试集算不出来（(M53) 数到 32px 真人 test 只有 6 张）、判官上游坏着且 32px 校准不过关。
唯一**不需要新数据来源**的修法是 `model/build_64to32.py` 那套 2×2 众数降采样（当年给 v11 造训练数据），
把它用到 **val/test 包**上就能得到 32px 参照集。但动手之前必须先数清楚：
**test 里到底有没有 64px 真人瓦片、覆盖 E_mat 的 272 个材质几个。**

⚠ 本脚本只回答"物理上有多少"，不回答"降采样出来的算不算真人参照"（那要另行预注册论证）。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "tools"))
from tiles_data import load                     # noqa: E402
from build_64to32 import mode_pool2             # noqa: E402  ← 逐字复用当年的构造，不重写
from prompts import is_material, load_set       # noqa: E402


def survives_downsample(s):
    """照抄 build_64to32.main() 的唯一筛除条件：降采样后实际用到的颜色 < 3 就丢。"""
    g = mode_pool2(np.asarray(s["idx"]))
    return int(len(np.unique(g)))


def main():
    out = {}

    print("=== 一、dataset_k16.json 里各尺寸 × split 的真人瓦片张数（decontam 默认开）===")
    sizes = sorted({16, 24, 32, 64})
    table = {}
    for size in sizes:
        row = {}
        for split in ("train", "val", "test"):
            row[split] = len(load(size, split))
        table[str(size)] = row
        print(f"  size={size:2d}  " + "  ".join(f"{k}={v}" for k, v in row.items()))
    out["tiles_by_size_split"] = table

    print("\n=== 二、64px 瓦片降到 32px 后还剩多少（构造＝build_64to32 的 2×2 众数 + <3 色丢弃）===")
    surv = {}
    for split in ("train", "val", "test"):
        rows = load(64, split)
        ks = [survives_downsample(s) for s in rows]
        kept = [k for k in ks if k >= 3]
        surv[split] = {"n_64": len(rows), "n_kept_32": len(kept),
                       "k_hist": dict(sorted(Counter(ks).items()))}
        print(f"  split={split:5s}  64px={len(rows):4d}  降采样后留下={len(kept):4d}"
              f"  颜色数分布={dict(sorted(Counter(ks).items()))}")
    out["downsample_survival"] = surv

    print("\n=== 三、判官集合覆盖：提示表里的材质，有多少能从 64px 拿到参照 ===")
    for set_name in ("V_mat", "E_mat"):
        prompts, split = load_set(set_name)
        want = {e["material"] for e in prompts}
        rows = [s for s in load(64, split)
                if s["material"] in want and is_material(s["material"])]
        kept = [s for s in rows if survives_downsample(s) >= 3]
        mats_all = {s["material"] for s in rows}
        mats_kept = {s["material"] for s in kept}
        # 对照：现有 32px 原生参照（= targets(set_name,32) 的口径）
        nat = [s for s in load(32, split)
               if s["material"] in want and is_material(s["material"])]
        rec = {"split": split, "n_prompts": len(want),
               "n_tiles_64": len(rows), "n_mats_64": len(mats_all),
               "n_tiles_64to32": len(kept), "n_mats_64to32": len(mats_kept),
               "n_tiles_native32": len(nat),
               "n_mats_native32": len({s["material"] for s in nat}),
               "packs_64": dict(sorted(Counter(s["pack"] for s in kept).items(),
                                       key=lambda kv: -kv[1]))}
        out[set_name] = rec
        print(f"  {set_name:6s} split={split:5s} 提示表材质={len(want):3d}"
              f" | 64px 命中瓦片={len(rows):4d} 材质={len(mats_all):3d}"
              f" | 降采样后={len(kept):4d} 材质={len(mats_kept):3d}"
              f" | 现有原生 32px={len(nat):3d} 材质={rec['n_mats_native32']:3d}")
        print(f"         降采样后按包分布（前8）: "
              f"{dict(list(rec['packs_64'].items())[:8])}")

    p = ROOT / "experiments/m64_ref64_count.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
