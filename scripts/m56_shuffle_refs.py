#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M56) 造"错配参考嵌入"那一臂的输入，并把判读器要的操作检验数落盘。

只做两件事，⛔ 不算任何判据：
  1. 把 /tmp/m55_refs 的 {提示词 -> 参考图嵌入} 做**固定种子的错排**（derangement，无不动点），
     写成 <out>/emb_shard0.pt —— 键集一个字不变，只把值换成**别的材质**的嵌入。
  2. 写 <stats> JSON：键集是否相同、不动点个数、V_mat 那 125 个提示词的参考嵌入
     **两两余弦相似度**的中位数/均值（查 `VOID_REF_COLLAPSE`）、以及每个提示词
     "自己的嵌入 vs 被错配到的嵌入"的余弦（登记用）。

用法（远程，仓库根目录）：
  python scripts/m56_shuffle_refs.py --src /tmp/m55_refs --out /tmp/m56_refs_shuf \
      --stats /tmp/m56_refstats.json --set V_mat
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/tmp/m55_refs"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/m56_refs_shuf"))
    ap.add_argument("--stats", type=Path, default=Path("/tmp/m56_refstats.json"))
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    ref = {}
    shards = sorted(a.src.glob("emb_shard*.pt"))
    for f in shards:
        ref.update(torch.load(f, map_location="cpu"))
    keys = sorted(ref)
    n = len(keys)
    if n < 2:
        raise SystemExit("M56_ABORT_REFS_EMPTY")

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)
    for i in range(n):                      # 消掉不动点：与下一个位置对调
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    fixed = int((perm == np.arange(n)).sum())

    shuf = {keys[i]: ref[keys[perm[i]]] for i in range(n)}
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(shuf, a.out / "emb_shard0.pt")
    (a.out / "shard0.done").write_text("m56\n", encoding="utf-8")

    from prompts import load_set
    prompts, _ = load_set(a.set)
    names = [e["prompt"] for e in prompts]
    miss = [p for p in names if p not in ref]

    E = torch.stack([ref[p][0].float() for p in names if p in ref])
    E = torch.nn.functional.normalize(E, dim=-1)
    C = (E @ E.T).numpy()
    iu = np.triu_indices(C.shape[0], k=1)
    off = C[iu]
    idx = {k: i for i, k in enumerate(keys)}
    self_vs_shuf = []
    for p in names:
        if p in ref:
            e0 = torch.nn.functional.normalize(ref[p][0].float(), dim=-1)
            e1 = torch.nn.functional.normalize(shuf[p][0].float(), dim=-1)
            self_vs_shuf.append(float((e0 * e1).sum()))

    stats = {
        "src": str(a.src), "out": str(a.out), "set": a.set, "seed": a.seed,
        "n_keys": n, "n_shards_read": len(shards),
        "keys_equal": sorted(shuf) == keys,
        "fixed_points": fixed,
        "n_set_prompts": len(names), "n_missing_in_refs": len(miss),
        "cos_median": float(np.median(off)), "cos_mean": float(off.mean()),
        "cos_p90": float(np.quantile(off, 0.9)), "cos_max": float(off.max()),
        "cos_self_vs_shuffled_mean": float(np.mean(self_vs_shuf)),
        "cos_self_vs_shuffled_max": float(np.max(self_vs_shuf)),
    }
    a.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    print("M56_SHUFFLE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
