"""离线复算结构尺子：换归一化/描述子而不重新生成。

配合 `struct_metric.py` 存下的 `experiments/struct_metric_tiles.json`。

要回答的是第 3 条为什么不通过：硬塞假结构到颗粒材质上只被罚 0.5%。
猜想是颗粒瓦片的自相关峰随机，那几个描述子维度的 MAD 很大，把惩罚除没了。
明显的修法是给 MAD 加下限——**而这正是最容易"调到自己通过"的地方**。

所以这个脚本对每个变体**同时报三条判据**。
只让第 3 条通过、把第 1 或第 2 条换掉的改动不是修复，是自欺。
基线变体 `none` 必须复现线上跑出来的数字，否则脚本本身就是错的。
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from struct_metric import descriptor                              # noqa: E402
from stability import split_half                                  # noqa: E402


def load(path="experiments/struct_metric_tiles.json"):
    d = json.loads(Path(path).read_text())
    for r in d:
        r["bands"] = [tuple(b) for b in r["bands"]]
    return d


def _hex(h):
    return np.frombuffer(bytes.fromhex(h), np.uint8).reshape(16, 16)


def scorer(train, bands, mode: str, floors: np.ndarray | None):
    """mode: none=线上版（9 维取平均）；floor=MAD 取全局下限；
    median=跨维取中位；clip=先截断再平均。

    线上版的毛病是**聚合非稳健**：逐维中位只有 0.7–3.4，
    总分却是 45（最大 357）——某些材质某一维 MAD 近零，除下去就炸，
    少数极端维度主导了整个分数，把真实响应埋掉。
    稳健归一化后再用平均聚合本来就是错的，这是对症不是调参。
    """
    D = np.stack([descriptor(t, bands) for t in train])
    med = np.median(D, 0)
    mad = np.median(np.abs(D - med), 0) * 1.4826 + 1e-3
    if mode == "floor" and floors is not None:
        mad = np.maximum(mad, floors)

    def f(ix):
        z = np.abs(descriptor(ix, bands) - med) / mad
        if mode == "median":
            return float(np.median(z))
        if mode == "clip":
            return float(np.mean(np.minimum(z, 10.0)))
        return float(np.mean(z))
    return f


def global_mads(data) -> np.ndarray:
    """各维 MAD 在所有材质上的中位数，作为下限的来源。

    下限取自**全体材质**，不看第 3 条的结果——
    否则就是照着判据调参数。
    """
    out = []
    for r in data:
        D = np.stack([descriptor(_hex(h), r["bands"]) for h in r["train"]])
        out.append(np.median(np.abs(D - np.median(D, 0)), 0) * 1.4826 + 1e-3)
    return np.median(np.stack(out), 0)


def evaluate(data, mode: str, floors=None, frac: float = 1.0) -> dict:
    S = {"artist": [], "seeded": [], "none": []}
    raw_seeded = []
    G = {"artist": [], "seeded": [], "none": [], "forced": []}
    for r in data:
        sc = scorer([_hex(h) for h in r["train"]], r["bands"], mode,
                    None if floors is None else floors * frac)
        t = S if r["structured"] else G
        t["artist"].append(sc(_hex(r["artist"])))
        per = [sc(_hex(h)) for h in r["seeded"]]
        t["seeded"].append(float(np.mean(per)))
        if r["structured"]:
            raw_seeded.append(per)
        t["none"].append(float(np.mean([sc(_hex(h)) for h in r["none"]])))
        if not r["structured"] and r["forced"]:
            G["forced"].append(float(np.mean([sc(_hex(h)) for h in r["forced"]])))
    a, s, n = (np.median(S[k]) for k in ("artist", "seeded", "none"))
    fk, nn = np.array(G["forced"]), np.array(G["none"])
    rel = (np.median(fk) - np.median(nn)) / max(np.median(nn), 1e-9) if len(fk) else 0.0
    pv = stats.wilcoxon(fk, nn).pvalue if len(fk) > 5 else float("nan")
    rep = split_half(raw_seeded)
    return {"artist": a, "seeded": s, "none": n,
            "c1": a < s, "c2": n > s, "c3": rel > 0.05 and pv < 0.05,
            "rel": rel, "p": pv, "stable": rep.stable, "rho": rep.rho,
            "gap": rep.half_gap}


def main():
    data = load()
    print(f"载入 {len(data)} 个材质"
          f"（有结构 {sum(r['structured'] for r in data)}）")
    fl = global_mads(data)
    print(f"\n{'变体':<20}{'真人':>8}{'有种子':>9}{'无种子':>9}"
          f"{'C1':>5}{'C2':>5}{'C3':>5}{'硬塞相对':>10}{'p':>10}")
    print("-" * 82)
    for name, mode, frac in (("none 线上版(均值)", "none", 1.0),
                             ("floor x1.0", "floor", 1.0),
                             ("median 跨维中位", "median", 1.0),
                             ("clip 截断10后均值", "clip", 1.0)):
        r = evaluate(data, mode, fl, frac)
        print(f"{name:<20}{r['artist']:>8.2f}{r['seeded']:>9.2f}{r['none']:>9.2f}"
              f"{'✓' if r['c1'] else '✗':>5}{'✓' if r['c2'] else '✗':>5}"
              f"{'✓' if r['c3'] else '✗':>5}{r['rel']:>9.1%}{r['p']:>10.3g}"
              f"{('稳定' if r['stable'] else '不稳'):>7}"
              f" ρ={r['rho']:+.2f} 差{r['gap']:.0%}")
    print("\n三条必须同时为 ✓。只修好 C3 而弄坏 C1/C2 的变体不算修复。")


if __name__ == "__main__":
    main()
