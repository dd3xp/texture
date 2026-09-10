"""前提检查：真人像素画在**各向同性**材质上到底抖不抖动（预注册 `41b0960`）。

`README.md:100-106` 列的真人四条性质里，第四条「抖动承担明暗过渡」被标着
**未被有效检验**。门拒绝的那半（各向同性）四条路已全关，第五条路
（抖动降采样器）该不该以「模仿真人惯例」立项，取决于这个前提。
本脚本纯本地、不用 GPU：只读 `data/tiles/dataset_k16.json`。

口径沿用 `analysis/paired/aniso_gate.py`：`size==16` 的真人瓦片，
idx 十六进制解成逐格调色板下标，按**管线同一个门**分组。

统计量 checker_rate：不跨边的 15x15 个 2x2 窗里，在「恰好含 2 个不同下标」
的窗中，对角交替（I[0,0]==I[1,1] 且 I[0,1]==I[1,0]）所占比例——
50% 棋盘抖动最不含糊的签名。零模型是**格内洗牌**（保住调色板与颜色直方图、
只毁空间排布）。洗牌下该比例约 0.14，故方向可解读：真抖动远高于零模型，
平涂色块/渐变**低于**零模型。

判据（预注册，不在此改）：门拒绝组内逐瓦片差值中位 >0 且符号检验 p<0.05，
且观测中位 >= 1.5x 洗牌中位。灵敏度上限也是预注册的：对角签名只抓 50%
棋盘抖动，25%/75% 有序抖动在 2x2 里与色块边缘同为 3+1、不可分，
故 3+1 只作描述性报告，证伪只能证伪 50% 棋盘这一种。
"""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "analysis"))
from downsample import dominant_period, anisotropy, W        # noqa: E402
from exact import binom_test                                 # noqa: E402

MIN_WINDOWS = 10      # 2 色窗少于此数则统计量不稳，弃用（预注册）
SHUFFLES = 200        # 零模型重复次数（预注册）


def tiles():
    """size==16 的真人瓦片，产出 (材质名, 逐格下标, 调色板)。"""
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text())
    for s in ds["samples"]:
        if s["size"] != 16:
            continue
        n = s["size"]
        idx = np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(n, n)
        yield s["material"], idx, np.array(s["palette"], dtype=float)


def window_classes(idx: np.ndarray):
    """把每个不跨边 2x2 窗分类。返回 (2 色窗数, 对角窗数, 3+1 窗数, 条纹窗数)。

    只在「恰好含 2 个不同下标」的窗里分类：这样分母不受平涂面积影响，
    统计量问的是「既然这里有两色，它们是怎么摆的」。
    """
    a, b = idx[:-1, :-1], idx[:-1, 1:]      # 左上、右上
    c, d = idx[1:, :-1], idx[1:, 1:]        # 左下、右下
    # 六个两两比较里「不同」的对数唯一确定了去重后的取值个数：
    # 1 种 -> 0，恰好 2 种 -> 3（3+1 分割）或 4（2+2 分割），3 种 -> 5，4 种 -> 6。
    uniq = (
        (a != b).astype(int) + (a != c).astype(int) + (a != d).astype(int)
        + (b != c).astype(int) + (b != d).astype(int) + (c != d).astype(int)
    )
    # 4 个值中恰好 2 种取值时，6 个两两比较里不同的对数只能是 3（3+1 分割）
    # 或 4（2+2 分割）；1 种取值 -> 0，3 种 -> 5，4 种 -> 6。
    two = (uniq == 3) | (uniq == 4)
    diag = two & (a == d) & (b == c) & (a != b)
    three_one = two & (uniq == 3)
    stripe = two & (uniq == 4) & ~diag
    return int(two.sum()), int(diag.sum()), int(three_one.sum()), int(stripe.sum())


def checker_rate(idx: np.ndarray):
    n2, nd, _, _ = window_classes(idx)
    return (nd / n2 if n2 else None), n2


def shuffled_rates(idx: np.ndarray, seed: int):
    """格内洗牌零模型：保住颜色直方图，只毁空间排布。"""
    rng = np.random.default_rng(seed)
    flat = idx.reshape(-1)
    out = []
    for _ in range(SHUFFLES):
        p = rng.permutation(flat).reshape(idx.shape)
        r, n2 = checker_rate(p)
        if r is not None and n2 >= MIN_WINDOWS:
            out.append(r)
    return out


def lum_ratio(idx: np.ndarray, pal: np.ndarray):
    """次判据①：对角对两色亮度差 / 该瓦片已用色任意两色亮度差（中位之比）。"""
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    uniq = (
        (a != b).astype(int) + (a != c).astype(int) + (a != d).astype(int)
        + (b != c).astype(int) + (b != d).astype(int) + (c != d).astype(int)
    )
    two = (uniq == 3) | (uniq == 4)
    m = two & (a == d) & (b == c) & (a != b)
    if not m.any():
        return None
    lum = pal @ W
    dl = np.abs(lum[a[m].astype(int)] - lum[b[m].astype(int)])
    used = np.unique(idx)
    if len(used) < 2:
        return None
    lu = lum[used.astype(int)]
    allpairs = np.abs(lu[:, None] - lu[None, :])[np.triu_indices(len(lu), 1)]
    denom = float(np.median(allpairs))
    if denom <= 1e-9:
        return None
    return float(np.median(dl)) / denom


def as_index_map(rgb: np.ndarray) -> np.ndarray:
    """已量化的瓦片 RGB -> 逐格调色板下标（颜色本身是什么无关，只看排布）。"""
    flat = rgb.reshape(-1, 3)
    _, inv = np.unique(flat, axis=0, return_inverse=True)
    return inv.reshape(rgb.shape[:2]).astype(np.uint8)


def pipeline_mode():
    """描述性（**非**预注册判据）：同一统计量量在管线自己的瓦片上。

    B23 的点采样每格取一个真实源像素，相邻格因此采到源图上相距很远的位置，
    格间变化是**空间不相干**的；box 平均则过度相干（成片平涂）。
    真人在 §上面那半的口径下 checker_rate 中位 0.041、比洗牌低 3 倍，
    所以这里量的是「两个降采样器各偏到哪一侧」。这把 B23 的
    「逐格噪声没用」从一条经验事实变成有度量支撑的解释。
    """
    from PIL import Image
    d = ROOT / "experiments/iso_point/16"
    if not d.is_dir():
        print(f"没有 {d}（B23 的瓦片，需从服务器取回）")
        return
    rows = []
    for bp in sorted(d.glob("*_box.png")):
        pp = bp.with_name(bp.name.replace("_box.png", "_point.png"))
        if not pp.exists():
            continue
        rec = {"material": bp.name[:-8].replace("_", " ")}
        for tag, path in (("box", bp), ("point", pp)):
            idx = as_index_map(np.asarray(Image.open(path).convert("RGB")))
            obs, n2 = checker_rate(idx)
            nulls = shuffled_rates(idx, seed=7)
            rec[tag] = obs
            rec[tag + "_null"] = float(np.mean(nulls)) if nulls else None
            rec[tag + "_n2"] = n2
        rows.append(rec)

    _guard(ROOT / "experiments/dither_pipeline.json")
    (ROOT / "experiments/dither_pipeline.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"管线自己的瓦片（B23 门拒绝的 {len(rows)} 个材质，16px）")
    for tag in ("box", "point"):
        v = [r[tag] for r in rows if r[tag] is not None]
        nl = [r[tag + "_null"] for r in rows if r.get(tag + "_null") is not None]
        print(f"  {tag:<5} checker_rate 中位 {statistics.median(v):.3f}"
              f"   自己的洗牌中位 {statistics.median(nl):.3f}"
              f"   富集 {statistics.median(v)/statistics.median(nl):.2f}x")
    both = [r for r in rows if r["box"] is not None and r["point"] is not None]
    up = sum(r["point"] > r["box"] for r in both)
    nz = sum(r["point"] != r["box"] for r in both)
    print(f"  逐材质 point > box：{up}/{nz}   符号检验 p={binom_test(up, nz):.3g}")
    print("  真人（门拒绝组）中位 0.041、富集 0.34x —— 对照上面两行看谁偏到了哪边")


USAGE = """用法：
  python analysis/premise/dither.py              # 真人瓦片上的前提检验
  python analysis/premise/dither.py --pipeline   # 改看管线自己出的瓦片
  ... --force                                    # 允许覆盖已存在的结果 JSON

⚠ 本脚本**会覆盖** experiments/dither_premise.json / dither_pipeline.json。
   它们是已提交的一手数据，覆盖前请确认你真的想重跑（加 --force）。"""


def _guard(out: Path) -> None:
    """别让一次手滑覆盖掉已提交的一手数据。

    此前用的是 `sys.argv` 成员判断，**任何不认识的参数都会静默跑完整分析
    并覆盖 JSON**——实测 `--help` 就这么把 dither_premise.json 冲了一次
    （所幸分析是确定性的，只有 p 值末位因 exact.py 换 logsumexp 变了 1e-14）。
    """
    if out.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{out.name} 已存在。重跑会覆盖这份已提交的数据，"
                         f"确认要重跑就加 --force。" + "\n\n" + USAGE)


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(USAGE)
        return
    unknown = [a for a in sys.argv[1:] if a not in ("--pipeline", "--force")]
    if unknown:
        raise SystemExit(f"不认识的参数：{' '.join(unknown)}" + "\n\n" + USAGE)
    # **在入口处就挡**，不要跑完三分钟分析才拒绝写（写文件处的守卫留作兜底）。
    _guard(ROOT / ("experiments/dither_pipeline.json" if "--pipeline" in sys.argv
                   else "experiments/dither_premise.json"))
    if "--pipeline" in sys.argv:
        pipeline_mode()
        return
    groups = {"isotropic": [], "periodic": []}
    for k, (mat, idx, pal) in enumerate(tiles()):
        rgb = pal[idx].astype(float)
        gated = dominant_period(rgb, lo=2, hi_frac=0.625) > 0 and anisotropy(rgb) >= 0.20
        obs, n2 = checker_rate(idx)
        if obs is None or n2 < MIN_WINDOWS:
            continue
        nulls = shuffled_rates(idx, seed=1000 + k)
        if not nulls:
            continue
        _, nd, n31, nst = window_classes(idx)
        groups["periodic" if gated else "isotropic"].append({
            "material": mat, "obs": obs, "null": float(np.mean(nulls)),
            "null_p95": float(np.percentile(nulls, 95)),
            "n2": n2, "diag": nd, "three_one": n31, "stripe": nst,
            "lum_ratio": lum_ratio(idx, pal),
        })

    out = {}
    for g, recs in groups.items():
        if not recs:
            continue
        obs = [r["obs"] for r in recs]
        nul = [r["null"] for r in recs]
        dif = [r["obs"] - r["null"] for r in recs]
        up = sum(x > 0 for x in dif)
        nz = sum(x != 0 for x in dif)
        p = binom_test(up, nz) if nz else 1.0
        mo, mn = statistics.median(obs), statistics.median(nul)
        above = sum(r["obs"] > r["null_p95"] for r in recs) / len(recs)
        lr = [r["lum_ratio"] for r in recs if r["lum_ratio"] is not None]
        tot2 = sum(r["n2"] for r in recs)
        out[g] = {
            "n": len(recs), "median_obs": mo, "median_null": mn,
            "enrich": mo / mn if mn > 1e-9 else float("inf"),
            "median_diff": statistics.median(dif), "up": up, "nz": nz, "p": p,
            "frac_above_null_p95": above,
            "median_lum_ratio": statistics.median(lr) if lr else None,
            "share_diag": sum(r["diag"] for r in recs) / tot2,
            "share_three_one": sum(r["three_one"] for r in recs) / tot2,
            "share_stripe": sum(r["stripe"] for r in recs) / tot2,
        }

    _guard(ROOT / "experiments/dither_premise.json")
    (ROOT / "experiments/dither_premise.json").write_text(
        json.dumps({"groups": out, "tiles": groups}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    for g in ("isotropic", "periodic"):
        if g not in out:
            continue
        s = out[g]
        tag = "门拒绝（各向同性）" if g == "isotropic" else "门触发（周期）· 对照"
        print(f"\n{tag}  {s['n']} 张")
        print(f"  checker_rate 中位 观测 {s['median_obs']:.3f}  洗牌 {s['median_null']:.3f}"
              f"   富集 {s['enrich']:.2f}x")
        print(f"  逐瓦片差值中位 {s['median_diff']:+.3f}   变大 {s['up']}/{s['nz']}"
              f"   符号检验 p={s['p']:.3g}")
        print(f"  超过自己洗牌 95 分位的占比 {s['frac_above_null_p95']:.0%}")
        print(f"  2 色窗构成：对角 {s['share_diag']:.0%}  3+1 {s['share_three_one']:.0%}"
              f"  条纹 {s['share_stripe']:.0%}")
        if s["median_lum_ratio"] is not None:
            print(f"  次判据① 对角对亮度差 / 全体已用色亮度差 = {s['median_lum_ratio']:.2f}"
                  "（抖动应明显 <1）")

    s = out.get("isotropic")
    if not s:
        print("\n没有门拒绝的瓦片，判据无法评估")
        return
    ok = s["median_diff"] > 0 and s["p"] < 0.05 and s["enrich"] >= 1.5
    print("\n--- 主判据（预注册 41b0960）---")
    print(f"  中位差 >0: {s['median_diff'] > 0}   p<0.05: {s['p'] < 0.05}"
          f"   富集 >=1.5x: {s['enrich'] >= 1.5}")
    if ok:
        print("  -> **前提成立**：真人在各向同性材质上确实用 50% 棋盘抖动。")
        print("     抖动降采样器可以按「模仿真人惯例」立项，作为第五条路。")
    else:
        print("  -> **前提不成立**：真人在各向同性材质上并不靠 50% 棋盘抖动。")
        print("     第五条路不以「模仿真人惯例」立项；README 那条假设改为已检验且不成立。")
        print("     注意预注册的灵敏度上限：只证伪了 50% 棋盘这一种，")
        print("     不许写成「真人完全不抖动」。")


if __name__ == "__main__":
    main()
