#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P3) 给接缝这把尺子做标定：双端锚点 + 真人参照带 + **反例**。零 GPU / 零 API / 零训练 / 零判官。

预注册：本文件先提交再跑；判据写死在下面，⛔ 一个字不许放宽。

## 为什么

体裁调研（`docs/paper_genre_survey.md` §七之二/七之三）给出三条：
1. 可平铺的量化指标只有 5/18 篇报，而**为自己的指标画地板/锚点的只有 2/18**；
2. SD-πXL 把 **bilinear 降采样**跑自己那套指标，得到比所有像素化方法**更高**的分数，
   由此逐字断言 "none of the metrics assesses directly the pixelization quality itself"
   —— 本轮照做（我们的反例更强，见下）；
3. Structured Pattern Expansion 的做法是**在已知可平铺的真人语料上算出参照带**，论证"我落在带内"，
   而不是"我比对手高"。

而我们自己的尺子有一个**双侧**失效方向，此前从未写明：
`eval/metrics.py::tile_seam_ratio` = 接缝邻格差 / 内部邻格差，**理想值 1**（接缝与内部一样连续）。
- 比 1 大 = 接缝看得见；
- **比 1 小 = 接缝比内部更平滑**（边缘被抹平）—— 而正式测试表里 **B2 = 0.470**、B4 = 0.862 落在这一侧。
  ⚠ 任何"越低越好"的朴素读法都会把 B2 读成最好。本轮就是把这件事量出来并写进论文。

## 尺子（两把，都报）

- `ratio`   = `tile_seam_ratio`（本项目已发表口径，理想 1）；主判据用 **|ratio − 1|**（离理想的距离）。
- `ts_raw`  = **接缝处每通道绝对差的均值**（Tiled Diffusion 的 `Tiling Score` 口径，越低越好，
  不做归一化）→ 跨论文可比的那个数。⚠ 它对"平涂"完全没有免疫力，这正是反例要打的地方。
  ⚠ Tiled Diffusion 对三次测量的聚合方式**自相矛盾**（定义句 maximum、后文 average）→ 本脚本
  **两种都报**（`ts_raw_max` / `ts_raw_mean`：接缝 + 两个内部偏移），⛔ 不替它选一个。

## 臂

| 臂 | 是什么 | 作用 |
|---|---|---|
| `artist` | 测试集真人 16px 瓦片（1265 张，12 个包） | **参照带**（按包重采样，见 OP3） |
| `TRD16` / `TRD16c_rr4` / `B1` / `B2` / `B4` / `B5` / `B7` | 已发表的七列 | 被标定的对象 |
| `A_flatborder` | 真人瓦片 + **把最外一圈像素替换成该瓦片均值**（内部不动） | **反例①**：接缝差→0 而内部不变 ⇒ ratio→0、ts_raw→0，**两把尺子都给"完美"分**，而它明显更差 |
| `A_blur` | 真人瓦片 2× 双线性降采样再升回 16px | **反例②**（SD-πXL 那一招的同构物） |
| `A_const` | 整张填该瓦片均值 | **反例③**：内部差→0 ⇒ `ratio` 返回 nan（尺子**拒答**＝这一侧它是安全的），但 `ts_raw`＝0（满分） |
| `A_halfswap` | **B2 瓦片左右对半互换** | **好端锚点**：新接缝＝原内部相邻列 ⇒ 接缝处定义上完美（Tiled Diffusion 的 Swapped 锚点） |
| `A_roll_artist` | 真人瓦片随机环移（可平铺瓦片环移后仍可平铺） | **不变性检验**：真可平铺的图，环移不该改变分数 |

## 判据（跑前写死）

- **(D1) `METRIC_NOT_A_CRITERION`**：若 `A_flatborder` 或 `A_const` 在某把尺子上**优于或等于**
  全部七列里最好的那一列（`ratio`: |ratio−1| 更小；`ts_raw`: 更小）
  ⇒ 该尺子**单独不构成判据**，论文里必须与质量尺子（判官/KID）联读。
  ⚑ 预测：两把尺子**都**会被反例打穿（这是要写进论文的话，不是失败）。
- **(D2) `IN_BAND` / `OUT_OF_BAND`**：`artist` 的 |ratio−1| 的**按包自助 95% 区间**即参照带；
  TRD 两列落在带内记 `IN_BAND`。⚠ **落在带下方不算"更好"**（那是反例①的方向）。
- **(D3) 好端锚点有效性**：`A_halfswap` 的 `ts_raw` 必须**显著低于** `B2` 本身（它构造上就该如此）；
  不成立则锚点无效、(D1)/(D2) 照样报但**不引用锚点**。
- **(D4) 不变性**：`A_roll_artist` 与 `artist` 的中位 |ratio−1| 差 < 0.02，否则说明实现有方向性 bug。
- **⛔ 本轮不授权任何配置变更、不产生任何"哪个方法更好"的结论**（那要判官/KID，不是接缝分）。

## 操作检验（任一不过 ⇒ 该臂作废）

- (OP1) 每个方法臂可用瓦片 ≥ 250；`artist` ≥ 200 张且 ≥ 5 个包。
- (OP2) 反例臂与其来源臂**逐材质一一对应**（同 slug，同一张源图变换而来）。
- (OP3) `artist` 的区间**必须按包重采样**（瓦片级只能当下界报）—— 本项目 (M16) 的规矩。
- (OP4) 复跑逐位一致（固定 seed）。

用法（服务器上跑，图都在 `experiments/baselines/`）：
    python analysis/arch/seam_calib.py --set E_mat --size 16 --out /tmp/seam_calib.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "model"))
from prompts import load_set                     # noqa: E402
from tiles_data import load as load_tiles        # noqa: E402

METHODS = ["TRD16", "TRD16c_rr4", "B1", "B2", "B4", "B5", "B7"]
B_BOOT, SEED = 2000, 0


def ratio(a):
    """eval/metrics.py::tile_seam_ratio 的同式实现（理想 1）。内部近平涂返回 nan。"""
    a = np.asarray(a, np.float64)
    seam = (np.abs(a[:, -1] - a[:, 0]).mean() + np.abs(a[-1, :] - a[0, :]).mean()) / 2
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    return float(seam / inner) if inner > 1e-6 else float("nan")


def ts_raw(a):
    """Tiling Score 口径：接缝处 + 两个内部偏移处的每通道绝对差均值。返回 (max, mean)。"""
    a = np.asarray(a, np.float64)
    w, h = a.shape[1], a.shape[0]
    vals = [(np.abs(a[:, -1] - a[:, 0]).mean() + np.abs(a[-1, :] - a[0, :]).mean()) / 2]
    for off in (w // 4, w // 2):                  # 两个固定偏移＝图自身的相邻列差（尺度参照）
        vals.append((np.abs(a[:, off] - a[:, off - 1]).mean()
                     + np.abs(a[off, :] - a[off - 1, :]).mean()) / 2)
    return float(max(vals)), float(np.mean(vals))


def flat_border(a):
    b = np.asarray(a, np.float64).copy()
    m = b.reshape(-1, b.shape[-1]).mean(0)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = m
    return b


def blur(a):
    im = Image.fromarray(np.asarray(a, np.uint8))
    s = im.size[0]
    return np.asarray(im.resize((s // 2, s // 2), Image.BILINEAR)
                        .resize((s, s), Image.BILINEAR), np.float64)


def const(a):
    b = np.asarray(a, np.float64)
    return np.broadcast_to(b.reshape(-1, b.shape[-1]).mean(0), b.shape).copy()


def halfswap(a):
    b = np.asarray(a, np.float64)
    return np.concatenate([b[:, b.shape[1] // 2:], b[:, :b.shape[1] // 2]], axis=1)


def load_dir(d, slugs):
    out = {}
    for s in slugs:
        p = d / f"{s}.png"
        if not p.exists():
            cands = [q for q in d.glob(f"{s}_*.png") if re.fullmatch(re.escape(s) + r"_\d+", q.stem)]
            p = sorted(cands, key=lambda q: int(q.stem.rsplit("_", 1)[1]))[0] if cands else None
        if p is not None and p.exists():
            out[s] = np.asarray(Image.open(p).convert("RGB"), np.float64)
    return out


def stats(tiles):
    r = np.array([ratio(t) for t in tiles], float)
    tm = np.array([ts_raw(t)[0] for t in tiles], float)
    ta = np.array([ts_raw(t)[1] for t in tiles], float)
    fin = np.isfinite(r)
    return {"n": len(tiles), "n_nan_ratio": int((~fin).sum()),
            "ratio_median": float(np.median(r[fin])) if fin.any() else float("nan"),
            "dev_median": float(np.median(np.abs(r[fin] - 1))) if fin.any() else float("nan"),
            "ts_max_median": float(np.median(tm)), "ts_mean_median": float(np.median(ta))}


def pack_band(tiles, packs, rng):
    """按包自助：真人参照带（(M16) 规矩，瓦片级只当下界）。返回 |ratio-1| 中位数的 95% 区间。"""
    by = {}
    for t, p in zip(tiles, packs):
        by.setdefault(p, []).append(np.abs(ratio(t) - 1))
    keys = [k for k in by if np.isfinite(np.array(by[k], float)).any()]
    boot = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals = np.concatenate([np.array(by[keys[i]], float) for i in pick])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            boot.append(np.median(vals))
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--root", type=Path, default=ROOT / "experiments/baselines")
    ap.add_argument("--out", type=Path, default=Path("/tmp/seam_calib.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    items = load_set(a.set)[0]
    slugs = [e["material"].rsplit(".", 1)[0] for e in items]
    res, bad = {}, []

    # 真人参照带（split=test，按包）
    art = [s for s in load_tiles(a.size, "test")]
    at = [np.asarray(s["palette"][s["idx"]], np.float64) for s in art]
    ap_ = [s["pack"] for s in art]
    res["artist"] = stats(at)
    lo, hi, n_pk = pack_band(at, ap_, rng)
    res["artist"]["band_dev_95"] = [lo, hi]
    res["artist"]["n_packs"] = n_pk
    if len(at) < 200 or n_pk < 5:
        bad.append("OP1/OP3：真人参照瓦片或包数不足")

    for m in METHODS:
        d = load_dir(a.root / m / str(a.size), slugs)
        res[m] = stats(list(d.values()))
        if len(d) < 250:
            bad.append(f"OP1：{m} 只有 {len(d)} 张")

    # 反例与锚点（(OP2) 逐材质一一对应：都从同一张源图变换而来）
    b2 = load_dir(a.root / "B2" / str(a.size), slugs)
    for name, src, fn in [("A_flatborder", at, flat_border), ("A_blur", at, blur),
                          ("A_const", at, const),
                          ("A_roll_artist", at, None),
                          ("A_halfswap", list(b2.values()), halfswap)]:
        if fn is None:
            tiles = [np.roll(t, (int(rng.integers(t.shape[0])), int(rng.integers(t.shape[1]))), (0, 1))
                     for t in src]
        else:
            tiles = [fn(t) for t in src]
        res[name] = stats(tiles)

    # ---- 判据 ----
    best_dev = min(res[m]["dev_median"] for m in METHODS)
    best_ts = min(res[m]["ts_max_median"] for m in METHODS)
    d1 = {}
    for cf in ("A_flatborder", "A_const"):
        d1[cf] = {"beats_on_ratio": bool(res[cf]["dev_median"] <= best_dev)
                  if np.isfinite(res[cf]["dev_median"]) else None,
                  "beats_on_ts": bool(res[cf]["ts_max_median"] <= best_ts)}
    verdict1 = ("METRIC_NOT_A_CRITERION"
                if any(v["beats_on_ratio"] or v["beats_on_ts"] for v in d1.values())
                else "METRIC_SURVIVED_COUNTEREXAMPLES")
    lo, hi = res["artist"]["band_dev_95"]
    d2 = {m: ("IN_BAND" if lo <= res[m]["dev_median"] <= hi else
              ("BELOW_BAND_flat_direction" if res[m]["dev_median"] < lo else "ABOVE_BAND_visible_seam"))
          for m in METHODS}
    d3 = bool(res["A_halfswap"]["ts_max_median"] < res["B2"]["ts_max_median"])
    d4 = bool(abs(res["A_roll_artist"]["dev_median"] - res["artist"]["dev_median"]) < 0.02)
    out = {"set": a.set, "size": a.size, "seed": SEED, "B_boot": B_BOOT, "rows": res,
           "D1": {"verdict": verdict1, "detail": d1, "best_dev": best_dev, "best_ts_max": best_ts},
           "D2": d2, "D3_anchor_valid": d3, "D4_roll_invariant": d4, "ops_failed": bad}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'臂':<16}{'n':>5}{'ratio中位':>11}{'|r-1|':>9}{'TS_max':>9}{'TS_mean':>9}{'nan':>6}")
    for k, v in res.items():
        print(f"{k:<16}{v['n']:>5}{v['ratio_median']:>11.3f}{v['dev_median']:>9.3f}"
              f"{v['ts_max_median']:>9.3f}{v['ts_mean_median']:>9.3f}{v['n_nan_ratio']:>6}")
    print(f"\n真人参照带 |ratio-1| 95%（按 {n_pk} 个包自助）= [{lo:.3f}, {hi:.3f}]")
    print(f"(D1) {verdict1}   detail={d1}")
    print(f"(D2) {d2}")
    print(f"(D3) 好端锚点有效={d3}   (D4) 环移不变={d4}")
    if bad:
        print("⚠ 操作检验未过：" + "；".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
