#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M26) 预注册：(M25) 的结论句本身又是一条**没量过的断言**。零 GPU / 零 API / 零判官 / 零训练。

## 为什么问这个

(M25)（`6598207`）量完 32px 的梳齿之后，结尾写下这么一句：

> "一张只吃 d/n 的表表达不了『两档各有一支、其中一支像素周期固定』
>  → 『替换/削弱归一化谐波支』的提案没被判死，可以另行预注册。"

这句话是**下一次重训的全部动机**，而它和上一轮被检验的那句括号是**同一种东西**：
一条听起来显然、却一次都没被量过的断言。(M25) 自己记的通用教训是
"『统计量能分开两个假设』这个隐含前提本可以用一行代码先验证"——这里适用的是同一条。

**它为什么不是自动成立的**：`ToroidalBias` 的谐波支吃的是**归一化偏移** u = d/n
（`trd.py:88-95`），所以"一张表对两档都成立"的充要条件不是"两档的像素周期相同"，
而是"**两档在同一个 u 上要求同一个符号**"。两档都有整数 d 的共享 u 只有四个：

    u = 0.125 / 0.250 / 0.375 / 0.500   <->   16px d = 2 / 4 / 6 / 8
                                        <->   32px d = 4 / 8 / 12 / 16

(M25) 已量到 **32px 在这四个 u 上全是峰**（C = +0.010 / +0.052 / +0.011 / +0.081）。
于是"共用表不矛盾"的条件就化成一个**从没被读过的数**：**16px 在 u=0.375（d=6）上是不是峰**。

- 若 **C_16(6) < 0**（16px 是谷、32px 是峰）→ 同一个输入 u 要求相反的符号
  → 共用归一化表**确实**表达不了两档 → (M25) 那句话成立，提案的动机**第一次有了证据**。
- 若 **C_16(6) > 0** → 四个共享 u 上两档**同号**，共用表在符号层面**没有矛盾**
  → (M25) 那句话在符号层面**不成立**，⛔ 不许再拿它当重训的理由。

⚠ 本轮的价值同样是**不对称**的：判成 `U_CONSISTENT` 直接省掉一次重训。

## 量什么（仪器一个字不改）

`analysis/arch/period_scale.py` 的 `contrast()` / `boot_stat()` / `pack_index()` **直接 import**，
它们自己又 import 的是 `scale_prior.tile_curves`。口径与 (M25) 主判据**逐字相同**：
`split="train"`、`decontam=True`、`k_used>=3`、**`extra=False`（原生池）**、B=2000、seed=0、
重采样单位 = **包**（(M16) 纪律）。**测试集一张不看，不碰模型、不碰任何胜负数字。**

    局部对比 C_n(d) = A_hat(d) - [A_hat(d-1) + A_hat(d+1)] / 2      （不含阈值）

## 为什么主判据选 u=0.375 而不是 u=0.125（跑前写死的理由）

纯平滑衰减的凸性**只会把 C 压负**，而这个压低量在 A_hat 陡的地方最大。
u=0.125 在 16px 上是 **d=2**（A_hat 最陡的近邻区），在 32px 上是 d=4（已经平了）
→ 拿它当主判据等于让凸性**替我造出**想要的那个矛盾，**反保守**。
u=0.375 在两档上分别是 d=6 与 d=12，**都在曲线尾部**，凸性贡献接近 0。
→ **主判据只读 u=0.375**；u=0.125 只作**描述性副读数**并标注"受凸性混杂"。
u=0.5 是环面折回的**单边点**（(M25) 第六节：A_hat(n/2+1)=A_hat(n/2-1)，天然被抬高）
→ ⛔ **不进任何判据**，只报告。

## 识别检验 (ID)（(M25) 教训的直接落实；先读，不过 -> 主判决作废）

这把尺子必须**有能力报告"两档一致"**，否则"不一致"这个读数毫无信息
（例如两档 C 的整体尺度不同，就会让任何 u 都显得矛盾）。
u=0.25 是两档**公认的峰**（(M25) 表：+0.056 / +0.052）→ 要求

    (ID) C_16(4) 包级 CI 下界 > 0  **且**  C_32(8) 包级 CI 下界 > 0

即同一条代码路径在 u=0.25 上**报出"两档同为峰"**。不过 -> `ID_FAIL`，主判决作废。

## 预注册判决（跑之前写死，不许改）

主统计量 = **C_16(6)**，包级自助 95% 百分位区间。

- **(D1) `U_CONFLICT`**：CI 上界 < 0
  -> 共用归一化表在符号层面**确实**装不下两档。
  授权：①把这个读数写进 `trd.py` docstring 与账本；
  ②**允许另行预注册**"替换/削弱归一化谐波支"，且该预注册必须带识别检验。
  ⛔ 本身**不**授权任何配置改动、重训、新臂、默认值改动、判官调用。
- **(D2) `U_CONSISTENT`**：CI 下界 > 0
  -> 四个共享 u 上两档同号 -> (M25) 那句结论在**符号层面**不成立。
  ⛔ 今后不许再拿"一张 d/n 的表表达不了两档"当重训/改架构的理由。
  ⚠ **边界（跑前写死）**：本轮只判**符号**。两档在同一 u 上**幅度**不同（例如 32px 的
  +0.010 vs 16px 的某个更大的正值）**仍可能**构成错设，但那是**另一个统计量、另一次预注册**；
  ⛔ 不许把 `U_CONSISTENT` 读成"归一化表是对的"，也⛔ 不许读成 (M9) 被洗清（那讲衰减）。
- **(D3) 其余一律 `UNDECIDED`**，什么都不改，照实记账。⛔ 不许挑合上的那一半讲。

## 操作检验（先读，任一不过 -> 主判决作废，照实写"没测到"）

- **(OP1)** 两档各 >= `scale_prior.MIN_TILES` 张 k_used>=3 的训练瓦片。
- **(OP2) 逐字复现 (M25)**：C_16(4)=+0.056、C_16(8)=+0.073、C_32(8)=+0.052、C_32(16)=+0.081
  与 Q(D_odd={4,12})=+0.0104，容差 5e-4。任一对不上 = 仪器/口径漂了 -> 作废。
- **(OP3)** 主判据用到的 d 都是**内点**（16px 的 6 < 8，32px 的 12 < 16）。

## 聚集检验（(M16) 纪律，主判据之后必跑）

逐包读数 + HHI + LOPO（逐个去掉一个 **16px** 包重算主判决）。任一包能翻转 -> 加 `_FRAGILE`，
**只能引方向、⛔ 不许引数值当一般结论**。

## 数据可见性披露（诚实记账）

本轮用的曲线 (M25) 已经算过并落盘在 `experiments/period_scale.json` 的 `curve_16` 里，
所以数据在仓库中**是可得的**。我在写完并提交本文件之前**没有读过 C_16(2) / C_16(6)**
（(M25) 的账本与 docstring 只登了 d=4/8 两个值）。判据与主/副读数的分工**写在本文件里、
先于运行提交**——与 (M17)/(M19) 在已落盘判官 JSON 上预注册是同一做法。

用法：
    python analysis/arch/table_conflict.py --out experiments/table_conflict.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import tiles_data  # noqa: E402
from scale_prior import tile_curves, MIN_TILES  # noqa: E402
from period_scale import contrast, pooled, pack_index, boot_stat, B_BOOT, SEED  # noqa: E402

# 两档都有整数 d 的共享归一化偏移 u = d/n
SHARED_U = [(0.125, 2, 4), (0.250, 4, 8), (0.375, 6, 12), (0.500, 8, 16)]
U_MAIN = 0.375                      # 主判据（跑前写死，理由见 docstring）
U_DESC = 0.125                      # 描述性副读数（受凸性混杂）
U_EDGE = 0.500                      # 单边点，不进任何判据

# (OP2) (M25) 已发表读数，容差 5e-4
M25_EXPECT = {("C16", 4): +0.056, ("C16", 8): +0.073,
              ("C32", 8): +0.052, ("C32", 16): +0.081}
M25_Q_EXPECT = +0.0104              # Q over D_odd={4,12}
OP2_TOL = 5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/table_conflict.json"))
    a = ap.parse_args()

    res = {"b_boot": B_BOOT, "seed": SEED, "u_main": U_MAIN, "shared_u": SHARED_U}
    data = {}
    for n in (16, 32):
        s = [x for x in tiles_data.load(size=n, split="train", extra=False) if x["k_used"] >= 3]
        data[n] = {"s": s, "c": tile_curves(s, n), "packs": pack_index(s)}
        print(f"[{n}px] train tiles k_used>=3: {len(s)}  packs {len(data[n]['packs'])}", flush=True)
    res["n_tiles"] = {str(n): len(data[n]["s"]) for n in (16, 32)}
    res["n_packs"] = {str(n): len(data[n]["packs"]) for n in (16, 32)}

    # ---------------- 四个共享 u 上两档的 C 与包级 CI ----------------
    rows = []
    for u, d16, d32 in SHARED_U:
        r = {"u": u, "d16": d16, "d32": d32}
        for n, d in ((16, d16), (32, d32)):
            pt, lo, hi = boot_stat(data[n]["c"], data[n]["packs"],
                                   lambda c, d=d, n=n: contrast(c, d, n))
            r[f"C{n}"] = pt
            r[f"ci{n}"] = [lo, hi]
        rows.append(r)
    res["shared_table"] = rows
    print("\nu       16px d  C_16      CI                    | 32px d  C_32      CI")
    for r in rows:
        tag = "  <- main" if r["u"] == U_MAIN else ("  (edge, no criterion)" if r["u"] == U_EDGE
                                                   else ("  (convexity-confounded)" if r["u"] == U_DESC else ""))
        print(f"{r['u']:.3f}   d={r['d16']:<2}   {r['C16']:+.4f}  [{r['ci16'][0]:+.4f},{r['ci16'][1]:+.4f}]"
              f"  | d={r['d32']:<2}   {r['C32']:+.4f}  [{r['ci32'][0]:+.4f},{r['ci32'][1]:+.4f}]{tag}",
              flush=True)

    by_u = {r["u"]: r for r in rows}

    # ---------------- 操作检验 ----------------
    op1 = all(len(data[n]["s"]) >= MIN_TILES for n in (16, 32))
    c16p, c32p = pooled(data[16]["c"]), pooled(data[32]["c"])
    got = {("C16", 4): contrast(c16p, 4, 16), ("C16", 8): contrast(c16p, 8, 16),
           ("C32", 8): contrast(c32p, 8, 32), ("C32", 16): contrast(c32p, 16, 32)}
    q_now = float(np.mean([contrast(c32p, d, 32) for d in (4, 12)]))
    op2_items = {f"{k[0]}({k[1]})": {"got": float(v), "expect": M25_EXPECT[k],
                                     "ok": bool(abs(v - M25_EXPECT[k]) <= OP2_TOL)}
                 for k, v in got.items()}
    op2_items["Q(D_odd={4,12})"] = {"got": q_now, "expect": M25_Q_EXPECT,
                                    "ok": bool(abs(q_now - M25_Q_EXPECT) <= OP2_TOL)}
    op2 = all(v["ok"] for v in op2_items.values())
    op3 = by_u[U_MAIN]["d16"] < 8 and by_u[U_MAIN]["d32"] < 16
    res["op"] = {"op1_n": bool(op1), "op2_reproduce_M25": op2_items, "op2_pass": bool(op2),
                 "op3_interior": bool(op3)}
    print(f"\n(OP1) both n>={MIN_TILES}: {op1}")
    print("(OP2) reproduce (M25):")
    for k, v in op2_items.items():
        print(f"      {k:<18} got {v['got']:+.4f}  expect {v['expect']:+.4f}  {'ok' if v['ok'] else 'MISMATCH'}")
    print(f"      -> {'pass' if op2 else 'FAIL => void'}")
    print(f"(OP3) main-criterion offsets are interior points: {op3}", flush=True)

    # ---------------- 识别检验 (ID) ----------------
    idr = by_u[0.250]
    id_ok = idr["ci16"][0] > 0 and idr["ci32"][0] > 0
    res["identification"] = {"u": 0.250, "C16": idr["C16"], "ci16": idr["ci16"],
                             "C32": idr["C32"], "ci32": idr["ci32"], "pass": bool(id_ok)}
    print(f"\n(ID) at u=0.250 the same code path reports BOTH canvases as peaks: "
          f"{'pass' if id_ok else 'FAIL => ID_FAIL, main verdict void'}", flush=True)

    # ---------------- 主判决 ----------------
    m = by_u[U_MAIN]
    lo16, hi16 = m["ci16"]

    def verdict_from(lo, hi):
        if hi < 0:
            return "U_CONFLICT"
        if lo > 0:
            return "U_CONSISTENT"
        return "UNDECIDED"

    v = verdict_from(lo16, hi16)
    res["main"] = {"stat": f"C_16({m['d16']}) at u={U_MAIN}", "point": m["C16"],
                   "ci": [lo16, hi16], "C32_same_u": m["C32"], "ci32": m["ci32"]}
    if not (op1 and op2 and op3):
        v = "OP_FAIL"
    elif not id_ok:
        v = "ID_FAIL"
    res["verdict_main"] = v
    print(f"\nMAIN  C_16({m['d16']}) = {m['C16']:+.4f}  pack CI [{lo16:+.4f}, {hi16:+.4f}]"
          f"   (32px same u: {m['C32']:+.4f})  -> {v}", flush=True)

    # ---------------- 描述性副读数：u=0.125（受凸性混杂） ----------------
    d = by_u[U_DESC]
    res["descriptive_u0125"] = {"C16": d["C16"], "ci16": d["ci16"],
                                "C32": d["C32"], "ci32": d["ci32"],
                                "note": "convexity-confounded: 16px d=2 sits in the steep part of A_hat"}

    # ---------------- (C) 聚集：16px 逐包 + HHI + LOPO ----------------
    pk16, c16 = data[16]["packs"], data[16]["c"]
    f_main = lambda c: contrast(c, m["d16"], 16)            # noqa: E731
    tbl = sorted(({"pack": p, "n": int(ids.size), "C": float(f_main(pooled(c16, ids)))}
                  for p, ids in pk16.items()), key=lambda r: -r["n"])
    res["by_pack_16"] = tbl
    tot = sum(r["n"] for r in tbl)
    res["pack_hhi_16"] = float(sum((r["n"] / tot) ** 2 for r in tbl))
    print(f"\n(C) 16px by pack: {len(tbl)} packs, HHI={res['pack_hhi_16']:.3f} (top 8)")
    for r in tbl[:8]:
        print(f"  {str(r['pack'])[:44]:<44} n={r['n']:>5}  C_16({m['d16']})={r['C']:+.4f}")

    flips = []
    if v in ("U_CONFLICT", "U_CONSISTENT"):
        for p in pk16:
            sub = {k: ids for k, ids in pk16.items() if k != p}
            if not sub:
                continue
            _, lo2, hi2 = boot_stat(c16, sub, f_main)
            if verdict_from(lo2, hi2) != v:
                flips.append(p)
    res["lopo_flips_16"] = flips
    res["verdict"] = v + ("_FRAGILE" if flips and v in ("U_CONFLICT", "U_CONSISTENT") else "")
    print(f"\nLOPO (16px): packs that flip the main verdict = {flips if flips else 'none'}")
    print(f"\nVERDICT: {res['verdict']}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
