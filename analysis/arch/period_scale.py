#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M25) 预注册：`ToroidalBias` 谐波支那句"周期性结构确实随画布缩放"从来没被量过。零 GPU、零 API、零判官。

## 为什么问这个

`model/trd.py::ToroidalBias` 的谐波支吃的是**归一化偏移** `d/n`（`trd.py:59-60`），
也就是说它的频率单位是**"每瓦片多少周"**，而且 16px 与 32px **共用同一张表**。
(M9)（`4e965b8`）已经用相关长度 L 指出这张共用表在**衰减/局部性**那一面是错设的；
但 `trd.py` 的 docstring 里还留着一句**与之相反、且一次都没被测量过**的括号断言：

> "谐波那一支原样保留（周期性结构确实随画布缩放：16px 的自相关峰在 d=4/8，32px 在 d=8/16）"

这句话是**整个"替换/削弱归一化谐波支"提案的前提**：如果它是真的，谐波支的坐标在周期那一面
没毛病，提案只剩衰减支可动；如果它是假的（32px 的自相关峰仍在 **4 个像素**上），
那么共用表在**两面都**错设——两档要求同一个输入 `d/n` 给出不同的值。

⚠ 本轮的价值是**不对称**的：判成 `P_CANVAS` 会**省掉一次重训**（把提案的动机判死），
判成 `P_CELL` 也只是让提案"值得另行预注册"，**本轮一个配置都不改**。

**这是零 GPU、零 API 的纯数据测量**：只看训练集真人瓦片自己，不碰模型、不碰测试集、
不看任何胜负数字。

## 量什么

仪器 = `analysis/arch/scale_prior.py::tile_curves`，**直接 import，一个字不改**：
每张瓦片的秩网格上量"隔 d 格同色阶"的超机遇比例 Â_n(d)，d=1..n/2，池化取均值。
环面对称：Â_n(d) = Â_n(n-d)，故 d>n/2 一律折回。

新统计量（**不是** L，不含任何阈值 → 不受 (M12) 那个悬崖统计量的问题影响）：

    **局部对比** C_n(d) = Â_n(d) - [Â_n(d-1) + Â_n(d+1)] / 2

梳齿（周期结构）在它的周期 p 的整数倍上是**峰**（C>0），在半周期处是**谷**（C<0）；
纯平滑衰减的曲线因为凸性给出**略负**的 C。

- 先在 **16px 池**上定出真人的像素周期：**p16 := argmax_{d=2..8} C_16(d)**（数据定，不是我挑的）。
- 再看 **32px 池**上 p16 的**奇数倍**（32px 上仍是峰 ⟺ 像素周期不变）：
  D_odd = {m·p16 : m 奇数, m·p16 <= 16}（p16=4 时 = {4, 12}）。
  - **(H-cell) 像素周期固定** → 32px 上 d=4 仍是峰 → C_32(4) > 0
  - **(H-canvas) 周期随画布缩放**（周期变 8）→ d=4 恰是**半周期 = 谷** → C_32(4) < 0
  两个假设在这个统计量上**符号相反**，不需要任何基线校正。
- 画布侧的对照分数：Q_canvas = mean C_32(d)，d ∈ D_canvas = {m·2p16 <= 16} = {8, 16}。

主统计量：**Q = mean_{d ∈ D_odd} C_32(d)**。

## 预注册判据（跑之前写死，不许改）

重采样单位 = **包**（(M16) 纪律：真人语料一律按包自助），B=2000，seed=0，95% 百分位区间。

- **(D1) `P_CELL`**：Q 的 CI 下界 > 0
  → 32px 真人瓦片上仍是**同一个像素周期**的梳齿 → 共用归一化表在**周期支上也是错设的**，
  `trd.py` 那句括号断言是**假的**。
  授权：①改掉 `trd.py` 里那句断言；②记一条候选，并**允许另行预注册**"替换/削弱归一化谐波支"。
  ⛔ 本身**不**授权任何配置改动、重训、新臂、默认值改动。
- **(D2) `P_CANVAS`**：Q 的 CI 上界 < 0 **且** Q_canvas 的 CI 下界 > 0
  （凸性只能把 C 压负，造不出正的梳齿分数 → 这一条不会被凸性伪造）
  → 那句断言在周期那一面**成立** → "替换谐波支"提案**在周期动机上判死**，
  ⛔ 今后不许再拿"周期也不随画布缩放"当理由。
  ⚠ **这不洗清 (M9)**：(M9) 说的是**衰减/局部性**那一面（相关长度 L），与周期是两回事。
- **(D3) 其余一律 `UNDECIDED`**，不改任何东西，照实记账。⛔ 不许挑合上的那一半讲。

## 操作检验（先读，任一不过 → 主判决作废，照实写"没测到"）

- **(OP1)** 两档各 >= 300 张 k_used>=3 的训练瓦片（沿用 `scale_prior.MIN_TILES`）。
- **(OP2) 正对照**：16px 池上 C_16(p16) 的包级 CI 下界 > 0
  （＝这把尺子确实能看见梳齿，且 16px 真的有周期结构）。不过 → `NO_COMB_16`，作废。
- **(OP3)** 2·p16 <= 16（两个竞争周期在 32px 的量程内都量得到）。

## 聚集检验（(M16) 纪律，主判据之后必跑）

逐包读数 + HHI；LOPO（逐个去掉一个 32px 包重算主判决）。任一包能翻转 → 判决加 **`_FRAGILE`**，
**只能引方向、⛔ 不许引 Q 的数值当"真人 32px 语料"的一般结论**（32px 只有 ~2.7 个有效包）。

## 口径

`split="train"`、`decontam=True`、`k_used>=3`；主判据用 **`extra=False`（原生瓦片）**——
理由跑前写死：`train_64to32.json` 是 64px **降采样两倍**得来的，它的像素周期是我们预处理的产物、
不是真人 32px 画法，混进来正好污染本轮要量的那个量。
实际训练池（`extra=packs_only+64to32`）只作**描述性**副读数，**不作判据**、不许用来翻主判决。
**测试集一张不看。**

用法：
    python analysis/arch/period_scale.py --out experiments/period_scale.json
    python analysis/arch/period_scale.py --extra train_extra_packs_only.json+train_64to32.json \
        --out /tmp/period_scale_extra.json      # 仅描述性副读数，须在服务器上跑
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import tiles_data  # noqa: E402
from scale_prior import tile_curves, MIN_TILES  # noqa: E402   仪器原样复用

B_BOOT = 2000
SEED = 0


def a_hat(curve, d, n):
    """Â_n(d)，环面折回：d 与 n-d 等价；curve[j] 对应 d=j+1。"""
    d = d % n
    if d > n // 2:
        d = n - d
    return curve[d - 1] if d >= 1 else 1.0


def contrast(curve, d, n):
    """局部对比 C_n(d) = Â(d) - [Â(d-1) + Â(d+1)]/2。"""
    return a_hat(curve, d, n) - 0.5 * (a_hat(curve, d - 1, n) + a_hat(curve, d + 1, n))


def comb_sets(p16):
    """D_odd（像素周期不变才是峰）、D_canvas（周期翻倍的峰）、D_null（描述性基线）。"""
    d_odd = [m * p16 for m in range(1, 17) if m % 2 == 1 and m * p16 <= 16]
    d_canvas = [m * 2 * p16 for m in range(1, 17) if m * 2 * p16 <= 16]
    comb = set(d_odd) | set(d_canvas)
    d_null = [d for d in range(2, 17) if d not in comb and d % p16 != 0]
    return d_odd, d_canvas, d_null


def pooled(curves, idx=None):
    return curves[idx].mean(0) if idx is not None else curves.mean(0)


def pack_index(samples):
    by = defaultdict(list)
    for i, s in enumerate(samples):
        by[s["pack"]].append(i)
    return {p: np.array(v) for p, v in by.items()}


def boot_stat(curves, packs, fn, b=B_BOOT, seed=SEED):
    """按包自助：fn(池化曲线) -> 标量。返回 (点估计, lo, hi)。"""
    keys = list(packs)
    rng = np.random.default_rng(seed)
    point = fn(pooled(curves))
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([packs[keys[i]] for i in pick])
        vals.append(fn(pooled(curves, idx)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def verdict_of(q_ci, qc_ci):
    if q_ci[1] > 0:
        return "P_CELL"
    if q_ci[2] < 0 and qc_ci[1] > 0:
        return "P_CANVAS"
    return "UNDECIDED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default=False, help="默认 False=原生瓦片（主判据口径）")
    ap.add_argument("--out", type=Path, default=Path("experiments/period_scale.json"))
    a = ap.parse_args()

    res = {"extra": a.extra, "b_boot": B_BOOT, "seed": SEED}
    data = {}
    for n in (16, 32):
        s = [x for x in tiles_data.load(size=n, split="train", extra=a.extra) if x["k_used"] >= 3]
        data[n] = {"s": s, "c": tile_curves(s, n), "packs": pack_index(s)}
        print(f"[{n}px] 训练瓦片 k_used>=3: {len(s)}  包 {len(data[n]['packs'])}", flush=True)
    res["n_tiles"] = {str(n): len(data[n]["s"]) for n in (16, 32)}
    res["n_packs"] = {str(n): len(data[n]["packs"]) for n in (16, 32)}
    for n in (16, 32):
        res[f"curve_{n}"] = pooled(data[n]["c"]).tolist()
        print(f"[{n}px] Â(d) =", " ".join(f"{v:.3f}" for v in pooled(data[n]['c'])))
        print(f"[{n}px] C(d) =", " ".join(f"{contrast(pooled(data[n]['c']), d, n):+.3f}"
                                          for d in range(2, n // 2 + 1)), flush=True)

    # ---------------- p16：16px 池上数据定出的像素周期 ----------------
    c16 = pooled(data[16]["c"])
    p16 = int(max(range(2, 9), key=lambda d: contrast(c16, d, 16)))
    d_odd, d_canvas, d_null = comb_sets(p16)
    res.update({"p16": p16, "d_odd": d_odd, "d_canvas": d_canvas, "d_null": d_null})
    print(f"\np16 = argmax C_16(d) = {p16}  ->  D_odd={d_odd}  D_canvas={d_canvas}")

    # ---------------- 操作检验 ----------------
    op1 = all(len(data[n]["s"]) >= MIN_TILES for n in (16, 32))
    pc = boot_stat(data[16]["c"], data[16]["packs"], lambda c: contrast(c, p16, 16))
    op2 = pc[1] > 0
    op3 = 2 * p16 <= 16
    res["op"] = {"op1_n": bool(op1), "op2_pos_control": {"C16_p16": pc[0], "ci": [pc[1], pc[2]],
                                                         "pass": bool(op2)},
                 "op3_range": bool(op3)}
    print(f"(OP1) 两档 n>={MIN_TILES}: {op1}")
    print(f"(OP2) 正对照 C_16({p16}) = {pc[0]:+.4f} 包级 CI [{pc[1]:+.4f}, {pc[2]:+.4f}] -> "
          f"{'过' if op2 else '不过 => NO_COMB_16，作废'}")
    print(f"(OP3) 2*p16<=16: {op3}", flush=True)

    # ---------------- 主判据：32px 上 D_odd 的局部对比 ----------------
    c32, pk32 = data[32]["c"], data[32]["packs"]
    f_q = lambda c: float(np.mean([contrast(c, d, 32) for d in d_odd]))          # noqa: E731
    f_qc = lambda c: float(np.mean([contrast(c, d, 32) for d in d_canvas]))      # noqa: E731
    f_qn = lambda c: float(np.mean([contrast(c, d, 32) for d in d_null]))        # noqa: E731
    q, qc, qn = (boot_stat(c32, pk32, f) for f in (f_q, f_qc, f_qn))
    res["Q"] = {"point": q[0], "ci": [q[1], q[2]]}
    res["Q_canvas"] = {"point": qc[0], "ci": [qc[1], qc[2]]}
    res["Q_null_descriptive"] = {"point": qn[0], "ci": [qn[1], qn[2]]}
    res["C32_by_d"] = {str(d): contrast(pooled(c32), d, 32) for d in range(2, 17)}
    print(f"\nQ        (D_odd={d_odd})    = {q[0]:+.4f}  包级 CI [{q[1]:+.4f}, {q[2]:+.4f}]")
    print(f"Q_canvas (D_canvas={d_canvas}) = {qc[0]:+.4f}  包级 CI [{qc[1]:+.4f}, {qc[2]:+.4f}]")
    print(f"Q_null   (描述性, d={d_null}) = {qn[0]:+.4f}  CI [{qn[1]:+.4f}, {qn[2]:+.4f}]", flush=True)

    v = verdict_of(q, qc)
    if not (op1 and op2 and op3):
        v = "NO_COMB_16" if not op2 else "OP_FAIL"
    res["verdict_main"] = v

    # ---------------- (C) 聚集：逐包 + HHI + LOPO ----------------
    packs_tbl = []
    for p, ids in sorted(pk32.items(), key=lambda kv: -len(kv[1])):
        packs_tbl.append({"pack": p, "n": int(ids.size),
                          "Q": f_q(pooled(c32, ids)), "Q_canvas": f_qc(pooled(c32, ids))})
    res["by_pack_32"] = packs_tbl
    tot = sum(p["n"] for p in packs_tbl)
    res["pack_hhi_32"] = float(sum((p["n"] / tot) ** 2 for p in packs_tbl))
    print(f"\n(C) 32px 逐包：{len(packs_tbl)} 个包，HHI={res['pack_hhi_32']:.3f}（前 8）")
    for p in packs_tbl[:8]:
        print(f"  {str(p['pack'])[:44]:<44} n={p['n']:>5}  Q={p['Q']:+.4f}  Qc={p['Q_canvas']:+.4f}")

    flips = []
    if v in ("P_CELL", "P_CANVAS"):
        for p in pk32:
            keep = np.concatenate([ids for k, ids in pk32.items() if k != p]) if len(pk32) > 1 else None
            if keep is None or keep.size == 0:
                continue
            sub_c, sub_pk = c32, {k: ids for k, ids in pk32.items() if k != p}
            # 重编号：boot_stat 直接吃原曲线 + 去掉该包的包索引
            q2 = boot_stat(sub_c, sub_pk, f_q)
            qc2 = boot_stat(sub_c, sub_pk, f_qc)
            if verdict_of(q2, qc2) != v:
                flips.append(p)
    res["lopo_flips"] = flips
    res["verdict"] = v + ("_FRAGILE" if flips and v in ("P_CELL", "P_CANVAS") else "")
    print(f"\nLOPO：去掉任一包会翻转主判决的包 = {flips if flips else '无'}")
    print(f"\n判决：{res['verdict']}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
