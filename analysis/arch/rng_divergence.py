#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M30) 预注册：(M29) 的守门差值里，有没有混进"两次独立训练"这一项？
零 GPU / 零 API / 零判官 / 零训练 / 零数据（只构造模型、只抽随机数）。

## 为什么问这个

(M29)（`4cdab69`）判 `GATE_FAILED`：`--bias_size_cond` 新臂在 16px 上 KID 8.316 vs 控制臂
v11d 6.198，差 +2.118 > 门槛 +1.0。账本当轮写死的下一步是一道硬门：

> 下一个候选必须先解释**为什么 16px 会被带着动**（本轮 16px 只改了偏置表的调制、KID 却动了 2.1）
> —— 不许直接换个解绑形式再试一遍。

本轮就是去答这一问。**配方差异已排除**：两个 `config.json` 逐键比对，除
`bias_size_cond` 与三个后加的默认值（`bias_cells=[]` / `p_tile16=0.0` / `pack_balance=0.0`）外
完全相同——同 `init_from`（`runs/trd_v10/last.pt`）、同 12000 步、同 batch 256、同 p32 0.7。

于是只剩两个竞争解释：
  (A) FiLM 确实改动了 16px 那张表 → 16px 样本质量真的变了；
  (B) 与 FiLM 无关：两臂其实是**两次独立训练**（看到的数据顺序不同），
      而"同配置重训"的漂移量本项目**从未标定过**
      （记忆里已有一条：`eval/noise_floor.py` 只量采样噪声、**不含训练随机性**）。

(B) 有一个**可以一行代码证伪**的机制：`model/train_trd.py:273` 只 `torch.manual_seed(0)` 一次，
模型在那之后构造；而训练循环的取批用的是 **CPU 全局 RNG**（`:504` 的 `torch.rand(())` 决定
这一步喂 16 还是 32、`:514` 的 `torch.randint(0, nD, (bs,))` 取样本下标）。
`ToroidalBias.__init__`（`model/trd.py:130`）在 `size_cond=True` 时多构造一个
`nn.Linear(1, 2*hidden)` —— 它虽然**随后被置零**（`:132-133`），但 `nn.Linear` 的**构造**
本身已经从 CPU 流里抽走了随机数。只要抽走的数不为零，两臂进入训练循环时的 RNG 状态就不同
→ **从第 0 步起看到的就是不同的数据**。

⚠ 这与 (S1) 不矛盾：(S1) 证的是"**第 0 步的偏置函数**逐元素相同"（零初始化保留），
本轮问的是"**第 0 步之后喂进去的数据**是否相同"。两者是不同的东西。

## 量什么

在**同一颗种子**下各构造一次模型，然后**紧接着**按训练循环的调用次序抽 `STEPS` 步的
`use32` 硬币与批次下标，逐位比对。除模型构造外两条路径逐字相同 → 任何差异只能来自构造。

## 识别检验（先跑；不过 → 主判决作废）

(ID) **同一份配置连抽两次**（A vs A）必须**逐位相同**。
这把尺子必须有能力报"没有分叉"，否则"分叉了"这个读数毫无信息。

## 预注册判决（跑之前写死，不许改）

- **(D1) `RNG_DIVERGES`**：A 与 B 的抽样序列不逐位相同
  → v11d 与 scond 是**两次独立训练**；(M29) 的 +2.118 里**必然**混有同配置重训漂移，
  而该漂移量从未被标定 → 解释 (A) **未被确立**，(B) **未被排除**。
  授权：①把这条写进账本与 `trd.py` 的相关注释；
  ②**今后任何"改架构 + 重训 + 对着控制臂守门"的预注册，必须先给出同配置重训漂移的标定**
  （或把守门画在该漂移之外）。
  ⛔ **不授权**：重开 (M29)（判决照旧生效）、改任何默认值、改 `final_test.sh`、
  任何新臂、任何判官调用、任何 32px 准入条件的放宽。
- **(D2) `RNG_IDENTICAL`**：逐位相同
  → 数据顺序不是解释；+2.118 只能来自 FiLM 本身（加 GPU 非确定性）→ 解释 (A) 站得住，
  账本那道硬门就**已经被答**，下一个候选可以直接往"为什么调制 16px 会伤 16px"走。

⚠ **两种判决都不给出 2.118 的归因**：本轮**不测量**漂移有多大（那要另起一次同配置重训），
只判"这一项在不在差值里"。⛔ 不许把 `RNG_DIVERGES` 读成"所以 (M29) 其实通过了"。

用法：
    python analysis/arch/rng_divergence.py --a /tmp/m30/v11d.json --b /tmp/m30/scond.json \
        --out experiments/rng_divergence.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.train_trd import model_from_args  # noqa: E402

STEPS = 200          # 抽多少步；跑前写死
P32_FALLBACK = 0.7
NDATA = 4096         # 训练集大小的替身：只要两臂用同一个值，比的就是 RNG 流本身


def draw(cfg, steps=STEPS, seed=0):
    """同一颗种子 -> 构造模型 -> 按训练循环的调用次序抽 steps 步。返回 (参数量, 抽样序列)。"""
    torch.manual_seed(seed)
    model = model_from_args(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    p32 = float(cfg.get("p32", P32_FALLBACK))
    bs = int(cfg.get("batch", 256))
    seq = []
    for _ in range(steps):
        use32 = torch.rand(()).item() < p32          # train_trd.py:504
        idx = torch.randint(0, NDATA, (bs,))         # train_trd.py:514
        seq.append((bool(use32), idx))
    return n_par, seq


def first_diff(sa, sb):
    """返回 (第一处分叉的步号, use32 不同的步数, 批次下标不同的步数)；无分叉返回 (None, 0, 0)."""
    first, n_coin, n_idx = None, 0, 0
    for i, ((ca, ia), (cb, ib)) in enumerate(zip(sa, sb)):
        d_coin = ca != cb
        d_idx = not torch.equal(ia, ib)
        if d_coin:
            n_coin += 1
        if d_idx:
            n_idx += 1
        if (d_coin or d_idx) and first is None:
            first = i
    return first, n_coin, n_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="控制臂 config.json（v11d）")
    ap.add_argument("--b", type=Path, required=True, help="新臂 config.json（scond）")
    ap.add_argument("--out", type=Path, required=True)
    g = ap.parse_args()

    ca = json.loads(g.a.read_text(encoding="utf-8"))
    cb = json.loads(g.b.read_text(encoding="utf-8"))

    # 先把"配方差异已排除"这条也当场复核一遍，别只信上一轮的眼睛
    IGNORE = {"out"}
    diff_keys = sorted(k for k in set(ca) | set(cb)
                       if k not in IGNORE and ca.get(k, "<missing>") != cb.get(k, "<missing>"))
    cfg_diff = {k: [ca.get(k, "<missing>"), cb.get(k, "<missing>")] for k in diff_keys}

    # (ID) 识别检验：同一份配置连抽两次必须逐位相同
    _, a1 = draw(ca)
    _, a2 = draw(ca)
    id_first, id_coin, id_idx = first_diff(a1, a2)
    id_pass = id_first is None
    print(f"(ID) A vs A: 分叉步={id_first} coin_diff={id_coin} idx_diff={id_idx} -> "
          f"{'PASS' if id_pass else 'FAIL'}", flush=True)

    npa, sa = draw(ca)
    npb, sb = draw(cb)
    first, n_coin, n_idx = first_diff(sa, sb)
    diverges = first is not None

    if not id_pass:
        verdict = "ID_FAIL"
    else:
        verdict = "RNG_DIVERGES" if diverges else "RNG_IDENTICAL"

    res = {
        "steps": STEPS, "seed": 0, "n_data_stub": NDATA,
        "config_diff_excluding_out": cfg_diff,
        "params_a": npa, "params_b": npb, "params_delta": npb - npa,
        "id_check": {"pass": id_pass, "first_diff_step": id_first},
        "first_diff_step": first,
        "steps_with_diff_use32": n_coin,
        "steps_with_diff_batch_idx": n_idx,
        "verdict": verdict,
    }
    g.out.parent.mkdir(parents=True, exist_ok=True)
    g.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"配置差异（除 out）：{cfg_diff}")
    print(f"参数量 A={npa} B={npb} 差={npb - npa}")
    print(f"A vs B: 第一处分叉步={first}；{STEPS} 步里 use32 不同 {n_coin} 步、批次下标不同 {n_idx} 步")
    print(f"判决：{verdict}")
    print(f"写入 {g.out}")


if __name__ == "__main__":
    main()
