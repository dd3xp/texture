#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M32) 预注册：把「成对设计」造出来，并当场验证它真的成对。
零 GPU / 零 API / 零判官 / 零训练 / 零新臂（本机 CPU：只构造模型、只抽随机数、只比权重）。

## 为什么是这件事

(M30)（`3ceda7f`）判 `RNG_DIVERGES`，授权栏只写了一条：

> 今后任何"改架构 + 重训 + 对着控制臂守门"的预注册，**必须先给出同配置重训漂移的标定，
> 或把守门画在该漂移之外**。

并列出了两条**均未实施**的路：①模型构造之后再 `torch.manual_seed()` 一次；②给取批单开
`torch.Generator`。(M31)（在跑）走的是"标定漂移"；本轮走另一半 —— **把漂移这一项直接消掉**。
两者互补：成对设计成立的话，今后的架构臂不必再向 Dmax 让出预算。

⛔ 本轮**不开任何架构新臂**（(M31) 出结果前的禁令照旧），不碰 (M29)/(M31) 的任何判决，
不改任何默认值。⛔ 且**在 `arch_drift` 跑完前不把 `train_trd.py` scp 到远程**
——(M31) 的 seed 2 还没启动，会去读远程那一份；虽然改动默认关、旧路径逐字节不变，
但"跑着的标定"不接受任何"在原理上无害"的扰动。

## 改动（本文件之外唯一的改动）

`model/train_trd.py`：新增 `--reseed_after_build`（**默认关 → 旧路径逐字节不变**）与
`seed_for_training(seed, enabled)`；后者在**训练循环入口**（`t0/n` 之后）被调用一次。
本脚本**直接 import 这个函数**来做检验 —— 不镜像、不复刻，测的就是交付物本身。

## 量什么（同一颗种子下，A = 控制臂、B = 只翻一个架构开关）

- **(ID) 识别检验**（先跑；不过 → 全部判决作废）：同一份配置连抽两次必须逐位相同。
- **(P1)** 开 `--reseed_after_build`：A/B 两臂 `STEPS` 步的 `use32` 硬币与批次下标逐位相同？
- **(P2) 阳性对照**：关它 → 必须仍然分叉（否则这把尺子看不见 (M30) 已证的东西，(P1) 无意义）。
- **(P3)** 两臂**共享权重**（除新模块外的同名参数）在构造时是否逐位相同？
  —— 新模块会把它之后构造的所有模块的初始化整体移位，预期 **不同**。
- **(P4)** 模拟 `--init_from`：把 A 的 `state_dict` 以 `strict=False` 载入 B（这正是
  `train_trd.py:351-368` 对源检查点做的事）后，共享权重是否逐位相同、缺的键是不是只有新模块、
  且新模块是不是全零。
- **(P5)** 两臂的 `nn.Dropout` 模块数与 `p` 是否一致 —— 每步前向消耗的随机数一样多，
  (P1) 的对齐才不会在第 1 步之后散掉。

## 预注册判决（跑之前写死，不许改）

- **`PAIRED_OK`**：(ID) 过、(P1) 相同、(P2) 仍分叉、(P4) 相同且缺键只有新模块、(P5) 一致。
  → 成对设计的配方 = **`--init_from` + `--reseed_after_build`**，且**必须同时具备**（(P3) 说明
  少了 `--init_from` 就不成对）。授权：今后"改架构 + 重训"的预注册**可以**用这个配方替代
  "把守门画在 (M31) 的 Dmax 之外"；⛔ 但**必须在该轮预注册里写明已跑过本脚本并附读数**
  （每个新模块都要重跑一次：移位量与 RNG 消耗都因模块而异）。
- **`PAIRED_NEEDS_CKPT`**：(P1)(P2)(P5) 如上，但 (P4) 不同 → 只授权"数据顺序成对"，
  权重侧仍不成对 → 守门仍须画在 Dmax 之外。
- **`PAIRED_FAILS`**：(P1) 仍分叉 → 改动无效，`--reseed_after_build` 作废、默认留关。
- **`ID_FAIL` / `CONTROL_FAIL`**：(ID) 或 (P2) 不过 → 尺子不可信，本轮什么都没测到。

⛔ **本轮不授权**：重开 (M29)（`GATE_FAILED` 一个字不改）、改任何默认值、改 `final_test.sh`、
任何新臂、任何判官调用、任何 32px 准入条件的放宽、任何对 (M31) 的干预。
⚠ **(P1) 只保证 CPU 流对齐**。远程 `dev=cuda`：增广与 dropout 走 CUDA 流，`nn.Linear` 的构造
**不消耗** CUDA 流 → 那条流本就对齐，(P5) 一致时保持对齐。但若今后某个新模块在**前向**里
消耗随机数（新的 dropout/采样），两条流都会散 → 那一轮必须重跑 (P5)。

用法：
    python analysis/arch/paired_rng.py --cfg remote_tmp/runs/trd_scond_09161230/config.json \
        --flag bias_size_cond --out experiments/paired_rng.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.train_trd import model_from_args, seed_for_training  # noqa: E402

STEPS = 200          # 抽多少步；跑前写死（与 (M30) 同）
NDATA = 4096         # 训练集大小的替身：两臂用同一个值，比的是 RNG 流本身
SEED = 0


def build(cfg, reseed):
    """同一颗种子 -> 构造模型 -> （可选）再播一次种。返回模型。"""
    torch.manual_seed(SEED)
    model = model_from_args(cfg)
    seed_for_training(SEED, reseed)          # 交付物本身，不是镜像
    return model


def draw(cfg, reseed, steps=STEPS):
    """构造模型后，按训练循环的调用次序抽 steps 步。返回 (模型, 抽样序列)。"""
    model = build(cfg, reseed)
    p32 = float(cfg.get("p32", 0.7))
    bs = int(cfg.get("batch", 256))
    seq = []
    for _ in range(steps):
        use32 = torch.rand(()).item() < p32          # train_trd.py 的 use32 硬币
        idx = torch.randint(0, NDATA, (bs,))         # train_trd.py 的批次下标
        seq.append((bool(use32), idx))
    return model, seq


def first_diff(sa, sb):
    """(第一处分叉步, use32 不同的步数, 批次下标不同的步数)；无分叉 -> (None, 0, 0)."""
    first, n_coin, n_idx = None, 0, 0
    for i, ((ca, ia), (cb, ib)) in enumerate(zip(sa, sb)):
        d_coin, d_idx = ca != cb, not torch.equal(ia, ib)
        n_coin += d_coin
        n_idx += d_idx
        if (d_coin or d_idx) and first is None:
            first = i
    return first, int(n_coin), int(n_idx)


def shared_param_diff(ma, mb):
    """两模型同名同形参数里有多少个不逐位相同；返回 (不同的个数, 共享的个数, 前几个名字)。"""
    sa, sb = ma.state_dict(), mb.state_dict()
    shared = [k for k in sa if k in sb and sa[k].shape == sb[k].shape]
    bad = [k for k in shared if not torch.equal(sa[k], sb[k])]
    return len(bad), len(shared), bad[:5]


def dropout_sig(m):
    return sorted(float(d.p) for d in m.modules() if isinstance(d, nn.Dropout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True, help="一条真实臂的 config.json")
    ap.add_argument("--flag", default="bias_size_cond", help="要翻的架构开关（B 臂开、A 臂关）")
    ap.add_argument("--out", type=Path, required=True)
    g = ap.parse_args()

    base = json.loads(g.cfg.read_text(encoding="utf-8"))
    ca = dict(base); ca[g.flag] = False          # A = 控制臂
    cb = dict(base); cb[g.flag] = True           # B = 新臂

    # (ID) 同一份配置连抽两次必须逐位相同
    _, a1 = draw(ca, False)
    _, a2 = draw(ca, False)
    id_first = first_diff(a1, a2)[0]
    id_pass = id_first is None

    # (P2) 阳性对照：不开 reseed 必须仍然分叉（(M30) 已证）
    ma_off, sa_off = draw(ca, False)
    mb_off, sb_off = draw(cb, False)
    off_first, off_coin, off_idx = first_diff(sa_off, sb_off)
    control_pass = off_first is not None

    # (P1) 开 reseed：必须逐位相同
    ma_on, sa_on = draw(ca, True)
    mb_on, sb_on = draw(cb, True)
    on_first, on_coin, on_idx = first_diff(sa_on, sb_on)
    p1_same = on_first is None

    # (P3) 构造时的共享权重
    n_bad3, n_shared3, names3 = shared_param_diff(ma_on, mb_on)

    # (P4) 模拟 --init_from：把 A 的 state_dict 载入 B
    missing, unexpected = mb_on.load_state_dict(ma_on.state_dict(), strict=False)
    n_bad4, n_shared4, names4 = shared_param_diff(ma_on, mb_on)
    new_all_zero = all(float(dict(mb_on.state_dict())[k].abs().max()) == 0.0 for k in missing) if missing else None
    only_new = bool(missing) and not unexpected

    # (P5) 前向的随机数消耗：dropout 结构必须一致
    da, db = dropout_sig(ma_on), dropout_sig(mb_on)
    p5_same = da == db

    if not id_pass:
        verdict = "ID_FAIL"
    elif not control_pass:
        verdict = "CONTROL_FAIL"
    elif not p1_same:
        verdict = "PAIRED_FAILS"
    elif n_bad4 == 0 and only_new and new_all_zero and p5_same:
        verdict = "PAIRED_OK"
    else:
        verdict = "PAIRED_NEEDS_CKPT"

    res = {
        "cfg": str(g.cfg), "flag": g.flag, "steps": STEPS, "seed": SEED, "n_data_stub": NDATA,
        "id_check": {"pass": id_pass, "first_diff_step": id_first},
        "P1_reseed_on": {"first_diff_step": on_first, "n_coin_diff": on_coin,
                         "n_idx_diff": on_idx, "identical": p1_same},
        "P2_reseed_off": {"first_diff_step": off_first, "n_coin_diff": off_coin,
                          "n_idx_diff": off_idx, "diverges": control_pass},
        "P3_params_at_build": {"n_shared": n_shared3, "n_differing": n_bad3, "examples": names3},
        "P4_after_init_from": {"n_shared": n_shared4, "n_differing": n_bad4, "examples": names4,
                               "missing_keys": list(missing), "unexpected_keys": list(unexpected),
                               "new_module_all_zero": new_all_zero},
        "P5_dropout": {"a": da, "b": db, "same": p5_same},
        "verdict": verdict,
    }
    g.out.parent.mkdir(parents=True, exist_ok=True)
    g.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"(ID) A vs A 分叉步={id_first} -> {'PASS' if id_pass else 'FAIL'}")
    print(f"(P2) reseed 关：第一处分叉步={off_first}，{STEPS} 步里硬币不同 {off_coin}、下标不同 {off_idx}")
    print(f"(P1) reseed 开：第一处分叉步={on_first}，硬币不同 {on_coin}、下标不同 {on_idx}")
    print(f"(P3) 构造时共享权重 {n_shared3} 个，不同 {n_bad3} 个，例：{names3}")
    print(f"(P4) 载入 A 的权重后共享 {n_shared4} 个，不同 {n_bad4} 个；缺键 {list(missing)}；"
          f"多键 {list(unexpected)}；新模块全零={new_all_zero}")
    print(f"(P5) dropout A={da[:3]}...({len(da)} 个) B={db[:3]}...({len(db)} 个) same={p5_same}")
    print(f"判决：{verdict}")
    print(f"写入 {g.out}")


if __name__ == "__main__":
    main()
