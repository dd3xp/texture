#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M35) 跑前自检：固定像素周期梳齿支 `--bias_pix 4` 装上去之后，

  (S1) 新列置零时，16/24/32 三档的偏置表与旧路径**逐元素相同**（max|delta| = 0）；
  (S2) 新列非零时，同一个 u=0.375 上 16px(d=6) 与 32px(d=12) 给出**相反符号**的增量
       —— 这正是 (M26) 量到的那个冲突，且**没有任何画布输入**；
  (S3) 新列非零时，三档的循环平移残差与**旧路径**同量级（= 可平铺性没被破坏）；
  (S4) 环面折回点单值：P 整除 n 时，d=+n/2 与 d=-n/2 处特征相同；P 不整除 n 时断言会响。

零 GPU、零数据、纯 CPU。跑法：python analysis/arch/pix_selfcheck.py
打印里只用 ASCII 与常用汉字（非 GBK 字形会让 Windows 控制台崩在写结果之前）。
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from trd import ToroidalBias  # noqa: E402

P_PIX = 4.0
SIZES = (16, 24, 32)
HEADS, HID, FREQS = 6, 128, 8


def build(pix, seed=0):
    torch.manual_seed(seed)
    b = ToroidalBias(HEADS, hidden=HID, freqs=FREQS, pix_periods=pix)
    return b


def copy_old_into_new(old, new):
    """照搬 train_trd.py 的 init_from：旧列照抄、新列置零。"""
    sd = dict(old.state_dict())
    w_old = sd["mlp.0.weight"]
    w = torch.zeros_like(new.state_dict()["mlp.0.weight"])
    assert w.shape[1] > w_old.shape[1], (w.shape, w_old.shape)
    w[:, :w_old.shape[1]] = w_old
    sd["mlp.0.weight"] = w
    missing, unexpected = new.load_state_dict(sd, strict=True)
    return missing, unexpected


def table(bias, n):
    """[n*n, n*n, heads] -> 按偏移 d 取一行：返回 g[0, :] 重排成 [n, n, heads]。"""
    with torch.no_grad():
        g = bias._table(n, torch.device("cpu"))
    return g[0].view(n, n, HEADS)


def main():
    out = {"P": P_PIX, "sizes": list(SIZES)}
    old = build(())
    new = build((P_PIX,))
    copy_old_into_new(old, new)

    # (S1) 零初始化 -> 三档逐元素相同
    s1 = {}
    for n in SIZES:
        d = (table(old, n) - table(new, n)).abs().max().item()
        s1[str(n)] = d
    out["S1_max_abs_diff"] = s1
    s1_ok = all(v == 0.0 for v in s1.values())

    # (S2) 新列非零 -> 同一个 u=0.375 上两档反号（无任何画布输入）
    # 只动新列：让梳齿支贡献 +cos(2pi*w/P)（w=像素偏移），其余权重不变。
    with torch.no_grad():
        w = new.mlp[0].weight
        n_old = old.mlp[0].weight.shape[1]
        w[:, n_old:] = 0.0
        w[:, n_old + 1] = 1.0   # dy 的 cos 那一列
    s2 = {}
    for n, d in ((16, 6), (32, 12)):
        delta = (table(new, n)[d, 0] - table(old, n)[d, 0]).mean().item()
        s2[f"n{n}_d{d}"] = delta
    # 判据 = **两档反号**（(M26) 要的就是这个）。⛔ 不判哪一边正：MLP 有 GELU + 第二层线性，
    # 同一列权重经非线性后落在哪个方向由学出来的权重定，跑前指定方向等于凭空加一条不成立的要求。
    s2_ok = s2["n16_d6"] * s2["n32_d12"] < 0
    out["S2_delta_at_u0.375"] = s2
    out["S2_opposite_sign"] = s2_ok

    # (S3) 循环平移残差：偏置只依赖环绕偏移 -> 沿对角平移整张表应当不变
    def shift_resid(bias, n):
        r = 0.0
        # 检验：把整张偏置矩阵的行与列按同一个循环平移重排，结果必须不变（= 平移等变/可平铺）
        with torch.no_grad():
            full = bias._table(n, torch.device("cpu"))          # [n*n, n*n, H]
        idx = torch.arange(n * n).view(n, n)
        for k in (1, 3, 7):
            p = torch.roll(idx, shifts=(k, k), dims=(0, 1)).flatten()
            r = max(r, (full - full[p][:, p]).abs().max().item())
        return r
    s3 = {str(n): {"new": shift_resid(new, n), "old": shift_resid(old, n)} for n in SIZES}
    out["S3_cyclic_shift_resid"] = s3
    s3_ok = all(v["new"] <= max(1e-4, 50 * v["old"] + 1e-6) for v in s3.values())

    # (S4) 折回点单值：w=+n/2 与 w=-n/2 在环面上是**同一个偏移**，特征必须相同。
    # sin(2pi*w/P) 在 w=+-n/2 相等 <=> sin(pi*n/P)=0 <=> P 整除 n。直接量这个数，
    # ⛔ 不拿 g[n//2] 和 g[n-n//2] 比（偶数 n 下那是同一个元素 = 什么也没量）。
    import math
    s4 = {}
    for n in SIZES:
        s4[f"fold_gap_n{n}_P{int(P_PIX)}"] = abs(2 * math.sin(2 * math.pi * (n / 2) / P_PIX))
        s4[f"fold_gap_n{n}_P5"] = abs(2 * math.sin(2 * math.pi * (n / 2) / 5.0))   # 反对照：5 不整除
    bad = build((5.0,))                       # 5 不整除 16/24/32
    try:
        table(bad, 16)
        s4["assert_fires"] = False
    except AssertionError:
        s4["assert_fires"] = True
    s4_ok = (s4["assert_fires"]
             and all(v < 1e-12 for k, v in s4.items() if k.endswith(f"P{int(P_PIX)}"))
             and all(v > 0.1 for k, v in s4.items() if k.endswith("P5")))
    s4["n_checked"] = len([k for k in s4 if k.startswith("fold_gap")])
    out["S4"] = s4

    # (S5) 参数量差：只多 4 列 * hidden
    out["S5_new_params"] = new.mlp[0].weight.numel() - old.mlp[0].weight.numel()
    s5_ok = out["S5_new_params"] == 4 * HID

    out["pass"] = {"S1": s1_ok, "S2": s2_ok, "S3": s3_ok, "S4": s4_ok, "S5": s5_ok}
    out["all_pass"] = all(out["pass"].values())
    p = Path("/tmp/m35_selfcheck.json") if Path("/tmp").exists() else ROOT / "m35_selfcheck.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(("全部通过" if out["all_pass"] else "有不通过项") + f" -> {p}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
