#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(P8) 开跑前自检：新开关的默认值必须与旧行为**逐位相同**，四条臂必须真的不同。

⚠ 这是 (M29)/(M30) 换来的纪律：零初始化/默认值"看起来不变"不算数，要逐元素比；
而且**改了默认路径的代码就必须证明默认路径没动**，否则整轮实验的控制臂是假的。

检查（全部 CPU、零 GPU、零数据）：
 (S1) `wrap="torus"` 的偏置表与**旧公式**（折回环面）逐元素 max|Δ| = 0。
 (S2) 四种配置两两不同：torus / none / broken / off。
 (S3) **平移等变**：torus 下，把网格循环平移 d 格，偏置矩阵应当只是对应的行列置换
      （即 bias[i+d, j+d] == bias[i, j]，环面意义下）→ max|Δ| ≈ 0；
      而 none / broken 下这一条**必须被破坏**（否则这两条臂没有区分度，实验白做）。
 (S4) `augment(roll=False)` 不做循环平移；`roll=True` 会做。
 (S5) 参数量：torus / none / broken 三者**完全相同**（只改折回方式，不改结构）。

    python analysis/arch/p8_selftest.py
"""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from trd import ToroidalBias                      # noqa: E402
from train_trd import augment                     # noqa: E402

N, HEADS, FREQS = 8, 2, 8


def legacy_feat(n, freqs):
    """旧公式（本次改动之前的 grid_offsets 主体），逐字抄写用于 (S1)。"""
    ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    ys, xs = ys.flatten().float(), xs.flatten().float()
    dy = (ys[:, None] - ys[None, :]) / n
    dx = (xs[:, None] - xs[None, :]) / n
    dy = (dy + 0.5) % 1.0 - 0.5
    dx = (dx + 0.5) % 1.0 - 0.5
    f = torch.arange(1, freqs + 1).float()
    ay, ax = 2 * math.pi * dy[..., None] * f, 2 * math.pi * dx[..., None] * f
    return torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], -1)


def table(wrap, off=False, seed=0):
    torch.manual_seed(seed)
    b = ToroidalBias(HEADS, hidden=32, freqs=FREQS, wrap=wrap)
    b.off = off
    with torch.no_grad():
        return b.forward(N, "cpu")[:, 16:, 16:].clone(), b


def shift_perm(n, d):
    idx = torch.arange(n * n).view(n, n)
    return torch.roll(idx, shifts=(d, 0), dims=(0, 1)).flatten()


def main():
    ok = True

    # (S1) 默认路径逐位不变
    torch.manual_seed(0)
    b = ToroidalBias(HEADS, hidden=32, freqs=FREQS, wrap="torus")
    with torch.no_grad():
        new = b.mlp(b.grid_offsets(N, "cpu"))
        old = b.mlp(legacy_feat(N, FREQS))
    d1 = (new - old).abs().max().item()
    print(f"(S1) torus 与旧公式 max|Δ| = {d1:.3e}  {'OK' if d1 == 0 else '**不一致**'}")
    ok &= d1 == 0

    # (S2) 四种配置互不相同
    t_torus, _ = table("torus")
    t_none, _ = table("none")
    t_broken, _ = table("broken")
    t_off, _ = table("torus", off=True)
    pairs = {"torus-none": (t_torus - t_none).abs().max().item(),
             "torus-broken": (t_torus - t_broken).abs().max().item(),
             "none-broken": (t_none - t_broken).abs().max().item(),
             "torus-off": (t_torus - t_off).abs().max().item()}
    print("(S2) 两两差异：" + "  ".join(f"{k}={v:.3f}" for k, v in pairs.items()))
    ok &= all(v > 1e-6 for v in pairs.values())
    ok &= float(t_off.abs().max()) == 0.0
    print(f"     off 臂是否整块为零：{float(t_off.abs().max()) == 0.0}")

    # (S3) 平移等变：torus 必须成立，none/broken 必须被破坏
    print("(S3) 循环平移等变（max|Δ|，越小越等变）：")
    for wrap, want in (("torus", True), ("none", False), ("broken", False)):
        t, _ = table(wrap)
        worst = 0.0
        for d in (1, 3):
            perm = shift_perm(N, d)
            worst = max(worst, (t[:, perm][:, :, perm] - t).abs().max().item())
        good = worst < 1e-5
        print(f"     {wrap:<7} {worst:.3e}  等变={good}  期望={want}  {'OK' if good == want else '**不符**'}")
        ok &= good == want

    # (S4) roll 开关
    g = torch.arange(2 * N * N).view(2, N, N) % 7
    torch.manual_seed(0)
    a_on = augment(g.clone(), roll=True)
    torch.manual_seed(0)
    a_off = augment(g.clone(), roll=False)
    rolled = not torch.equal(a_on, a_off)
    same_rows = all(torch.equal(torch.sort(a_off[i].flatten())[0], torch.sort(g[i].flatten())[0]) for i in range(2))
    print(f"(S4) roll=True 与 False 不同：{rolled}；roll=False 只做翻转（元素集合不变）：{same_rows}")
    ok &= rolled and same_rows

    # (S5) 参数量一致
    ns = {w: sum(p.numel() for p in table(w)[1].parameters()) for w in ("torus", "none", "broken")}
    print(f"(S5) 参数量：{ns}  {'OK' if len(set(ns.values())) == 1 else '**不一致**'}")
    ok &= len(set(ns.values())) == 1

    print("\n" + ("全部通过，可以开跑" if ok else "**有检查未过：不许开跑**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
