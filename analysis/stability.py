"""样本量纪律：任何逐材质的定量结论都必须过折半检验。

为什么是一个模块而不是文档里的一条规矩：这条规矩写在 docs 里之后
**同一个会话里仍然翻车两次**——

- A3q：6 样本说错缝法赢、8 样本说它输、40 样本说它赢 7/10。
- 头条：4 样本给出「距真人 0.035、比基线接近 6 倍、p=1.5e-08」，
  12 样本复核是 0.128 / 1.4 倍 / p=0.032。

两次都是"每个材质采几张就下结论"。A3f 定过"画廊必须多样本"，
但那条只被用在看图上，没被用在统计量上，因为它只是一句话。

所以改成函数：报结果必须先拿到 `StabilityReport`，
不稳的时候它会直接把警告印在结果里，绕不过去。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class StabilityReport:
    n_items: int
    n_samples: int
    rho: float                 # 两半之间的相关
    half_gap: float            # 两半点估计的差
    stable: bool
    note: str

    def __str__(self) -> str:
        flag = "稳定" if self.stable else "⚠ 不稳定，不要据此下结论"
        return (f"折半检验：{self.n_items} 项 × {self.n_samples} 样本  "
                f"ρ={self.rho:+.3f}  两半相对差={self.half_gap:.1%}  [{flag}]"
                + (f"\n  {self.note}" if self.note else ""))


def split_half(per_item: list[list[float]], min_samples: int = 12,
               min_rho: float = 0.8, max_rel_gap: float = 0.10,
               rng_seed: int = 0) -> StabilityReport:
    """per_item: 每个材质的若干次采样值。

    把每个材质的样本**随机**分两半各求均值，看两半是否给出同样的图景。
    随机而非前后对半，避免采样种子的顺序效应伪装成稳定。

    三条门：样本数够、两半相关高、两半点估计接近。
    任何一条不过就判不稳定——这时候唯一正确的动作是加样本，不是解释。

    第三条用**相对**差（差 / 量级）而不是绝对差。
    最初写成绝对 0.02，那是照平坦度（量程 0–1）定的；
    换到量程 17–62 的结构描述子距离上，7% 的相对差被误判成不稳定。
    一个只在某一个量纲下成立的阈值本身就是错的。
    """
    ns = [len(v) for v in per_item]
    n = min(ns) if ns else 0
    if not per_item or n < 2:
        return StabilityReport(len(per_item), n, float("nan"), float("nan"),
                               False, "样本不足以折半")
    rng = np.random.default_rng(rng_seed)
    a, b = [], []
    for v in per_item:
        idx = rng.permutation(len(v))
        h = len(v) // 2
        a.append(float(np.mean([v[i] for i in idx[:h]])))
        b.append(float(np.mean([v[i] for i in idx[h:2 * h]])))
    a, b = np.array(a), np.array(b)
    rho = float(stats.spearmanr(a, b).correlation) if len(a) > 2 else float("nan")
    scale = float(np.median(np.abs(np.concatenate([a, b])))) + 1e-9
    gap = float(abs(np.median(a) - np.median(b))) / scale

    fails = []
    if n < min_samples:
        fails.append(f"每项仅 {n} 样本 < {min_samples}")
    if not np.isnan(rho) and rho < min_rho:
        fails.append(f"两半相关 {rho:.2f} < {min_rho}")
    if gap > max_rel_gap:
        fails.append(f"两半相对差 {gap:.1%} > {max_rel_gap:.0%}")
    return StabilityReport(len(per_item), n, rho, gap, not fails,
                           "；".join(fails))


def compare(per_item_a: list[list[float]], per_item_b: list[list[float]],
            target: list[float], name_a="A", name_b="B", **kw) -> None:
    """两种做法各自距目标多远，并对两边都做折半检验后再报显著性。

    显著性只在两边都稳定时才印——不稳的 p 值是本项目撤回过的东西。
    """
    ra, rb = split_half(per_item_a, **kw), split_half(per_item_b, **kw)
    t = np.array(target)
    da = np.abs(np.array([np.mean(v) for v in per_item_a]) - t)
    db = np.abs(np.array([np.mean(v) for v in per_item_b]) - t)
    print(f"{name_a} 距目标中位 {np.median(da):.4f}")
    print(f"{name_b} 距目标中位 {np.median(db):.4f}")
    print(f"  {name_a} {ra}")
    print(f"  {name_b} {rb}")
    if ra.stable and rb.stable:
        p = stats.wilcoxon(da, db).pvalue
        print(f"  Wilcoxon p={p:.3g}  距离比 "
              f"{np.median(da)/max(np.median(db),1e-9):.2f}×")
    else:
        print("  ⚠ 有一边不稳定，不报 p 值——先加样本")
