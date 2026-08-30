"""结构先验：从训练数据估计布局，作为种子交给模型填细节。

为什么需要它（证据链见 docs/plan.md 的 A3g–A3l）：

- A3g 多样本重验确认：砖块类是唯一在多样本下稳固的失败点。
- A3h 判别实验：给模型看砖块上面 8–12 行，它能正确延续砖行。
  **所以是生成问题不是表示问题**——模型知道砖长什么样，
  只是在空白网格上建立不起全局结构。
  这与掩码率曲线（模型优势从 +0.246 崩到 +0.032）互为定性/定量印证。
- 于是：把全局布局交给先验，细节交给模型。

**能力边界（实测，不是猜测）**：

1. 只覆盖**周期性**结构。191 个材质里 42% 检出周期（砖、木板等），
   57% 无周期——后者正确地不加种子，因为模型本来就做得好。
2. 只能捕捉**在作者之间一致**的结构。横缝位置跨作者高度一致；
   竖缝相位常常不一致（`default_brick` 就是如此），
   跨作者平均后互相抵消，检不出来。
3. 周期先验不覆盖非周期的全局结构。为此另加**边框先验**
   （`learn_border`）：191 个材质里 33 个（17%）有显著且方向一致的边框，
   其中包括周期先验漏掉的 `tin_block`。
   但它也有盲点——`carts_cart_side` 视觉上有边框，
   上下边一亮一暗相互抵消，均值差只有 +0.01、方向一致率 50%，检不出来。
"""

from __future__ import annotations

import numpy as np


def _periodic_axis(prof: np.ndarray, min_std: float = 0.15,
                   min_score: float = 0.25) -> tuple[list[int], float]:
    """从暗度廓线估周期与相位。

    用自相关而不是固定阈值：砖层本质是周期结构。
    无周期时返回空列表——这是刻意的，颗粒材质不该被硬塞结构。
    """
    p = prof - prof.mean()
    if p.std() < min_std:
        return [], 0.0
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag, score = max(((l, ac[l]) for l in range(3, 9)), key=lambda t: t[1])
    if score < min_score:
        return [], float(score)
    phase = int(np.argmin([p[i::lag].mean() for i in range(lag)]))
    return sorted(set(range(phase, len(prof), lag))), float(score)


def learn_prior(tiles: list[np.ndarray], palette_sizes: list[int],
                size: int = 16, joint_thresh: float = -0.8) -> dict:
    """从同一材质的多个作者版本估计布局先验。

    tiles: 各版本的索引图（调色板按亮度排序，档号越小越暗）
    返回 rows（横缝行）、joints（逐层竖缝）、seam（砖缝档位比例）、score。

    **竖缝逐层单独估**，不用贯通竖线——错缝砌法相邻层竖缝错开半块，
    贯通竖线会把它切成网格（A3j 的失败原因）。
    """
    if not tiles:
        return {"rows": [], "joints": {}, "seam": 0.0, "score": 0.0}

    n = len(tiles)
    prof = np.zeros(size)
    quant = []
    for ix, nk in zip(tiles, palette_sizes):
        a = ix.astype(float)
        prof += (a.mean(1) - a.mean()) / (a.std() + 1e-9)
        quant.append(np.percentile(a, 15) / max(nk - 1, 1))
    rows, score = _periodic_axis(prof / n)

    joints: dict[int, tuple[int, int, list[int]]] = {}
    if rows:
        bands, prev = [], 0
        for r in rows + ([size] if rows[-1] != size - 1 else []):
            if r > prev:
                bands.append((prev, r))
            prev = r + 1
        for bi, (y0, y1) in enumerate(bands):
            col = np.zeros(size)
            for ix in tiles:
                seg = ix[y0:y1].astype(float)
                if seg.size == 0 or seg.std() < 1e-9:
                    continue
                col += (seg.mean(0) - seg.mean()) / (seg.std() + 1e-9)
            col /= n
            joints[bi] = (y0, y1, [c for c in range(size) if col[c] < joint_thresh])

    return {"rows": rows, "joints": joints,
            "seam": float(np.median(quant)), "score": score}


def make_seed(prior: dict, n_colors: int, size: int = 16) -> np.ndarray:
    """把先验变成种子网格。-1 表示留给模型填。"""
    seed = np.full((size, size), -1, int)
    if not prior["rows"]:
        return seed                      # 无周期结构 -> 不加种子
    idx = int(round(prior["seam"] * (n_colors - 1)))
    for r in prior["rows"]:
        seed[r, :] = idx
    for _, (y0, y1, cols) in prior["joints"].items():
        for c in cols:
            seed[y0:y1, c] = idx
    return seed


def fill_from_seed(net, seed: np.ndarray, palette: np.ndarray, n_colors: int,
                   material_id, steps: int = 64, device: str = "cuda",
                   seed_rng: int | None = None, temperature: float = 1.3):
    """按种子填充其余格子。随机顺序解掩码——
    置信度顺序在近均匀分布下会塌成纯色（D5 实测 0.97 vs 真人 0.293）。

    **temperature 默认 1.3 而非 1.0**：种子里大量同色会把模型往
    "继续同一个色"上拖，导致砖面用色数从 12 掉到 9、面内同色占比
    0.224 升到 0.346（p=7e-8）。T=1.3 把这两项拉回真人水平
    （13 色 / 0.232，真人 13 / 0.209），
    全局平坦度距真人也从 0.131 降到 0.035（基线是 0.207）。
    T≥1.6 会开始破坏结构。
    """
    import torch

    size = seed.shape[0]
    idx = torch.full((1, size, size), net.MASK, dtype=torch.long, device=device)
    given = torch.tensor(seed >= 0, device=device)
    if given.any():
        idx[0][given] = torch.tensor(seed[seed >= 0], device=device)

    K = net.k
    pal = torch.zeros(1, K, 3, device=device)
    pal[0, :n_colors] = torch.tensor(palette.astype(np.float32) / 255.0, device=device)
    valid = torch.zeros(1, K, device=device)
    valid[0, :n_colors] = 1.0
    mid = torch.as_tensor([material_id], device=device)

    cells = torch.nonzero((~given).view(-1)).squeeze(-1)
    if seed_rng is not None:
        torch.manual_seed(seed_rng)
    perm = cells[torch.randperm(len(cells), device=device)]
    for chunk in torch.chunk(perm, max(1, min(steps, len(perm)))):
        logits = net(idx, pal, valid, mid)
        logits = logits.masked_fill(
            (valid < 0.5)[:, None, :].expand(-1, size * size, -1), float("-inf"))
        prob = (logits[0] / max(temperature, 1e-6)).softmax(-1)
        samp = torch.multinomial(prob, 1).squeeze(-1)
        idx[0].view(-1)[chunk] = samp[chunk]
    return idx[0].cpu().numpy()


def learn_border(tiles: list[np.ndarray], min_gap: float = 0.5,
                 min_consist: float = 0.8) -> dict:
    """检测最外一圈相对内部是否一致地更暗或更亮。

    周期先验只处理平移对称的结构，边框不是那一类。
    要求**跨作者方向一致**（而不只是幅度大）——
    单个作者画了边框不代表该材质有边框。

    盲点：上下边一亮一暗的材质会互相抵消（`carts_cart_side` 就是如此），
    这里用整圈均值，检不出那种情况。
    """
    diffs = []
    for ix in tiles:
        a = ix.astype(float)
        if a.std() < 1e-9:
            continue
        m = np.zeros(a.shape, bool)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
        diffs.append((a[m].mean() - a[~m].mean()) / a.std())
    if len(diffs) < 4:
        return {"has_border": False, "gap": 0.0, "consist": 0.0, "level": 0.0}
    d = np.array(diffs)
    consist = float(max((d > 0).mean(), (d < 0).mean()))
    gap = float(np.median(d))
    if abs(gap) < min_gap or consist < min_consist:
        return {"has_border": False, "gap": gap, "consist": consist, "level": 0.0}
    # 边框档位：取各版本边框像素的中位档，归一化
    lv = []
    for ix in tiles:
        a = ix.astype(float)
        m = np.zeros(a.shape, bool)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
        lv.append(np.median(a[m]) / max(a.max(), 1))
    return {"has_border": True, "gap": gap, "consist": consist,
            "level": float(np.median(lv))}


def add_border(seed: np.ndarray, border: dict, n_colors: int) -> np.ndarray:
    """把边框先验叠加到种子上。不覆盖已有的周期种子。"""
    if not border.get("has_border"):
        return seed
    idx = int(round(border["level"] * (n_colors - 1)))
    idx = max(0, min(n_colors - 1, idx))
    out = seed.copy()
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        blank = out[sl] < 0
        out[sl] = np.where(blank, idx, out[sl])
    return out
