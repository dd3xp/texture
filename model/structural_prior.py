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


_BOND_SELECT: dict | None = None


def load_bond_select(path="data/tiles/bond_select.json") -> dict:
    """读逐材质的错缝选择结果（`bond_select.py` 产出）。缺文件就返回空。"""
    global _BOND_SELECT
    if _BOND_SELECT is None:
        import json
        import pathlib
        f = pathlib.Path(path)
        _BOND_SELECT = json.loads(f.read_text()) if f.exists() else {}
    return _BOND_SELECT


def learn_prior(tiles: list[np.ndarray], palette_sizes: list[int],
                size: int = 16, joint_thresh: float = -0.8,
                material: str | None = None,
                min_seam_gap: float = 0.08) -> dict:
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

    # 缝的档位：**在检出的缝行上实测**，不用全图 15 百分位。
    # 那个常数把缝画成了近乎黑线：66 个有横缝的材质里，
    # 常数给 0.133 而真人实测是 0.400，56 个材质被画得比真人更暗；
    # 真人的缝只比面暗 0.146，常数造成的落差约 0.40，接近三倍
    # （`analysis/structure_grain/seam_level.py`）。
    # 仍然逐材质学而不是换一个常数——`default_bookshelf`(0.13)、
    # `default_desert_stone_brick`(0.07) 是真的深缝。
    seam_p15 = float(np.median(quant))
    seam, seam_gap = seam_p15, 0.0
    if rows:
        obs, face = [], []
        m_ = np.zeros(size, bool)
        m_[[r for r in rows if r < size]] = True
        for ix, nk in zip(tiles, palette_sizes):
            a = ix.astype(float) / max(nk - 1, 1)
            obs.append(np.median(a[m_]))
            face.append(np.median(a[~m_]))
        seam = float(np.median(obs))
        seam_gap = float(np.median(face) - np.median(obs))
        # 缝若不比面暗，那检出的"行"就不是暗缝——不该画。
        # `nc_woodwork_plank` 实测缝档位 0.43、面 0.43，
        # 硬画上去反而把纹理抹平（`experiments/seam_vs.png`）。
        if seam_gap < min_seam_gap:
            rows, joints = [], {}

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

    bond = learn_bond(tiles, rows, size)
    if bond.get("active") and material is not None:
        # 选择结果里明确判负的材质，退回平均法
        bond["active"] = bool(load_bond_select().get(material, {}).get("use_bond", False))

    return {"rows": rows, "joints": joints, "seam": seam, "seam_p15": seam_p15,
            "seam_gap": seam_gap, "score": score, "bond": bond}


def _col_period(seg: np.ndarray, min_std: float = 0.15,
                min_score: float = 0.20) -> tuple[int | None, int | None]:
    """单个砖层内的竖缝周期与相位。"""
    c = seg.mean(0)
    c = c - c.mean()
    if c.std() < min_std:
        return None, None
    ac = np.correlate(c, c, "full")[len(c) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lag, sc = max(((l, ac[l]) for l in range(3, 9)), key=lambda t: t[1])
    if sc < min_score:
        return None, None
    return lag, int(np.argmin([c[i::lag].mean() for i in range(lag)]))


def learn_bond(tiles: list[np.ndarray], rows: list[int], size: int = 16,
               min_detect: float = 0.5, min_period: float = 0.6,
               max_phase: float = 0.6, select: dict | None = None) -> dict:
    """错缝砌法：估竖缝周期，判断相位是否需要生成时采样。

    A3l 把"竖缝检不出"归给"相位跨作者不一致"，但那个诊断
    把周期和相位混在一起了。分开看（`analysis/structure_grain/joint_phase.py`）：

    `default_brick` 逐作者检出 16–21/28，**周期压倒性是 8**（各层 62–89%），
    **相位却散在 0,1,2,3,6,7**（众数仅 33–53%）。
    跨作者平均之所以抹平竖缝，是因为每个作者选的全局相位不同。

    进一步看同一作者相邻层的相位差：44 对里 27 对正好是 4（半个周期），
    61% 落在 3–5，只有 20% 接近 0——**是错缝，不是每层独立随机**。

    所以：周期从数据估（材质属性），相位在生成时采一个全局值
    （作者的自由选择），相邻层交替偏移半周期。

    先按证据筛出候选：检出率 ≥50%、周期众数 ≥60%、相位众数 <60%。
    这一步选出 10 个材质，但错缝法只在其中 4 个上胜过平均法
    （`bond_eval.py`），所以**候选不等于该启用**。

    试过两个数据侧的门，都不预测胜负，都已撤掉：
    - 「平均法找到的竖缝根数不足应有的一半」：只砍掉赢家 vessels_shelf
      （0.06→0.28），六个输家一个没拦住。
    - 「跨作者平均的列廓线还能否检出周期」：`default_brick` 检出 0.80
      却该赢，`wool_blue` 检出 0.00 却该输，方向都不对。

    所以改成**逐材质离线选择**（`bond_select.py`）：两种种子各生成若干张，
    量竖缝可检出率，取更接近训练瓦片真人均值的那个。
    只用训练数据，不碰测试集，也不需要人工标注。
    `select` 缺省时退回候选判据（即不选择，全部启用）。
    """
    from collections import Counter
    if not rows or len(tiles) < 4:
        return {"active": False}
    bands = []
    prev = 0
    for r in rows + ([size] if rows[-1] != size - 1 else []):
        if r > prev:
            bands.append((prev, r))
        prev = r + 1
    if len(bands) < 2:
        return {"active": False}

    pers, phs, n_try = [], [], 0
    for t in tiles:
        for y0, y1 in bands:
            n_try += 1
            p, ph = _col_period(t[y0:y1].astype(float))
            if p is not None:
                pers.append(p)
                phs.append(ph)
    if not pers or len(pers) / n_try < min_detect:
        return {"active": False, "detect": len(pers) / max(n_try, 1)}
    period, cnt = Counter(pers).most_common(1)[0]
    f_per = cnt / len(pers)
    f_ph = Counter([p for p, q in zip(phs, pers) if q == period]).most_common(1)[0][1]         / max(cnt, 1)
    if f_per < min_period or f_ph >= max_phase:
        return {"active": False, "period": period,
                "f_period": f_per, "f_phase": f_ph,
                "detect": len(pers) / n_try}
    return {"active": True, "period": int(period), "offset": int(period) // 2,
            "f_period": f_per, "f_phase": f_ph, "detect": len(pers) / n_try}


def make_seed(prior: dict, n_colors: int, size: int = 16, rng=None) -> np.ndarray:
    """把先验变成种子网格。-1 表示留给模型填。

    `rng` 只在错缝先验启用时用到：竖缝相位是作者的自由选择，
    每次生成采一个全局值，相邻层交替偏移半周期。
    不传 rng 就退回平均法（可复现，也是 A3n 之前的行为）。
    """
    seed = np.full((size, size), -1, int)
    if not prior["rows"]:
        return seed                      # 无周期结构 -> 不加种子
    idx = int(round(prior["seam"] * (n_colors - 1)))
    for r in prior["rows"]:
        seed[r, :] = idx

    bond = prior.get("bond", {})
    if bond.get("active") and rng is not None:
        per, off = bond["period"], bond["offset"]
        base = int(rng.integers(per)) if hasattr(rng, "integers") else rng.randrange(per)
        for bi, (y0, y1, _) in enumerate(
                (v for _, v in sorted(prior["joints"].items()))):
            ph = (base + off * bi) % per
            for c in range(ph, size, per):
                seed[y0:y1, c] = idx
        return seed

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

    **整圈版本已弃用**，保留是为了复现 A3n 的数字。
    盲点：上下边一亮一暗的材质会互相抵消（`carts_cart_side` 差值仅 +0.01）。
    生产用 `learn_edges`，它四条边分开判。
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


EDGES = ("top", "bottom", "left", "right")


def _edge_mask(shape, which: str) -> np.ndarray:
    m = np.zeros(shape, bool)
    if which == "top":
        m[0, :] = True
    elif which == "bottom":
        m[-1, :] = True
    elif which == "left":
        m[:, 0] = True
    else:
        m[:, -1] = True
    return m


def learn_edges(tiles: list[np.ndarray], min_gap: float = 0.5,
                min_consist: float = 0.8) -> dict:
    """四条边分开检测，取代整圈均值。

    整圈均值的盲点是上下边方向相反时互相抵消——
    `carts_cart_side` 视觉上有明显边框，整圈差值却只有 +0.01、一致率 50%。
    分边之后每条边各自判断方向与一致性，互不干扰。
    """
    out: dict[str, dict] = {}
    for e in EDGES:
        vals = []
        for ix in tiles:
            a = ix.astype(float)
            if a.std() < 1e-9:
                continue
            m = _edge_mask(a.shape, e)
            vals.append((a[m].mean() - a[~m].mean()) / a.std())
        if len(vals) < 4:
            out[e] = {"active": False, "gap": 0.0, "consist": 0.0, "level": 0.0}
            continue
        d = np.array(vals)
        consist = float(max((d > 0).mean(), (d < 0).mean()))
        gap = float(np.median(d))
        active = abs(gap) >= min_gap and consist >= min_consist
        lvl = 0.0
        if active:
            lv = []
            for ix in tiles:
                a = ix.astype(float)
                m = _edge_mask(a.shape, e)
                lv.append(np.median(a[m]) / max(a.max(), 1))
            lvl = float(np.median(lv))
        out[e] = {"active": active, "gap": gap, "consist": consist, "level": lvl}
    return out


def add_edges(seed: np.ndarray, edges: dict, n_colors: int) -> np.ndarray:
    """把分边先验叠加到种子上。不覆盖已有的周期种子。"""
    out = seed.copy()
    for e in EDGES:
        info = edges.get(e, {})
        if not info.get("active"):
            continue
        idx = max(0, min(n_colors - 1, int(round(info["level"] * (n_colors - 1)))))
        m = _edge_mask(out.shape, e)
        out[m & (out < 0)] = idx
    return out


def add_border_union(seed: np.ndarray, border: dict, edges: dict,
                     n_colors: int) -> np.ndarray:
    """整圈与分边两种检测取并集。

    实测两者互补而非替代（`experiments/edge_vs_ring.png`）：
    - `carts_cart_side` 上下边方向相反，整圈抵消（+0.01/50%）检不出，
      分边能抓到（top +1.31/88%、bottom −1.37/94%）。
    - `tin_block` 整圈过关（−1.23/84%）能给出完整边框，
      但拆开后只有 bottom 单独达标（四条边各自的一致率不如整体），
      只用分边反而丢了三条边。

    所以先叠整圈（若命中），再叠分边补漏，都不覆盖已有的周期种子。
    """
    out = add_border(seed, border, n_colors)
    return add_edges(out, edges, n_colors)
