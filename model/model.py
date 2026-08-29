"""调色板索引空间上的掩码预测模型。

为什么是这个设计：

- **输出是索引不是 RGB。** 调色板合规、零抗锯齿、硬边由构造保证，
  不需要事后量化。这是整条技术路线的核心。
- **掩码预测（MaskGIT 风格）而非自回归。** 纹理没有天然的扫描顺序，
  自回归的光栅顺序会引入不该有的方向性偏置。
  掩码预测还顺带白送了区域约束：区域外的格子标成 BG 且永不解掩码。
- **调色板作为条件 token 输入。** 同一个索引在不同调色板下含义不同，
  模型必须看到调色板本身，否则学到的是"第 3 档"而不是"这个颜色"。

词表 = K 个调色板索引 + MASK + BG。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int, drop: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class PixelTextureModel(nn.Module):
    """在 N×N 调色板索引上做掩码预测。

    序列 = [材质 token] + [K 个调色板 token] + [N*N 个格子 token]
    只在格子位置上算损失。
    """

    def __init__(self, k: int = 16, n_materials: int = 521, size: int = 16,
                 d: int = 256, depth: int = 6, heads: int = 8, drop: float = 0.1):
        super().__init__()
        self.k, self.size = k, size
        self.MASK, self.BG = k, k + 1
        vocab = k + 2

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, size * size, d))
        nn.init.normal_(self.pos, std=0.02)

        self.mat = nn.Embedding(n_materials, d)
        # 调色板每一档编码成一个 token：RGB + 该档的序号
        self.pal_in = nn.Linear(3, d)
        self.pal_pos = nn.Parameter(torch.zeros(1, k, d))
        nn.init.normal_(self.pal_pos, std=0.02)
        # 无效档（实际用色少于 K）用一个可学习的占位向量，避免混入零向量
        self.pal_null = nn.Parameter(torch.zeros(d))

        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, k)   # 只预测真实调色板档位，不预测 MASK/BG

    def forward(self, idx, palette, pal_valid, material):
        B, N, _ = idx.shape
        g = self.tok(idx.view(B, -1)) + self.pos

        p = self.pal_in(palette) + self.pal_pos
        p = torch.where(pal_valid[..., None] > 0, p, self.pal_null.expand_as(p))
        m = self.mat(material)[:, None]

        x = torch.cat([m, p, g], 1)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(x[:, 1 + self.k:])          # (B, N*N, K)


class FiLM(nn.Module):
    """把条件向量变成逐通道的缩放和偏移。

    卷积架构没有 token 序列可以拼接条件，所以用 FiLM 注入——
    这是条件卷积网络的标准做法，比把条件铺成额外通道更省参数。
    """

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * ch)

    def forward(self, x, c):
        g, b = self.to_gb(c).chunk(2, -1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class ConvBlock(nn.Module):
    """残差卷积块。3x3 卷积提供局部性先验——
    这正是纯 transformer 缺的东西：它不知道"相邻格子"意味着什么。
    """

    def __init__(self, ch: int, cond_dim: int, drop: float):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = FiLM(cond_dim, ch)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x, c):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.film(h, c)
        h = self.c2(self.drop(F.gelu(self.n2(h))))
        return x + h


class ConvTextureModel(nn.Module):
    """卷积版掩码预测。接口与 PixelTextureModel 完全一致，
    所以 training_step 和 generate 都不用改。

    与 transformer 版的唯一区别是**空间归纳偏置**：
    3x3 卷积天然知道邻域，而 transformer 的位置信息全靠可学习嵌入。
    砖缝、板条这类布局依赖局部规律，这是这次改架构要验证的假设。
    """

    def __init__(self, k: int = 16, n_materials: int = 521, size: int = 16,
                 d: int = 128, depth: int = 6, heads: int = 8, drop: float = 0.1):
        super().__init__()
        self.k, self.size = k, size
        self.MASK, self.BG = k, k + 1

        self.tok = nn.Embedding(k + 2, d)
        self.pos = nn.Parameter(torch.zeros(1, d, size, size))
        nn.init.normal_(self.pos, std=0.02)

        cond = d
        self.mat = nn.Embedding(n_materials, cond)
        self.pal_in = nn.Linear(3, cond)
        self.pal_null = nn.Parameter(torch.zeros(cond))
        self.cond_mix = nn.Sequential(nn.Linear(2 * cond, cond), nn.GELU(),
                                      nn.Linear(cond, cond))

        self.blocks = nn.ModuleList([ConvBlock(d, cond, drop) for _ in range(depth)])
        self.out_norm = nn.GroupNorm(8, d)
        self.head = nn.Conv2d(d, k, 1)

    def forward(self, idx, palette, pal_valid, material):
        B, N, _ = idx.shape
        x = self.tok(idx).permute(0, 3, 1, 2) + self.pos

        p = self.pal_in(palette)
        p = torch.where(pal_valid[..., None] > 0, p, self.pal_null.expand_as(p))
        # 调色板按有效档求均值，得到一个"这套配色长什么样"的向量
        p = (p * pal_valid[..., None]).sum(1) / pal_valid.sum(1, keepdim=True).clamp(min=1)
        c = self.cond_mix(torch.cat([self.mat(material), p], -1))

        for b in self.blocks:
            x = b(x, c)
        return self.head(F.gelu(self.out_norm(x))).flatten(2).transpose(1, 2)


class SpatialAttn(nn.Module):
    """在 16x16 展平成的 256 个位置上做一次全局自注意力。

    A3 发现 depth=4 的卷积感受野约 9x9，小于整块瓦片，
    砖块布局、条纹分行这些**全局**结构学不到。
    注意力一步覆盖全图，正好补上卷积缺的那部分。
    A3b 证实：加深卷积（depth8，感受野 17x17）无效，加注意力有效——
    缺的是显式全局交互，不是感受野尺寸。

    **条件 token**：A3b 的画廊显示模型"学会了条纹但方向错了"
    （sandstone 出竖纹，真人是横向砖行）。原因是这一层是纯空间自注意力，
    **决定布局时看不到自己在画什么材质**。
    把条件向量作为一个额外的 key/value token 接进来，
    每个位置就能在决定布局时查询材质信息。
    """

    def __init__(self, ch: int, heads: int, drop: float):
        super().__init__()
        # 通道数不一定被 heads 整除（换宽度时很容易踩），取一个能整除的最大值
        heads = max((h for h in range(1, heads + 1) if ch % h == 0), default=1)
        self.norm = nn.GroupNorm(8, ch)
        self.cond_proj = nn.Linear(ch, ch)
        self.attn = nn.MultiheadAttention(ch, heads, dropout=drop, batch_first=True)

    def forward(self, x, c=None):
        B, C, H, W = x.shape
        q = self.norm(x).flatten(2).transpose(1, 2)
        kv = q if c is None else torch.cat([self.cond_proj(c)[:, None], q], 1)
        h, _ = self.attn(q, kv, kv, need_weights=False)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class HybridTextureModel(ConvTextureModel):
    """卷积提供局部性，注意力提供全局性。

    卷积负责"相邻像素怎么配"，注意力负责"整块的布局是什么"。
    其余（条件注入、输出头、接口）全部继承 ConvTextureModel 不变。

    `attn_every=1` 表示每个卷积块后都插注意力。A3b 用的是 2，
    而对照显示全局交互是有效杠杆、堆深度无效，所以往有效方向加密。
    """

    def __init__(self, k=16, n_materials=521, size=16, d=96, depth=4,
                 heads=6, drop=0.1, attn_every=1):
        super().__init__(k=k, n_materials=n_materials, size=size, d=d,
                         depth=depth, heads=heads, drop=drop)
        self.attns = nn.ModuleList([
            SpatialAttn(d, heads, drop) if (i + 1) % attn_every == 0
            else nn.Identity() for i in range(depth)])

    def forward(self, idx, palette, pal_valid, material):
        B, N, _ = idx.shape
        x = self.tok(idx).permute(0, 3, 1, 2) + self.pos

        p = self.pal_in(palette)
        p = torch.where(pal_valid[..., None] > 0, p, self.pal_null.expand_as(p))
        p = (p * pal_valid[..., None]).sum(1) / pal_valid.sum(1, keepdim=True).clamp(min=1)
        c = self.cond_mix(torch.cat([self.mat(material), p], -1))

        for blk, at in zip(self.blocks, self.attns):
            x = blk(x, c)
            x = at(x, c) if isinstance(at, SpatialAttn) else at(x)
        return self.head(F.gelu(self.out_norm(x))).flatten(2).transpose(1, 2)


def build_model(arch: str, **kw):
    """按名字构建。两种架构接口一致，可直接互换做对照。"""
    if arch == "conv":
        return ConvTextureModel(**kw)
    if arch == "hybrid":
        return HybridTextureModel(**kw)
    if arch == "transformer":
        return PixelTextureModel(**kw)
    raise ValueError(f"未知架构: {arch}")


def mask_schedule(t: torch.Tensor) -> torch.Tensor:
    """余弦掩码率：t=0 全掩码，t=1 几乎不掩码。MaskGIT 的标准做法。"""
    return torch.cos(t * math.pi / 2).clamp(1e-3, 1.0)


def training_step(model, batch, device) -> tuple[torch.Tensor, dict]:
    idx = batch["idx"].to(device)
    pal = batch["palette"].to(device)
    val = batch["pal_valid"].to(device)
    mat = batch["material"].to(device)
    B, N, _ = idx.shape

    t = torch.rand(B, device=device)
    rate = mask_schedule(t)[:, None, None]
    m = torch.rand(B, N, N, device=device) < rate
    m[:, 0, 0] |= ~m.view(B, -1).any(1)              # 保证每个样本至少掩一格

    inp = torch.where(m, torch.full_like(idx, model.MASK), idx)
    logits = model(inp, pal, val, mat)

    # 无效档（超出该样本实际用色）不该被预测到
    invalid = (val < 0.5)[:, None, :].expand(-1, N * N, -1)
    logits = logits.masked_fill(invalid, float("-inf"))

    tgt = idx.view(B, -1)
    sel = m.view(B, -1)
    loss = F.cross_entropy(logits[sel], tgt[sel])
    with torch.no_grad():
        acc = (logits[sel].argmax(-1) == tgt[sel]).float().mean()
    return loss, {"loss": loss.item(), "acc": acc.item(),
                  "mask_frac": sel.float().mean().item()}


@torch.no_grad()
def generate(model, palette, pal_valid, material, size, steps=16,
             region=None, temperature=1.0, order="random", device="cuda"):
    """迭代解掩码采样。

    order="random"（默认）：按随机顺序分批解掩码。
    order="confidence"：MaskGIT 原版的按置信度优先。**实测会塌成纯色**——
    模型在全掩码时的边缘分布近乎均匀（熵占上限 0.99），
    此时"最高置信度"几乎是任意的，填下去又被模型条件强化，形成正反馈；
    步数越多越严重（8 步最常见单色占比 0.971，32 步 1.000）。
    随机顺序没有这个偏置：同样的模型，最常见单色占比 0.246，
    而真人原生是 0.293。保留该选项只为复现这个对比。

    region 为 None 时生成整幅；否则区域外填 BG 且永不解掩码。
    """
    B = palette.shape[0]
    idx = torch.full((B, size, size), model.MASK, dtype=torch.long, device=device)
    if region is None:
        region = torch.ones(B, size, size, dtype=torch.bool, device=device)
    idx = torch.where(region, idx, torch.full_like(idx, model.BG))

    def logits_now():
        lg = model(idx, palette, pal_valid, material)
        invalid = (pal_valid < 0.5)[:, None, :].expand(-1, size * size, -1)
        return lg.masked_fill(invalid, float("-inf"))

    if order == "random":
        for b in range(B):
            cells = torch.nonzero(region[b].view(-1)).squeeze(-1)
            perm = cells[torch.randperm(len(cells), device=device)]
            for ch in torch.chunk(perm, min(steps, max(len(perm), 1))):
                p = (logits_now()[b] / max(temperature, 1e-6)).softmax(-1)
                samp = torch.multinomial(p, 1).squeeze(-1)
                idx[b].view(-1)[ch] = samp[ch]
        return idx

    todo = region.clone()
    total = todo.view(B, -1).sum(1)
    for s_ in range(steps):
        prob = (logits_now() / max(temperature, 1e-6)).softmax(-1)
        samp = torch.multinomial(prob.view(-1, model.k), 1).view(B, size, size)
        conf = prob.view(B, size, size, -1).gather(-1, samp[..., None])[..., 0]
        conf = torch.where(todo, conf, torch.full_like(conf, -1.0))
        keep_n = (total.float() * (1 - mask_schedule(
            torch.tensor((s_ + 1) / steps, device=device)))).long().clamp(min=1)
        flat = conf.view(B, -1)
        for b in range(B):
            k = int(min(keep_n[b].item(), int(todo[b].sum().item())))
            if k <= 0:
                continue
            pick = flat[b].topk(k).indices
            idx[b].view(-1)[pick] = samp[b].view(-1)[pick]
            todo[b].view(-1)[pick] = False
    if todo.any():
        am = logits_now().argmax(-1).view(B, size, size)
        idx = torch.where(todo, am, idx)
    return idx
