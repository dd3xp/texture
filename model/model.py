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
